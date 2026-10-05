"""Measured inference-only batch selection and exact CFG reference-bank reuse."""
import json
import time

import torch


class CFGReferenceCache:
    """Bounded per-reference/timestep cache for one frozen inference model."""
    def __init__(self, original):
        self.original = original
        self.key = None
        self.inputs = None
        self.value = None
        self.snapshots = None
        self.values = {}
        self.hits = self.misses = 0

    def clear(self):
        self.key = self.inputs = self.value = None
        self.snapshots = None
        self.values = {}

    def __call__(self, model, reference, position_ids, sigma, face_mask, maximum):
        if torch.is_grad_enabled() or model.training:
            raise RuntimeError('CFG reference reuse is inference-only')
        inputs = (reference, position_ids, face_mask)
        key = (id(model), tuple((id(t), t._version, tuple(t.shape), t.dtype, t.device) for t in inputs), maximum)
        if key != self.key:
            same_reference = (self.key is not None and self.key[0] == id(model) and self.key[-1] == maximum
                              and all(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device
                                      and torch.equal(a, b) for a, b in zip(inputs, self.snapshots)))
            if not same_reference:
                self.values = {}
                self.snapshots = tuple(t.detach().clone() for t in inputs)
            self.key, self.inputs = key, inputs
        timestep = tuple(sigma.detach().cpu().reshape(-1).tolist())
        if timestep not in self.values:
            # The fixed protocol has20 sigma values. Bound memory even if reused
            # by another schedule, rather than retaining an unbounded history.
            if len(self.values) >= 24:
                self.values.pop(next(iter(self.values)))
            self.values[timestep] = self.original(model, reference, position_ids, sigma, face_mask, maximum)
            self.misses += 1
        else:
            self.hits += 1
        return self.values[timestep]


@torch.no_grad()
def qualify(m, backend, model, run, config, step, cache):
    """Time real paired CFG predictions on a full 12-prompt reference group."""
    from ba_dit.nn import isolated_reference
    rows = m.read_manifest(config['data']['validation_manifest'])
    masks = json.loads((run/'routing_masks.json').read_text())['samples']
    # Largest cached reference layout gives a conservative inference workload.
    groups = [rows[i:i+12] for i in range(0, len(rows), 12)]
    group = max(groups, key=lambda g: m.load_pair(config, g[0], 'cpu', negative=True)[0]['reference_tokens'].shape[1])
    ref = m.cache_path(config, group[0], 'vae')
    assert len(group) == 12 and all(m.cache_path(config, r, 'vae') == ref for r in group)
    loaded = [m.load_pair(config, row, 'cuda', negative=True)[0] for row in group]
    height, width = config['data']['target_size']
    native = torch.cat([m.load_file(run/'native'/f"{r['sample_id']}.safetensors")['latent'] for r in group]).cuda()
    noise = torch.cat([torch.randn(native[i:i+1].shape, generator=torch.Generator().manual_seed(row['seed']),
                                 dtype=native.dtype) for i, row in enumerate(group)]).cuda()
    alpha = torch.cat([m.routing_token_alpha(m.face_alpha((width, height), masks[r['sample_id']]['face_bbox'],
                                 config['branch']['mask_feather_pixels']), config) for r in group]).cuda()
    sigma = noise.new_tensor([.5])
    noisy = m.scene_latent(noise, native, noise, .5, alpha)
    results, baseline = [], None
    original = cache.original
    prior_path = run/'validation_execution_policy.json'
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    cases = ((2, False, False), (2, True, False), (2, True, True)) if prior else (
        (2, False, False), (2, True, False), (6, True, False), (12, True, False), (2, True, True))
    for size, reuse, warm_reference in cases:
        tensors = {k: loaded[0][k] if k == 'reference_mask' else torch.cat([r[k] for r in loaded[:size]])
                   for k in loaded[0]}
        tensors['target_face_mask'] = alpha[:size].flatten(1)
        isolated_reference.reference_bank = cache if reuse else original
        cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # One warm-up and one measurement, each with a cold positive bank.
        for iteration in range(2):
            if not warm_reference or iteration == 0:
                cache.clear()
            torch.cuda.synchronize(); started = time.monotonic()
            positive = backend.predict(model, tensors, noisy[:size], sigma, config, True)
            negative = backend.predict(model, tensors, noisy[:size], sigma, config, True, negative=True)
            torch.cuda.synchronize(); seconds = time.monotonic()-started
        pair = (positive[:2].detach().clone(), negative[:2].detach().clone())
        if baseline is None:
            baseline = pair
        differences = [(a.float()-b.float()) for a, b in zip(pair, baseline)]
        exact = all(torch.equal(a, b) for a, b in zip(pair, baseline))
        relative_rms = max(float(d.square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-8))
                           for d, b in zip(differences, baseline))
        if size == 2:
            assert exact, 'Reference cache changed the frozen prediction'
        reserved = torch.cuda.max_memory_reserved()
        result = {'batch_size': size, 'reference_cache': reuse, 'cfg_pair_seconds': seconds,
                  'reference_already_cached': warm_reference,
                  'seconds_per_image': seconds/size, 'first_two_outputs_exact': exact,
                  'max_abs_difference': max(float(d.abs().max()) for d in differences),
                  'relative_rms_difference': relative_rms,
                  'peak_reserved_gib': reserved/2**30,
                  'reserved_fraction': reserved/torch.cuda.get_device_properties(0).total_memory}
        results.append(result)
        print('Validation throughput probe: '+json.dumps(result), flush=True)
        del positive, negative, tensors
    isolated_reference.reference_bank = cache
    cache.clear()
    # Batch reshaping may select a different BF16 GEMM kernel. Bound and record
    # that numerical variation; exact same-size CFG reuse is mandatory above.
    accepted = [r for r in results if r['reference_cache'] and r['reserved_fraction'] < .80
                and r['relative_rms_difference'] < .005]
    assert accepted, 'No safe qualified validation configuration'
    best = min(accepted, key=lambda r: r['seconds_per_image'])
    policy = {'batch_size': best['batch_size'], 'reference_cache': True, 'qualified_checkpoint': step,
              'cache_scope': 'all20 timesteps across identical same-reference prompt batches',
              'probe_samples': [r['sample_id'] for r in group], 'measurements': results,
              'predicted_denoising_speedup': results[0]['seconds_per_image']/best['seconds_per_image'],
              'scope': 'inference only; unchanged prompts/seeds/masks/schedule/CFG/precision/model/training',
              'user_authorized_larger_validation_batches': True}
    cold = next(r for r in results if r['batch_size'] == best['batch_size'] and r['reference_cache']
                and not r['reference_already_cached'])
    warm = next((r for r in results if r['batch_size'] == best['batch_size'] and r['reference_already_cached']), cold)
    batches_per_reference = 12/best['batch_size']
    amortized = (cold['seconds_per_image']+(batches_per_reference-1)*warm['seconds_per_image'])/batches_per_reference
    policy['predicted_denoising_speedup'] = results[0]['seconds_per_image']/amortized
    if prior:
        policy['previous_batch_measurements'] = prior['measurements']
    m.write(run/'validation_execution_policy.json', policy)
    print('Selected validation execution: '+json.dumps(policy), flush=True)
    return policy
