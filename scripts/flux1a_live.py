"""Live Comet lifecycle and per-pair decoding around the frozen FLUX1a runner."""
import argparse
import copy
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/FLUX1a_vast9B_20261004'


class StreamingDecode:
    def __init__(self, module, run, config, step):
        self.m, self.run, self.config, self.step = module, run, config, step
        self.folder = run/f'validation-{step:06d}'
        self.receipt = run/f'live_decode_{step:06d}.json'
        self.sha = module.file_hash(run/f'checkpoint-{step:06d}'/'adapters.safetensors')
        self.records = json.loads(self.receipt.read_text()) if self.receipt.exists() else {}
        assert all(r['checkpoint_sha256'] == self.sha for r in self.records.values())
        self.masks = json.loads((run/'routing_masks.json').read_text())['samples']
        self.vae = None
        self.api = None
        self.write = module.write

    def image(self, folder, row):
        m = self.m
        key = row['sample_id']
        native_path = self.run/'native'/f'{key}.png'
        assert m.file_hash(native_path) == self.masks[key]['baseline_image_sha256']
        with m.safe_open(folder/f'{key}.safetensors', framework='pt') as f:
            latent = f.get_tensor('latent')
        image = m.backend_module(self.config).decode(self.vae, latent, self.config)
        native = m.Image.open(native_path).convert('RGB')
        alpha = m.face_alpha(native.size, self.masks[key]['face_bbox'], self.config['branch']['mask_feather_pixels'])
        result = m.preserve_background(image, native, alpha)
        difference = m.np.abs(m.np.asarray(result).astype(float)-m.np.asarray(native).astype(float))
        support = alpha.squeeze().numpy() > 0
        audit = {'sample_id': key,
                 'background_max_abs': float(difference[~support].max()) if (~support).any() else 0.,
                 'face_mean_abs': float(difference[support].mean())}
        assert audit['background_max_abs'] == 0
        return image, result, audit

    def publish(self, report):
        m = self.m
        assert report['checkpoint_sha256'] == self.sha
        if self.vae is None:
            # Loading a decoder must not change the sampler's RNG state.
            with m.torch.random.fork_rng(devices=[0]):
                self.vae = m.backend_module(self.config).load_vae(self.config)
            # Compare this exact decode/composition against a previously completed
            # panel before streaming any new image. No extra denoising is run.
            zero = self.run/'validation-000000'
            if (zero/'quality_summary.json').exists():
                row = json.loads((zero/'validation.json').read_text())['samples'][0]
                raw, image, _ = self.image(zero, row)
                assert m.np.array_equal(m.np.asarray(raw), m.np.asarray(m.Image.open(zero/f"{row['sample_id']}_raw.png")))
                assert m.np.array_equal(m.np.asarray(image), m.np.asarray(m.Image.open(zero/row['image'])))
                self.write(self.run/f'live_decode_parity_{self.step:06d}.json',
                           {'raw_pixels_exact': True, 'composed_pixels_exact': True, 'sample_id': row['sample_id']})
            from comet_ml.api import API
            from scripts.vast_gpu import env_value
            self.api = API(api_key=env_value(ROOT/'.env', 'COMET_API_KEY'), cache=False).get_experiment_by_key(
                json.loads((self.run/'comet_experiment.json').read_text())['experiment_key'])
        for row in report['samples']:
            key = row['sample_id']
            record = self.records.get(key)
            if record and record.get('uploaded'):
                continue
            if record is None:
                with m.safe_open(self.folder/f'{key}.safetensors', framework='pt') as f:
                    assert f.metadata()['checkpoint_sha256'] == self.sha
                raw, image, audit = self.image(self.folder, row)
                raw.save(self.folder/f'{key}_raw.png')
                image.save(self.folder/row['image'])
                record = {'checkpoint_sha256': self.sha, 'audit': audit,
                          'image_sha256': m.file_hash(self.folder/row['image']), 'uploaded': False}
                self.records[key] = record
                self.write(self.receipt, self.records)
            for attempt in range(3):
                try:
                    result = self.api.log_image(str(self.folder/row['image']), image_name=f'fixed96/{key}',
                                                step=self.step, metadata={'prompt': row['prompt'], 'seed': row['seed']})
                    if result is None:
                        raise RuntimeError('Comet image upload was not acknowledged')
                    record['uploaded'] = True
                    record['uploaded_at_unix'] = time.time()
                    self.write(self.receipt, self.records)
                    print(f'Validation {self.step}: uploaded {key} to Comet immediately after decode '
                          f'({sum(r["uploaded"] for r in self.records.values())}/96)', flush=True)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2)
        self.write(self.run/f'background_audit_{self.step}.json',
                   [self.records[row['sample_id']]['audit'] for row in report['samples']])


def worker(module, arguments):
    # The supervised publisher owns run lifetime. Stage SDK sessions still flush
    # and close normally, but cannot mark the whole experiment ended.
    from comet_ml._online import Experiment
    Experiment._mark_as_ended = lambda self: None
    if module != 'scripts.online_face_ba':
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name='__main__')
        return
    from scripts import online_face_ba as m
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('train', 'infer', 'decode', 'summarize'))
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--step', type=int, default=0)
    parser.add_argument('--resume', type=int, default=0)
    args = parser.parse_args(arguments)
    if args.step == 6000:
        from scripts.continue_flux1a_6k import continuation_config
        original_verify = m.verify
        def verify_extended(run):
            config, identity = original_verify(run)
            return continuation_config(run, config), identity
        m.verify = verify_extended
        if args.action == 'train':
            import ba_dit.logging as logging
            logging.total_steps = lambda run: 6000
            import ba_dit.continuation as continuation
            from scripts.extend_flux1a_labels import verify_resume
            strict_extension = continuation.verify_extension
            original_train = m.train_segment
            def train_extended(*positional, **keywords):
                continuation.verify_extension = lambda checkpoint, config, original_digest: verify_resume(
                    checkpoint, config, original_digest, strict_extension)
                try:
                    return original_train(*positional, **keywords)
                finally:
                    continuation.verify_extension = strict_extension
            m.train_segment = train_extended
    original_infer, original_decode = m.infer, m.decode

    def infer(run, config, step):
        from ba_dit.nn import isolated_reference
        from scripts.flux1a_validation_speed import CFGReferenceCache, qualify
        config = copy.deepcopy(config)
        backend = m.backend_module(config)
        cache = CFGReferenceCache(isolated_reference.reference_bank)
        original_load = m.load_adapters
        policy_path = run/'validation_execution_policy.json'
        def load(model, checkpoint, loaded_config, mode):
            print(f'Validation {step}: loading frozen model; completed images will be reused', flush=True)
            result = original_load(model, checkpoint, loaded_config, mode)
            policy = json.loads(policy_path.read_text()) if policy_path.exists() else None
            if policy is None or not policy.get('cache_scope'):
                policy = qualify(m, backend, model, run, config, step, cache)
            config['validation']['batch_size'] = policy['batch_size']
            isolated_reference.reference_bank = cache
            print(f'Validation {step}: batch_size={policy["batch_size"]}; '
                  'reuse exact frozen reference bank across CFG and same-reference prompt batches', flush=True)
            return result
        m.load_adapters = load
        stream = StreamingDecode(m, run, config, step)
        original_write = m.write
        def write(path, value):
            original_write(path, value)
            if path == stream.folder/'validation.json':
                stream.publish(value)
        m.write = write
        try:
            original_infer(run, config, step)
        finally:
            m.write = original_write
            m.load_adapters = original_load
            isolated_reference.reference_bank = cache.original

    def decode(run, config, step):
        receipt = run/f'live_decode_{step:06d}.json'
        if not receipt.exists():
            return original_decode(run, config, step)
        records = json.loads(receipt.read_text())
        report = json.loads((run/f'validation-{step:06d}'/'validation.json').read_text())
        assert len(report['samples']) == len(records) == 96
        for row in report['samples']:
            record = records[row['sample_id']]
            assert record['uploaded'] and record['checkpoint_sha256'] == report['checkpoint_sha256']
            assert record['image_sha256'] == m.file_hash(run/f'validation-{step:06d}'/row['image'])
        print(f'Validation {step}: all96 images already decoded and uploaded during generation', flush=True)

    m.infer, m.decode = infer, decode
    m.main(args)


def main():
    if ROOT != Path('/workspace/rsrch_FLUX1abc'):
        raise RuntimeError('Live wrapper is scoped to the existing GB10 FLUX1a run')
    if len(sys.argv) > 1 and sys.argv[1] == '--worker':
        worker(sys.argv[2], sys.argv[3:])
        return
    original_popen = subprocess.Popen
    class LivePopen(original_popen):
        def __init__(self, command, *args, **kwargs):
            command = list(command)
            if len(command) > 2 and command[1] == '-m' and command[2] in {
                    'scripts.online_face_ba', 'scripts.evaluate_metrics', 'scripts.evaluate_face_quality'}:
                command = command[:2]+['scripts.flux1a_live', '--worker']+command[2:]
            super().__init__(command, *args, **kwargs)
    subprocess.Popen = LivePopen
    sys.argv = ['scripts.run_multi_id_face_ba', '--run', str(RUN), '--config', str(RUN/'resolved_config.yaml'),
                '--native-bundle', str(ROOT/'runs/native_fixed96_bundle'), '--id-clip-only', '--resume']
    runpy.run_module('scripts.run_multi_id_face_ba', run_name='__main__')


if __name__ == '__main__':
    main()
