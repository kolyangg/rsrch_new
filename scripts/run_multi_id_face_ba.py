"""Serial FLUX4B multi-ID controller; optional two-rank training, serial validation."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ba_dit.config import ROOT, digest, load_config


def latest_checkpoint(run, until):
    """Only atomic, complete training checkpoints can be resumed."""
    steps = []
    for path in run.glob('checkpoint-*'):
        if not all((path/name).is_file() for name in
                   ('manifest.json', 'adapters.safetensors', 'training_state.pt')):
            continue
        step = json.loads((path/'manifest.json').read_text())['step']
        if step <= until:
            steps.append(step)
    if not steps:
        raise RuntimeError('No complete checkpoint; initialization must finish first')
    return max(steps)


def archive_uncheckpointed_metrics(run, step):
    """Preserve crash evidence without duplicating rolled-back updates on resume."""
    path = run/'metrics.jsonl'
    if not path.exists():
        return
    kept, tail = [], []
    for line in path.read_text().splitlines(keepends=True):
        try:
            committed = json.loads(line)['step'] <= step
        except (ValueError, KeyError):
            committed = False
        (kept if committed else tail).append(line)
    if tail:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        (run/f'uncheckpointed_metrics_{stamp}.jsonl').write_text(''.join(tail))
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(kept)); temporary.replace(path)


def freeze_native(config, baseline, bundle, rows):
    from PIL import Image, ImageDraw
    from ba_dit.data.manifest import file_hash
    from ba_dit.validation_masks import load_masks
    from ba_dit.nn.masked_face_flow import face_alpha
    from scripts.online_face_ba import write

    record = load_masks(config, [r['sample_id'] for r in rows])
    missing = [r['sample_id'] for r in rows if not record['samples'][r['sample_id']]['face_bbox']]
    if missing:
        raise ValueError(f'Native face masks need review: {missing}. Inspect {baseline}; '
                         'resume with --mask-overrides JSON (sample_id -> [x0,y0,x1,y1]).')
    (bundle/'native').mkdir(parents=True, exist_ok=True)
    sheet = Image.new('RGB', (6*192, ((len(rows)+5)//6)*216), 'white')
    samples = {}
    for index, row in enumerate(rows):
        key = row['sample_id']; item = dict(record['samples'][key])
        if file_hash(baseline/f'{key}.png') != item['baseline_image_sha256']:
            raise ValueError(f'Frozen mask belongs to a different native image: {key}')
        item['latent_sha256'] = file_hash(baseline/f'{key}.safetensors')
        samples[key] = item
        for suffix in ('.png', '.safetensors'):
            shutil.copyfile(baseline/f'{key}{suffix}', bundle/'native'/f'{key}{suffix}')
        image = Image.open(baseline/f'{key}.png').convert('RGB')
        overlay = Image.new('RGBA', image.size, (255,50,50,0))
        alpha = face_alpha(image.size,item['face_bbox'],config['branch']['mask_feather_pixels'])
        overlay.putalpha(Image.fromarray((alpha.squeeze().numpy()*100).astype('uint8')))
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        x, y = index % 6*192, index//6*216
        sheet.paste(image.resize((192,192)), (x,y))
        ImageDraw.Draw(sheet).text((x+3,y+195), key, fill='black')
    sheet.save(bundle/'mask_overlays.png')
    write(bundle/'routing_masks.json', {'source_signature':record['signature'],
          'feather_pixels':config['branch']['mask_feather_pixels'], 'samples':samples})
    write(bundle/'ownership_boxes.json', {k:v['face_bbox'] for k,v in samples.items()})
    shutil.copyfile(baseline/'quality_summary.json', bundle/'quality_summary.json')


def main(args):
    config_path = args.config.resolve()
    run = args.run.resolve()
    setup = run.with_name(run.name+'_setup')
    if args.resume:
        for folder in (run,setup):
            if (folder/'resolved_config.yaml').exists():
                config_path = folder/'resolved_config.yaml'
                break
    config = load_config(config_path)
    if args.conditioning:
        if args.resume and args.conditioning != config['data'].get('conditioning','cached'):
            raise ValueError('Cannot change conditioning mode on resume; use a fresh run')
        config['data']['conditioning'] = args.conditioning
    panel_size = 12 if args.pilot_panel else 96
    panel_name = 'manual_val_12_pilot.jsonl' if args.pilot_panel else 'manual_val_96.jsonl'
    if (config['model']['arch'] not in {'flux2_klein_4b','flux2_klein_9b'} or
            config['branch'].get('kind') != 'masked_face_qkvo' or
            config['data']['train_limit'] is not None or config['validation']['limit'] != panel_size or
            Path(config['data']['validation_manifest']).resolve() != (ROOT/'data/validation'/panel_name).resolve() or
            not config['logging']['enabled']):
        raise ValueError('Requires masked FLUX Klein, full multi-ID training and the declared fixed panel')
    steps = sorted({0, config['training']['steps'], *range(config['training']['validation_every'],
                   config['training']['steps'], config['training']['validation_every'])})
    world = config['training'].get('world_size', 1)
    plan = {'run':str(run), 'train_manifest':config['data']['train_manifest'],
            'microbatch_per_gpu':config['training'].get('microbatch_size',1), 'world_size':world,
            'effective_batch':world*config['training']['grad_accum']*config['training'].get('microbatch_size',1),
            'validation_steps':steps, 'trainable':'BA Q/K/V/output LoRA only',
            'initialization':'fresh adapters, not one-ID weights',
            'conditioning':config['data'].get('conditioning','cached'),
            'stages':['import downloaded Large images if needed', 'preflight',
                      f'cache fixed{panel_size} validation only' if config['data'].get('conditioning') == 'online' else 'cache full training and validation inputs',
                      'pretrained parity / gradient / exact resume admission',
                      f'native fixed{panel_size} / masks / scoring', 'train and validate serially']}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return  # Deliberately no data reads, cache preparation, CUDA calls or writes.

    import torch
    from ba_dit.data.manifest import read_manifest, assert_disjoint, file_hash
    from ba_dit.data.cache import cache_path
    from scripts.online_face_ba import SOURCES, verify, write

    (ROOT/'runs').mkdir(exist_ok=True)
    lock = (ROOT/'runs/face_flow_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    dedicated_encoder = config['data'].get('encoder_device') == 'cuda:1'
    visible = 2 if dedicated_encoder else world
    minimum = config['hardware']['min_vram_gb'] if world > 1 or dedicated_encoder else max(45,config['hardware']['min_vram_gb'])
    if torch.cuda.device_count() < visible or any(torch.cuda.get_device_properties(i).total_memory < minimum*10**9 for i in range(visible)):
        raise RuntimeError(f'Requires {visible} visible GPU(s), each with at least {minimum} GB VRAM')
    # A killed controller must not leave a child that a second controller can duplicate.
    os.set_inheritable(lock.fileno(), True)
    envs = Path(os.environ.get('BA_ENVS_DIR', ROOT/'envs')).resolve()
    metrics = str(envs/'metrics/bin/python'); quality = str(envs/'face-quality/bin/python')
    for python in (metrics, quality):
        if not Path(python).is_file():
            raise FileNotFoundError(f'Missing {python}; complete scripts/setup_machine.sh flux48 first')
    if (run.exists() or setup.exists()) and not args.resume:
        raise FileExistsError('Run/setup already exists; use --resume or a fresh --run')
    setup.mkdir(parents=True, exist_ok=True)
    if os.getenv('BA_COMET_OFFLINE') == '1' and config['logging']['enabled']:
        from ba_dit.logging import connect
        startup = connect(config, setup, name=run.name)
        startup.log_parameters({'cluster/job_id': os.getenv('SLURM_JOB_ID'),
                                'cluster/cuda_allocator': os.getenv('PYTORCH_CUDA_ALLOC_CONF', 'default'),
                                'training/effective_batch': world * config['training']['grad_accum'],
                                'training/microbatch_per_gpu': 1})
        startup.log_other('cluster/stage', 'dataset_verification')
        startup.log_other('cluster/slurm_state', 'RUNNING')
        startup.end()  # Close the startup archive immediately for the login uploader.
        write(setup/'status.json', {'stage':'dataset_verification', 'status':'running',
                                   'controller_pid':os.getpid()})
    if args.mask_overrides and (run/'comet_experiment.json').exists():
        raise ValueError('Masks are immutable after initialization; use a fresh run for changed masks')
    state_dir = setup
    def call(key, command):
        receipt = state_dir/f'{key}.done.json'
        if receipt.exists():
            return
        write(state_dir/'status.json', {'stage':key, 'status':'running', 'controller_pid':os.getpid(),
                                      'updated_at':datetime.now(timezone.utc).isoformat()})
        print(f'{key}: {state_dir / (key+".log")}', flush=True)
        with (state_dir/f'{key}.log').open('a') as stream:
            child = subprocess.Popen(list(map(str,command)), cwd=ROOT, stdout=stream,
                                     stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            write(state_dir/'status.json', {'stage':key, 'status':'running', 'controller_pid':os.getpid(),
                                          'child_pid':child.pid})
            try:
                code = child.wait()
            except BaseException:
                # Keep the lock until the child exits, even if the controller is interrupted.
                child.wait()
                raise
        if code:
            write(state_dir/'status.json', {'stage':key, 'status':'failed', 'exit_code':code})
            raise RuntimeError(f'{key} failed ({code}); inspect {state_dir / (key+".log")}')
        write(receipt, {'command':list(map(str,command)), 'finished':datetime.now(timezone.utc).isoformat()})

    train_path = Path(config['data']['train_manifest'])
    if not train_path.exists():
        if not args.images_root:
            raise FileNotFoundError(f'{train_path}: pass --images-root for the downloaded adjusted Large dataset')
        command = [sys.executable, ROOT/'scripts/prepare_dataset.py', 'large', '--images-root',
                   args.images_root.resolve(), '--output', train_path]
        if args.metadata:
            command += ['--metadata',args.metadata.resolve()]
        call('import_large', command)
    audit = json.loads(train_path.with_suffix('.audit.json').read_text())
    pinned = json.loads((ROOT/'locks/datasets.json').read_text())['large']['metadata_sha256']
    aliases = json.loads((ROOT/'data/validation/large_dataset_identity_aliases.json').read_text())
    if (audit.get('pinned_metadata_sha256') != pinned or audit.get('dataset') != 'large' or
            audit['manifest_sha256'] != file_hash(train_path) or audit['identity_aliases'] != aliases or
            audit['selection']['requested_pairs'] is not None):
        raise ValueError('Expected full pinned adjusted Large import with held-out identity aliases; no pilot/one-ID substitution')
    train = read_manifest(train_path, training=True)
    validation = read_manifest(config['data']['validation_manifest'])
    assert_disjoint(train, validation)
    if len(validation) != panel_size or len({r['identity_id'] for r in train}) < 2 or any(
            r.get('split_policy') == 'one_id_diagnostic' for r in train+validation):
        raise ValueError('Expected multiple training identities and the declared identity-disjoint panel')
    extra_sources = ('ba_dit/distributed_training.py', 'scripts/run_clust_v100.sh') if world > 1 else ()
    guard = {'config':digest(config), 'train':file_hash(train_path),
             'validation':file_hash(config['data']['validation_manifest']),
             'sources':{p:file_hash(ROOT/p) for p in (*SOURCES, 'scripts/run_multi_id_face_ba.py',
                         'scripts/check_online_face_ba.py', 'ba_dit/validation_masks.py',
                         'ba_dit/data/conditioning.py', *extra_sources)}}
    guard_path = setup/'identity.json'
    if guard_path.exists() and json.loads(guard_path.read_text()) != guard:
        raise ValueError('Setup config/data/source changed; use a fresh run name')
    write(guard_path, guard)

    initialized = all((run/name).is_file() for name in
                      ('identity.json','comet_experiment.json','checkpoint-000000/training_state.pt'))
    if not initialized:
        # Recover partial initialization using the original setup config, not a moved run path.
        frozen_config = setup/'resolved_config.yaml'
        import yaml
        frozen_config.write_text(yaml.safe_dump(config,sort_keys=False))
        config_path = frozen_config
        # Conditioning precision controls cache size; include a 10% margin.
        # No backbone hidden-state caches: these are text embeddings and VAE latents only.
        entries = {}
        h,w = config['data']['target_size']; ref = config['data']['reference_size']
        cached_rows = validation if config['data'].get('conditioning') == 'online' else train+validation
        scalar_bytes = 4 if config['model'].get('conditioning_dtype') == 'float32' else 2
        text_width = 12288 if config['model']['arch'].endswith('9b') else 7680
        for row in cached_rows:
            entries[cache_path(config,row,'encoder')] = 512*text_width*scalar_bytes+65536
            entries[cache_path(config,row,'vae')] = (h*w+ref*ref)*scalar_bytes+65536
        entries[cache_path(config,{**validation[0],'prompt':''},'encoder')] = 512*text_width*scalar_bytes+65536
        missing = sum(size for path,size in entries.items() if not path.exists())
        cache_root = Path(config['data']['cache_dir']); cache_root.mkdir(parents=True,exist_ok=True)
        required = int(missing*1.1)
        if shutil.disk_usage(cache_root).free < required:
            raise RuntimeError(f'Need approximately {required/2**30:.1f} GiB free for missing input caches at {cache_root}')
        checkpoint_budget = (config['training']['steps']//config['training']['checkpoint_every']+1)*.4*2**30+8*2**30
        shared_disk = cache_root.stat().st_dev == setup.stat().st_dev
        if shutil.disk_usage(setup).free < checkpoint_budget+(required if shared_disk else 0):
            raise RuntimeError('Insufficient run/cache disk space; reserve caches plus checkpoints and validation images')
        cli = [sys.executable,'-m','ba_dit.cli']
        call('preflight', cli+['preflight','--config',config_path,'--split','train'])
        cache_splits = ('validation',) if config['data'].get('conditioning') == 'online' else ('train','validation')
        for split in cache_splits:
            call('cache_'+split, cli+['precompute','--config',config_path,'--split',split])
        # Preserve failed admission attempts; never reuse an incomplete replay check.
        admission_receipt = setup/'admission_path.json'
        if admission_receipt.exists():
            admission = Path(json.loads(admission_receipt.read_text())['path'])
        else:
            admission = setup/f'admission-{len(list(setup.glob("admission-*"))):03d}'
            call(admission.name,[sys.executable,'-m','scripts.check_online_face_ba',
                                '--config',config_path,'--output',admission])
            write(admission_receipt,{'path':str(admission)})
        baseline = setup/f'native{panel_size}'
        command = cli+['infer','--config',config_path,'--mode','native','--output-dir',baseline,
                       '--no-comet','--no-quality-metrics','--skip-output-masks']
        if baseline.exists(): command += ['--resume-validation']
        call('native96',command)
        command = [metrics,ROOT/'scripts/build_validation_output_masks.py','--validation',baseline]
        if args.mask_overrides:
            command += ['--overrides',args.mask_overrides.resolve()]
        # An explicit correction gets a new receipt and invalidates pre-init native scoring.
        mask_key = 'masks_'+(file_hash(args.mask_overrides)[:12] if args.mask_overrides else 'auto')
        call(mask_key,command)
        call('native_score_'+mask_key,[metrics,'-m','scripts.evaluate_metrics','--validation',baseline,'--no-comet'])
        bundle = setup/'native_bundle'
        freeze_native(config,baseline,bundle,validation)
        if run.exists():
            # Preserve evidence of an interrupted initialization before rebuilding step zero.
            run.rename(setup/f'interrupted-init-{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")}')
        if (setup/'initialize.done.json').exists():
            raise RuntimeError('A previously initialized run lost required files; inspect before rebuilding')
        call('initialize',[sys.executable,'-m','scripts.online_face_ba','init','--run',run,
                           '--config',config_path,'--admission',admission,'--native-source',bundle])

    config,_ = verify(run)
    if not (run/'checkpoint-000000/training_state.pt').exists():
        raise RuntimeError('Initialization incomplete; preserve this run and restart with a fresh --run')
    state_dir = run
    (run/'controller.pid').write_text(str(os.getpid())+'\n')
    write(run/'execution_plan.json', plan)
    for step in steps:
        if step and not (run/f'train_{step}.done.json').exists():
            resume = latest_checkpoint(run,step)
            if resume < step:
                archive_uncheckpointed_metrics(run,resume)
                command = [sys.executable,'-m','scripts.online_face_ba','train','--run',run,
                           '--step',step,'--resume',resume]
                if world > 1:
                    command = [sys.executable,'-m','torch.distributed.run','--standalone',
                               '--nproc-per-node=2', *command[1:]]
                call(f'train_{step}',command)
        for action in ('infer','decode','score','face_quality','summarize'):
            if action == 'face_quality' and args.id_clip_only:
                continue
            verify(run)
            if action in ('score','face_quality'):
                module = 'scripts.evaluate_metrics' if action == 'score' else 'scripts.evaluate_face_quality'
                command = [metrics if action=='score' else quality,'-m',module,
                           '--validation',run/f'validation-{step:06d}','--log-dir',run,'--global-step',step]
                if action == 'score': command += ['--ownership-boxes',run/'ownership_boxes.json']
            else:
                command = [sys.executable,'-m','scripts.online_face_ba',action,'--run',run,'--step',step]
            call(f'{action}_{step}',command)
        write(run/'status.json',{'status':'running','last_completed_validation':step})
    write(run/'status.json',{'status':'completed','training_steps':steps[-1],
                            'last_completed_validation':steps[-1]})
    print(f'Completed training and scoring: {run}; best checkpoint: best_checkpoint.json',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/flux4b_48_multi_id_large.yaml')
    parser.add_argument('--images-root', type=Path, help='Already extracted adjusted Large image directory')
    parser.add_argument('--metadata', type=Path, help='Pinned filtered_ids3_adj.json; defaults to private metadata bundle')
    parser.add_argument('--mask-overrides', type=Path, help='Reviewed native-image boxes; allowed only before initialization')
    parser.add_argument('--conditioning', choices=('online','cached'),
                        help='Default config uses online encoding; cached explicitly prepares all training inputs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--pilot-panel', action='store_true', help='Explicitly use the existing fixed12 held-out pilot')
    parser.add_argument('--id-clip-only', action='store_true', help='Keep original ID/CLIP metrics; omit seven additional face-quality models')
    parser.add_argument('--dry-run', action='store_true', help='Print configuration/plan only; no data preparation or CUDA')
    main(parser.parse_args())
