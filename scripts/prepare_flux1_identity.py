"""Prepare training-only face landmarks/identity labels, preserving every flow-training row."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ba_dit.config import load_config
from ba_dit.data.geometry import target_geometry
from ba_dit.data.manifest import file_hash, read_manifest


APP = None


def initialize_detector(threads=2):
    global APP
    import onnxruntime
    import cv2
    from unittest.mock import patch
    from insightface.app import FaceAnalysis
    from insightface.model_zoo.model_zoo import ModelRouter
    cv2.setNumThreads(threads)
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # InsightFace 0.7.3 get_model drops sess_options; inject at the router,
    # which actually constructs the ONNX session. Avoid hundreds of threads
    # per process when preparing labels in a Slurm CPU allocation.
    original = ModelRouter.get_model
    def configured(router, **kwargs):
        return original(router, sess_options=options, **kwargs)
    with patch.object(ModelRouter, 'get_model', configured):
        APP = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'],
                           allowed_modules=['detection', 'recognition'])
    assert all(model.session.get_session_options().intra_op_num_threads == threads
               for model in APP.models.values()), 'ONNX worker thread limit was not applied'
    APP.prepare(ctx_id=-1, det_size=(640, 640))


def prepare_target(job):
    import cv2
    from insightface.utils.face_align import estimate_norm
    from ba_dit.metrics import bbox_iou
    row, target_size = job
    app = APP
    with Image.open(row['target']) as original:
        image, geometry = target_geometry(original.convert('RGB'), target_size, row['target_box'])
    scale = np.asarray(geometry['resize_wh']) / np.asarray(geometry['source_wh'])
    left, top = geometry['crop_xyxy'][:2]
    box = np.asarray(row['target_box']) * [*scale, *scale] - [left, top, left, top]
    bgr = np.asarray(image)[:, :, ::-1].copy()
    faces = []
    for size in (640, 576, 512, 448, 384, 320, 256):
        app.det_model.input_size = (size, size)
        faces = app.get(bgr)
        if faces:
            break
    faces = sorted(faces, key=lambda face: bbox_iou(face.bbox, box), reverse=True)
    accepted = bool(faces and bbox_iou(faces[0].bbox, box) >= .3 and float(faces[0].det_score) >= .6)
    if len(faces) > 1 and abs(bbox_iou(faces[0].bbox, box) - bbox_iou(faces[1].bbox, box)) < .05:
        accepted = False
    item = {'sample_id': row['sample_id'], 'target_sha256': row['target_hash'],
            'target_size': target_size, 'target_box': row['target_box'],
            'accepted': accepted, 'identity_id': row['identity_id']}
    if accepted:
        face = faces[0]
        matrix = estimate_norm(face.kps, image_size=112)
        item['aligned_to_image'] = cv2.invertAffineTransform(matrix).tolist()
        embedding = face.embedding.astype(np.float32)
        item['embedding'] = (embedding / np.linalg.norm(embedding)).tolist()
    return item


def main(args):
    import cv2

    config = load_config(args.config)
    root = args.output.resolve() if args.output else Path(config['data']['identity_supervision'])
    config['data']['identity_supervision'] = str(root)
    if (root / 'manifest.json').exists():
        from ba_dit.nn.online_identity_loss import supervision_identity
        if args.limit:
            raise FileExistsError('Use a separate directory for partial smoke preparation')
        supervision_identity(config)
        print('Existing complete supervision verified; no changes.')
        return
    root.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(config['data']['train_manifest'], training=True, limit=args.limit)
    full_count = len(rows)
    if args.scheduled_only:
        if args.limit:
            raise ValueError('Scheduled coverage cannot be combined with a partial smoke limit')
        from ba_dit.nn.online_identity_loss import scheduled_sample_ids
        required = scheduled_sample_ids(config)
        rows = [row for row in rows if row['sample_id'] in required]
        assert {row['sample_id'] for row in rows} == required
    scope = 'scheduled_training_rows' if args.scheduled_only else 'full_manifest'
    initialize_detector(args.threads)
    recognition = APP.models['recognition']
    preparation = {'train_manifest_sha256':file_hash(config['data']['train_manifest']),
                   'target_size':config['data']['target_size'], 'arcface_sha256':file_hash(recognition.model_file),
                   'geometry_sha256':file_hash(Path(__file__).resolve().parents[1]/'ba_dit/data/geometry.py'),
                   'preparation_code_sha256':file_hash(__file__), 'scope':scope,
                   'schedule_seed':config['training']['seed'],
                   'schedule_updates':config['training']['steps'], 'full_training_rows':full_count}
    guard = root/'preparation_identity.json'
    if guard.exists() and json.loads(guard.read_text()) != preparation:
        raise ValueError('Preparation data, recognizer or geometry changed; use a fresh directory')
    guard.write_text(json.dumps(preparation,indent=2)+'\n')
    records, embeddings = [], []
    # Per-target receipts allow safe continuation of a long CPU preparation.
    items = root / 'items'
    items.mkdir(exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    import multiprocessing
    pending = [(row, config['data']['target_size']) for row in rows
               if not (items / (row['sample_id'] + '.json')).exists()]
    pool = (ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                initializer=initialize_detector, initargs=(args.threads,))
            if args.workers > 1 else nullcontext())
    with pool as executor:
        prepared = executor.map(prepare_target, pending, chunksize=8) if executor else map(prepare_target, pending)
        for index, row in enumerate(rows):
            path = items / (row['sample_id'] + '.json')
            if path.exists():
                item = json.loads(path.read_text())
                assert item['target_sha256'] == row['target_hash'] and item['target_size'] == config['data']['target_size']
                assert item['target_box'] == row['target_box']
            else:
                item = next(prepared)
                assert item['sample_id'] == row['sample_id']
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(item) + '\n')
                temp.replace(path)
            item = dict(item)
            if item['accepted']:
                if not (root / 'arcface_parity.npz').exists():
                    with Image.open(row['target']) as original:
                        image, _ = target_geometry(original.convert('RGB'), config['data']['target_size'], row['target_box'])
                    bgr = np.asarray(image)[:, :, ::-1].copy()
                    matrix = cv2.invertAffineTransform(np.asarray(item['aligned_to_image']))
                    aligned = cv2.warpAffine(bgr, matrix, (112,112), borderValue=0.0)
                    normalized = (aligned[:, :, ::-1].astype(np.float32).transpose(2,0,1)[None]-127.5)/127.5
                    expected = recognition.session.run(None, {recognition.input_name:normalized})[0]
                    np.savez(root/'arcface_parity.npz', input=normalized, embedding=expected)
                item['embedding_index'] = len(embeddings)
                embeddings.append(item.pop('embedding'))
            records.append(item)
            if (index + 1) % 100 == 0:
                print(f'Prepared {index + 1}/{len(rows)} training targets; accepted {len(embeddings)}', flush=True)
    if not embeddings:
        raise RuntimeError('No valid identity supervision; flow training rows were not removed')
    np.save(root / 'target_embeddings.npy', np.asarray(embeddings, dtype=np.float32))
    (root / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    manifest = {**preparation, 'complete': args.limit is None, 'training_targets_only': True, 'validation_images_used': False,
                'train_manifest_sha256': file_hash(config['data']['train_manifest']),
                'target_size': config['data']['target_size'], 'rows': len(rows), 'accepted': len(embeddings),
                'accepted_fraction': len(embeddings) / len(rows), 'arcface_path': str(Path(recognition.model_file).resolve()),
                'arcface_sha256': file_hash(recognition.model_file), 'files': {name: file_hash(root / name) for name in
                    ('records.jsonl', 'target_embeddings.npy', 'arcface_parity.npz', 'preparation_identity.json')}}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--scheduled-only', action='store_true', help='Cover every exact scheduled training sample; keep the full training pool/order unchanged')
    parser.add_argument('--limit', type=int, help='Named partial CPU smoke; cannot qualify a training run')
    args=parser.parse_args()
    if min(args.workers,args.threads)<1: parser.error('workers and threads must be positive')
    main(args)
