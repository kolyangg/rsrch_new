# Deploy on a new GPU machine

The branch runner is independent of a PhotoMaker installation. Only the original dataset metadata/photos and identity lookup files are reused.

## 1. Transfer the current source and validation data

If using Git, transfer a revision containing the current changes. Prepared archives live in `data/bundles/` and can be transferred directly. To package a newer working checkout without committing it, use unused output filenames (the packers refuse to overwrite an existing archive):

```bash
python3 scripts/package_project.py --output data/bundles/rsrch_new-code.tar.gz
envs/qwen21/bin/python scripts/package_validation_data.py --include-smoke \
  --output data/bundles/validation-and-smoke-with-metrics.tar.gz
envs/qwen21/bin/python scripts/package_training_metadata.py \
  --legacy-data /home/kolyangg/rsrch/dataset_full \
  --output data/bundles/training-metadata.tar.gz
scp data/bundles/rsrch_new-code.tar.gz HOST:/workspace/
scp data/bundles/validation-and-smoke-with-metrics.tar.gz HOST:/workspace/
scp data/bundles/training-metadata.tar.gz HOST:/workspace/
```

The source archive includes tracked/unignored working files. The validation archive includes reference photos, original identity embeddings, reference masks, any completed backbone output-mask sets, and optionally the 38-pair smoke dataset. The training-metadata archive contains both original dataset JSONs with their pinned hashes; image archives are downloaded separately. Credentials, model weights, environments and training runs are excluded. The packaging commands do not upload anything.

On the new host:

```bash
cd /workspace
tar -xzf rsrch_new-code.tar.gz
cd rsrch_new
scripts/setup_machine.sh flux48 --download-weights
# Or flux80, qwen48, qwen80. Each backend uses its own environment.

envs/flux-toolkit/bin/python scripts/download_dataset.py \
  --archive /workspace/validation-and-smoke-with-metrics.tar.gz \
  --destination data/imported_validation
# Preserve the existing tracked fixed-panel metadata while adding private assets.
cp -a data/imported_validation/validation/. data/validation/
cp -a data/imported_validation/train_smoke data/
cp -a data/imported_validation/train_pairs_smoke.jsonl data/
```

Use `envs/qwen21/bin/python` for data tools on a Qwen-only host. Set `COMET_API_KEY` securely in the shell or `.env`; set `HF_TOKEN` when the selected model requires your HF access. The setup script never copies credentials from another machine.

Space depends on the chosen model and dataset. Locked source weights alone are about 16.2 GB for FLUX 4B components, 34.9 GB for FLUX 9B components, or 33.1 GB for Qwen. Leave room for environments, raw archives plus extracted data, conditioning caches and checkpoints; the original plan recommends 250 GB for small pilots of both families, preferably 500 GB for iteration. The setup uses `uv` and chooses CUDA 12.6 wheels on driver 565.77; newer drivers retain the previously tested CUDA 13.0 FLUX or CUDA 12.8 Qwen pins. The selected wheel and runtime are printed during setup and recorded in the run's preflight. CUDA 12.6 is a deployment deviation to recheck with a real training update.

## 2. Download and import the training data

Both original dataset options work with `flux48`, `flux80`, `qwen48`, and `qwen80`. Their sources, archive layouts and metadata hashes are recorded in [locks/datasets.json](../locks/datasets.json). Each produces its own manifest:

| Option | Original data | Training manifest |
| --- | --- | --- |
| `cosmic` | Cosmic Large scene targets plus the reference-face bank | `data/train_pairs_cosmic.jsonl` |
| `large` | Adjusted Large Dataset, `large_dataset_adj/large_dataset` | `data/train_pairs_large.jsonl` |

### Original Google Drive sources

- Cosmic targets: [cosmic_dataset_images.tar](https://drive.google.com/file/d/16upgN9HNXRKdASiWsYvo1pw1-ndpC3uO/view?usp=sharing).
- Cosmic references: [cosmic_large.tar.gz](https://drive.google.com/file/d/1YV1sGZIFE4qQ1Ju-5Yqo9NvcKt2q16zO/view?usp=sharing).
- Large: `large_dataset_adj.tar`. Its original Drive link is still pending. The old setup conversation used an example ID, then reported a manual download; no actual Large URL was saved in the checked repository/history. Supply it with `--url` / `LARGE_DATASET_URL`, or reuse an extracted image directory.

For a rented Vast instance when the original Large link is unavailable, transfer the 17 GB adjusted image folder from the old workstation. The command resumes partial transfers and also sends the two private metadata/validation bundles:

```bash
python3 scripts/sync_large_dataset_vast.py 53574065
# On the instance, after cloning this repository and creating the FLUX environment:
envs/flux-toolkit/bin/python scripts/download_dataset.py \
  --archive /workspace/training-metadata.tar.gz --destination data/datasets/metadata
envs/flux-toolkit/bin/python scripts/prepare_dataset.py large \
  --images-root /workspace/datasets/large_dataset \
  --sample-pairs 4096 --output data/train_pairs_large_4096.jsonl
scripts/run_profile.sh flux48 train --mode branch_only \
  --train-manifest data/train_pairs_large_4096.jsonl --run-name flux48_large4096_branch
```

The 4,096-pair manifest is a named disk-bounded training pilot. Stable hashes select pairs across the complete Large metadata before image preparation, avoiding alphabetical first-N bias. The Large preset automatically maps seven matching IMDb identities to held-out validation names using [large_dataset_identity_aliases.json](../data/validation/large_dataset_identity_aliases.json). The full release remains available on disk for a larger run. Training, data order, selection method and the immutable Comet key are saved with the pilot.

FLUX validation supports an optional `--batch-size 8` experiment when prompts share the exact cached reference and token mask. On the rented RTX 6000 Ada, batch eight showed no per-sample speed gain over serial inference and changed the generated latents, so the production profile keeps batch size one. Use a separate named run when comparing batch settings.

If the process is interrupted before the first optimizer update, resume its fixed step-zero validation and then training with `scripts/run_profile.sh flux48 train --resume-run runs/flux48_large4096_branch --quality-metrics`. It reuses completed latents and the saved Comet experiment key.

For a shorter wiring check, `configs/flux4b_48_pilot12.yaml` selects 12 items from the fixed panel across all eight identities. This is a separate named experiment with its own output-mask set and Comet run:

```bash
scripts/run_profile.sh flux48-pilot12 preflight --split train \
  --train-manifest data/train_pairs_large_4096.jsonl
scripts/run_profile.sh flux48-pilot12 train --mode branch_only --quality-metrics \
  --train-manifest data/train_pairs_large_4096.jsonl \
  --run-name flux48_large4096_branch_pilot12
envs/flux-toolkit/bin/python scripts/compare_validation_steps.py \
  runs/flux48_large4096_branch_pilot12 --later-step 2000
```

The two Cosmic links were recovered from the previous project's actual download commands, and their Drive pages returned the archive names above on 30 September 2026. Those commands renamed the downloads to `LAION-5B-Filtered-Large.tar` and `LAION-5B-Filtered-Large-Faces.tar.gz`; these are the same two Drive files. Both are required. The later BigCelebs release is a separate dataset and is not substituted for Large.

First unpack the private metadata bundle transferred in step 1:

```bash
envs/flux-toolkit/bin/python scripts/download_dataset.py \
  --archive /workspace/training-metadata.tar.gz \
  --destination data/datasets/metadata

# Download both original Cosmic archives, resolve their roots, and import pairs.
envs/flux-toolkit/bin/python scripts/prepare_dataset.py cosmic

# Download/import Large when its original archive link has been supplied.
envs/flux-toolkit/bin/python scripts/prepare_dataset.py large \
  --url "$LARGE_DATASET_URL"
```

Use `envs/qwen21/bin/python` instead on a Qwen-only host. `prepare_dataset.py cosmic --show` and `prepare_dataset.py large --show` display sources without downloading. Metadata hashes are checked before downloading; a different metadata release requires the generic importer below. Downloads are resumable, archives are retained, and completed extractions have source/hash receipts. Repeat an interrupted command to reuse completed archives. Existing output manifests require a new `--output` name.

The full archives are intended for the deployment machine. This laptop has approximately 45 GB free; full image archives have not been downloaded again. The metadata bundle and existing real local image samples are available for verification.

### Reuse local files or a different dataset release

To reuse the existing Large images on this workstation (no download):

```bash
envs/flux-toolkit/bin/python scripts/prepare_dataset.py large \
  --metadata /home/kolyangg/rsrch/dataset_full/filtered_ids3_adj.json \
  --images-root /home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset
```

Both preparation paths accept `--identity-aliases`, `--exclude-identities`, `--max-pairs`, and `--sample-pairs`. Aliases are a JSON mapping from dataset IDs to validation/canonical names; exclusions are a JSON list of held-out dataset IDs. Supply them when identities use another naming system (for example IMDb IDs versus the panel's short names): raw string equality cannot establish semantic identity separation. Image hashes additionally reject exact train/validation duplicates. Use `--sample-pairs` for a representative disk-bounded pilot and an explicit `--output data/train_pairs_<dataset>_pilot.jsonl`.

The generic importer remains available for other releases, including BigCelebs-style nested identity metadata:

```bash
envs/flux-toolkit/bin/python scripts/import_training_dataset.py \
  --format large --metadata /path/to/filtered_ids3_adj.json \
  --images-root /path/to/large_dataset --output data/train_pairs_custom.jsonl

envs/flux-toolkit/bin/python scripts/import_training_dataset.py \
  --format cosmic --metadata /path/to/gathered_data_cosmic_large_filtered.json \
  --images-root /path/containing/LAION-5B-Filtered-Large \
  --reference-root /path/containing/LAION-5B-Filtered-Large-Faces \
  --output data/train_pairs_cosmic_custom.jsonl
```

Cosmic uses the old target-path records with `face_paths`, `face_bboxes`, `face_crop_new`, captions and optional body crops. The separate reference root handles its second archive without copying the face bank. Large uses `identity -> image_id -> {new_face_crop, text, ...}`, with images at `identity/image_id.jpg`; its legacy body-crop ordering is preserved.

The importer chooses the first eligible sorted Cosmic reference or the next distinct Large view. Pairing policy, input hashes, dataset name, source receipts and exclusions are saved beside each manifest in `.audit.json`. The initial cache-based pilot uses deterministic pairs without random augmentation.

Select either dataset with `--train-manifest` for preflight, precompute and training. For example:

```bash
scripts/run_profile.sh flux48 preflight --split train \
  --train-manifest data/train_pairs_cosmic.jsonl
scripts/run_profile.sh qwen80 train --mode branch_only --quality-metrics \
  --train-manifest data/train_pairs_large.jsonl --run-name qwen80_large_branch
```

## 3. Admit the hardware profile

```bash
scripts/run_profile.sh flux48 preflight --split train \
  --train-manifest data/train_pairs_smoke.jsonl
scripts/run_profile.sh flux48 train --mode branch_only --smoke-steps 2 \
  --train-manifest data/train_pairs_smoke.jsonl --limit 2 \
  --output-dir runs/flux48_hardware_smoke
```

Repeat with `flux80`, `qwen48` or `qwen80`. The transferred smoke data is sufficient for this check. It uses that profile's actual target/reference geometry and normal accumulation, saves the adapters, records first-update gradients and peak memory, and rejects a run exceeding 90% reserved VRAM. A local test at 256 pixels does not admit a 768/1024-pixel run on another GPU.

Run native full-panel validation once per profile to create its output face masks, then start the branch pilot:

```bash
scripts/run_profile.sh flux48 infer --mode native --quality-metrics \
  --output-dir runs/flux48_native_full96
scripts/run_profile.sh flux48 train --mode branch_only --quality-metrics \
  --train-manifest data/train_pairs_cosmic.jsonl --run-name flux48_cosmic_branch
```

For matched controls use separate run names with `--mode lora_only` or `--mode lora_plus_branch`. Use the same manifest, geometry, seed, update budget and validation settings. Validation and training run serially on the GPU.

## 4. Inspect artifacts and resume

- `resolved_config.yaml`: actual parameters and paths.
- `comet_experiment.json`: immutable project/run key, reused on resume.
- `optimizer_inventory.json`: exact trainable names/shapes.
- `metrics.jsonl`: every loss, gradient, memory and step-time scalar, plus progress percentage, remaining steps, ETA seconds and updates/hour. These are also sent to Comet at each optimizer update; the console prints a compact progress bar every 25 updates.
- `checkpoint-NNNNNN/`: adapter tensors, optimizer/scheduler/RNG/cursor, source/data/config fingerprints, relocatable resume config.
- `validation-NNNNNN/`: latents, PNGs, per-image timing/parity, identity/CLIP scores and the seven face-quality curves (scoring is on by default).
- `data/validation/output_masks/<backbone>/<geometry-and-schedule>/`: native-baseline face masks consumed by validation scoring.

Move the checkpoint together with the same code, dataset and weight revisions. Project-local paths in `resume_config.yaml` expand against the checkout on the destination host. External dataset paths must remain valid or be deliberately relocated as a new experiment. Full resume checks the data fingerprint, configuration and training implementation.

For a worker started before progress logging was added, run `python -m scripts.live_training_progress RUN_DIR` from the matching training environment. It tails the existing metrics file and logs progress to the same Comet experiment without restarting training. `--once --no-comet` prints a local snapshot.

The original face-quality models download their pretrained weights on first scoring. Both scoring environments use CPU after generation exits. Face-quality scoring uses up to eight CPU threads by default; `scripts/evaluate_face_quality.py --threads N` overrides this. Per-model progress and elapsed times are recorded because MANIQA can dominate CPU evaluation time. `BA_ENVS_DIR` optionally moves all four environments together; setup and launch scripts use the same root. Output-mask PNGs can be reconstructed from the frozen manifest boxes when transferring metadata alone, with their recorded hashes checked afterward.

```bash
scripts/run_profile.sh qwen80 train --mode branch_only --quality-metrics \
  --resume runs/your_run/checkpoint-000500
```

This retains the checkpoint's saved geometry. For expanded geometry use `--init-adapter` with the new profile to start a separate run. The 4B and 9B FLUX adapters have different shapes and are intentionally incompatible.
