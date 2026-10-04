# Deploy on a new GPU machine

The branch runner is independent of a PhotoMaker installation. Only the original dataset metadata/photos and identity lookup files are reused.

## FLUX4B multi-ID Q/K/V/O BA on 48GB

Use `configs/FLUX1_vast_4b.yaml` and `scripts/run_flux4b_multi_id.sh`
for the full-denoiser experiment continued from the successful online one-ID
architecture. Transfer the current working source (these additions are not yet
committed), set up `flux48` below, download the adjusted Large image archive,
and install the private validation and training-metadata bundles. Supply
`COMET_API_KEY` through the environment or untracked `.env`.

```bash
# On the NEW machine, after environment/weights/data download:
bash scripts/run_flux4b_multi_id.sh \
  --run runs/flux4b_large_qkvo_r128 \
  --images-root /workspace/datasets/large_dataset

# Resume interrupted preparation, training or scoring using the same run:
bash scripts/run_flux4b_multi_id.sh \
  --run runs/flux4b_large_qkvo_r128 --resume \
  --images-root /workspace/datasets/large_dataset

# Optional full-cache experiment (requires substantially more disk):
bash scripts/run_flux4b_multi_id.sh \
  --run runs/flux4b_large_qkvo_r128_cached \
  --images-root /workspace/datasets/large_dataset --conditioning cached

# Configuration/plan only: no data reads, CUDA calls or preparation:
bash scripts/run_flux4b_multi_id.sh --run runs/flux4b_large_qkvo_r128 --dry-run
```

`--images-root` must point to the extracted directory expected by the Large
metadata, not its parent archive folder. `--metadata /path/filtered_ids3_adj.json`
can override the default private bundle location. If a full imported
`data/train_pairs_large.jsonl` already exists, the launcher verifies its import
audit and reuses it. The original Large archive URL is not saved in this
repository; downloading it requires `LARGE_DATASET_URL` or transferring the
existing archive. This launcher starts from downloaded images.

| Setting | Value |
|---|---|
| Backbone | Frozen FLUX.2-klein **Base 4B**, full denoiser in the gradient path |
| Trainable | Q/K/V/output BA LoRA only, rank128, 25,165,824 parameters |
| Training data | Full pinned adjusted Large, held-out validation identities removed via saved aliases |
| Pairing | Distinct same-ID views, deterministic next-view pairs, shuffled each epoch |
| Batch | **Microbatch1 × accumulation8 = effective8**; one GPU |
| Geometry | Target768×768, reference512 |
| Optimizer | AdamW, LR5e-5, warmup100, gradient checkpointing |
| Budget | 10,000 optimizer updates; full checkpoints every500 |
| Validation | Original fixed96 order/prompts/seeds/references, at0/2000/4000/6000/8000/10000 |
| Generation | 20 sampling steps, CFG4; masks/background from matching native generations |

Fresh adapters are initialized; the one-ID checkpoint is not loaded. The
launcher imports pairs, checks identity/image disjointness, caches only fixed96
validation inputs, runs pretrained native/off/gradient/frozen-weight/memory and exact
optimizer-resume checks, then generates and scores the native panel. It freezes
these native face masks and starts serial training/validation in one Comet run
under `rsrch_new`. ID_sim/CLIP, seven face-quality curves, paired face panels,
background preservation checks and `best_checkpoint.json` are retained. Missing
or ambiguous native faces stop initialization for review; correct their boxes
in JSON and resume with `--mask-overrides /path/reviewed_boxes.json`. Masks cannot
change after initialization. The native images are under `RUN_setup/native96/`.

Training uses `data.conditioning: online`: frozen text encoder and VAE remain
loaded alongside the denoiser and encode each selected pair under no-grad.
In the default online mode, no training embeddings or latents are written to disk.
`--conditioning cached` enables full precomputation as a separate named run.
Conditioning mode is frozen on resume; both modes retain the same loss and data. Encoder execution
preserves the flow noise/timestep RNG stream. Only the fixed96 validation
inputs are cached and reused across checkpoints. The earlier400–500GiB estimate
was for an unnecessary full-dataset embedding cache, explicitly rejected by the
user; it is **not** a requirement of this setup. Disk is needed for raw data,
weights, environments, checkpoints and validation outputs. `BA_ENVS_DIR` is
supported. The authorized destination is documented in `MACHINES.md`.

Admission requires at least45GB CUDA-addressable memory and peak reserved memory below90%. This is
a conservative starting batch configuration, not a measured48GB throughput
optimum. Only syntax, configuration, resume-selection and dry-run checks were
performed while preparing these scripts; no local data preparation or training
was run. Earlier one-ID runs retain their original source snapshots; use commit
`3714e7d` to replay that completed run exactly.

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
python3 scripts/sync_large_dataset_vast.py 53994096 --jobs 8
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
# Quick Cosmic hardware probe

For an explicitly named short hardware check, with a prepared Cosmic paired
manifest and the usual locked FLUX environment/weights:

```bash
envs/flux-toolkit/bin/python -m scripts.run_cosmic_smoke \
  --run runs/flux4b_cosmic_hardware15_qkvo_r128
```

The supplied config uses the 15 pairs available locally on 2026-10-03 because
the original Drive downloads became quota-blocked. It is **not** a full Cosmic
experiment. The probe trains 100 updates with live conditioning, effective
batch8 and rank128 BA only. It checks native/off parity and finite updates,
then saves/reloads the actual optimizer after2 updates. It omits generated
image panels and ID/CLIP scoring at the user's explicit request to confirm
hardware quickly. Only one validation input is cached for encoding parity;
the original fixed96 manifest is also read to check data overlap.

To use another prepared Cosmic manifest, copy the YAML to a new named config
and change `data.train_manifest`; pass `--config` and a fresh `--run` path.
The normal data command remains `python scripts/prepare_dataset.py cosmic`
(with project `PYTHONPATH=.`), using the original two Drive links. An HTTP200
HTML quota page is a download failure. The downloader falls back to Google's
public download endpoint when gdown's share-page parsing fails, but still
rejects access/quota error pages and checks download lengths.

## GB10: FLUX.2-klein Base 9B pilot

After account approval for the gated official9B repository, download the pinned
`locks/weights-flux80.json` components into the existing FLUX environment.
`configs/FLUX1_vast_9b_batch8.yaml` is a separate full-Large, fixed12
held-out pilot:768px, reference512, rank128 Q/K/V/output BA, fresh adapters,
online conditioning,2000 updates and validation0/1000/2000. The fixed12 panel
is intentionally smaller than the normal96 protocol. Native-generated9B
masks must be prepared by its controller; do not reuse4B masks/checkpoints.

Benchmark true microbatch1/2/4 at effective batch8 and choose by measured
throughput with memory headroom. The benchmark needs a small paired manifest
with original image paths, not a full training cache:

```bash
envs/flux-toolkit/bin/python -m scripts.benchmark_flux9b \
  --config configs/FLUX1_vast_9b_batch8.yaml \
  --manifest data/train_pairs_large_benchmark8.jsonl \
  --output runs/flux9b_gb10_benchmark

bash scripts/run_flux4b_multi_id.sh \
  --run runs/flux9b_large_qkvo_r128_pilot12 \
  --config runs/flux9b_gb10_benchmark/selected_config.yaml \
  --images-root /workspace/datasets/large_dataset \
  --pilot-panel --id-clip-only
```

The shared launcher supports both4B and9B despite its historical filename.
The second command performs pretrained parity, gradients and exact resume
admission, then native12/masks, initialization, training and serial scoring.
It retains original ID_sim/CLIP while omitting seven optional face-quality
models. True batches require identical token shapes without padding; the
adjusted Large images satisfy that layout. Each example retains its own
reference-face keys and fresh noise/timestep. Native LoRA is disabled.

For a resumed transfer with a slow last directory, use
`python3 scripts/sync_large_dataset_vast.py 53994096 --jobs 16 --balance-files`
Stop only the old transfer to that
same destination before launching replacement streams.

On3October the user explicitly requested immediate training. The dedicated
`scripts.run_flux9b_training_first` controller starts the full-Large9B pilot
without the preceding image panel/replay sequence. It saves actual checkpoint0,
trains to1000, evaluates0/1000 with newly generated native9B masks, then
trains/evaluates2000. This is an explicitly deferred initial evaluation.
Use the selected benchmark config, the same `--run`, and `--resume` only for
an existing training-first run. Its immutable training identity and ordinary
checkpoint compatibility checks remain enabled.

### Five-hour9B pilot on GB10

Use `configs/FLUX1_vast_9b.yaml` with
`scripts.run_flux9b_training_first --profile-validation`. This keeps9B,768px
and rank128 but uses effectivebatch1. Two thousand updates therefore process
2000 examples, versus16000 for effectivebatch8. It is a separate scientific
experiment. The training worker saves checkpoint0 and measures a conservative
validation timing estimate after update2 without advancing training RNG.
Measured initial projection:4.47h including all fixed12 panels and30min
loading/scoring reserve; see the run's five_hour_budget.json for actual values.
