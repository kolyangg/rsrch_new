# Branched attention for FLUX.2 and Qwen-Image-2.1

This repository adapts the reference-read branch from the PhotoMaker CL39 research to FLUX.2-klein Base 4B/9B and Qwen-Image-2.1 7B. The first comparison keeps native attention intact and adds a trainable low-rank correction to target tokens' read of reference tokens. It compares four modes on the same paired identity data: native, attention LoRA, branch only, and LoRA plus branch.

The implementation specification is [plans/260930/CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md](plans/260930/CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md). It is a design, not evidence that these model profiles fit or have been trained. [plans/260930/IMPLEMENTATION_NOTES.md](plans/260930/IMPLEMENTATION_NOTES.md) records measured progress and blockers.

## Layout

- `configs/`: 48 GB and 80 GB proposed profiles, including matched 80 GB comparisons.
- `scripts/`: pinned source acquisition, selective weight audit, validation-panel import, and launch helpers.
- `data/validation/`: the fixed PhotoMaker `manual_val` prompts and metadata. Source images and generated masks remain local and are ignored by Git.
- `ba_dit/`: new model/data/training integration code.
- `sources/`, `weights/`, `cache/`, `envs/`, `runs/`: local sources and large artifacts, ignored by Git.

## Setup and local validation

```bash
cd /home/kolyangg/rsrch_new
scripts/clone_sources.sh
python3 scripts/import_validation_panel.py --source /home/kolyangg/rsrch_apr_test/dataset_full/val_dataset
python3 scripts/build_face_masks.py
# After auditing/downloading the pinned model components (see the plan):
envs/flux-toolkit/bin/python scripts/validate_native_flux.py --arch 4b --steps 50 --output-dir runs/native_flux4b_local_50step
envs/qwen21/bin/python scripts/validate_native_qwen21.py --steps 40 --output-dir runs/native_qwen21_local_40step
```

Use the backend-specific environments and metadata-first `scripts/weights_manifest.py` audit in the plan before any model download. `--branch-initialized` on either validation script exercises the registered zero-initialized branch with a backend-specific reference mask. This has been checked for one sample at two steps; it is not a trained branch result. The six research YAML files are **proposed `ba_dit` schema**, not valid upstream Toolkit or Diffusers trainer inputs. `configs/flux4b_upstream_smoke.yaml` and `configs/flux9b_upstream_smoke.yaml` are upstream Toolkit LoRA smoke configurations. The fixed validation panel is the first 96 items of the original 12-reference × 12-prompt ordering, seed 0, with one generated image per item. The local feasibility check generated only its first item. Only the reference face boxes supply branch key masks; generated-image boxes are evaluation labels, never inference inputs.

Comet experiments use project `rsrch_new`; keep the API key only in untracked `.env` or the environment. `scripts/log_validation_comet.py` records the validation image, panel hash, source pin, model revisions, memory telemetry, and immutable experiment key. The planned research trainer must also record its resolved config and validate at step 0 and every 2,000 optimizer updates, matching the previous project's standard panel.

The local 16 GB GPU completed native FLUX 4B and Qwen 7B inference at 512 pixels; Qwen reserved almost the entire device. The training profiles target 48 GB and 80 GB GPUs and must pass the plan's measured memory gates before research runs. `scripts/train_flux_native_lora_smoke.sh 48|80` and `scripts/train_qwen_native_lora_smoke.sh 48|80` prepare native paired-LoRA plumbing checks on those GPU classes. The branch trainer, trained-branch validation, and identity metrics remain to be implemented; [implementation notes](plans/260930/IMPLEMENTATION_NOTES.md) list the exact remaining work and measured results.

`scripts/run_native_validation_profile.sh 48|80 flux|qwen [limit]` serially runs up to all 96 native panel items on the selected GPU class and uploads the result to Comet. The 48 GB profile uses FLUX 4B or Qwen 7B at 512 pixels; the 80 GB profile uses FLUX 9B or Qwen 7B at 1024 pixels with 768-pixel reference preprocessing. These launchers have not been run on a 48/80 GB host.
