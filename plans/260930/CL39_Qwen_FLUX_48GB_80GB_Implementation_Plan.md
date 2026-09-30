# CL39-style branched attention on Qwen and FLUX: 48 GB development → 80 GB research

**Updated:** 30 September 2026.  
**Replaces:** the Qwen/FLUX implementation instructions in `CL39_80GB_Repositories_LoRA_Branched_Attention_Plan.md`.  
**Scope:** FLUX.2-klein **Base 4B and Base 9B**, and **Qwen-Image-2.1 7B**. HiDream and LLaDA are deliberately outside this revision; the preceding four-model plan is retained separately as a historical reference.  
**Purpose:** an implementation handoff for another agent: acquire source and weights, isolate environments, establish paired LoRA training, add a registered branched-attention path, run memory-safe experiments, and scale the successful design.  
**Execution status:** source files and release documentation were inspected; the proposed configurations have **not been trained or memory-profiled on a 48 GB or 80 GB GPU**. “Fits” below means an engineering target subject to the mandatory acceptance tests, not a measured result. The accompanying source-clone and weight-audit helpers are supplied; the `ba_dit` research trainers and model patches are specifications to implement, not already finished software.

## Navigation

| Need | Read |
|---|---|
| Which small/large models actually exist? | 1–2 |
| Hardware profiles and memory acceptance | 3 |
| Clone sources, environments and weight acquisition | 4–5 |
| Paired data and exact caching requirements | 6 |
| Shared LoRA/branched-attention mathematics | 7 |
| Exact FLUX integration and 4B → 9B migration | 8 |
| Exact Qwen integration and 48 → 80 GB migration | 9 |
| Agent package, configuration and launch contracts | 10–12 |
| Tests, checkpointing and experiment promotion | 13–14 |
| Disk-space budgets and capacity recommendations | 15 |
| Ordered implementation work packages | 16 |
| Full research configurations and acquisition helpers | Appendices A–C |
| Source links | Source ledger |

---

## 1. Recommendation and what changes from the preceding plan

**For the user's specific strategy—test on a genuinely smaller model, then repeat on a larger version—start with FLUX.2-klein Base 4B on 48 GB, and promote the architecture to Base 9B on 80 GB.** The same native AI Toolkit implementation defines both models, so the code path, reference-token handling and branch insertion points can be shared. However, their widths differ: **transfer the implementation and experimental finding, not the trained 4B adapter tensors.** Reinitialize and retrain on 9B. [F4] [F5]

**For Qwen, use the same Qwen-Image-2.1 7B visual generator on both GPUs.** I verified one official 2.1 generator release, not a 4B/7B/14B size family. Its 48 GB profile saves memory through staged conditioning, a smaller image-token budget and parameter-efficient training. Its 80 GB profile supports a higher-resolution experiment and later additional reference images. This is a **small-compute → larger-compute configuration**, not small-model → large-model. Do not relabel an older 20B Qwen-Image/Edit checkpoint as a larger 2.1 sibling. [Q1] [Q2]

| Route | 48 GB development model | 80 GB next step | What transfers |
|---|---|---|---|
| **FLUX, recommended size-scaling route** | `black-forest-labs/FLUX.2-klein-base-4B` | `black-forest-labs/FLUX.2-klein-base-9B` | Branch/backend code, data policy, test suite and hypothesis; **not** width-specific BA/LoRA weights or text embeddings. |
| **Qwen, recommended second architecture** | `Qwen/Qwen-Image-2.1` | The **same checkpoint**, first identical settings, then higher resolution/reference budget | BA/LoRA weights can transfer unchanged when module structure/rank/sites are identical; geometry-dependent caches must be regenerated when their inputs change. |

The previous Qwen-first research recommendation and this FLUX-first development recommendation address different priorities. Qwen's causal prefix is attractive for reference routing. FLUX wins here because the requested **real 4B → 9B progression already exists in one training backend**. Neither recommendation is a measured quality ranking or a promise of SOTA.

### Non-negotiable distinctions

- **Use Base/undistilled FLUX checkpoints**, not the four-step distilled or separate KV checkpoint as a supposedly equivalent small model. Changing distillation/attention semantics is a separate experiment. [F1] [F2]
- A **Qwen3-4B or Qwen3-8B text encoder is not the Qwen-Image visual generator**. The optional Qwen-Image prompt-rewriting checkpoints are not larger image generators either. [F4] [Q1]
- Quantizing a 7B backbone does not produce a smaller-parameter architecture. Treat quantized training as a separate fallback configuration.
- Tiny synthetic transformer configurations may validate tensor plumbing on CPU, but their image-quality results say nothing about the pretrained model.
- Retain the paired identity task: **one adapter that generalizes to unseen people**, not one identity-specific LoRA per person.

## 2. Verified source, dimensions and ownership

### 2.1 Repositories and audited pins

| Workspace directory | Repository | Role | Revision policy |
|---|---|---|---|
| `sources/diffusers-qwen` | `https://github.com/huggingface/diffusers.git` | **Actual Qwen model, pipeline and paired LoRA trainer to modify** | `fef717ffb01f407d2637584ed936c16db908587a` |
| `sources/ai-toolkit-flux` | `https://github.com/ostris/ai-toolkit.git` | **Actual FLUX paired-training runner and native transformer to modify** | `ecee894ed2b1f3716d9d7326693061ec1a3105bb` |
| `sources/qwen-release` | `https://github.com/QwenLM/Qwen-Image-2.1.git` | Official release/inference conventions | Record resolved HEAD immediately after clone. |
| `sources/flux2-official` | `https://github.com/black-forest-labs/flux2.git` | Official native reference implementation | Record resolved HEAD immediately after clone. |
| Existing research worktree, or a new separate one | `https://github.com/kolyangg/rsrch.git` | New `ba_dit` modules/configs/tests | Record the user's actual starting commit; **do not reset or modify the existing PhotoMaker experiment branch**. |

These two audited training pins are carried forward intentionally. A package manager must not silently replace them with today's default branch. A GitHub **file blob SHA is not a repository commit**. Capture both source commits and changed-file hashes in every run.

### 2.2 Architecture facts that the installer must assert

| Property | FLUX Base 4B | FLUX Base 9B | Qwen-Image-2.1 |
|---|---:|---:|---:|
| Visual model family | klein Base | klein Base | Qwen Image 2.1 |
| Hidden width | 3,072 | 4,096 | 4,096; assert from loaded configuration |
| Attention heads / head width | 24 / 128 | 32 / 128 | 32 / 128; assert from loaded configuration |
| Transformer depth | 5 double + 20 single blocks | 8 double + 24 single blocks | 32 single-stream blocks |
| Conditioning encoder | Qwen3-4B | Qwen3-8B | Qwen3-VL 8B component bundled with the pipeline |
| FLUX text-feature width | 7,680 | 12,288 | Not interchangeable with either FLUX encoder output |
| Reference computation | Joint, target-dependent | Joint, target-dependent | Block-causal prefix; target-independent when its producers are frozen |
| Native image representation relevant here | Packed 128-channel tokens | Packed 128-channel tokens | 64-channel VAE latent space; image tokens grouped into 2×2 slots |

FLUX values are explicitly defined in `Klein4BParams` and `Klein9BParams`; do not derive the model size by changing only its marketing name. [F4] [F5] Qwen's release specifies the 7B/32-layer design, and its transformer/trainer specify causal conditioning and latent-slot handling. The agent must compare the remaining numerical dimensions against the actual weight configuration before installing adapters. [Q1] [Q3] [Q4]

**Access/licensing:** FLUX Base 4B is Apache 2.0; Base 9B is gated under the FLUX non-commercial model licence. Qwen 2.1 is distributed under the Qwen Research licence. Preserve the licence files and obtain authorized model access before downloading; a source-code licence does not replace a checkpoint licence. [F1] [F2] [W1]

### 2.3 Exact executable files

```text
# Qwen: under sources/diffusers-qwen/
examples/dreambooth/README_qwenimage21.md
examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py
src/diffusers/models/transformers/transformer_qwenimage21.py
src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py

# FLUX: under sources/ai-toolkit-flux/
run.py
requirements.txt
requirements_base.txt
extensions_built_in/diffusion_models/flux2/flux2_klein_model.py
extensions_built_in/diffusion_models/flux2/flux2_model.py
extensions_built_in/diffusion_models/flux2/src/model.py
extensions_built_in/diffusion_models/flux2/src/sampling.py
extensions_built_in/diffusion_models/flux2/src/pipeline.py
```

Both families have usable paired LoRA routes. **Neither upstream repository already implements this CL39-inspired branch.** Qwen's release repository is not itself the paired trainer: use Diffusers. FLUX's selected runtime is AI Toolkit's vendored native model: editing a Diffusers `Flux2AttnProcessor` in a different environment will not affect it. [Q3] [F3] [F4] [F6]

## 3. Two hardware setups, four primary research profiles

### 3.1 Starting configurations

All profiles freeze the base generator, train small adapters, use one GPU, use BF16 for the generator, and start with **one reference and microbatch 1**. Gradient accumulation keeps the effective batch at eight pairs. The pilot resolutions are proposals to measure, not claims that a native 2K model has already been validated at every lower resolution.

| Setting | `flux4b_48` | `flux9b_80` | `qwen7b_48` | `qwen7b_80` |
|---|---|---|---|---|
| GPU budget | 48 GB class | 80 GB class | 48 GB class | 80 GB class |
| Generator | klein Base 4B | klein Base 9B | Qwen 2.1 7B | Same Qwen 2.1 7B |
| Target / reference pilot size | 768 / 512 px | 1024 / 768 px | 768 / 512 px | 1024 / 768 px |
| Microbatch × accumulation | 1 × 8 | 1 × 8 | 1 × 8 | 1 × 8 |
| Branch adapters | K and V, rank 16 | K and V, rank 16 | K and V, rank 16 | Same modules/rank as 48 GB |
| Selected branch sites | 4 double + 4 single | 4 double + 4 single | 8 single-stream blocks | Same 8 blocks |
| Native LoRA comparison | Rank 8, attention-only updates at the same sites | Same policy | Same policy | Same policy |
| Checkpointing | Non-reentrant | Non-reentrant | Non-reentrant | Non-reentrant |
| Encoders/VAE during training | Not resident on GPU; omitted from training-only loader | Same | Same | Same |
| Deep reference caches | Off | Off | Off initially | Off initially |
| Identity/perceptual image loss | Off initially | Off initially | Off initially | Off initially |
| Validation | Separate process, serial samples | Same | Same | Same |

**Mandatory matched-size promotion profiles:** `flux9b_80_matched.yaml` and `qwen7b_80_matched.yaml` retain **768 target / 512 reference**, all other pilot training settings and the same data split. Run these before the expanded 1024/768 profiles. Otherwise a “large model” result also changes resolution and available reference detail.

For the very first native/BA smoke test, use **512/512** on either family, one or two selected branch sites, and no image decoding during optimization. This tests code, not quality. Afterward run the exact intended 768/512 profile; passing 512/512 does not establish that it fits.

### 3.2 Why 48 GB is a credible target

The large frozen encoders do not need to be on the GPU during the optimization step. Separate preparation from training:

```text
Process A: text / multimodal conditioning encoder → CPU/disk outputs → exit
Process B: VAE target + reference encoding → CPU/disk latents and metadata → exit
Process C: generator + small trainable adapters + one cached pair → optimize
Process D: serial validation / decoding in a separate process
```

This is more reliable than setting an `offload` flag while accidentally retaining the entire pipeline on CUDA. It also avoids a second full generator or a PhotoMaker-style duplicated denoiser. The native model already contains reference-image representations; the new branch reads those features.

| Frozen visual weights only | Approximate checkpoint-scale BF16 storage | Meaning |
|---|---:|---|
| FLUX 4B | 7.75 GB ≈ 7.2 GiB | Not a complete training-memory estimate. |
| FLUX 9B | 18.2 GB ≈ 17.0 GiB | Still leaves substantial 80 GB headroom before activation accounting. |
| Qwen 2.1 | About 14.2 GB ≈ 13.2 GiB | The separate ~17.5 GB encoder is staged out of the training step. |

Sizes are rounded observed checkpoint figures, not allocator measurements. [W1] [W2] [W3] [W6] Actual resident tensors can differ from stored bytes due to dtype, buffers, module loading and padding.

**Do not mistake “frozen” for “no backward graph.”** Later backbone operations must remain differentiable with respect to earlier adapter outputs. FLUX reference states also receive target-dependent changes through joint attention. Do not put the whole transformer or its reference rows under `no_grad()` merely because base weights are frozen.

With eight sites, K/V branch LoRA has approximately `4 × hidden_width × rank × sites` parameters: **1.57 million for 4B and 2.10 million for 9B/Qwen at rank 16**, excluding tiny gates. At 16 bytes of training state per parameter, that is only roughly 25–34 MB. Activations, encoder residency, sequence length and attention kernels—not this small optimizer state—are the dominant design concerns. This is arithmetic for the specified branch, not the size of an arbitrary upstream all-layer LoRA.

Full-model training is out of scope: at an illustrative 16 bytes per trainable parameter, a 4B model already requires ~64 GB before activations. Gradient accumulation does not fix an OOM occurring at microbatch 1.

### 3.3 Resource acceptance and fallback order

Use the device's reported capacity, not a literal assumption that “48 GB” is 48 GiB. Require peak reserved CUDA memory **≤90% of reported device memory**, plus an external free-memory/NVML check for other processes. Run at least 100 optimization steps after a warmup, covering the largest target/reference/text layout, save/resume and separate validation. Record step time, peak allocated/reserved bytes, process GPU usage and process RSS.

If the 48 GB profile fails, investigate in this order:

1. Verify that no text encoder, VAE, second generator, dense attention capture, full-model EMA or full frozen-weight gradient has survived into training.
2. Verify native memory-efficient attention and non-reentrant checkpointing; bound diagnostic query chunks and the extra branch's reference support.
3. Reduce **actual native reference pixels** and/or target size; lower-resolution references must be re-encoded, not just masked at the branch. Never report a branch-only token cap as a reduction in native sequence length.
4. Reduce active sites for an engineering smoke test; lower rank only if parameter state is actually material. Fewer sites do not remove all later backward activations.
5. Use an explicitly named frozen-base **NF4 QLoRA** experiment for Qwen only after a BF16 control and gradient tests. The paired trainer documents a bitsandbytes quantization configuration, but the custom branch still needs validation. Quantized inference formats such as arbitrary GGUF/ConvRot files are not drop-in training checkpoints. [Q3]
6. CPU activation/offload or block swapping is a last, separately measured engineering option. Do not insert an inference-only offload hook into training without checking autograd correctness.

For 80 GB, validate **1024/768 first**. Qwen 1536 or native 2048 targets and multiple references are later token-budget sweeps; **no 2K-plus-multi-reference training fit is asserted here**. Keep microbatch 1: Qwen's trainer requires a compatible shared token layout within a batch, not merely enough RAM. [Q3]

Host provisioning estimates: **64 GB CPU RAM is a lean starting point; 128 GB is preferred** for both environments and staging. For larger datasets, multiple workers, conversion copies or CPU offload, 128–256 GB is safer. Use bounded per-worker caches and disk streaming; do not materialize all conditioning tensors as a Python list in RAM.

## 4. Sources and isolated environments

### 4.1 Workspace

Use a persistent Linux filesystem on the GPU host. On WSL, prefer the Linux filesystem for numerous cache files rather than placing the working tree on a Windows-mounted drive. Existing PhotoMaker environments and results remain untouched.

```bash
export BA_ROOT="${BA_ROOT:-$HOME/work/cl39_dit}"
mkdir -p "$BA_ROOT"/{sources,envs,configs,data,runs,locks,weights,cache,scratch}
export HF_HOME="$BA_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$BA_ROOT/cache/pip"
export TORCH_HOME="$BA_ROOT/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$BA_ROOT/cache/inductor"
export TMPDIR="$BA_ROOT/scratch"
```

Set these variables **before Python imports**. All environments use the same model-file cache, but have separate site-packages. See Appendix B for the complete clone script. It refuses to overwrite an existing source directory and pins the two effective training repositories.

The code to implement can live under a new `diffusion_template/ba_dit/` package in the research repository or a separate package in this workspace. Keep legacy `lora.py`/`lora2.py` changes minimal or zero; do not make the new DiT runtime conditional on importing the PhotoMaker pipeline.

### 4.2 Preflight before installations

```bash
nvidia-smi
python3.11 --version
# Needed only when building a CUDA extension, not for ordinary prebuilt Torch wheels:
command -v nvcc >/dev/null && nvcc --version || true
```

A GPU's VRAM size does not determine driver, wheel or kernel compatibility. In particular, a previously working CUDA 12.x PhotoMaker installation is not evidence that a CUDA 13 wheel will work. Do not change the host driver as an unannounced dependency-install side effect.

### 4.3 Qwen environment

Use the audited Diffusers checkout and a separate environment. The following preserves the original plan's published Torch 2.10 / torchvision 0.25 CUDA 12.8 starting pair and Qwen's Transformers requirement. It is a **candidate complete environment**, not an already-tested lock. [Q1] [Q3] [T1]

```bash
set -euo pipefail
export BA_ROOT="${BA_ROOT:-$HOME/work/cl39_dit}"
python3.11 -m venv "$BA_ROOT/envs/qwen21"
source "$BA_ROOT/envs/qwen21/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
cd "$BA_ROOT/sources/diffusers-qwen"
python -m pip install -e .
python -m pip install -r examples/dreambooth/requirements_flux.txt
python -m pip install 'transformers>=5.17,<6' 'peft>=0.14' \
  datasets tensorboard pillow pyyaml pytest
python -m pip check
python -c 'from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel; from transformers import Qwen3VLProcessor; print("Qwen imports OK")'
python -m pip freeze > "$BA_ROOT/locks/qwen21.pip-freeze.txt"
```

Pin the resolved Transformers/PEFT/Accelerate versions after the first successful model-level backward test. Subsequent machines must reproduce that tested resolution rather than resolving the inequalities again. Do not install optional prompt-enhancement models, vLLM/SGLang servers or a separate FlashAttention build for the first training experiment.

### 4.4 FLUX / AI Toolkit environment

At the audited Toolkit revision, its installation instructions request Torch **2.13.0**, torchvision **0.28.0**, CUDA **13.0**. Its `requirements_base.txt` pins Transformers **5.5.3**, PEFT **0.18.1**, and its own Diffusers Git revision. Keep that environment separate from Qwen. [F3] [F7]

```bash
set -euo pipefail
export BA_ROOT="${BA_ROOT:-$HOME/work/cl39_dit}"
python3.11 -m venv "$BA_ROOT/envs/flux-toolkit"
source "$BA_ROOT/envs/flux-toolkit/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cu130
cd "$BA_ROOT/sources/ai-toolkit-flux"
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pip check
python -c 'from extensions_built_in.diffusion_models.flux2.flux2_klein_model import Flux2Klein4BModel, Flux2Klein9BModel; print(Flux2Klein4BModel.arch, Flux2Klein9BModel.arch)'
python -m pip freeze > "$BA_ROOT/locks/flux-toolkit.pip-freeze.txt"
```

**Stop on unavailable wheels, incompatible drivers or a dependency conflict.** The upstream-requested Torch pair is not independently validated here as a complete compatible stack. Resolve a supported environment for this pinned Toolkit in a fresh environment and record the change; do not silently downgrade Torch while retaining unrelated binary packages. Python 3.12 is recommended by Toolkit, while its documented minimum also admits 3.11. Node/UI and torchaudio are not needed for this image-only command-line workflow. [F3]

Before each training run, log the imported model/block file paths with `inspect.getfile`, all installed package versions, GPU/driver/runtime, source commits and the adapter parameter inventory. An import from an unpatched checkout is a failed preflight even if its class name looks right.

## 5. Acquire exactly the needed weights

### 5.1 Selected components

| Logical local alias | Download source | What to retain |
|---|---|---|
| `weights/flux4b` | `black-forest-labs/FLUX.2-klein-base-4B` | **Root** `flux-2-klein-base-4b.safetensors`, licence/readme; not the additional Diffusers transformer copy. |
| `weights/flux9b` | `black-forest-labs/FLUX.2-klein-base-9B` | **Root** `flux-2-klein-base-9b.safetensors`, licence/readme. |
| `weights/flux4b_text` | `Qwen/Qwen3-4B` | Safetensors shards, config, index, tokenizer/chat-template files. |
| `weights/flux9b_text` | `Qwen/Qwen3-8B` | Same categories, for this distinct encoder. |
| `weights/flux_vae` | `ai-toolkit/flux2_vae` | `ae.safetensors`; shared by the native Toolkit klein routes. |
| `weights/qwen21` | `Qwen/Qwen-Image-2.1` | Diffusers `transformer/`, `text_encoder/`, `vae/`, `processor/`, `scheduler/`, `model_index.json` and licence. |

These choices follow the **actual loaders**. Toolkit normally retrieves its native root checkpoint, Qwen3 text model and a separate VAE. The proposed cached loader must accept their **locked local paths**, including a text-encoder path override; its stock hard-coded remote encoder ID is not a reproducibility mechanism. [F4] [F6]

Qwen's Qwen3-VL component and FLUX's Qwen3-8B text encoder are different models; **do not deduplicate them by name or substitute one for the other**. Do not download optional 9B prompt rewriters, duplicate ComfyUI weight layouts or several quantization formats for this initial plan.

### 5.2 Audit before download

Appendix C supplies `scripts/weights_manifest.py`. Its `audit` command uses HF metadata only, verifies required file groups and non-missing byte sizes, resolves each model to a full commit, and writes an immutable file-selection lock. Its `download` command consumes that lock and creates local symlinks into the shared HF cache. Authentication uses the normal HF token/login mechanism. No training or automatic model-format conversion occurs.

```bash
# In either environment with huggingface_hub installed; authenticate if needed:
hf auth login

# Acquire just the 48 GB pilots first:
python scripts/weights_manifest.py audit --set small \
  --lock "$BA_ROOT/locks/weights-small.json"
# Review selected files, total bytes, permissions and available disk before this step:
python scripts/weights_manifest.py download \
  --lock "$BA_ROOT/locks/weights-small.json" --weights-dir "$BA_ROOT/weights"

# On promotion, audit all components with a NEW lock filename:
python scripts/weights_manifest.py audit --set all \
  --lock "$BA_ROOT/locks/weights-all.json"
```

**Promotion revision check:** before downloading `weights-all.json`, compare overlapping components against `weights-small.json`. If an upstream main revision changed, do not replace the old aliases. Either retain the already-locked revision for that component in the new reviewed lock, or use a separately named alias/run and rerun parity. The helper intentionally refuses to replace an alias pointing at another snapshot. `--set flux48`, `flux80` or `qwen` also audits a single route.

File sizes displayed in this document are rounded source observations; the helper supplies the exact selected-byte totals on the implementation host. Its network execution was not performed in this session. An authentication failure is not permission to download an unofficial mirror.

Set `local_files_only=True` where the backend supports it during training; missing files should stop the run rather than silently download a newer revision. Do not copy a whole HF snapshot with symlink dereferencing into every environment. The HF cache uses snapshot references and stored blobs; physical deduplication across unrelated repositories is not guaranteed. [C1]

## 6. Paired identity data and bounded conditioning caches

### 6.1 Input manifest

Use one JSONL row per reference/target **cross-view pair**, with split by identity. Paths may be relative to a declared dataset root; resolve and validate them before caching.

```json
{"sample_id":"person004_view01_to_view07","identity_id":"person004","reference_images":["images/person004/01.jpg"],"reference_face_masks":["masks/person004/01_face_head.png"],"target_image":"images/person004/07.jpg","target_face_mask":"masks/person004/07_face_head.png","prompt":"A photograph of the person in the reference, smiling outdoors in a red jacket.","split":"train"}
```

`target_face_mask` is an optional **supervision/evaluation label**. It must not be supplied as an oracle conditioning input in a result claimed to be ordinary reference-only generation. The primary branch pilot uses an all-target output gate; a later learned localizer consumes model features at inference, not this label.

Reject missing/empty references, accidental same-file pairs, malformed masks and identity overlap between training and evaluation. Keep a fixed pilot panel plus a broader held-out-identity evaluation. Ordinary per-person DreamBooth examples only prove trainer plumbing; they do not prove generalization to unseen identities.

### 6.2 Geometry rules

Preserve original images. Write deterministic transform metadata separately for each reference and target: resize, crop, padding, flip, encoded pixel dimensions, latent grid, token grid, and original-to-token mapping. **Reference and target transforms are independent for cross-view identity pairs.** Do not inherit an aligned editing transform that crops both images using one image's coordinates. [Q3]

For each input, require the face/head crop to survive preprocessing and transform its mask identically. Non-square examples must be supported through explicit `(height,width)` metadata even if the first four profiles are square. Do not infer a square grid from `sqrt(sequence_length)`; Qwen's joint sequence contains text and image-slot structure, and FLUX single blocks contain text plus all images.

Qwen requires sizes divisible by **32** and opaque alpha channels for RGB inputs because its VAE consumes RGBA. Its VLM image-slot accounting must match the clean-reference latent grid. A 512-pixel reference with a 768-pixel target is a **custom-loader capability to implement and assert**, not an existing flag on the aligned `_img2img` trainer. Begin with equal-size pairs for upstream smoke tests; unequal sizes are accepted only after the exact native inference layout and slot-count test pass. [Q3] [Q4]

For FLUX, use its native reference preprocessing and position-ID creation. The stock wrapper can use a **1024²** reference-pixel limit even when target resolution is smaller. Merely adding a `reference_size: 512` key to YAML changes nothing until the loader consumes it and logs the encoded reference grid. [F6]

### 6.3 Cache contents and keys

**FLUX cache entry:** exact text embeddings as expected by `batched_prc_txt`; target latent sample or stored posterior parameters according to the chosen policy; reference packed latent sequence; corresponding native position IDs; reference/target token grids and transformed masks. If the native VAE path computes an additional packing, normalization or channel rearrangement, cache **after reproducing that path** rather than guessing an SDXL scaling constant.

**Qwen cache entry:** joint text-plus-reference VLM embeddings, their mask and image-pad/slot layout; clean target/reference VAE latents; original image-shape metadata; transformed masks; relevant processor outputs used to reconstruct the joint token metadata. A prompt-only cache is wrong: its encoder representations depend on the reference image. [Q3]

At minimum, the cache key hashes:

```text
schema version + backend + source revision + model/encoder/VAE revisions
+ exact prompt/chat template/processor options
+ ordered reference content hashes + target content hash for target latents
+ resize/crop/flip/padding/alpha conventions + mask content hashes / mask-schema version
+ actual encoded pixel/latent/token grids
+ posterior-sample or posterior-mean policy and sample seed
+ selected encoder hidden layers/output convention + stored dtype
```

Factor keys by component so FLUX can reuse identical text conditioning across pairs when it is truly identical. Do not store ten thousand physical copies of one common prompt tensor. Qwen multimodal embeddings generally cannot be deduplicated by prompt text alone.

Store **exact native returned tensors and layout initially**, including meaningful padded positions. Do not strip FLUX padding or change text length as a disk optimization unless native attention-mask equivalence has been proved. Qwen's processor does not expose the old `--max_sequence_length` shortcut; prevent unbounded prompts through dataset curation and explicit admission checks, not an unsupported trainer flag. [Q3]

Caching VAE posterior means rather than samples, or fixing one sample per image rather than sampling every epoch, changes the training distribution. Choose and record the policy for all comparison arms. The recommended first cached baseline follows the upstream latent-sampling convention with a fixed cache seed. A later cache can store posterior parameters to resample, but should not be introduced silently mid-comparison.

### 6.4 Required loader changes—not just configuration flags

| Backend | Existing behavior | Required 48 GB implementation |
|---|---|---|
| Qwen | Per-sample conditioning and optional latent caching in the paired trainer | A **disk-backed**, bounded cache loader; a training-only process that loads the transformer without recreating the VLM/VAE pipeline; support independently sized reference/target grids and exact slot metadata. |
| FLUX | `cache_latents_to_disk` exists, but `get_noise_prediction` can still encode control images through the VAE every step | Add a cache path for **reference packed tokens and position IDs**; bypass live reference encoding when a validated cache exists; add a transformer-only load mode and local encoder/VAE path overrides. |

Cache preparation must write atomically and resume safely. A cache miss in a long training run should fail with the required key, not reload encoders on CUDA. Keep a bounded CPU cache (e.g. 4 GB), zero persistent CUDA sample cache, and a small worker count initially. DataLoader workers should not each load their own model or an entire cached dataset.

**Do not persist per-layer reference KV for every pair.** This is not needed for either primary profile. Qwen deep prefix caching is a later optimization, and FLUX's ordinary reference stream is not independently cacheable. Disk implications are quantified in Section 15.

## 7. Shared trainable branch: preserve native attention, then correct its reference read

### 7.1 What transfers from CL39

Retain the principle of a **separately controllable reference attention message**, optionally restricted to face/head support and applied locally to target-image tokens. Do not transplant the old doubled SDXL batch, timestep schedule, 2,048-dimensional PhotoMaker ID encoder or square-grid mask helper into a joint-attention DiT.

The earlier handoff described a safer DiT adaptation: native attention remains intact, and the extra branch changes how the model reads reference features. This is **CL39-inspired**, not a claim that the exact original `reference_message − target_self_attention` calculation is unchanged.

For a selected layer and target query rows:

```text
N  = native attention output, with all native text/target/reference support
R0 = Attention(Q_native_target, K_native_reference, V_native_reference)
R1 = Attention(Q_native_target, K_adapted_reference, V_adapted_reference)
Δ  = R1 − R0
Y_target = N_target + gamma × G_target × Δ
Y_non_target = N_non_target
```

Apply this **before the original attention-output projection**, with the original projection, dropout, modulation gate and residual applied exactly once. R0 and R1 use **the same gathered reference indices**, including the same position IDs and attention support. The extra reference read is normalized over its selected reference keys; it is not claimed to be the exact reference contribution inside the native joint-attention softmax.

A literal `R − N` transplant can be retained as a separately named ablation, but it subtracts text/composition-containing information from joint attention. It is not the default port.

### 7.2 Low-rank K/V adaptation without a second model copy

At reference rows, obtain the exact hidden states entering the native projection after that block's native normalization/modulation:

```text
K0_pre = native_effective_K_projection(H_ref)
V0     = native_effective_V_projection(H_ref)
K1_pre = K0_pre + (alpha/rank) × B_K(A_K(H_ref))
V1     = V0     + (alpha/rank) × B_V(A_V(H_ref))
K0 = native_RoPE(native_K_norm(K0_pre), original_reference_positions)
K1 = native_RoPE(native_K_norm(K1_pre), original_reference_positions)
```

The native Q supplied to both reads already has its correct normalization and RoPE. Preserve head layout, dtype and attention scaling. Apply RoPE only once to each adapted key; do not add a projection-space residual to an already rotated K and call it equivalent.

**Native-effective means native LoRA is included when enabled.** In a joint LoRA+BA run, R0 must not silently read the original frozen K/V weights while R1 reads LoRA-modified weights. This would make BA nonzero for the wrong reason and break the zero-effect baseline.

For fused FLUX projections, use the effective K/V slices described in Section 8. Do not allocate full dense copied Q/K/V weights per site. Four low-rank matrices per site are enough for this initial branch.

### 7.3 Initialization that preserves the base and still learns

Initialize each A with a normal low-rank initialization, each B to zero, `alpha=rank`, and set a fixed nonzero `gamma=0.1` for the first pilot, with a configured maximum of 0.3. The exact gamma is a proposed research hyperparameter, not inherited validation evidence.

Then `R1=R0` at initialization, but B matrices can receive gradients. **Do not initialize both a multiplicative gate and the projection update to zero**; that can block the first useful gradient. Initially A gradients may be zero because B is zero; B should receive a finite nonzero gradient on a non-degenerate reference example, followed by A after B changes.

Keep R0 differentiable with respect to target Q and any trainable upstream path. Detaching R0 can preserve forward values at step zero while changing input gradients. Test forward **and input-gradient** parity against the same native model, including a native-LoRA-enabled test case.

### 7.4 Native LoRA baseline and parameter ownership

Implement four modes:

| Mode | Trainable parameters | Purpose |
|---|---|---|
| `native` | None | Pretrained baseline and exact BA-off parity. |
| `lora_only` | Native attention LoRA | Benefit from paired training without the separate branch. |
| `branch_only` | Branch K/V low-rank matrices | Causal value of the new path. |
| `lora_plus_branch` | Both groups | Interaction; compare against both simpler arms. |

In the proposed research profiles, native attention LoRA uses **rank 8**, while BA K/V uses **rank 16** at the same sites. For square D×D projections, native Q/K/V/O rank 8 and BA K/V rank 16 each have approximately `128D` parameters per site. This gives a useful matched-budget control. It does not mean the stock all-layer Toolkit/DreamBooth smoke configuration already has that exact parameter budget.

Qwen native LoRA targets `to_q`, `to_k`, `to_v` and `to_out.0` **only at selected sites**. Use PEFT or an equivalent registered LoRA wrapper, then print the exact expanded module list. Do not let a name matcher also wrap branch A/B modules.

For FLUX, the research `attention_only_rows` policy must handle fused layers explicitly:

- Double blocks: native image `qkv` and image attention-output projection. Use separate rank-8 A/B pairs for each Q, K and V slice, not one shared rank-8 factor spanning the combined 3D output; the latter has a different parameter count. Use separate rank-8 A/B pairs for each Q, K and V slice, not one shared rank-8 factor spanning the combined 3D output; the latter has a different parameter count.
- Single blocks: independent Q/K/V slice adapters on `linear1`, excluding its parallel-MLP rows; and the attention-input columns of `linear2`, excluding its MLP-input columns.
- A whole-matrix LoRA on single-block `linear1`/`linear2` also changes MLP computation. That is a valid **different** baseline, not an attention-only update. Stock Toolkit LoRA can be used for the initial smoke test, but the matched control needs these slice-aware wrappers.

Register modules **before** optimizer construction and `accelerator.prepare`. Count each parameter once. Base parameters stay frozen; frozen normalization parameters are not accidentally re-enabled. Optimizer groups have separate branch/native-LoRA learning rates even when both start at `1e-4`. Do not make trainable modules lazily inside a processor call. Keep trainable adapter parameters and optimizer state in FP32, with projection computation under BF16 autocast and outputs returned in the native activation dtype. Avoid a later whole-model BF16 cast that unintentionally downcasts the registered trainable parameters.

### 7.5 Mask support, efficiency and the full CL39-style progression

The primary 48 GB experiment changes only the reference read. It uses a face/head **reference-key mask**, a deterministic spatially stratified cap of 512 keys for the extra reads, and `G_target=1` on target-image rows. It is a broad target correction, **not yet the complete spatial CL39 port**. Native attention still sees its full original reference image.

Gather valid reference keys rather than multiplying background hidden states by zero. A zero key can still participate in a softmax. Use identical support in R0 and R1. For an empty valid reference mask, return exactly zero correction and record the event; never invoke an all-masked softmax or substitute an unrelated reference silently.

Chunk target queries for R0/R1, initially 128 at a time. Do not materialize full native attention probability matrices to calculate diagnostics. A cap only on the new read saves **branch** memory; only reducing the encoded reference image reduces **native** sequence memory.

After the simple branch and matched LoRA control work, add these as separate experiments:

**Spatial gate.** Introduce a small target localizer, for example `Linear(D+4,64) → SiLU → Linear(64,1)`, using detached pre-branch target hidden features plus four fixed sigma features. Supervise it with the downsampled clean-target face/head mask through a balanced BCE loss, but use its predicted probabilities for routing at inference. Warm up the localizer with the generator frozen and branch disabled; log performance across sigma bins. Its mask label is training supervision, not an inference input. A high-noise suppression schedule must use normalized sigma and be the same during training/inference. Compare against the global gate; poor localization should not be misdiagnosed as lack of branch value.

**Confidence attenuation.** Add a detached, normalized reference-attention entropy or another calibrated confidence estimator using the actual valid reference support. Compute it in chunks. Recalibrate thresholds for the new architecture; do not blindly copy CL39's 0.25 floor or entropy thresholds.

**Frequency split.** On the target's explicit `(H_tokens,W_tokens)` grid, split Δ into low/high spatial components and gate their strengths over normalized sigma. Apply to target-image tokens only; preserve padded/missing positions with mask-aware normalization if sparse query support is used. Initial gains should reproduce the unfactored Δ so the extra filtering has its own no-op test. Do not blur over concatenated text/reference positions.

These later modules must remain identical between 48/80 GB matched runs. They can plausibly improve localization/detail, but none has been measured on these new backbones in this plan.

## 8. FLUX implementation: one backend for 4B and 9B

### 8.1 Establish an unmodified paired LoRA smoke test first

Use `Flux2Klein4BModel` (`arch: flux2_klein_4b`) in Toolkit, not a FLUX.1 model class. The 9B class is `Flux2Klein9BModel` (`arch: flux2_klein_9b`). Both inherit the reference-aware `Flux2Model.get_noise_prediction`. [F4] [F6]

For upstream smoke data, export paired images into separate target/reference directories, using the same **pair basename** in each, with a `.txt` instruction next to the target. Do not overwrite one identity's different target/reference pairings by exporting with identity ID alone. Prefer filesystem links where valid; record the export mapping. A short `sd_trainer` configuration is included in Appendix A separately from the new research schema.

Run 100 steps at 512/512 with sampling disabled. The stock path can load text encoder/VAE and re-encode references; it is a functionality baseline, **not the final 48 GB transformer-only design**. Its broader all-layer LoRA is also not the matched-parameter research control.

### 8.2 Exact native sequence and patch seams

In `Flux2Model.get_noise_prediction`, capture layout **before** concatenation:

```text
batched_prc_img(target_latents) → target packed tokens + target IDs
encode_image_refs(...)        → reference packed tokens + reference IDs
x = concatenate(target tokens, reference tokens)
```

Retain the native conditioning convention, clean reference tokens, t/1000 handling, output slice to target length, and native flow-matching velocity loss `noise − clean_target_latents`. Do not apply the old PhotoMaker reference-noising process to an input the new model expects to be clean. [F6]

Modify `Flux2.forward` to accept an optional immutable `branch_context`, pass it to the selected blocks, and include it explicitly in checkpoint closures/calls. Derive all widths/head counts from the loaded model and assert them against the intended profile. [F5]

**Double-stream insertion:**

```text
image/text native normalization and modulation
img_attn.qkv(image_hidden), text qkv
native Q/K normalization + native positional handling
native joint attention over text + target + references
split text output / image output
    >>> image_output[target_indices] += gated_reference_read_delta <<<
img_attn.proj → native image gate/residual → native MLP
```

Q/K/V occupy three D-width slices of fused `img_attn.qkv`. Obtain the K/V low-rank inputs from the exact modulated image hidden state used there. Only target image output rows receive the correction; the reference/text rows from **this block** remain unchanged directly.

**Single-stream insertion:**

```text
x = [text, target, references]
x_mod → linear1 → split(qkv, parallel_mlp_input)
native Q/K normalization + positional handling + attention
    >>> attention_output[target_indices_in_joint_sequence] += delta <<<
concatenate(attention_output, activation(parallel_mlp_input))
linear2 → native gate/residual
```

The first `3D` rows of `linear1` produce Q/K/V; later rows belong to the parallel MLP. Add the branch to the attention component **before** concatenation with that MLP component. Offset image token indices by the actual text length. Do not infer reference rows as the entire suffix without retaining each image's size and position IDs. [F5]

### 8.3 Exact 48 GB loader changes

Add a training-only construction path:

1. Resolve the native generator's local root safetensors file from the locked alias.
2. Load its state on CPU and construct `Flux2` using `Klein4BParams`; preserve native dtype/weight-loading behavior.
3. Install native attention-only LoRA and/or BA modules, then move the single generator to GPU.
4. Construct the native scheduler and cached-pair adapter **without instantiating text encoder or VAE**.
5. Replace live `encode_image_refs` calls with validated cached packed sequences and IDs. Preserve all tensor dtypes, channel order, normalization, time/position IDs and target output slicing.
6. Attach the branch parameters to the actual Toolkit optimizer and add branch-aware save/load hooks. Implement the cached route as a dedicated trainer extension or a narrow option in the existing trainer; do not duplicate the entire training framework.
7. Install the same native branch wrappers in a serial inference runner. A stock unpatched Diffusers pipeline cannot execute this native branch checkpoint automatically.

The present `Flux2Model.load_model` normally constructs encoders as well, and generation pipeline creation can move them to CUDA. A YAML “offload” assertion alone is not sufficient. The preflight must inspect live modules and peak memory. [F6]

### 8.4 Site maps and 4B → 9B promotion

Proposed zero-indexed sites:

```text
4B: double_blocks [1,2,3,4]; single_blocks [3,8,13,18]
9B: double_blocks [2,4,6,7]; single_blocks [4,10,16,22]
```

These cover comparable portions of the respective stacks, but are not a claim of optimal or exact layer correspondence. Keep them in configuration; fail on nonexistent sites rather than clipping indices.

Promotion procedure:

1. Establish branch value on 4B over its own pretrained and matched native-LoRA baselines, including quality/pose metrics and branch-only reference interventions.
2. Acquire the 9B **Base** generator and Qwen3-8B encoder. Reuse the same AE only after revision and preprocessing equivalence checks.
3. Construct the same branch code using D=4096/32 heads and the 9B site map. **Initialize new BA/native LoRA weights; do not load the 4B tensors with `strict=False`, interpolate them, or pad them.** Such transfer would itself be a new research method.
4. Recompute text embeddings: Qwen3-4B's 7,680-wide conditioning is not the 9B model's 12,288-wide conditioning. Reuse image latents only if exact AE revision, normalization, transform and cache policy match.
5. Start a **new optimizer/scheduler/run**. Train `flux9b_80_matched` using the original 768/512 data budget and settings.
6. Compare the **within-backbone BA improvement**, not just the larger model's absolute score against 4B. Larger generators can improve baseline quality while reducing BA's incremental value.
7. Only then run `flux9b_80` at 1024/768, followed by optional rank/site changes as separate ablations.

There is no need to download a 32B FLUX.2-dev checkpoint for this progression. It changes memory, conditioning and engineering constraints and is deliberately not the planned “big” model.

## 9. Qwen implementation: the same 7B model on 48 and 80 GB

### 9.1 Native paired LoRA smoke command

The existing trainer is `examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py`. It supports a dataset with target image, condition image and caption columns. The condition enters through both the VLM and clean latent sequence; loss is on target output only. [Q3]

For a local dataset, write a small Hugging Face datasets builder/loader compatible with this upstream trainer and verify its `load_dataset` path, or use an already accessible paired dataset for the smoke test. Do not automatically upload private identity photographs to the Hub. Our final local JSONL manifest support is a **new loader**, not an existing `--manifest` flag.

```bash
source "$BA_ROOT/envs/qwen21/bin/activate"
cd "$BA_ROOT/sources/diffusers-qwen"
accelerate launch --num_processes 1 \
  examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py \
  --pretrained_model_name_or_path="$BA_ROOT/weights/qwen21" \
  --dataset_name="$PAIRED_DATASET_NAME_OR_SUPPORTED_LOCAL_BUILDER" \
  --cond_image_column=cond_image --image_column=image --caption_column=caption \
  --instance_prompt="Photograph the person from the reference in the requested scene." \
  --output_dir="$BA_ROOT/runs/qwen_native_lora_smoke" \
  --mixed_precision=bf16 --resolution=512 --train_batch_size=1 \
  --gradient_accumulation_steps=8 --gradient_checkpointing \
  --rank=16 --lora_alpha=16 --learning_rate=1e-4 \
  --lr_scheduler=constant --lr_warmup_steps=0 --max_train_steps=100 \
  --cache_latents --offload --seed=42
```

Check `--help` at the pinned revision before use. This native command is a smoke starting point; the final 48 GB disk-backed transformer-only route is described below. Omit validation prompts/images during the initial optimizer smoke so the trainer does not add pipeline decoding to the peak.

### 9.2 Exact attention insertion

At the inspected revision, `QwenImage21AttnProcessor.__call__` calls `_qwenimage21_prepare_qkv`, computes the native attention outputs, flattens the head dimensions, and then applies `attn.to_out[0]` and `[1]`. Add BA **after native head output formation and before `to_out`**. Preserve all original block gates and MLPs. [Q4]

Patch model → block → attention processor with an optional immutable branch context. Preserve existing arguments:

```text
rotary_emb, layer_cache, kv_cache_mode, cache_write_slice, segments, key_valid
```

Construct metadata from the native image layout. Reference image tokens can be interleaved with text in the prefix; **prefix is not synonymous with reference face**. The 2×2 image-slot expansion also means VLM slot indices cannot be used as latent-token indices without native mapping.

Use proposed zero-indexed sites `[4,8,12,16,20,24,28,31]` on both GPUs. Apply branch K/V low-rank projections to reference hidden states before native K normalization and reference RoPE. Keep native Q unchanged. Write the correction only to target-query rows.

### 9.3 Build the 48 GB training process

Adapt the paired trainer rather than copying the old SDXL loss loop:

1. Add the validated local JSONL dataset/cache loader from Section 6.
2. Add separate encoder-precompute and VAE-precompute entry points. Reuse the upstream processor's exact prompt/image semantics; do not implement a different caption template for convenience.
3. In cached training mode, load **only** `QwenImage21Transformer2DModel` plus its shipped scheduler/configuration and small trainable modules. Do not instantiate `QwenImage21Pipeline.from_pretrained(...).to('cuda')` and then merely hope that unused modules are released.
4. Reconstruct original embeddings, masks, image-slot expansion and target/reference latent concatenation from the cached payload. For unequal sizes, explicitly pass the correct individual image grids.
5. Preserve the upstream training noise/sigma distribution and prediction convention. The paired documentation specifically distinguishes the shipped scheduler's inference dynamic shift from training sigmas; avoid copying FLUX's sampling shift or PhotoMaker's epsilon target. [Q3]
6. Extend optimizer parameter collection and Accelerate save/load hooks for the branch. Disable full-backbone EMA and image-space identity losses initially.
7. Start with the native **segmented SDPA `QwenImage21AttnProcessor`**. The code warns that uncompiled FlexAttention can create dense FP32 score matrices. Compilation/Flex is a later numerically checked optimization. [Q4]

For primary training **and initial custom inference**, disable prefix KV reuse while retaining the native **causal mask and zero-timestep prefix modulation**. If the pipeline lacks a public cache toggle, implement one in the custom runner. **Do not set `causal_condition=False` to disable caching**: that changes the model's attention/modulation semantics.

### 9.4 Later optional prefix optimization

Qwen's block-causal prefix can be evaluated independently when its producer is frozen. It is reasonable to add an optimized route later, but validate it against the complete native forward first.

A valid optimization must preserve the complete prefix (text plus all image blocks), t=0 modulation, prefix causality, padding and RoPE. It needs native per-layer K/V for target attention and enough pre-projection reference features for BA. Recompute trainable BA K/V on every update. Native shared LoRA makes prefix representations trainable, so the frozen-prefix shortcut must be disabled or recomputed with the correct graph in `lora_only`/`lora_plus_branch` modes.

Caches cannot cross unrelated samples or optimizer updates with changed producers. Use immutable keys and short bounded lifetimes. Native cached inference's Q index space is target-only while K/V indices include the prefix; the branch context must represent both spaces. If caching source slices, use owned copies where needed—Qwen's source explicitly uses `clone()` to avoid retaining full prefill buffers through prefix views. [Q4]

### 9.5 48 → 80 GB migration

There are two different operations:

**Hardware-only move:** identical checkpoint revision, BA sites/rank, native LoRA scope, data transforms and resolution. Load the full adapter/resume state and run `qwen7b_80_matched`. Outputs and a training step should match within the measured native hardware/kernel tolerance. No new backbone download or adapter reinitialization is required.

**Higher-budget continuation:** retain the adapter tensors but run `qwen7b_80` at 1024/768 after regenerating changed cache entries. This is a new resolution curriculum stage, not a bit-identical resume. Preserve the original checkpoint, use a new run ID, explicitly choose whether to retain optimizer moments, and restart/log the new learning-rate schedule. A conservative starting proposal is to retain weights, reset the scheduler, and reduce the adapter learning rate from `1e-4` to `5e-5`; test this rather than assuming it is optimal.

If changing rank or adding sites, state-dict and optimizer compatibility change. Initialize new sites/parameters deliberately and record the mapping; do not advertise this as an ordinary same-shape resume. Multiple-reference training also needs a new dataset/collator because the initial paired trainer contract is one condition image. Implement ordered per-image slots and masks before increasing `max_reference_images`.

Do not claim this creates a larger-parameter Qwen model. An older 20B Qwen-Image/Edit experiment would be a separate architecture port with different processor/attention assumptions.

## 10. New package and interfaces the implementation agent must build

Keep substantive new logic separate from upstream model files. The only narrow upstream changes should be optional context plumbing, the two FLUX attention seams or Qwen processor installation, and trainer load/optimizer/checkpoint hooks.

```text
ba_dit/
  pyproject.toml
  ba_dit/
    config.py                    # strict schema, path expansion, resolved config
    types.py                     # immutable layout/context and cached-pair types
    nn/reference_read_delta.py   # registered K/V low-rank adapters
    nn/sliced_native_lora.py     # FLUX fused-row/column attention-only LoRA
    nn/routing.py                # optional later localizer/confidence/frequency
    backends/qwen21.py           # native processor, slot/layout and cache handling
    backends/flux2_native.py     # native double/single-block integration
    data/manifest.py             # identity split and source hashes
    data/geometry.py             # independent ref/target transforms and masks
    data/cache.py                # bounded disk-backed cache and keys
    data/qwen21_pairs.py         # processor/slot reconstruction
    data/flux_pairs.py           # packed reference tokens and IDs
    trainers/qwen21.py           # adapt upstream paired training loop
    trainers/flux_toolkit.py     # Toolkit extension / narrow trainer integration
    checkpoint.py               # branch + native LoRA + resumable state
    evaluate.py                 # same native backend as training
    cli.py                      # preflight/precompute/train/infer/evaluate
  tests/
    test_zero_effect.py
    test_input_gradient_parity.py
    test_reference_support.py
    test_qwen_slot_geometry.py
    test_flux_fused_slices.py
    test_cache_equivalence.py
    test_checkpoint_recompute.py
    test_optimizer_inventory.py
    test_export_resume.py
```

Suggested immutable per-forward context:

```python
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class BranchContext:
    sample_ids: tuple[str, ...]
    target_query_indices: torch.Tensor
    reference_key_indices: tuple[torch.Tensor, ...]
    target_grid_hw: tuple[int, int]
    reference_grid_hw: tuple[tuple[int, int], ...]
    reference_key_valid: tuple[torch.Tensor, ...]
    target_gate: torch.Tensor | None
    sigma: torch.Tensor
    kv_mode: str  # "full", "prefill", or "cached_decode"
```

A frozen dataclass does not freeze the contents of its tensors. Never mutate these tensors while their checkpointed forward is outstanding. The backend may extend this schema for batch-specific slot maps and positions. Do not reassign a global `current_reference`, `current_mask` or `current_step` before backward recomputation completes.

### Required CLI behavior

The following is a **new research CLI contract**, not an upstream command that already exists:

```text
python -m ba_dit.cli preflight  --config FILE [--model-smoke-steps N]
python -m ba_dit.cli precompute --config FILE --stage encoder|vae|all --split train|validation
python -m ba_dit.cli train      --config FILE --mode native|lora_only|branch_only|lora_plus_branch
python -m ba_dit.cli infer      --config FILE --checkpoint PATH --manifest FILE --output-dir DIR
python -m ba_dit.cli evaluate   --config FILE --generated-dir DIR --manifest FILE
```

`native` is a forward/evaluation mode, not an optimizer mode with an empty parameter list. `precompute --stage all` must sequence the two preparation processes or explicitly release each model before the next; it must not load all components simultaneously on CUDA.

Configuration loader requirements:

- Reject unknown fields. Do not accept a configuration that looks correct but leaves its memory settings unused.
- Resolve `${BA_ROOT}` from the environment and `${RUN_NAME}` from `run.name`; stop on unresolved variables.
- Verify the checkpoint against `revision_source` and save the resolved configuration and hashes with the run.
- Interpret step counts as **optimizer updates**, not examples or microsteps. At batch 1 and accumulation 8, 2,000 updates expose 16,000 pairs, excluding resampling details.
- Use mode to decide which groups train; do not infer it from adapter filenames. `branch_only` must freeze native LoRA, and `lora_only` must bypass BA arithmetic entirely.
- Load only the required component classes for the selected stage. The preflight must list every CUDA module and confirm no encoder/VAE resides there during cached training.
- Use safe, explicit new run directories. A promotion is not permission to overwrite the original checkpoint or caches.

Keep the new package's unconditional dependencies lightweight; do not install one combined Transformers/Diffusers dependency set that upgrades the incompatible Qwen and Toolkit environments into the same stack. Backend dependencies stay in their separate environment locks.

The full four primary configs and two matched-promotion configs are provided in the companion `configs/` directory. Appendix A embeds the four primary files for a self-contained handoff.

## 11. Training launch sequence and acceptance milestones

Run the staged workflow after implementing the CLI above. These commands intentionally distinguish the future research trainer from the supplied acquisition helper and existing upstream smoke trainers.

### 11.1 FLUX 4B on 48 GB

```bash
export BA_ROOT="${BA_ROOT:-$HOME/work/cl39_dit}"
source "$BA_ROOT/envs/flux-toolkit/bin/activate"
export PYTHONPATH="$BA_ROOT/sources/ai-toolkit-flux:${PYTHONPATH:-}"
# Install the NEW package after the implementation agent has created it:
python -m pip install -e "$BA_ROOT/sources/ba_dit"
python -m ba_dit.cli preflight --config "$BA_ROOT/configs/flux4b_48.yaml"
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/flux4b_48.yaml" --stage encoder --split train
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/flux4b_48.yaml" --stage vae --split train
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/flux4b_48.yaml" --stage all --split validation
python -m ba_dit.cli preflight --config "$BA_ROOT/configs/flux4b_48.yaml" --model-smoke-steps 100
# Give each mode its own resolved run name/output directory:
python -m ba_dit.cli train --config "$BA_ROOT/configs/flux4b_48.yaml" --mode lora_only
python -m ba_dit.cli train --config "$BA_ROOT/configs/flux4b_48.yaml" --mode branch_only
```

The launcher should append the mode to the run ID or require an explicit unique name. Do not run these training processes concurrently on a single GPU.

### 11.2 Qwen on 48 GB

```bash
source "$BA_ROOT/envs/qwen21/bin/activate"
python -m pip install -e "$BA_ROOT/sources/ba_dit"
python -m ba_dit.cli preflight --config "$BA_ROOT/configs/qwen7b_48.yaml"
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/qwen7b_48.yaml" --stage encoder --split train
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/qwen7b_48.yaml" --stage vae --split train
python -m ba_dit.cli precompute --config "$BA_ROOT/configs/qwen7b_48.yaml" --stage all --split validation
python -m ba_dit.cli preflight --config "$BA_ROOT/configs/qwen7b_48.yaml" --model-smoke-steps 100
python -m ba_dit.cli train --config "$BA_ROOT/configs/qwen7b_48.yaml" --mode lora_only
python -m ba_dit.cli train --config "$BA_ROOT/configs/qwen7b_48.yaml" --mode branch_only
```

Use the same pairing order, target noise seeds, training budget, validation inputs and cache policy across matched modes. Do not compare a branch trained on a much larger identity set against a LoRA baseline trained only on a few example images.

### 11.3 Promote only after the simple result is interpretable

**Engineering milestone:** native LoRA works; BA step-zero and gradient parity pass; BA weights update; the exact 48 GB profile passes memory/save/resume tests.

**Initial research milestone:** use a modest identity-diverse pilot (e.g. 1,000–5,000 pairs) and a fixed 2,000-update budget as an initial proposal. Inspect more than the aggregate ID_SIM. Small or non-significant effects at this stage should be reported with uncertainty; they are not final architecture conclusions.

**Promotion milestone:** a promising result survives more held-out identities, another training seed, branch-off tests and a matched-budget LoRA control without a material pose/quality regression. Do not impose a made-up universal ID_SIM threshold; report uncertainty and paired per-identity differences.

For FLUX, retrain fresh on `flux9b_80_matched`, then test the expanded profile. For Qwen, first reproduce the same adapter on `qwen7b_80_matched`, then continue under a new higher-resolution stage. Repeat native and LoRA baselines at the new resolution as well.

## 12. Memory instrumentation and speed controls

A model-level smoke run must report **actual** tensors and hardware, not just requested YAML values. Record:

```text
GPU name, reported total memory, driver/runtime, other GPU processes
exact generator and conditioning checkpoint revisions
native target tokens, native reference tokens, text/slot-expanded tokens
extra branch reference-key count, selected sites and adapter parameters
live encoder/VAE/transformer CUDA residency
peak allocated memory, peak reserved memory, external process GPU usage
CPU RSS, cache hit/miss counts, step time after warmup
native loss, gradient norm, BA B-matrix gradient/update norms
```

Instrument after synchronization around warmup and measured steps; reset peak stats only at the intended measurement boundary. Include optimizer initialization, its first real update, checkpoint writing and resume. A forward-only benchmark is not a training-memory result.

```python
# Integrate around the real train_one_optimizer_step function; not a standalone trainer.
import time
import torch

def measure_steps(train_one_optimizer_step, measured_steps=100):
    for _ in range(5):
        train_one_optimizer_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(measured_steps):
        train_one_optimizer_step()
    torch.cuda.synchronize()
    total = torch.cuda.get_device_properties(0).total_memory
    result = {
        "seconds_per_optimizer_step": (time.perf_counter() - start) / measured_steps,
        "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
        "total_GiB": total / 2**30,
        "reserved_fraction": torch.cuda.max_memory_reserved() / total,
    }
    if result["reserved_fraction"] > 0.90:
        raise RuntimeError(f"Insufficient safety margin: {result}")
    return result
```

Also inspect `nvidia-smi`/NVML because PyTorch's allocator counters do not include all device usage. A successful 100-step run is an admission test, not evidence that every future bucket or prompt length fits. Enforce maximum actual token counts at data admission.

Avoid speed optimizations that change the experiment silently. TF32, fused kernels, compile mode, quantization and reduced text/reference budgets belong in the recorded configuration. A newer 48 GB GPU and an older 80 GB GPU need not have the same step time; no throughput estimate is supplied without hardware measurements.

## 13. Checkpoints and tests that prevent false-positive branch results

### 13.1 Checkpoint format

Save a compact model-adaptation checkpoint containing:

```text
manifest.json
  schema; source and model commits; backbone ID; backend; tensor widths/heads
  selected sites; native LoRA target policy/rank; branch rank/variant
  cache/geometry/processor schemas; training mode; configuration hash
native_lora.safetensors     # only when that mode has native LoRA
branch.safetensors         # BA parameters; optional routing modules with explicit keys
training_state/            # optimizer, scheduler, RNG, global update, sampler position
resolved_config.yaml
metrics.json
```

The manifest must identify **which native backend executes the checkpoint**. Ordinary LoRA export formats will not automatically save arbitrary BA modules. Do not publish a LoRA-only file and conclude that the branch survived because some images still look good.

Do not repeatedly save frozen generator weights. Use adapter-aware Accelerate/Toolkit save hooks so a generic `save_state()` does not accidentally serialize the entire base model. Save optimizer state for reproducibility, but keep no more than three latest states and one best state by default. Preserve an extra checkpoint before a curriculum/promotion stage.

### 13.2 Required tests before long training

| Test | Required result |
|---|---|
| Import origin / pin check | Effective classes come from the intended patched repositories and exact checkpoint family. |
| Native round trip | Weight loading and cached inputs reproduce the native backend output at several sigmas before BA exists. |
| BA off | BA bypass reproduces native output; optional native LoRA remains enabled in both sides of the comparison. |
| BA initialized on | Forward output matches native within measured numerical tolerance. |
| Input-gradient parity | Gradients to synthetic target hidden states and an upstream trainable adapter match when BA's B matrices are zero. |
| First update | B receives finite, nonzero gradients for a non-degenerate example; after updates the branch output changes. A need not receive gradients on the first step. |
| Parameter inventory | Every trainable BA/LoRA parameter occurs once in the optimizer; frozen generator has no accumulated parameter gradients. |
| Target-only write | This layer's BA modification is zero at non-target output rows; in FLUX later layers may legitimately propagate the target change into references. |
| Correct key support | Excluded reference keys are absent from both R0/R1 softmax support; empty support gives zero correction, no NaN. |
| Geometry | Different reference/target sizes, rectangular buckets, padding and original position IDs pass alignment checks. |
| Qwen slot accounting | Processor image slots and expanded clean-reference latent tokens agree exactly. Batch/layout checks do not get disabled to “make it run.” |
| FLUX fused projections | Native MLP slices are unchanged in BA-only mode, and native attention-only LoRA touches only its documented rows/columns. |
| Checkpoint recomputation | Gradients agree with/without checkpointing; two sequential microbatches do not reuse mutable masks/references. |
| Input-cache equivalence | Live versus cached encoder/latent paths agree for the chosen sampling/geometry policy; changed prompts or crops cannot hit an old cache. |
| Qwen cache disable | Native causality and t=0 prefix modulation are unchanged when KV reuse is off. |
| Save/load and resume | Adapter outputs match after reload; a resumed optimizer step matches the continuous run within tolerance. |
| Memory acceptance | Largest admitted layout passes full forward/backward/update, serialization and separate validation with the safety margin. |
| Promotion guard | 4B BA weights cannot load into 9B; same-shape Qwen hardware migration can load; incompatible ranks/sites fail loudly. |

Establish tolerances by repeated native-native execution under the chosen dtype/kernel rather than selecting a very loose threshold after a failed port. Tiny synthetic tests can run on CPU, but real-checkpoint tests and memory validation require the target GPU.

## 14. Evaluation protocol and what counts as useful improvement

Use the same held-out reference images, requested scenes and identity split across native, LoRA-only, branch-only and joint modes. Keep the previously used CL39 evaluation panel for continuity, but do not make it the only test set. Record identity similarity, face detection failure rate, face quality/artifacts, prompt compliance, pose/expression diversity and reference-copying behavior.

**Interventions must isolate BA from native image conditioning.** BA-off is simple. For a wrong-reference test, leave the native reference image/conditioning unchanged and replace **only the additional branch's reference bank**. Both R0 and R1 must use the same substituted bank and layout. In FLUX, obtaining that bank may require a separate donor forward with its own target-conditioned features; budget it as an evaluation intervention, not the production training path. Do not change the input image globally and call the resulting identity loss evidence for BA.

Report within-backbone changes:

```text
BA value        = metric(backbone + BA) − metric(same pretrained backbone)
Beyond-LoRA     = metric(backbone + BA) − metric(same backbone + matched LoRA)
Joint increment = metric(backbone + LoRA + BA) − metric(backbone + LoRA)
```

Also inspect paired per-identity differences and uncertainty over identities/seeds. Better absolute 9B images do not establish that the BA idea scales; a gain only in ID_SIM with worse texture or reduced pose freedom is not an unqualified improvement.

The expanded Qwen profile is a higher-resolution/higher-conditioning-budget experiment, not evidence about larger-parameter Qwen scaling. Native 2K evaluation, additional reference inputs and a current public benchmark comparison can follow once the smaller controlled experiment is sound. No SOTA claim is justified by the plan alone.

## 15. Disk space: weights are only the first component

All disk figures below use **decimal GB** (`1 GB = 10^9 bytes`), unless explicitly labeled GiB. They are new working-space requirements, excluding the OS, existing PhotoMaker models and existing datasets unless stated. Observed file sizes are rounded; environment/cache/run allowances are **engineering estimates**.

### 15.1 New model files to retain

| Route/component | Generator | Conditioning encoder | VAE / metadata | Approximate selected total |
|---|---:|---:|---:|---:|
| **FLUX Base 4B, native Toolkit** | 7.75 GB | Qwen3-4B: 8.06 GB | AE: 0.336 GB + small config/tokenizer files | **~16.2 GB** |
| **FLUX Base 9B, native Toolkit** | 18.2 GB | Qwen3-8B: 16.4 GB | Same AE: 0.336 GB + small files | **~34.9 GB** |
| **Qwen-Image-2.1** | ~14.2 GB | Bundled Qwen3-VL: ~17.5 GB | VAE ~1.35 GB + processor/scheduler | **~33.1 GB** |
| **Both 48 GB pilots** | FLUX 4B + Qwen 2.1 | Included above | No duplicate layouts | **~49.3 GB; reserve 55–60 GB** |
| **All three generators: FLUX 4B + 9B + Qwen** | All retained | Both distinct FLUX encoders + Qwen VLM | One native FLUX AE copy | **~83.9 GB; reserve 90 GB** |

Sources: official HF repository trees for FLUX generators, Qwen text encoders, the native Toolkit AE and Qwen pipeline; Qwen's BF16 generator figure is also cross-checked against the Comfy-Org distribution linked by the native pipeline documentation. [W1] [W2] [W3] [W4] [W5] [W6] [W7] [W8] [W9]

Combined totals assume **one shared filesystem/model cache**. Independent servers without shared storage each need their own appropriate files; moving only the adapters does not eliminate the base-weight requirement on the destination.

Moving Qwen from 48 to 80 GB adds **no new base-weight files** if the revision is unchanged. Retaining the 4B setup while adding FLUX 9B adds approximately **34.6 GB** for the larger generator and its distinct text encoder; the existing native AE can be reused.

A root-level `snapshot_download` of every file is not always efficient. The official FLUX 4B repository displays **23.7 GB**, including both a 7.75 GB native checkpoint and a Diffusers layout with components. The selected Toolkit route needs the native file plus its explicitly chosen encoder/AE, not both generator formats. [W2]

Do not save multiple fp32/bf16/quantized model copies unless they are part of a documented experiment. Casting a file to BF16 on load does not shrink the original file on disk. Conversions can temporarily require source and destination files simultaneously.

### 15.2 Environments, code and operational overhead

| Item | Suggested disk allowance | Basis / caveat |
|---|---:|---|
| Qwen Python environment, CUDA-enabled Torch and training dependencies | **12–25 GB** | Estimate; depends on resolved wheels and libraries. |
| FLUX Toolkit environment, its CUDA stack and broader dependencies | **15–30 GB** | Estimate; separate from Qwen. No UI/node_modules included. |
| Both installed environments | **30–55 GB** | Plan conservatively rather than assume CUDA libraries deduplicate across environments. |
| Source clones, editable research package and Git objects | **2–5 GB** | Partial/shallow clones help; retained history can grow. |
| Download/install/build caches and temporary working space | **15–30 GB** | More for local CUDA builds, weight conversion or multiple new revisions. |
| Adapter/resume checkpoints across a small experiment set | **10–30 GB** | Retain bounded states, not a full base-model copy per save. |
| Validation images, tables and diagnostic outputs | **5–20 GB** | Depends strongly on image count, formats and resolution. |
| New raw training photos / derived exports | **Add actual dataset size** | Example only: 20,000 unique images averaging 1 MB would be ~20 GB. |

Measure after installation with `du`; these are not claimed installed sizes from the user's server. Shared `HF_HOME` avoids model downloads per environment, but does not automatically share installed PyTorch libraries. Using an existing system CUDA toolkit does not necessarily eliminate bundled wheel runtime files.

### 15.3 Conditioning caches: compute from actual tensors

For an uncompressed BF16 tensor, disk payload is approximately `number_of_elements × 2 bytes`, plus headers/metadata. Do not assume generic compression will significantly compress dense neural embeddings.

**Illustrative encoder cache budget for 10,000 distinct cached conditioning records:**

| Stored representation | Example shape per record | Payload per record | Payload for 10,000 |
|---|---|---:|---:|
| FLUX 4B text embeddings | 512 × 7,680 × BF16 | 7.86 MB | **78.6 GB** |
| FLUX 9B text embeddings | 512 × 12,288 × BF16 | 12.58 MB | **125.8 GB** |
| Qwen multimodal conditioning | 1,024 × 4,096 × BF16 | 8.39 MB | **83.9 GB** |

These token counts are **examples**, not a claim that every model always produces that shape. Log the actual stored output; extra selected encoder layers, padding, longer prompts or more reference images change the size. Repeated FLUX prompts can reduce the number of unique records substantially. Qwen records depend on the image as well as the prompt, so it is generally less reducible by prompt reuse alone.

**Illustrative latent payloads**, storing one target and one reference sample per pair, without posterior variances or duplicate copies:

| Family / target-reference size | Payload formula | 10,000 pairs |
|---|---|---:|
| FLUX 768 / 512 | `(48×48 + 32×32) × 128 × 2` bytes | **8.52 GB** |
| FLUX 1024 / 768 | `(64×64 + 48×48) × 128 × 2` bytes | **16.38 GB** |
| Qwen 768 / 512 | `(48×48 + 32×32) × 64 × 2` bytes | **4.26 GB** |
| Qwen 1024 / 768 | `(64×64 + 48×48) × 64 × 2` bytes | **8.19 GB** |

These calculations follow the stated packed/latent grids. The agent must verify actual tensors from the selected native AE/processor, including any frame axis, before using them as a storage forecast. Position IDs, masks, posterior parameters and multiple cached augmentations add overhead. Deduplicating repeated images by their content/transform key can reduce it.

**Why not save Qwen deep prefix KV per pair?** At an illustrative 32 layers, 2,048 prefix tokens, hidden width 4,096, K and V, BF16:

```text
32 × 2048 × 4096 × 2(K,V) × 2 bytes = 1,073,741,824 bytes ≈ 1 GiB per prefix
10,000 different prefixes ≈ 10.7 TB
```

That excludes additional pre-projection features for trainable BA. Even eight layers would be ~2.7 TB across that many prefixes. Cache encoder outputs and VAE latents, not an unbounded per-layer feature bank.

### 15.4 Recommended space by stage

| Stage | What this assumes | Recommended NEW available working space |
|---|---|---:|
| One lean FLUX 4B pilot | One environment, native weights, around 1,000 cached pairs, few retained runs | **150 GB** |
| Both 48 GB pilots, deliberately small data | Two environments, ~49 GB weights, around 1,000 unique conditioning records per family, bounded outputs | **250 GB minimum planning target** |
| Both 48 GB pilots with useful iteration room | More pairs, caches, multiple controlled runs and diagnostics | **500 GB preferred** |
| Both FLUX sizes plus Qwen, ~10,000 unique records per family | ~84 GB weights, both environments, potentially ~300+ GB conditioning/latent caches, dataset and run space | **1 TB preferred** |
| Large datasets, many augmentation caches, 2K/multi-reference work, numerous retained runs | Must calculate from the actual data/cache design | **2 TB or more may be appropriate** |

These are workload budgets, not rigid minimums. A tiny pilot can use less; many images or saved tensor banks can use far more. Keep **15–20% free space** rather than plan to fill the volume completely. A nominal disk already containing the OS or old experiments does not provide its nominal capacity as new free space.

For a concrete all-model 10,000-record illustration, the example prompt caches total ~288 GB. Keeping the example FLUX 4B, expanded FLUX 9B and expanded Qwen latent caches adds ~33 GB. Together with ~84 GB weights, ~45 GB environments, ~20–40 GB data and bounded run/scratch space, a working total can approach **600–700 GB before a comfortable reserve**. That is why 1 TB is more realistic than a volume sized only for checkpoints. This is arithmetic from the assumptions above, not an observation of the user's dataset.

### 15.5 Disk controls to implement

Store one model cache across environments and use local snapshot aliases. Keep one primary checkpoint format. Deduplicate conditioning by exact component key, not just sample filename. Limit checkpoints and avoid full-backbone saves. Disable full-layer/full-step attention dumps. Use a disk quota/report before precomputing a large cache.

Useful commands on the implementation machine:

```bash
df -h "$BA_ROOT"
du -sh "$BA_ROOT"/{envs,sources,cache,weights,data,runs,scratch}
# Symlinks in weights/ are aliases; do not add dereferenced weights/ to cache/ again.
du -sh "$HF_HUB_CACHE"
```

Before a large cache build, encode 100–500 representative pairs and sum their actual file sizes; extrapolate by unique cache-key count and add planned augmentation/resolution variants. This is more accurate than using a marketing model-parameter count for total disk needs.

## 16. Ordered implementation work packages and final handoff

| Work package | Deliverable | Acceptance / stop condition |
|---|---|---|
| **WP0: preserve and pin** | Separate source clones, environment locks, selected weight manifest, licence/access check | Existing PhotoMaker worktree untouched; imported code origin recorded. |
| **WP1: native paired LoRA** | Upstream 512/512 smoke for FLUX 4B and Qwen | Correct pair alignment, finite loss, real adapter updates, no unsupported CLI fields. |
| **WP2: bounded conditioning cache** | Native-equivalent encoder and target/reference latent caches; transformer-only training loaders | Live/cached outputs match; no encoder/VAE on CUDA in training; independent geometry tests pass. |
| **WP3: shared BA and sliced native LoRA** | Registered low-rank modules, strict parameter selection and context types | Zero-forward/input-gradient tests; first B update; exact optimizer inventory. |
| **WP4: FLUX 4B integration** | Double/single seams, same backend inference, save/resume, 48 GB profile | MLP/native reference semantics preserved; largest admitted layout passes memory gate. |
| **WP5: Qwen integration** | Slot-aware processor, causal semantics, cache-off inference, 48 GB profile | Independent-size token slots and native-loss equivalence pass; largest layout passes memory gate. |
| **WP6: controlled research pilot** | Native, matched-LoRA, BA-only, optional joint comparison | Per-identity results, artifacts/pose checks, interventions and uncertainty reported. |
| **WP7: scale** | Fresh FLUX 9B matched run; Qwen same-adapter hardware replication; then expanded profiles | No illicit 4B tensor transfer; changed caches rebuilt; separate new run/stage records. |
| **WP8: optional full CL39-style routing** | Learned spatial gate, confidence and frequency ablations | Each feature tested independently; no target-ground-truth conditioning at evaluation. |

The implementation agent should deliver: patches and their source pins; exact install locks; resolved configs; weight and cache manifests; compact trainable checkpoints; test outputs; measured 48/80 GB memory/step-time tables; native-versus-branch evaluation; and a short record of any departures from this design.

**Bottom line:** there is a practical, coherent 48 GB design. FLUX provides the genuine **Base 4B → Base 9B** experiment requested. Qwen provides a **same-7B, staged-memory and resolution** experiment. Keep those conclusions separate, preserve native conditioning, and prove the added branch is responsible for any gain.


---

## Appendix A. Full proposed profiles and an upstream smoke configuration

The next four configurations belong to the **new `ba_dit` schema**. They do not work by passing them unchanged to `ai-toolkit/run.py` or the upstream DreamBooth script. Implement the strict configuration/CLI contract in Section 10 first. The companion package contains identical standalone YAML files.

The configs keep branch rank/sites fixed between the primary and expanded stage where possible. Native LoRA is configured but is trainable only in the appropriate mode. `max_optimizer_steps: 2000` is an initial budget to investigate, not a claim that 2,000 updates is sufficient for generalizable identity training.

### A.1 `flux4b_48.yaml`

```yaml
schema_version: 1
status: proposed_ba_dit_schema_not_an_upstream_config
run:
  name: flux4b_48
  output_dir: ${BA_ROOT}/runs/${RUN_NAME}
  seed: 42
hardware:
  device: cuda:0
  world_size: 1
  nominal_vram_gb: 48
  max_reserved_fraction: 0.9
model:
  backend: flux2_native_toolkit
  checkpoint: ${BA_ROOT}/weights/flux4b
  revision_source: ${BA_ROOT}/locks/weights-small.json
  dtype: bfloat16
  freeze_base: true
  quantization: none
  text_encoder_checkpoint: ${BA_ROOT}/weights/flux4b_text
  vae_checkpoint: ${BA_ROOT}/weights/flux_vae/ae.safetensors
  arch: flux2_klein_4b
data:
  manifest: ${BA_ROOT}/data/train_pairs.jsonl
  validation_manifest: ${BA_ROOT}/data/validation_pairs.jsonl
  target_size:
  - 768
  - 768
  reference_size:
  - 512
  - 512
  max_reference_images: 1
  independent_pair_geometry: true
  crop_policy: fixed_identity_preserving
  random_flip: false
  caption_dropout: 0.0
  max_pixels_policy: assert_actual_encoded_grid
cache:
  root: ${BA_ROOT}/cache/conditioning
  encoder_outputs: true
  target_latents: true
  reference_latents: true
  reference_position_ids: true
  store_exact_native_layout: true
  storage_dtype: bfloat16
  deep_reference_kv: disabled
  allow_missing_entries: false
  max_cpu_cache_gb: 4
  max_gpu_cache_gb: 0
  load_encoders_during_train: false
training:
  mode: branch_only
  microbatch: 1
  gradient_accumulation_steps: 8
  max_optimizer_steps: 2000
  checkpoint_every_optimizer_steps: 500
  optimizer: adamw
  learning_rate_branch: 0.0001
  learning_rate_native_lora: 0.0001
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_schedule: constant_with_warmup
  warmup_optimizer_steps: 100
  gradient_checkpointing: true
  checkpoint_use_reentrant: false
  native_loss_and_sigma_sampling: true
  identity_loss_weight: 0.0
  ema: false
  trainable_parameter_dtype: float32
native_lora:
  rank: 8
  alpha: 8
  target_policy: attention_only_rows
  site_policy: branch_sites
  dropout: 0.0
  initialize_A: kaiming_uniform
  initialize_B: zeros
branch:
  variant: reference_read_delta
  projection_targets:
  - k
  - v
  rank: 16
  alpha: 16
  initialize_A: kaiming_uniform
  initialize_B: zeros
  gamma_init: 0.1
  gamma_max: 0.3
  gamma_trainable: false
  target_gate: all_target
  reference_key_policy: face_head_mask_then_spatially_stratified_cap
  max_reference_keys: 512
  empty_reference_action: zero_correction
  query_chunk_size: 128
  confidence: disabled
  frequency_filter: disabled
  sites:
    double_blocks:
    - 1
    - 2
    - 3
    - 4
    single_blocks:
    - 3
    - 8
    - 13
    - 18
inference:
  separate_process: true
  prefix_kv_cache: disabled_preserve_causal_mask
  sequential_cfg: true
  steps: 50
  guidance_scale: 4.0
  resize_masks_from_explicit_grid: true
checkpoint:
  format: native_lora_plus_branch
  save_frozen_backbone: false
  keep_latest: 3
  keep_best: 1
  save_optimizer: true
  strict_backbone_and_shapes: true
runtime:
  import_origin_assertions: true
  checkpoint_context: explicit_immutable_argument
  compile: false
  dense_attention_maps: false
  log_actual_token_shapes: true
  log_optimizer_parameter_inventory: true
```

### A.2 `flux9b_80.yaml`

```yaml
schema_version: 1
status: proposed_ba_dit_schema_not_an_upstream_config
run:
  name: flux9b_80
  output_dir: ${BA_ROOT}/runs/${RUN_NAME}
  seed: 42
hardware:
  device: cuda:0
  world_size: 1
  nominal_vram_gb: 80
  max_reserved_fraction: 0.9
model:
  backend: flux2_native_toolkit
  checkpoint: ${BA_ROOT}/weights/flux9b
  revision_source: ${BA_ROOT}/locks/weights-all.json
  dtype: bfloat16
  freeze_base: true
  quantization: none
  text_encoder_checkpoint: ${BA_ROOT}/weights/flux9b_text
  vae_checkpoint: ${BA_ROOT}/weights/flux_vae/ae.safetensors
  arch: flux2_klein_9b
data:
  manifest: ${BA_ROOT}/data/train_pairs.jsonl
  validation_manifest: ${BA_ROOT}/data/validation_pairs.jsonl
  target_size:
  - 1024
  - 1024
  reference_size:
  - 768
  - 768
  max_reference_images: 1
  independent_pair_geometry: true
  crop_policy: fixed_identity_preserving
  random_flip: false
  caption_dropout: 0.0
  max_pixels_policy: assert_actual_encoded_grid
cache:
  root: ${BA_ROOT}/cache/conditioning
  encoder_outputs: true
  target_latents: true
  reference_latents: true
  reference_position_ids: true
  store_exact_native_layout: true
  storage_dtype: bfloat16
  deep_reference_kv: disabled
  allow_missing_entries: false
  max_cpu_cache_gb: 4
  max_gpu_cache_gb: 0
  load_encoders_during_train: false
training:
  mode: branch_only
  microbatch: 1
  gradient_accumulation_steps: 8
  max_optimizer_steps: 2000
  checkpoint_every_optimizer_steps: 500
  optimizer: adamw
  learning_rate_branch: 0.0001
  learning_rate_native_lora: 0.0001
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_schedule: constant_with_warmup
  warmup_optimizer_steps: 100
  gradient_checkpointing: true
  checkpoint_use_reentrant: false
  native_loss_and_sigma_sampling: true
  identity_loss_weight: 0.0
  ema: false
  trainable_parameter_dtype: float32
native_lora:
  rank: 8
  alpha: 8
  target_policy: attention_only_rows
  site_policy: branch_sites
  dropout: 0.0
  initialize_A: kaiming_uniform
  initialize_B: zeros
branch:
  variant: reference_read_delta
  projection_targets:
  - k
  - v
  rank: 16
  alpha: 16
  initialize_A: kaiming_uniform
  initialize_B: zeros
  gamma_init: 0.1
  gamma_max: 0.3
  gamma_trainable: false
  target_gate: all_target
  reference_key_policy: face_head_mask_then_spatially_stratified_cap
  max_reference_keys: 512
  empty_reference_action: zero_correction
  query_chunk_size: 128
  confidence: disabled
  frequency_filter: disabled
  sites:
    double_blocks:
    - 2
    - 4
    - 6
    - 7
    single_blocks:
    - 4
    - 10
    - 16
    - 22
inference:
  separate_process: true
  prefix_kv_cache: disabled_preserve_causal_mask
  sequential_cfg: true
  steps: 50
  guidance_scale: 4.0
  resize_masks_from_explicit_grid: true
checkpoint:
  format: native_lora_plus_branch
  save_frozen_backbone: false
  keep_latest: 3
  keep_best: 1
  save_optimizer: true
  strict_backbone_and_shapes: true
runtime:
  import_origin_assertions: true
  checkpoint_context: explicit_immutable_argument
  compile: false
  dense_attention_maps: false
  log_actual_token_shapes: true
  log_optimizer_parameter_inventory: true
```

### A.3 `qwen7b_48.yaml`

```yaml
schema_version: 1
status: proposed_ba_dit_schema_not_an_upstream_config
run:
  name: qwen7b_48
  output_dir: ${BA_ROOT}/runs/${RUN_NAME}
  seed: 42
hardware:
  device: cuda:0
  world_size: 1
  nominal_vram_gb: 48
  max_reserved_fraction: 0.9
model:
  backend: qwen21_diffusers
  checkpoint: ${BA_ROOT}/weights/qwen21
  revision_source: ${BA_ROOT}/locks/weights-small.json
  dtype: bfloat16
  freeze_base: true
  quantization: none
  text_encoder_checkpoint: ${BA_ROOT}/weights/qwen21/text_encoder
  vae_checkpoint: ${BA_ROOT}/weights/qwen21/vae
data:
  manifest: ${BA_ROOT}/data/train_pairs.jsonl
  validation_manifest: ${BA_ROOT}/data/validation_pairs.jsonl
  target_size:
  - 768
  - 768
  reference_size:
  - 512
  - 512
  max_reference_images: 1
  independent_pair_geometry: true
  crop_policy: fixed_identity_preserving
  random_flip: false
  caption_dropout: 0.0
  max_pixels_policy: assert_actual_encoded_grid
cache:
  root: ${BA_ROOT}/cache/conditioning
  encoder_outputs: true
  target_latents: true
  reference_latents: true
  reference_position_ids: true
  store_exact_native_layout: true
  storage_dtype: bfloat16
  deep_reference_kv: disabled
  allow_missing_entries: false
  max_cpu_cache_gb: 4
  max_gpu_cache_gb: 0
  load_encoders_during_train: false
training:
  mode: branch_only
  microbatch: 1
  gradient_accumulation_steps: 8
  max_optimizer_steps: 2000
  checkpoint_every_optimizer_steps: 500
  optimizer: adamw
  learning_rate_branch: 0.0001
  learning_rate_native_lora: 0.0001
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_schedule: constant_with_warmup
  warmup_optimizer_steps: 100
  gradient_checkpointing: true
  checkpoint_use_reentrant: false
  native_loss_and_sigma_sampling: true
  identity_loss_weight: 0.0
  ema: false
  trainable_parameter_dtype: float32
native_lora:
  rank: 8
  alpha: 8
  target_policy: attention_only_rows
  site_policy: branch_sites
  dropout: 0.0
  initialize_A: kaiming_uniform
  initialize_B: zeros
branch:
  variant: reference_read_delta
  projection_targets:
  - k
  - v
  rank: 16
  alpha: 16
  initialize_A: kaiming_uniform
  initialize_B: zeros
  gamma_init: 0.1
  gamma_max: 0.3
  gamma_trainable: false
  target_gate: all_target
  reference_key_policy: face_head_mask_then_spatially_stratified_cap
  max_reference_keys: 512
  empty_reference_action: zero_correction
  query_chunk_size: 128
  confidence: disabled
  frequency_filter: disabled
  sites:
    transformer_blocks:
    - 4
    - 8
    - 12
    - 16
    - 20
    - 24
    - 28
    - 31
inference:
  separate_process: true
  prefix_kv_cache: disabled_preserve_causal_mask
  sequential_cfg: true
  steps: 40
  guidance_scale: 1.0
  resize_masks_from_explicit_grid: true
checkpoint:
  format: native_lora_plus_branch
  save_frozen_backbone: false
  keep_latest: 3
  keep_best: 1
  save_optimizer: true
  strict_backbone_and_shapes: true
runtime:
  import_origin_assertions: true
  checkpoint_context: explicit_immutable_argument
  compile: false
  dense_attention_maps: false
  log_actual_token_shapes: true
  log_optimizer_parameter_inventory: true
  attention_processor: QwenImage21AttnProcessor
```

### A.4 `qwen7b_80.yaml`

```yaml
schema_version: 1
status: proposed_ba_dit_schema_not_an_upstream_config
run:
  name: qwen7b_80
  output_dir: ${BA_ROOT}/runs/${RUN_NAME}
  seed: 42
hardware:
  device: cuda:0
  world_size: 1
  nominal_vram_gb: 80
  max_reserved_fraction: 0.9
model:
  backend: qwen21_diffusers
  checkpoint: ${BA_ROOT}/weights/qwen21
  revision_source: ${BA_ROOT}/locks/weights-small.json
  dtype: bfloat16
  freeze_base: true
  quantization: none
  text_encoder_checkpoint: ${BA_ROOT}/weights/qwen21/text_encoder
  vae_checkpoint: ${BA_ROOT}/weights/qwen21/vae
data:
  manifest: ${BA_ROOT}/data/train_pairs.jsonl
  validation_manifest: ${BA_ROOT}/data/validation_pairs.jsonl
  target_size:
  - 1024
  - 1024
  reference_size:
  - 768
  - 768
  max_reference_images: 1
  independent_pair_geometry: true
  crop_policy: fixed_identity_preserving
  random_flip: false
  caption_dropout: 0.0
  max_pixels_policy: assert_actual_encoded_grid
cache:
  root: ${BA_ROOT}/cache/conditioning
  encoder_outputs: true
  target_latents: true
  reference_latents: true
  reference_position_ids: true
  store_exact_native_layout: true
  storage_dtype: bfloat16
  deep_reference_kv: disabled
  allow_missing_entries: false
  max_cpu_cache_gb: 4
  max_gpu_cache_gb: 0
  load_encoders_during_train: false
training:
  mode: branch_only
  microbatch: 1
  gradient_accumulation_steps: 8
  max_optimizer_steps: 2000
  checkpoint_every_optimizer_steps: 500
  optimizer: adamw
  learning_rate_branch: 0.0001
  learning_rate_native_lora: 0.0001
  weight_decay: 0.0
  max_grad_norm: 1.0
  lr_schedule: constant_with_warmup
  warmup_optimizer_steps: 100
  gradient_checkpointing: true
  checkpoint_use_reentrant: false
  native_loss_and_sigma_sampling: true
  identity_loss_weight: 0.0
  ema: false
  trainable_parameter_dtype: float32
native_lora:
  rank: 8
  alpha: 8
  target_policy: attention_only_rows
  site_policy: branch_sites
  dropout: 0.0
  initialize_A: kaiming_uniform
  initialize_B: zeros
branch:
  variant: reference_read_delta
  projection_targets:
  - k
  - v
  rank: 16
  alpha: 16
  initialize_A: kaiming_uniform
  initialize_B: zeros
  gamma_init: 0.1
  gamma_max: 0.3
  gamma_trainable: false
  target_gate: all_target
  reference_key_policy: face_head_mask_then_spatially_stratified_cap
  max_reference_keys: 512
  empty_reference_action: zero_correction
  query_chunk_size: 128
  confidence: disabled
  frequency_filter: disabled
  sites:
    transformer_blocks:
    - 4
    - 8
    - 12
    - 16
    - 20
    - 24
    - 28
    - 31
inference:
  separate_process: true
  prefix_kv_cache: disabled_preserve_causal_mask
  sequential_cfg: true
  steps: 40
  guidance_scale: 1.0
  resize_masks_from_explicit_grid: true
checkpoint:
  format: native_lora_plus_branch
  save_frozen_backbone: false
  keep_latest: 3
  keep_best: 1
  save_optimizer: true
  strict_backbone_and_shapes: true
runtime:
  import_origin_assertions: true
  checkpoint_context: explicit_immutable_argument
  compile: false
  dense_attention_maps: false
  log_actual_token_shapes: true
  log_optimizer_parameter_inventory: true
  attention_processor: QwenImage21AttnProcessor
```

### A.5 Matched promotion configuration derivation

The companion files `flux9b_80_matched.yaml` and `qwen7b_80_matched.yaml` are full configurations, not implicit YAML inheritance. They differ from the corresponding expanded 80 GB files only in `run.name` and `data.target_size: [768,768]` / `data.reference_size: [512,512]`.

For Qwen, an exact hardware-only reproduction additionally loads the identical adapter/resume state and retains its noise/order/scheduler settings. For FLUX 9B, the branch starts fresh; its separate site map and conditioning encoder are already represented in the model-specific configuration.

To run a different mode, keep all model/data settings the same and set a unique output run ID. For a higher-resolution Qwen continuation, explicitly edit the new stage's learning rates/schedule as discussed in Section 9 rather than letting a changed configuration be treated as an exact resume.

### A.6 Existing Toolkit schema: FLUX 4B upstream LoRA smoke

This is a **proposed smoke config using the upstream Toolkit schema**, adapted from its examples and inspected model wrapper. Replace `/ABS/BA_ROOT` with the absolute workspace path. The stock encoder loader still uses its documented remote Qwen3-4B ID; record the resolved revision for this smoke. The reproducible research loader must use the local override from the main configs.

```yaml
job: extension
config:
  name: flux4b_upstream_lora_smoke
  process:
  - type: sd_trainer
    training_folder: /ABS/BA_ROOT/runs
    device: cuda:0
    network:
      type: lora
      linear: 16
      linear_alpha: 16
    save:
      dtype: bf16
      save_every: 50
      max_step_saves_to_keep: 3
      push_to_hub: false
    datasets:
    - folder_path: /ABS/BA_ROOT/data/pairs_export/target
      control_path: /ABS/BA_ROOT/data/pairs_export/reference
      caption_ext: txt
      caption_dropout_rate: 0.0
      shuffle_tokens: false
      cache_latents_to_disk: true
      resolution:
      - 512
    train:
      batch_size: 1
      steps: 100
      gradient_accumulation_steps: 8
      train_unet: true
      train_text_encoder: false
      gradient_checkpointing: true
      noise_scheduler: flowmatch
      optimizer: adamw8bit
      lr: 0.0001
      dtype: bf16
      skip_first_sample: true
      disable_sampling: true
      ema_config:
        use_ema: false
    model:
      name_or_path: /ABS/BA_ROOT/weights/flux4b
      arch: flux2_klein_4b
      vae_path: /ABS/BA_ROOT/weights/flux_vae/ae.safetensors
      quantize: false
      quantize_te: false
      model_kwargs:
        match_target_res: true
    sample:
      sampler: flowmatch
      sample_every: 10000
      width: 512
      height: 512
      prompts: []
      seed: 42
      guidance_scale: 4.0
      sample_steps: 50
meta:
  name: '[name]'
  version: '1.0'
```

```bash
source "$BA_ROOT/envs/flux-toolkit/bin/activate"
cd "$BA_ROOT/sources/ai-toolkit-flux"
python run.py "$BA_ROOT/configs/flux4b_upstream_smoke.yaml"
```

`match_target_res: true` is included to constrain the reference processing to the target's pixel budget in this equal-size smoke. Always inspect the actual resulting grid. The native smoke does not replace the new cached loader, registered branch or matched-budget LoRA implementation.

## Appendix B. Source-clone helper

Save as `scripts/clone_sources.sh`, or use the supplied identical file. This clones code only; it neither installs an environment nor downloads model weights. It deliberately refuses to overwrite an existing directory.

```bash
#!/usr/bin/env bash
set -euo pipefail
export BA_ROOT="${BA_ROOT:-$HOME/work/cl39_dit}"
mkdir -p "$BA_ROOT"/{sources,envs,configs,data,runs,locks,weights,cache,scratch}
clone_pinned() {
  local remote="$1" name="$2" ref="$3" dest
  dest="$BA_ROOT/sources/$name"
  [[ ! -e "$dest" ]] || { printf 'Refusing to overwrite: %s\n' "$dest" >&2; return 1; }
  git clone --filter=blob:none "$remote" "$dest"
  git -C "$dest" fetch origin "$ref"
  git -C "$dest" switch --detach "$ref"
  git -C "$dest" switch -c research/cl39-ba
  git -C "$dest" submodule update --init --recursive
  git -C "$dest" rev-parse HEAD > "$BA_ROOT/locks/$name.commit"
}
clone_pinned https://github.com/huggingface/diffusers.git diffusers-qwen fef717ffb01f407d2637584ed936c16db908587a
clone_pinned https://github.com/ostris/ai-toolkit.git ai-toolkit-flux ecee894ed2b1f3716d9d7326693061ec1a3105bb
for spec in 'QwenLM/Qwen-Image-2.1 qwen-release' 'black-forest-labs/flux2 flux2-official'; do
  read -r remote name <<< "$spec"
  dest="$BA_ROOT/sources/$name"
  [[ ! -e "$dest" ]] || { printf 'Refusing to overwrite: %s\n' "$dest" >&2; exit 1; }
  git clone --depth 1 "https://github.com/$remote.git" "$dest"
  git -C "$dest" rev-parse HEAD > "$BA_ROOT/locks/$name.commit"
done
printf 'Source clones only. Install environments and audit weights as described in the plan.\n'
```

## Appendix C. Metadata-first weight audit and selective download helper

Save as `scripts/weights_manifest.py`, or use the supplied identical file. `audit` accesses metadata and writes a lock; only an explicit `download` action retrieves weight files. The selected-file total is not incremental download size: existing cached blobs may reduce new downloads, whereas temporary files and environments add disk use. Byte-size checks are not a replacement for the Hub client's transport/integrity verification.

```python
#!/usr/bin/env python3
"""Audit selected HF files without downloading weights; download only from a saved lock.

Requires huggingface_hub. Authentication uses HF_TOKEN or the normal HF login cache.
No architecture code is installed, no training is launched, and no files are deleted.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPECS: dict[str, dict[str, Any]] = {
    "qwen21": {
        "repo": "Qwen/Qwen-Image-2.1",
        "patterns": ["model_index.json", "LICENSE*", "README.md", "processor/*", "scheduler/*.json",
                     "text_encoder/*.json", "text_encoder/*.safetensors", "transformer/*.json",
                     "transformer/*.safetensors", "vae/*.json", "vae/*.safetensors"],
        "required_groups": [["model_index.json"], ["processor/*"], ["text_encoder/*.safetensors"],
                            ["transformer/*.safetensors"], ["vae/*.safetensors"]],
    },
    "flux4b": {
        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
        "patterns": ["flux-2-klein-base-4b.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["flux-2-klein-base-4b.safetensors"]],
    },
    "flux9b": {
        "repo": "black-forest-labs/FLUX.2-klein-base-9B",
        "patterns": ["flux-2-klein-base-9b.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["flux-2-klein-base-9b.safetensors"]],
    },
    "flux4b_text": {
        "repo": "Qwen/Qwen3-4B",
        "patterns": ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        "required_groups": [["*.safetensors"], ["config.json"], ["tokenizer.json", "tokenizer.model"]],
    },
    "flux9b_text": {
        "repo": "Qwen/Qwen3-8B",
        "patterns": ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
        "required_groups": [["*.safetensors"], ["config.json"], ["tokenizer.json", "tokenizer.model"]],
    },
    "flux_vae": {
        "repo": "ai-toolkit/flux2_vae",
        "patterns": ["ae.safetensors", "LICENSE*", "README.md"],
        "required_groups": [["ae.safetensors"]],
    },
}
SETS = {
    "flux48": ["flux4b", "flux4b_text", "flux_vae"],
    "flux80": ["flux9b", "flux9b_text", "flux_vae"],
    "qwen": ["qwen21"],
    "small": ["flux4b", "flux4b_text", "flux_vae", "qwen21"],
    "all": ["flux4b", "flux4b_text", "flux_vae", "qwen21", "flux9b", "flux9b_text"],
}


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, p) for p in patterns)


def bytes_from_sibling(sibling: Any) -> int:
    size = getattr(sibling, "size", None)
    if size is None:
        lfs = getattr(sibling, "lfs", None)
        size = lfs.get("size") if isinstance(lfs, dict) else getattr(lfs, "size", None)
    if size is None or int(size) < 0:
        raise ValueError(f"No trustworthy size metadata for {sibling.rfilename}; refusing a zero estimate")
    return int(size)


def select_files(spec: dict[str, Any], siblings: list[Any]) -> list[dict[str, Any]]:
    result = [{"path": s.rfilename, "bytes": bytes_from_sibling(s)} for s in siblings
              if matches(s.rfilename, spec["patterns"])]
    names = [f["path"] for f in result]
    for group in spec["required_groups"]:
        if not any(matches(n, group) for n in names):
            raise ValueError(f"Missing required group {group} in {spec['repo']}")
    return sorted(result, key=lambda f: f["path"])


def audit(set_name: str) -> dict[str, Any]:
    from huggingface_hub import HfApi
    api = HfApi()
    components = []
    for key in SETS[set_name]:
        spec = SPECS[key]
        info = api.model_info(spec["repo"], revision="main", files_metadata=True)
        if not info.sha:
            raise ValueError(f"Could not resolve a commit for {spec['repo']}")
        files = select_files(spec, info.siblings or [])
        components.append({"key": key, "repo": spec["repo"], "revision": info.sha,
                           "files": files, "bytes": sum(f["bytes"] for f in files)})
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "set": set_name, "components": components,
            "bytes": sum(c["bytes"] for c in components)}


def validate_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1:
        raise ValueError("Unknown lock schema")
    seen = set()
    for c in lock["components"]:
        key = c["key"]
        if key in seen or key not in SPECS or c["repo"] != SPECS[key]["repo"]:
            raise ValueError(f"Invalid or duplicate component {key}")
        seen.add(key)
        if len(c["revision"]) != 40 or any(ch not in "0123456789abcdef" for ch in c["revision"].lower()):
            raise ValueError("A full HF commit SHA is required")
        for f in c["files"]:
            p = Path(f["path"])
            if p.is_absolute() or ".." in p.parts or not matches(f["path"], SPECS[key]["patterns"]):
                raise ValueError(f"Unexpected file path {p}")
            if not isinstance(f["bytes"], int) or f["bytes"] < 0:
                raise ValueError("Invalid file size")
        names = [f["path"] for f in c["files"]]
        for group in SPECS[key]["required_groups"]:
            if not any(matches(n, group) for n in names):
                raise ValueError(f"Incomplete component {key}")
        if c["bytes"] != sum(f["bytes"] for f in c["files"]):
            raise ValueError("Component total is inconsistent")
    if lock["bytes"] != sum(c["bytes"] for c in lock["components"]):
        raise ValueError("Lock total is inconsistent")


def report(lock: dict[str, Any]) -> None:
    for c in lock["components"]:
        print(f"{c['key']:14s} {c['bytes']/1e9:9.3f} GB  {c['bytes']/2**30:9.3f} GiB  "
              f"{len(c['files']):3d} files  {c['revision']}")
    print(f"Selected-file total: {lock['bytes']/1e9:.3f} GB / {lock['bytes']/2**30:.3f} GiB")
    print("This is selected source-file size, not missing-download size, peak disk or GPU memory.")
    print("Budget separately for environments, data, conditioning caches, checkpoints and temporary space.")


def download(lock: dict[str, Any], weights_dir: Path, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download
    weights_dir.mkdir(parents=True, exist_ok=True)
    # Do not overwrite pre-existing model directories or symlinks pointing elsewhere.
    for c in lock["components"]:
        destination = weights_dir / c["key"]
        expected_suffix = Path("snapshots") / c["revision"]
        if os.path.lexists(destination):
            if not destination.is_symlink() or not str(destination.resolve()).endswith(str(expected_suffix)):
                raise FileExistsError(f"Refusing to replace {destination}")
    for c in lock["components"]:
        snapshot = Path(snapshot_download(repo_id=c["repo"], revision=c["revision"],
                        allow_patterns=[f["path"] for f in c["files"]], cache_dir=cache_dir))
        for f in c["files"]:
            path = snapshot / f["path"]
            if not path.is_file() or path.stat().st_size != f["bytes"]:
                raise RuntimeError(f"Missing file or size mismatch: {path}")
        destination = weights_dir / c["key"]
        if os.path.lexists(destination):
            if not destination.is_symlink() or destination.resolve() != snapshot.resolve():
                raise FileExistsError(f"Refusing to replace {destination}")
        else:
            destination.symlink_to(snapshot.resolve(), target_is_directory=True)
        print(f"Verified files; local alias: {destination} -> {snapshot}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "download"])
    parser.add_argument("--set", choices=SETS, default="small")
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--cache-dir", default=os.getenv("HF_HUB_CACHE"))
    args = parser.parse_args()
    if args.action == "audit":
        if args.lock.exists():
            raise FileExistsError("Use a new lock filename; existing locks are immutable")
        lock = audit(args.set)
        validate_lock(lock)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        with args.lock.open("x", encoding="utf-8") as handle:
            json.dump(lock, handle, indent=2)
            handle.write("\n")
        report(lock)
    else:
        if args.weights_dir is None:
            parser.error("download requires --weights-dir")
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        validate_lock(lock)
        report(lock)
        download(lock, args.weights_dir, args.cache_dir)


if __name__ == "__main__":
    main()
```

## Source ledger

Source links establish released model/loader/trainer facts. Proposed branch equations, memory budgets, layer selections, learning rates, configurations and promotion policy are engineering recommendations in this document, not reported upstream training results. The original 80 GB handoff provided the starting organization and code map; this revision adds verified klein 4B support, the Qwen size-family limitation, explicit staged loaders, matched promotion profiles and disk accounting.

[Q1]: https://github.com/QwenLM/Qwen-Image-2.1/blob/main/README.md "Official Qwen-Image-2.1 release: 7B visual model, 32 layers, conditioning and optional prompt rewriters"
[Q2]: https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21 "Official Diffusers Qwen-Image-2.1 pipeline documentation"
[Q3]: https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/examples/dreambooth/README_qwenimage21.md "Audited paired LoRA documentation, image slots, geometry, caches and quantization"
[Q4]: https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/src/diffusers/models/transformers/transformer_qwenimage21.py "Audited Qwen transformer, processors, prefix modulation and caches"
[Q5]: https://github.com/huggingface/diffusers/blob/fef717ffb01f407d2637584ed936c16db908587a/examples/dreambooth/train_dreambooth_lora_qwenimage21_img2img.py "Actual upstream paired Qwen trainer"
[F1]: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B "Official klein Base 4B checkpoint and model licence"
[F2]: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B "Official klein Base 9B checkpoint and gated model licence"
[F3]: https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/README.md "Audited Toolkit installation and runner documentation"
[F4]: https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/extensions_built_in/diffusion_models/flux2/flux2_klein_model.py "Native 4B/9B classes, encoder IDs and checkpoint names"
[F5]: https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/extensions_built_in/diffusion_models/flux2/src/model.py "Native 4B/9B dimensions, double/single blocks and checkpointing"
[F6]: https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/extensions_built_in/diffusion_models/flux2/flux2_model.py "Actual model loading, control-image encoding and target/reference sequence construction"
[F7]: https://github.com/ostris/ai-toolkit/blob/ecee894ed2b1f3716d9d7326693061ec1a3105bb/requirements_base.txt "Audited Toolkit dependency pins"
[F8]: https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora "BFL's klein LoRA training guide; supporting feasibility, not a custom-BA memory measurement"
[W1]: https://huggingface.co/Qwen/Qwen-Image-2.1/tree/d0aa88ebf9ce42c3d055fcae2c138bf0582fcc31 "Observed official Qwen pipeline file tree, about 33.1 GB"
[W2]: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B/tree/main "Observed native 7.75 GB file and additional Diffusers layout"
[W3]: https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B/tree/main "Observed native 18.2 GB checkpoint"
[W4]: https://huggingface.co/Qwen/Qwen3-4B/tree/main "Observed 8.06 GB encoder repository"
[W5]: https://huggingface.co/Qwen/Qwen3-8B/tree/main "Observed 16.4 GB encoder repository"
[W6]: https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models "Cross-check of Qwen BF16 visual checkpoint size; not the recommended training download layout"
[W7]: https://huggingface.co/ai-toolkit/flux2_vae/tree/main "Native Toolkit AE file, about 336 MB"
[W8]: https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main/text_encoder "Qwen bundled vision-language encoder, about 17.5 GB"
[W9]: https://huggingface.co/Qwen/Qwen-Image-2.1/tree/main/vae "Qwen VAE file, about 1.35 GB"
[T1]: https://pytorch.org/get-started/previous-versions/ "Published Torch/torchvision wheel pairs, including 2.10.0/0.25.0"
[C1]: https://huggingface.co/docs/huggingface_hub/guides/manage-cache "Official HF cache structure, snapshots, shared stored files and cache paths"
