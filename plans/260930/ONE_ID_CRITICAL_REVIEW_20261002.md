# Full-denoiser BA fits 16GB; the cached face head is the weak experiment

**Date:** 2 October 2026. **Scope:** critical review of local one-ID training,
comparison with PhotoMaker CL14 and the April one-ID replay, and a bounded local
hardware check. Existing checkpoints and training code are preserved. No long
training run was started. Labels distinguish `[measured]`, `[code]`, `[report]`
and `[hypothesis]`.

## Executive conclusion

**Keep FLUX.2-klein Base 4B and restore BA inside the pretrained denoiser.**
Meaningful experiments are feasible on the local 16GB GPU. A new eight-update
test ran the full 768px transformer with fresh noise/timesteps, eight rank-128
reference branches and activation checkpointing: **`9.449 GiB` peak reserved,
`2.088 s` per microbatch/update, `12,582,912` trainable BA parameters**.
All branch tensors updated, gradients were finite, and native/BA-off predictions
were bit-exact before and after training. This establishes bounded training
feasibility, not useful identity learning or sustained thermal performance.
[measured]

I pushed the earlier simplification too far. Replacing the native face-velocity
prediction with a separate head made updates fast and branch influence obvious,
but changed the research problem. Increasing that head's width/depth and training
longer has not produced useful generated-face improvement over the native model.
The optimization is functioning; the current experiment is a poor basis for
deciding whether BA can work in FLUX. [code][measured]

The primary acceptance metric remains the original ownership-matched generated
image `id_sim`, together with CLIP and visual face quality. Cached regression
loss, identity training loss, GPU utilization and number of updates are not
sufficient acceptance metrics.

## 1. Results and immutable run identities

| Experiment | Immutable Comet key | Evidence |
| --- | --- | --- |
| Latest deterministic deep head | `75e5d8961bdf40efab63ffd2770711ca` | Completed at `5,000`; best additional step `500` |
| Earlier eight-block suffix training | `7beb18408145451896ccdeaa9a6722bb` | Rank `256`, BA output projections and learned face routers |
| Its fixed24 checkpoint evaluation | `65297da706af4cd6a0308b78fc18f9ca` | Checkpoints `0/1,000/2,000`; same panel as the native comparison below |
| Earlier Vast one-ID, full denoiser | `25604bb4dc58429d9378efea58b43ed2` | Separate fixed12 panel and `50` sampling steps |
| Historical CL14 production | `6fe0028be92242c38056b3d36665fdd6` | Cosmic multi-ID, `24,000` updates, fixed96 |
| New local hardware probe | No Comet experiment; bounded profiling artifact | Eight disposable updates; no saved model or quality claim |

The latest head uses the fixed24 prompt/seed panel at 768px. Its step zero is
already a trained parent checkpoint, not a new random branch. [code]

| Delivered pipeline / additional updates | ID_sim | CLIP |
| --- | ---: | ---: |
| Native backbone on the same panel | `0.331399` | `28.0729` |
| Latest head at `0` | `0.229714` | `26.1648` |
| Latest head at `500`, best | `0.251904` | `26.2695` |
| Latest head at `2,000` | `0.241371` | `26.0794` |
| Latest head at `5,000` | `0.225265` | `26.2110` |

These are delivered-pipeline comparisons: the head uses face CFG `1`, while the
native background/baseline uses CFG `4`. Thus the native-versus-head gap does not
isolate architecture alone. Within the latest run all settings are matched.
`20/24` cases improve from its start to `500`, but `15/24` worsen from `500` to
`5,000`. At `5,000`, mean identity is below the already weak starting point.
The controller's plateau flag means its patience was exhausted after regressions;
it does not establish convergence to a useful model. [measured][code]

All three paired face sheets were reviewed. Eye/mouth and patchy texture defects
remain, notably `oneid_01_s1`, `oneid_05_s1`, `oneid_10_s1`; both skiing cases lose
goggle detail. The native counterparts are visibly cleaner. Preserving pixels
outside the mask is enforced by compositing, so exact exterior equality does not
show that BA learned spatial locality. [measured][code]

The earlier eight-block suffix is also an important negative result. On the
matched fixed24 panel its ID rises only `0.331399 -> 0.339220` at `2,000`, with
only `10/24` cases improving. Its prompt-cluster bootstrap 95% interval for the
mean gain is `[-0.02296, +0.03788]`. CLIP falls `28.0729 -> 26.8572`, and
TOPIQ-Face falls `0.689587 -> 0.636220`. Simply restoring that exact fixed-cache,
rank-256, learned-router configuration is not the recommendation. [report]

The older full-denoiser Vast run produced `0.310866 -> 0.338955` at `1,500`, then
`0.331836` at `2,000`. It proves branch influence, not monotonic progress. Its
fixed12/50-step scores must not be pooled with fixed24/20-step scores. [report]

## 2. Why training is fast, and how CL14 differs

| Property | Current deep head | Historical CL14 |
| --- | --- | --- |
| Target image size | `768px` | `1024px` |
| Trainable computation | Three reference reads and MLPs after captured final-block Q/K/V | Branched self-attention throughout the pretrained U-Net |
| Frozen backbone during each update | Absent from the main flow update; features precomputed | Executed, with gradients through downstream frozen operations |
| Spatial processing after BA | Each target query independent of other target queries | Remaining native spatial attention, convolutions, residual blocks and output layers |
| Output task | Entire face latent velocity from learned core plus head | Modify attention messages, then use the native denoising network |
| Training inputs | Fixed bank: `1,140` fit cases from `19` photos, fixed noise/sigma/token samples | Newly sampled diffusion noise and timesteps |
| Batch meaning | `256` cached cases, `64` sampled face tokens each | `2` target/reference image pairs through the U-Net |
| Trainable parameters | `30,831,747`; frozen earlier head adds `5,327,360` | `219,217,920` total: BA `127,795,200`, generic LoRA `30,474,240`, PhotoMaker-default LoRA `60,948,480` |
| Measured step time | `0.515 s`, local laptop GPU, excludes cache construction | Historical production `2.21 s`, A100, batch `2`; different computation/hardware |

CL14 is a Cosmic multi-ID experiment, not the historical April one-ID run.
Its BA rank is `128`; generic/PhotoMaker ranks are `32/64`. It jointly trains
all three paths. Despite the inherited `train_ba_only` flag, the resolved
parameter contract contains those co-adapters. PhotoMaker identity conditioning
also supplies a pretrained identity prior. CL14's validation restores the
pretrained PhotoMaker-default adapter and uses RealVisXL; these historical
train/validation differences should not be copied into a new clean FLUX test.
[code][report]

The April one-ID replay config instead specifies rank `32`, batch `2`, separate
target/reference images of the same ID, BA across all steps, reference and target
branch weights, and a face-masked/full diffusion-loss alternation every `2`
updates. It caches conditioning, while target noising and U-Net execution remain
online. A verified timing for that exact April run was not located; the `2.21 s`
number above belongs to CL14. [code]

CL14's key operation is conceptually a spatial split **inside attention**:

```text
target face queries + reference-face K/V -> face attention message
target background queries + target K/V -> background attention message
mask routes these messages -> pretrained output projection / residual / U-Net
```

It does not ask a newly initialized head to recreate the final face denoising
velocity from scratch. The current head stops the backbone before the final
block's native attention output/MLP and final velocity projection, detaches the
features, and trains a replacement readout. Upstream captured features still
contain spatial context, but the learned head has no new target-to-target
spatial interaction. [code]

Also, the current `branch.rank: 256` in the inherited resolved YAML is not the
rank of its trained head. The factory builds dense width-`1024` projections and
two additional read blocks from `identity.json`. Changing that legacy rank field
alone would not strengthen this head. [code]

## 3. What is limiting the current experiment

1. **Loss of the pretrained face-denoising output path.** Learning complete face
   velocity from a small fixed dataset is a harder and different task than
   adapting pretrained attention. Random/zero face output helped test visibility
   of branch influence; it is a poor quality-training initialization. [hypothesis
   supported by code and the paired images]
2. **Finite regression bank reused excessively.** At `5,000 x 256 / 1,140`, the
   current stage presents about `1,123` bank-equivalents to the optimizer, on top
   of prior training. Sampling with replacement adds no new images, noise or
   sigma values. Inference recaptures features on its generated trajectories;
   exact cache/live parity for an identical input does not guarantee coverage of
   those trajectories. [code][calculation]
3. **Objective improvement does not transfer.** Fit flow MSE falls
   `0.4598 -> 0.2065`; the same-photo/different-noise probe changes
   `0.7105 -> 0.6754`. Auxiliary identity loss falls `0.1533 -> 0.0422` across
   the first/last full cycles after step `500`, while generated ID regresses.
   The identity objective uses the scoring recognizer family and a decoded
   one-step clean estimate. It is not an independent quality validation. [measured]
4. **Capacity is in the wrong place.** Thirty-one million parameters after the
   final feature capture cannot replace the effect of BA followed by pretrained
   spatial computation. Larger batches on nineteen photos and a wider head can
   make fitting this bank easier without improving generation. [hypothesis]
5. **One ID proves a limited claim.** It can establish learnability and useful
   prompted generation for that person. It cannot establish unseen-ID transfer;
   a model can memorize that person and ignore the reference. The current
   `refiner_off` ablation leaves the frozen core's reference read active. [code]

### Confidence and what is not established

| Claim | Confidence / basis |
| --- | --- |
| Current optimization updates real parameters | High: finite logs, all `35` tensors change, frozen core unchanged, resume verified |
| Current generated faces fall short of native output | High for this delivered fixed24 panel: metrics and paired images |
| Missing gradient or unloaded checkpoint explains the result | Evidence argues against it: prior checkpoint/provenance/parity audits pass |
| The dense final head differs materially from CL14 | High: source and resolved parameter contracts |
| Full-denoiser rank-128 BA can fit local16GB at 768px | High for the bounded measured configuration |
| Cache reuse, lost spatial processing, guidance or loss is the single root cause | Not established: the history bundles several changes |
| More rank or 48GB alone will improve identity | Not established |
| One-ID success would prove reference-conditioned generalization | No: requires additional identities and appropriate controls |

## 4. Recommended settings and two experiments

### First: a native-initialized BA baseline inside FLUX

Proposed configuration name: `flux4b_oneid_online_face_qkvo_r128_768`.
This is a new baseline, not an implemented or scored configuration. The primary
scientific comparison is **trained versus untrained adapters on this identical
architecture**, with native BA-off as an additional control. Comparing it with
the old head would be a bundled redesign, not a single-cause ablation.

| Setting | Proposed initial value |
| --- | --- |
| Backbone | Same pinned, undistilled FLUX.2-klein Base 4B |
| Image/reference geometry | `768 / 512`; use a separately named `512` experiment only if admission fails |
| BA sites | Eight sites, initially the existing four dual + four single block indices |
| Trainable scope | Branch-local target Q, reference K/V and attention-output LoRA only |
| Rank / estimated parameters | `128`; approximately `25.2M` for eight sets of four square `3072`-wide LoRA projections |
| Initialization | Reuse native projection weights; LoRA updates start at zero |
| Face routing | CL14-like replacement of the attention message at masked face queries; preserve native output projection, MLP, residuals and final prediction layer |
| Inference mask | Frozen native prompt/seed generation mask, with recorded provenance; no target-photo input |
| Training mask | Training-photo face supervision only |
| Training data | Same one-ID photos/reference; newly sample noise and native timesteps every microbatch |
| Safe caches | Text embeddings, VAE latents and immutable reference inputs; no fixed target hidden-state bank |
| Precision / memory | BF16 frozen transformer, FP32 trainable LoRA/Adam, activation checkpointing; stage encoder/VAE separately |
| Batch | Microbatch `1`, accumulation `4`, effective batch `4` |
| Optimizer | AdamW, initial LR `5e-5`, warmup `100`, gradient clip `1` |
| Objective | Native flow target with face-focused supervision; begin without auxiliary ArcFace loss |
| Generation | Full-image generation from prompt/reference; `20` steps, CFG `4` in the same patched positive/negative paths |
| Validation | Existing named fixed24 panel at `0/500/1,000/2,000`; unchanged original metrics and paired crops |

The masked split replaces an **attention message**, not the model's final
velocity. Native projection weights provide a pretrained starting point.
BA-on need not equal native at initialization under hard routing; record its own
step-zero panel. BA-off must remain bit-exact native. The whole computation after
the earliest active branch must remain differentiable even though its native
weights are frozen. Preserve backgrounds using the existing explicit spatial
composition contract and record that this is an imposed constraint.

**Hypothesis:** pretrained downstream spatial processing plus fresh noise allows
branch learning to improve new prompt-generated faces without the replacement
head's artifacts. **Prediction:** paired ID gains versus both its untrained
branch and native baseline, with preserved expression, eyewear, CLIP and face
quality. **Risk:** hard routing can still remove useful native face information;
nineteen photos can still overfit. **Decision gates:** finite gradients at every
site, exact native BA-off, checkpoint replay, actual reserved memory `<90%`, then
positive paired ID change supported across prompt clusters without material
quality regression. Do not continue toward `100k` merely because training loss
falls. A noisy or regressing `2k` panel should trigger diagnosis of this baseline.

This architecture's exact memory and quality remain to be measured. Today's
successful profile used the existing K/V-only delta at rank `128` (`12.6M`),
not the proposed Q/K/V/output hard-routing variant (`25.2M`). The new estimate is
`8 x 4 x 2 x 3072 x 128`; native projection weights are reused, not copied into
the trainable set. The old eight-site rank-256 experiment changed only output
projections and used fixed cached prefixes/learned routers; it is a different
configuration.

### Second: increase branch capacity only after the baseline is useful

Proposed configuration name: `flux4b_oneid_online_face_qkvo_r256_768`.
**Single scientific change:** rank `128 -> 256`, keeping sites, data/order,
fresh-noise draws, masks, effective batch, native loss, sampler and validation
fixed. Estimated trainable parameters: `50.3M`.

**Hypothesis:** branch projection capacity limits a baseline that is already
learning useful identity. **Prediction:** a repeatable improvement over rank
`128` at matched updates and images seen. **Risk:** greater overfitting and no
benefit on only one ID. **Gate:** retain the wider arm only if paired ID and
visual quality improve, rather than only train loss. If it fails memory admission
on16GB, run the same 4B model on an explicitly authorized 48GB host. Broader
insertion depth or `1024px` resolution should each be a later separate change.

48GB is useful for more activation headroom, larger true image microbatches,
deeper integration and eventual higher resolution. It is not required just to
start a proper4B BA experiment. Keep4B when changing hardware to preserve an
interpretable comparison; increasing backbone size simultaneously would obscure
whether the BA training design improved.

## 5. Local feasibility evidence and implementation order

The new probe used RTX4090 Laptop, Torch `2.13.0+cu130`, pinned Toolkit source
`ecee894ed2b1f3716d9d7326693061ec1a3105bb`, real pretrained4B weights, existing
768px latent/text inputs and newly sampled noise/timesteps. Two warmup updates
were excluded from the six-update timing window. Peak reserved fraction was
`0.5909`; all `32` branch tensors changed. No VAE or text encoder was resident
during optimization. The full transformer was resident and executed each update.
[measured]

At that measured rate, the existing K/V-only path with accumulation4 would take
about `8.35 s/update`, or `4.64 h` for2k updates before validation. This is an
arithmetic projection from a cool, short run, not an ETA for the proposed wider
architecture. Prior local sustained runs reached thermal throttling; measure
steady-state speed before promising a duration. Validation still legitimately
costs much more per image because it runs a multi-step sampler, CFG and decoding.

Implementation order:

1. Retain the completed head and its best checkpoint as diagnostic evidence.
2. Reuse `ba_dit.training`, the pinned patched backend and native timestep/loss
   path. Add a small imported module for branch-local Q/K/V/output adaptation and
   explicit training/generated masks; do not build another standalone predictor.
3. Keep target/reference indices, normalization, original RoPE and projection
   placement explicit. Recompute hidden states; detach no path after BA.
4. Inventory exact trainable names/parameters and strip inactive legacy config
   fields from this new experiment's metadata.
5. Run the bounded parity/gradient/resume/memory gates above before the named
   training run. Use one GPU process at a time and serial validation.
6. Evaluate generated images, then change only rank in the second experiment.
   Later, test reference sensitivity with controlled replacement/disablement of
   all relevant BA reference paths, and add more identities to test transfer.

## 6. Reproducing this review

Existing artifacts, without retraining:

```bash
cat runs/critical_review_20261002/online_ba_r128_profile.json
cat runs/flux4b_deep_identity1024_det_20261001/final_training_audit.json
cat runs/flux4b_deep_identity1024_det_20261001/final_score_audit.json
cat runs/flux4b_face_one_id_24_val_20261001/paired_bootstrap.json
```

The bounded profile source and resolved config are preserved under
`runs/critical_review_20261002/`. It can be rerun, when the GPU is idle, with:

```bash
envs/flux-toolkit/bin/python runs/critical_review_20261002/profile_online_ba.py
```

This performs eight updates and overwrites only that probe's result files; it
does not save a model. Archive that directory before collecting a new timing.
No new long-run launch command is supplied because the proposed masked Q/K/V/O
module is not implemented or admitted yet.

## 7. Sources

- [Current experiment and audits](IDENTITY_FLOW.md),
  [earlier full-suffix evaluation](FACE_BA_24_VALIDATION.md),
  [original full-denoiser one-ID run](ONE_ID_DIAGNOSTIC.md).
- [Detached feature capture](../../ba_dit/nn/face_crop_flow.py),
  [dense head](../../ba_dit/nn/deep_identity_flow.py),
  [head factory and optimizer](../../scripts/identity_face_flow.py),
  [fixed cache builder](../../scripts/expand_face_flow_cache.py),
  [native fresh-noise loss](../../ba_dit/backends/flux_runtime.py).
- Previous project: `diffusion_template/src/configs/clean_full_runs.json`,
  `large_dataset_joint_r128_24k.yaml`, `CL14_cosmic_joint_shadow_sa128_softmask_24k.yaml`,
  `one_id_rhca_apr2026_replay.yaml`; `src/model/photomaker_branched/attn_processor_cleanest.py`
  and `lora2.py`; `analysis/2026-08-16_training_pipeline_processor_lookup_fix.md`
  and `analysis/2026-08-22_serv_training_code_inventory_cl14_cl19_cl23_cl27_cl39.md`.
  CL14 historical numbers use its recorded contract/audit; its sealed historical
  source snapshot is not present locally, so current source is lineage evidence.
- BFL identifies [FLUX.2-klein Base4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
  as a training/fine-tuning model. Its [official LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
  describes a24GB recipe. That recipe is not a16GB guarantee; our16GB feasibility
  claim above is based on the measured staged-encoder, checkpointed BA workload.
