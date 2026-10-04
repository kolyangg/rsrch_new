# BA capacity and stability investigation — 2026-10-01

## Observed failure

The 100k continuation was superseded after its completed 60k validation when
the user requested architecture improvements. The partially executed 60k–80k
segment is not an evaluated checkpoint. All completed checkpoints are retained.

| Update | Prompted 24-image ID similarity |
| --- | ---: |
| 6,000 | 0.21402 |
| 20,000 | 0.19886 |
| 40,000 | 0.18437 |
| 60,000 | 0.15408 |

Fit flow MSE improves while separate-noise probe MSE worsens, from 1.06053
at 6k to 1.14427 at 60k. There are only 228 fixed fit cases (19 images,
6 sigmas, 2 noise seeds). More updates repeatedly fit the same features.
There is no identity-specific loss, so improving fit MSE does not imply
improving prompted identity similarity.

Fixed cached-input decomposition, averaged across the probe sigma groups:

| Update | Attention entropy / log(key count) | Reference output RMS | Other output RMS | Path cosine |
| --- | ---: | ---: | ---: | ---: |
| 6k | .00303 | .769 | 1.339 | -.424 |
| 20k | .00103 | 1.392 | 1.927 | -.746 |
| 40k | .00073 | 2.641 | 3.023 | -.903 |
| 60k | .00040 | 3.146 | 3.473 | -.924 |

The unbounded learned Q/K logits saturate attention, and growing branch and
query/noise contributions largely cancel. Reference-read-off MSE therefore
must not be presented as proof of identity transfer. Evidence is in
`runs/face_flow_capacity_investigation_20261001/path_diagnostics.json`.
The diagnostic used fixed cached inputs, identical checkpoints, and no new
noise or training updates. The probe is a development set of new noise on the
same photographs, not an independent identity/generalization test set.

## Candidate and controls

The old head's 256 is a projection bottleneck width, **not** native-attention
LoRA rank. The native FLUX backbone has no trainable LoRA in this experiment.
A simple 512-wide control tests capacity. `ConditionedFaceFlow` additionally:

- bounds cosine Q/K attention logits to [-4, 4];
- normalizes query/read features before mixing;
- explicitly conditions on sigma through a small MLP;
- adds a nonlinear residual layer before the zero-initialized velocity output.

Both heads still start with zero face flow, use the same frozen feature seam,
and predict all face velocity without adding native velocity. Only BA is
trainable. Native-generated masks, native exterior, reference, prompts, seeds,
24-image panel, flow target and metric definitions remain unchanged. Raw VAE
outputs and final exterior-preserving composites are both retained.

Controls use lr .001, batch8, seed142, the exact same case order and native
face-token flow MSE. Capacity and stability are first compared at 2k. Fresh
full-backbone cases with stratified continuous sigmas can be generated with
`scripts/expand_face_flow_cache.py`; these must be a separate named experiment.
No synthetic feature perturbation is claimed to be fresh backbone training.

## Checks and limitations

The focused head check verifies exact zero flow, finite gradients reaching all
parameters after the first output update, timestep/reference dependence and
checkpoint reload. A 512-wide fresh-process step-51 check differed in one
FP32 weight tensor by 1.862645149e-9; all others were exact. The first control
stopped at that strict equality assertion. Its evidence is retained; the retry
checks atol 1e-8 / rtol 1e-6 and records the actual differences. This is not
reported as bit-exact resume.

Image validation, actual pretrained native-off parity, cached/live prediction
parity, peak memory and Comet records are required before accepting a candidate.
The proposed changes do not guarantee monotonically improving ID scores.

## Controlled 2k cache results

All use the same fixed 228 fit / 114 probe cases, batch8, lr .001,
seed142 and native flow MSE:

| Head | Fit MSE | Separate-noise MSE |
| --- | ---: | ---: |
| Original width256 | .83184 | 1.07052 |
| Original width512 | .76296 | 1.09727 |
| Conditioned width512 | .57129 | 1.03028 |

Width alone worsened probe error. The conditioned head improved it, but
its growing fit/probe gap still motivates fresh-noise training. Its prompted
24-image validation is the deciding measurement, not cache loss.

Retained control runs:
`flux4b_face_original512_control_r1_20261001` and
`flux4b_face_conditioned512_control_r1_20261001`.
The latter has 5,327,360 trainable parameters. Both resumed step 51 within the documented
FP32 tolerance. All gradients are finite and every parameter tensor updates.

The captured native queries already carry pretrained text/reference conditioning.
Added attention is an additional identity route, not an exclusive identity input.
The conditioned candidate combines normalization, bounded logits, timestep
conditioning and a nonlinear layer; these controls cannot attribute an improvement
to any one of those changes individually.

## Initial image result and cache-size follow-up

Conditioned width512 at 2k scores ID **0.18814**, below the original width256
at 2k (**0.19661**), despite its lower flow probe error. Visual inspection of
six paired crops shows substantial face distortion in both heads; the native
backbone remains cleaner. No quality improvement is claimed from the loss
probe. The candidate's 6k fit/probe MSE is .33062/1.11965, another clear fit/probe
split. Full 6k prompted-image scoring is required to complete this control.

A separate planned run keeps the conditioned architecture but uses 684 fit
cases (228 original plus 456 new full-backbone cases with fresh noise and
stratified continuous sigmas), retains the same 114 probe cases, and changes
lr to .0003 and AdamW weight decay to .01. The original 19 photographs remain
unchanged. This is a bundled intervention, not an isolated capacity comparison.
The expanded cache is still finite and is not advertised as fresh-noise online
training; further long training must be justified by actual image metrics.

The completed fixed-cache conditioned control scored **0.18814 at 2k →
0.17927 at 6k**, with one unowned face at 6k. It is rejected as a quality
improvement. Bounded logits did prevent saturation (entropy fraction .917–.960
on six checked probe cases), but did not solve the data/loss problem. Its final
comparison plots and 24 paired crops are preserved in the control run.
The shared logger subsequently gained dynamic cache counts for the diverse run;
replaying the controls requires their recorded source snapshots.

## Reproduction

No new dependencies are needed beyond the FLUX toolkit and separate metrics
environments. Commands run from the repository root. GPU stages are serial.

```bash
# Add 456 fresh full-backbone cases (old probe and original fit retained).
envs/flux-toolkit/bin/python -m scripts.expand_face_flow_cache \
 --source runs/flux4b_masked_face_flow_24_stable_20261001 \
 --destination runs/face_flow_diverse_cache_20261001

# New experiment, fresh random/zero BA, same images/reference/prompted panel.
envs/flux-toolkit/bin/python -m scripts.conditioned_face_flow init \
 --run runs/flux4b_face_conditioned512_diverse_20261001 \
 --cache runs/face_flow_diverse_cache_20261001 \
 --lr .0003 --weight-decay .01 --total 10000
```

For an existing run, `train --resume N --step M` restores both BA and AdamW;
`infer`, `decode`, and `score --step M` run in that order. All commands take
`--run`. Use train-to-50 and resume-from-50 first for the fresh-process restart
check. Infer/decode/score step 0 verifies and records the noise initialization.
`scripts.review_conditioned_face_flow` creates score plots, all 24 paired face
crops, and a summary/audit and logs these to the run's immutable Comet key.
The current diverse experiment validates 0/2k/10k on the same named 24-image
panel. It is a diagnostic experiment, not the default 96-item protocol.

The diverse run's `execution_plan.json` records the explicit 0/2k/10k image
validation schedule. CLI inference stages override the inherited generic
`validation_every: 2000` field, which this experimental runner does not use to
schedule image validation. Checkpoints still save every 2k, and loss probes every 1k.

## Expanded-cache run admission

`runs/flux4b_face_conditioned512_diverse_20261001`,
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/9faec049f9a544948c4faeb2e2fc3a7b).
The 456 fresh cases required approximately 320 seconds of full-backbone work.
The retained head has **5,327,360 trainable parameters**; everything outside
BA is frozen. All first-step/second-step gradients pass, every tensor updates,
and the step 51 process-resume difference is at most 1.862645149e-9.

The actual step-zero 24-image generation scores **0.0202404061**, exactly
matching the previous noise initialization. Pretrained BA-off parity is exact.
At 2k, fit/probe MSE is **.89322/1.02343**. Mean cached update time through 2k is
.0100 seconds; peak reserved training memory is 1.334 GiB. This excludes cache
building and image validation, and must not be described as full FLUX training
throughput. Image scores at 2k/10k determine whether this run is useful.

At 10k the diverse run reaches fit/probe MSE **.51707/.80847**; the probe improves
through the longer interval instead of reversing. It improves all six probe
sigma groups versus the rejected fixed-cache conditioned head at 6k (a bundled
comparison with different update counts). The fresh-process step 2001 check has
maximum difference 7.450580597e-9, within the declared tolerance. All 10k losses
and gradients are finite and all 11 BA parameter tensors update. See
`runs/face_flow_capacity_investigation_20261001/sigma_probe_comparison.json`.

## Completed prompted-image results

| Diverse-noise conditioned BA | ID similarity | CLIP text similarity |
| --- | ---: | ---: |
| 0 | .02024 | 28.1677 |
| 2,000 | .16421 | 26.8878 |
| 10,000 | **.22660** | 26.1877 |

**21/24 images improve in ID score from 2k to 10k**; mean paired increase is
.062384 (+38.0%). The 10k score exceeds the previous best BA checkpoint's .21402
at 6k by .01257 (+5.9%). That is a comparison of best observed checkpoints,
not a matched10k architecture-only comparison. Native remains higher at .33140.
CLIP text similarity declines by .7001 from 2k to 10k. Face crops show improved
structure in several scenes, but visible texture/shape artifacts remain and
some expressions differ from the native scene. This is an ID improvement on
the fixed one-ID panel, not a general image-quality or multi-ID result.

All 24 trained images at 2k and10k have an owned face under the unchanged scoring
rules. The full72-image set has exactly zero exterior pixel error (explicit
post-decoder compositing, with raw outputs also retained). All latent/image
records point to the correct checkpoint hash. Actual cached/live predictions
match exactly at both trained checkpoints. Native-off parity is exact. The
10,000 loss/gradient records are finite; every BA parameter tensor changed.
Process resume passes the documented numerical tolerance. Peak CUDA reserved
memory is **1.334 GiB for cached training /9.396 GiB for inference**. Mean cached
update time is .01007 seconds, about 101 seconds of optimizer updates for 10k;
cache generation, probes, loading, decoding, scoring and logging are additional.

Training is complete at 10k; no 100k retry was launched. More capacity alone did
not help. The retained intervention combines a wider conditioned head, bounded
attention, more noise/timestep coverage and gentler optimization. Its results
justify this direction but do not demonstrate monotonic improvement through
100k. A future longer run should refresh backbone feature caches periodically
and retain image-based checkpoint selection instead of blindly reusing this
finite pool. An identity-specific objective is another untested option; it
was not used in these results.

Artifacts in the retained run: `comparison_metrics.png`, `paired_faces_1.png`
through `paired_faces_3.png`, `comparison_summary.json`, `probes.json`,
`source_snapshot.zip`, and all raw/final validation images. Plots, summaries, source snapshots and
final images are logged with the run's immutable Comet key; raw decoder outputs
and latent tensors remain local. The source snapshot and this document
separate rejected controls, proposed future changes, and measured results.
