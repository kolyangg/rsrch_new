# Deeper BA with a decoded face identity objective

## Previous run result (2026-10-01, 19:34 UTC)

`flux4b_reference_refiner1024_b128_20261001` stopped after its complete 5,000-step
24-image validation. ID_sim: 0.226596 at initialization, **0.229714 at 500**,
0.205587 at 2,000, and **0.192748 at 5,000**. Best checkpoint 500 is preserved.
The fit flow loss continued falling; separate-noise flow loss improved through
2,000, then also worsened. Paired crops show face texture/structure degradation.
This is an objective/generalization problem, not evidence of disconnected BA:
finite gradients, parameter updates, loaded checkpoints and image changes were
verified. Native baseline ID_sim remains 0.331399 on this panel.

## New experiment — completed with an early best

Final status: stopped after the scored 5,000-step panel on October 1 at 21:44 UTC.
Best generated-image ID_sim remains **.251904 at 500**. The documented two-check
stopping rule fired after regression at 2k and 5k; this is a development plateau
of the best score, not statistical convergence or a solved visual-quality result.
See the final review below. Do not restart either the failed original or completed
deterministic continuation.

Original run: `runs/flux4b_deep_identity1024_20261001` on the local 16GB GPU only.
It stopped on a resume-check failure after its completed500 validation; see the
recovery section below before taking action. Original Comet key: `c76d2a9a84814005bcd86287d95c2771`;
[experiment](https://www.comet.com/nikolay-2104/rsrch-new/c76d2a9a84814005bcd86287d95c2771).
Read the actual run status and processes before launching anything. The previous
run is complete; do not restart it. Check the new run's status for launch progress.

The existing 1024-wide reference refiner starts from its best 500-step checkpoint.
Two additional 16-head cross-attention/MLP blocks read the reference face again.
Their output projections start at zero, preserving the parent's predictions.
Queries remain independent across target positions, preserving sampled-token
training/full-scene inference parity. The native FLUX backbone and the trained
512-wide BA core remain frozen. Only the BA refiner trains.

Loss: retain native flow MSE on the same fixed cases, and add periodic identity
supervision on 19 training photos at sigmas .2/.4/.6/.8. Decode the clean estimate
`x0 = noisy - sigma * velocity` using the frozen FLUX2 VAE, sample an aligned
112px face using **training-photo** landmarks, and backpropagate `1 - cosine`
through frozen ArcFace to BA. Training reference embedding comes from the fixed
reference image, not the validation embedding file. VAE and ArcFace stay frozen.
ArcFace executor is reused from `rsrch_apr_test/.../arcface_identity_aux.py`.

The VAE sees a contiguous packed-latent training crop with 48px context around
the aligned face. Its context differs from full-image decoding; report this
approximation. Fresh full-backbone features are cached for all 76 identity cases.
The main flow cache/order, 24 generated-image panel, prompts, seeds, masks,
native backgrounds, sampler and metric definitions remain unchanged.

Identity loss uses the same recognition model family as ID_sim. This is an
explicit development objective, not independent evidence of perceptual quality.
Judge the rerun using generated images, original ID_sim/CLIP and paired crops;
do not equate lower identity training loss with successful generalization.
Depth, objective, batch size and LR change together in this named improvement
experiment. A gain would support the combination; it would not isolate which
change caused it. The unchanged main flow cache has 1,140 fit and 114 probe
cases; these are same-photo noise probes, not held-out identities.

## Admission and continuation

Before long training: check ArcFace ONNX parity, nonzero differentiable VAE/ID
gradients, exact initial parent prediction, frozen parameters, actual CUDA
memory below 90%, training throughput and optimizer resume including an ID step.
Choose batch size and identity weight from these measured probes. New Comet run.
Validate 0/500/2,000/5,000 and every 5,000 thereafter; keep the best checkpoint.
Stop after two generated-image checks with best ID gain below .003. The 100k
safety limit is not a convergence claim. Serial controller owns the GPU lock.

Implementation: `ba_dit/nn/deep_identity_flow.py`, `face_identity_loss.py`,
`arcface_identity.py`; `scripts/prepare_identity_flow.py`,
`identity_face_flow.py`, `run_identity_flow.py`.
Dependency setup: `bash scripts/setup_identity_aux.sh` (uv, pinned ONNX).

## Measured admission

- Trainable BA: **30,831,747 parameters**; frozen BA core: 5,327,360.
- Initial predictions match the parent's best checkpoint **exactly** on four
  actual fit/probe cached inputs. The two new reads initially add exactly zero.
- Differentiable ArcFace versus original ONNX: cosine 0.99999994, relative RMS
  1.19e-6, maximum element difference 3.46e-6. Disable cuDNN TF32 for this path;
  its default produced relative RMS 5.73e-4 and failed the tighter check.
- Real decoded identity loss has finite nonzero BA gradients on three training
  faces, including the largest 704px region. ID gradient norms 4.64/8.10/7.77;
  native flow norms 2.48/2.71/2.60. Frozen VAE has no parameter gradients.
- Identity weight **0.5 every second update**, selected from gradient norms
  with a .5 cap. One identity region per active update; no validation photos.
- Batch64/128/256: 314/506/**732 flow cases/s**, including a largest-region
  identity loss every second update. Batch256: **.350 s/update**, peak reserved
  **11.06 GiB (69.2%)**. This is a short measured benchmark, not a long-run ETA.
- Selected batch256, LR5e-5, warmup100, AdamW weight decay .01. Optimizer starts
  fresh; learned BA weights come from the best parent, not its regressed endpoint.
- Focused query-independence/zero-residual/gradient invariant check passed.

```bash
bash scripts/setup_identity_aux.sh
# Existing training caches/weights and one-ID dataset must be present.
envs/metrics/bin/python -m scripts.prepare_identity_flow align \
  --source runs/flux4b_reference_refiner1024_b128_20261001 --destination runs/NEW_AUX
envs/flux-toolkit/bin/python -m scripts.prepare_identity_flow cache \
  --source runs/flux4b_reference_refiner1024_b128_20261001 --destination runs/NEW_AUX
envs/flux-toolkit/bin/python -m scripts.check_identity_flow \
  --parent runs/flux4b_reference_refiner1024_b128_20261001 --aux runs/NEW_AUX
envs/flux-toolkit/bin/python -m scripts.identity_face_flow init \
  --run runs/NEW_RUN --parent runs/flux4b_reference_refiner1024_b128_20261001 \
  --aux runs/NEW_AUX --batch-size 256 --id-weight .5 --lr .00005 --total 100000
envs/flux-toolkit/bin/python -m scripts.run_identity_flow --run runs/NEW_RUN
```

Admission evidence: `runs/face_identity_aux_20261001/admission.json`.
Raw old-run review: `runs/flux4b_reference_refiner1024_b128_20261001/comparison_summary.json`.

## First live updates (19:49 UTC)

The controller completed 50 updates and started fresh step-zero image validation.
All35 trainable tensors have nonzero updates and finite gradients; all11 frozen
core tensors remain bit-exact. Mean live update time .276s; peak CUDA reserved
11.693GiB (73.1%). The run prepares expected post-resume states for two updates,
including one identity-loss update; fresh-process comparison runs at51/52.
No generated-image improvement is established yet.

Step-zero validation finished at19:57 UTC: **ID_sim .2297139211**, exactly the
parent best500 score. All24 newly generated images are **pixel-identical** to
that parent panel (`initial_parent_image_parity.json`). Native BA-off and the
actual full-backbone cached/live check both pass exactly; inference peak CUDA
reserved9.400GiB. The controller has resumed training toward500.

Fresh-process optimizer resume passed at51/52, including identity backprop at52.
Largest parameter difference from the uninterrupted expected state:3.27e-7,
within recorded atol2e-7/rtol2e-5 (numerical agreement, not bit equality).
Training passed100 updates with finite losses/gradients. The established
30-minute follow-up monitors newly scored panels and the convergence rule.

Review new panels with:
`envs/flux-toolkit/bin/python -m scripts.review_identity_flow --run runs/flux4b_deep_identity1024_20261001`.
This selects the correct experiment metadata and labels for the new model.

## Follow-up at 20:01 UTC

Completed500 updates; all500 loss/gradient records finite. All35 trainable
tensors updated and all11 frozen core tensors remain exact. Verified the500
checkpoint hash/identity and all20 immutable source hashes. Peak reserved
11.744GiB (73.4%); recent100 updates averaged.507s, slower than the initial
short benchmark, with sampled GPU utilization100%. One controller and one
inference child are active. The500-step24-image generation has started;
no trained-panel ID score is available yet. Baseline/best remains.2297139211.

## 20:30 UTC review and deterministic continuation

The500 panel improves ID_sim **.229714 → .251904** (+.02219;20/24 images improve).
CLIP26.1648 →26.2695; no missing, unowned or ambiguous selected faces. All48
exterior images preserve native pixels exactly. Actual paired crops show modest
changes with artifacts remaining; native ID.331399 is still higher. The verified
500 checkpoint remains best, SHA256 `74fb3205f688ac76665f61cb48384901a92b2454e1559d1d566e0522cc0ca5d3`.

Training failed at the resumed502 identity update: max parameter difference from
the stored expected update4.37e-7, exceeding its absolute2e-7 test near zero.
The preceding501 flow-only update was bit-exact. CUDA `grid_sample` backward
has no deterministic implementation in the installed Torch2.13; requesting
determinism reproduces that explicit error. The previous trainer did not enforce
the config's deterministic flag. Original logs, expected states and checkpoints
are preserved; no old source hashes or manifests were rewritten.

Current continuation: `runs/flux4b_deep_identity1024_det_20261001`, new Comet
key `75e5d8961bdf40efab63ffd2770711ca`. It imports unchanged0/500 validation
and training history through500, with explicit provenance. Model weights and
Adam state resume from500; original speculative/logged501 is not checkpointed
and is excluded from the continuation history. Its original record is retained.

`ba_dit/nn/deterministic_identity_loss.py` implements the same border-clamped
bilinear alignment using gathers. Global deterministic algorithms, cuDNN
determinism and fixed cuBLAS workspace are enforced in the new training script.
Focused CPU forward/gradient parity passes. CUDA forward max error versus
grid_sample2.38e-7; repeated input gradients match exactly. Fresh-process actual
501/502 replay passed **exact parameter equality** in two separate processes
at20:39 UTC, with max parameter difference0.0 for both updates. Peak reserved
7.990GiB for these two updates. Architecture, cache,
loss weights, optimizer, reference, prompts, masks and metric definitions stay
the same. New source files and runtime settings are separately hashed.

Continue with `scripts.run_deterministic_identity_flow`; its entry gate requires
the two exact replay checks. Review with `scripts.review_identity_flow` using
the new continuation directory. Use only one GPU process; do not restart the
old failed run. Preserve both parent checkpoints and all recovery evidence.

At20:42 UTC the actual continuation passed600 optimizer steps. Its501/502
checks are still exact in the production training process. Peak reserved
12.424GiB (77.7%), with96% sampled GPU utilization; recent100-update mean
approximately.29s. Losses/gradients remain finite. The new continuation keeps
both the best500 checkpoint and the same stopping rule;2k image validation is
next. The existing follow-up now tracks this directory and Comet key.

```bash
# Recovery was initialized and replayed twice before this launch.
envs/flux-toolkit/bin/python -m scripts.run_deterministic_identity_flow \
  --run runs/flux4b_deep_identity1024_det_20261001
```

Do not launch that command while its recorded controller/child is still alive.

## 21:04 UTC — 2,000-step validation

| Additional BA updates | ID_sim | CLIP |
| ---: | ---: | ---: |
| 0 | .229714 | 26.1648 |
| 500 (best) | **.251904** | **26.2695** |
| 2,000 | .241371 | 26.0794 |

At2k,15/24 images improve over step0, but only9/24 improve over500. Mean ID
change versus500 is−.0105325. All faces are owned/unambiguous, and all72
reviewed images preserve the exterior exactly. All three paired-face sheets
were inspected: texture/structure artifacts remain; the later checkpoint shows
no consistent visual gain. Native ID.331399 remains higher.

All2,000 loss/gradient records are finite, all35 trainable tensors updated,
and all11 frozen core tensors stayed exact. Verified all23 source hashes and
checkpoint2k SHA256 `a9f980e0ab4bbc8e15ee4e05be82139da31bce28b454457d7dd22be1f1857ce6`.
Full-backbone cache/live parity and native BA-off parity are exact. Training
peak reserved12.424GiB; inference9.400GiB. Mean update time501–2000:.460s.
The next process resumed2001/2002 **bit-exactly**, including the ID update.

Flow fit/probe MSE at500→2k: .446398→.357367 / .709667→.692778. Over full76-case
identity cycles, mean auxiliary loss fell .153275 (501–652) to .074797
(1849–2000), while generated-image ID fell. Improving cached objectives does
not establish improving prompted generation. Best500 remains preserved.

The controller has one insufficient-gain check, so training continues toward
5k under the existing patience-two rule. At21:06 UTC it passed2,200 updates.
No architecture or objective was changed during this continuation. Comparison
charts, paired crops and audits are logged in the continuation's Comet run.

## 21:44 UTC — final 5,000-step review

| Additional BA updates | ID_sim | CLIP |
| ---: | ---: | ---: |
| 0 (imported parent best) | .229714 | 26.1648 |
| **500 (preserved best)** | **.251904** | **26.2695** |
| 2,000 | .241371 | 26.0794 |
| 5,000 | .225265 | 26.2110 |

The controller completed generation, decoding and scoring, then stopped after
two gains over the previous best below .003. There is no remaining training or
validation process. This is regression after an early gain, not steadily
improving identity quality. Native remains higher at .331399. At 5k, 9/24 images
improve over best500 and 15/24 worsen; mean change is −.026639. Against step0,
14/24 improve but the mean changes by −.004449. All scored faces are owned and
unambiguous. The ownership-matched ID score above is the unchanged acceptance
metric; the legacy best-detection score at 5k is .227529.

All three 24-case paired-face sheets were inspected. Facial texture and eye/mouth
artifacts persist (especially `oneid_01_s1`, `oneid_05_s1`, `oneid_10_s1`);
the ski-goggle cases lose detail inside the face mask. The final panel does not
show a consistent visual gain over500. All **96** reviewed composites preserve
native exterior pixels exactly; raw decoded images remain available.

Final checks: all 5,000 sequential metric records finite; all35 trainable tensors
updated; all11 frozen core tensors bit-exact. Verified all23 source hashes, the
5k checkpoint manifest and per-image checkpoint provenance. Native BA-off and
full-backbone cache/live prediction checks are exact. Production resume2001/2002
is bit-exact, including the identity-loss update; the two recovery501/502 replays
also passed. Peak reserved CUDA memory is **12.424 GiB /77.7%**, inference9.400
GiB. Mean update time501–5000 is .515s (~497 cached flow cases/s at batch256),
excluding cache creation, probes and image validation.

Flow fit/probe MSE falls from .459813/.710477 to .206468/.675383. Mean auxiliary
identity loss over the first and last complete76-case cycles after500 falls
.153275 → .042233. These improvements do not translate into sustained prompted
image ID gains. Finite cache reuse, crop decoding context and one-step clean
estimation are candidate causes; no single cause has been isolated by this run.

Best checkpoint and Adam state remain in
`runs/flux4b_deep_identity1024_det_20261001/checkpoint-000500/`.
Verified best branch SHA256:
`74fb3205f688ac76665f61cb48384901a92b2454e1559d1d566e0522cc0ca5d3`.
Verified best Adam SHA256:
`9db958195bff1010b2c994c6ad254903937198509da86051b1385aead711a498`.
Final5k branch SHA256:
`de66469daa7e899586cdc83010dbc4651c5f98851186c43a73e84de84a57b7e3`.

`scripts.review_identity_flow` logged the final curves, paired images, source
snapshot and audits to continuation Comet `75e5d8961bdf40efab63ffd2770711ca`.
Additional evidence: `final_training_audit.json`, `final_score_audit.json`,
`comparison_summary.json`, `convergence.json`, `best_checkpoint.json` and
`completed_commands.json`. The report builder is
`reports/261001_deep_identity/build_report.py`. Final export and upload are
verified below.

## Final architecture report and follow-up closure

The **23-page** [architecture PDF](../../reports/261001_deep_identity/flux4b_deep_identity1024_architecture.pdf)
covers the whole model, three BA reads and frozen core, actual code snippets,
CL14/CL39 comparison, decoded identity objective, deterministic recovery, final
ID/CLIP/loss results, mask overlays and all24 paired face cases. All pages were
rendered from the PDF; the contact sheet and detailed architecture/results pages
were inspected. A loss-card overlap and an overflowing reproduction line were
repaired before the final upload. Individual model/BA diagrams are also exported
as PDFs; source hashes and run provenance are in `source_audit.json`.

Uploaded and content-hash verified at **2026-10-01T21:50:17.461300+00:00** to:
`Apps/temp/rsrch_new/2026-10-01/flux4b_deep_identity1024_architecture.pdf`
(helper API path `/rsrch_new/2026-10-01/flux4b_deep_identity1024_architecture.pdf`). Size: **5,059,984 bytes**.
Final PDF SHA256: `0ac45675d1ec943ef2327602028cfcb95c2d29094fb75f9e9e8ae1fd22c79136`.

The PDF, source audit and final scientific checks are logged to the current
Comet experiment. Upload proof: `dropbox_report_upload.json`; visual review:
`final_visual_review.json`; closure: `followup_completion.json` in the run.

Best500 weights and optimizer state are preserved. The scheduled follow-up
`review-local-ba-refiner-convergence-and-report` is now **PAUSED** after verifying
the completed stopping rule, checkpoint and report. No further GPU work is queued.
