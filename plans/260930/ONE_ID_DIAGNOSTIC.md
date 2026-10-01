# One-ID pipeline diagnostic

User requested this sequence on 2026-10-01: finish the current FLUX pilot at 2,000 updates including validation, then train on the original one-person dataset; if validation images do not change, investigate/fix branch wiring and retry on the same rented machine.

## Prepared experiment

- Host: existing Vast instance `53574065`; use the SSH credentials through `scripts/vast_gpu.py` and `SKILLS.MD`.
- Previous run: `/workspace/rsrch_new/runs/flux48_large4096_branch_pilot12`, parent PID `4092`, Comet key `77a3271d63794483a988f654b5e7e6ef`. Its saved config already caps training at 2,000 updates and then runs the final 12-image validation/scoring before exiting.
- Next run: `/workspace/rsrch_new/runs/flux48_one_id_diagnostic`, fresh zero-initialized branches with `configs/flux4b_48_one_id.yaml`, profile `flux48-one-id`. New Comet experiment under `rsrch_new`; read its immutable key from `comet_experiment.json` after launch.
- Training: same FLUX Base 4B, 768 px targets, 512 px references, rank 16, branch scale 0.1, LR 1e-4, microbatch one / accumulation eight, 2,000 updates. Validation at 0, 500, 1,000, 1,500, and 2,000; twelve fixed prompts, reference `51.jpg`, seed zero.
- Source: the original `nm0005092` one-ID metadata, 19 adjusted images, `ref/51.jpg`, and `id_embeds_one_id.pth`, copied from `/home/kolyangg/rsrch/dataset_full/one_id`. `rsrch_apr_test`'s archived configs reference this dataset, although its current checkout no longer contains the files. Prompts come from the imported original `data/validation/prompts_10.txt`.
- Import command: `envs/flux-toolkit/bin/python -m scripts.import_one_id --source /home/kolyangg/rsrch/dataset_full/one_id`. Output is ignored `data/datasets/one_id/`; transfer that directory to the same path on Vast. Audit includes hashes of every copied file.
- Deliberate adaptation: deterministic cyclic pairs of distinct views, original captions, no random flips. Original one-ID validation reference also occurs among the training images; this is intentional for the diagnostic and is not a generalization result. Both manifests explicitly opt into `one_id_diagnostic`, which accepts exactly one shared identity. Normal runs retain their disjointness checks. Train manifest SHA256 `fca34ce7041276fa7c45edc6102f7f2560019884c8cb940f0eb6b75f4e0a3262`; validation SHA256 `e08a8d86cb3264d5b5c837a61e7bea08a5952a18a339b2e285584208d5f8063e`.
- Step-zero inference creates new FLUX masks in a directory keyed by this panel's hash. Historical PhotoMaker output masks are not imported. Masks are used only for scoring. Identity metrics use the original one-ID embedding, with the existing cosine/face-selection definitions.

## Queue and follow-up

Run `envs/flux-toolkit/bin/python -m scripts.queue_one_id --previous-run runs/flux48_large4096_branch_pilot12 --previous-pid 4092` from the remote project root. Keep the launcher detached with stdout/stderr redirected to `/workspace/flux48_one_id_queue.log`.

The queue takes an exclusive lock, records state in `runs/one_id_queue.json`, waits for the original process identity to exit, verifies checkpoint 2,000, all validation PNGs, and both scoring summaries, compares the previous run's step-zero/2,000 outputs, and starts the new experiment. It refuses duplicate run directories and stops with a recorded error if prerequisites fail. It never interrupts the previous training or validation. After one-ID finishes, it compares each validation to step zero. Do not start a duplicate job while the queue or its child is running.

The scheduled follow-up should inspect the queue, the immutable run key, gradients/updated B matrices, checkpoint steps, and completed validation artifacts. At each new completed validation, run `python -m scripts.compare_validation_steps RUN_DIR --later-step STEP`, inspect the per-image differences and paired images, and record metric changes. Evaluate meaningful visual changes as well as exact pixel equality. Stochastic loss on changing images/noise/timesteps is not sufficient by itself to establish a wiring failure.

If outputs are identical or only trivially different after updates, inspect adapter parameter changes, checkpoint loading into the validation model, branch enablement, gates, reference masks/token indices, and whether branch-on versus branch-off changes the prediction for identical cached tensors, noise, and timestep. Use fixed-noise/timestep loss probes to assess learning when stochastic loss is inconclusive. Fix evidence-backed faults, verify the smallest relevant invariant and paired pretrained inference, then launch a fresh named one-ID retry on this host and repeat validation. Preserve failed-run evidence and compare each retry against its own step-zero images. Run one GPU job at a time.

Pause the follow-up after a completed diagnostic has shown finite branch updates and meaningful validation changes, with images and metrics reported; distinguish changed outputs from improved identity/quality. Record blockers rather than starting duplicate or speculative retries. Do not rent another machine or terminate this instance under this task.

## Verification so far

Local import checked all face boxes and target/reference preprocessing geometry. The original embedding contains identity key `51`. Focused split checks passed: unmarked identity overlap, partial opt-in, and multiple identities in diagnostic mode are rejected. Python syntax, shell syntax, and whitespace checks passed. The one-ID pretrained run has not started yet.

Deployed commit `dd14647` and the dataset to Vast. All transferred file and manifest hashes passed verification. Queue PID `5913` is detached, with phase `waiting_for_previous_training_and_validation`; the previous worker was alive at step 1,215. Checked that incomplete final artifacts reject handoff and the one-ID run directory does not yet exist. Active thread follow-up `review-one-id-training-and-repair-branch-wiring` runs every 30 minutes to perform the conditional diagnosis/repair above. The queue itself runs on Vast and does not need an open SSH session.

## 2026-10-01 02:39 UTC handoff

The previous pilot completed 2,000 updates and both final scoring passes, exited, and was replaced by the queued one-ID run (parent PID `6851`, queue PID `5913`, phase `one_id_running`). The new immutable Comet key is `25604bb4dc58429d9378efea58b43ed2`: [one-ID experiment](https://www.comet.com/nikolay-2104/rsrch-new/25604bb4dc58429d9378efea58b43ed2). Its training cache is prepared and step-zero validation is generating; no optimizer update or trained one-ID validation is available yet.

The completed previous pilot changed all 12 images, with per-image mean absolute pixel differences of 13.755–39.100 on the 0–255 scale. A paired contact sheet was inspected: changes include face shape, expression, hair, framing, and background details, and are visibly larger than rounding noise. Identity similarity rose 0.484306→0.535192; legacy best-face identity similarity rose 0.357385→0.399997; CLIP text similarity fell 24.394645→23.951896. TOPIQ-Face rose 0.822842→0.841349, MUSIQ 71.599759→72.311925, MANIQA 0.646104→0.652906, and TOPIQ 0.613007→0.624934. All 12 images retained detected faces and mask ownership. These are results on the named small panel, not generalization conclusions.

Step-zero and step-2,000 resolved validation configs are equal, the final report records `branch_only` and checkpoint `checkpoint-002000` (manifest SHA256 `efd409495254cde1adb34adbfe92bfecb24472ca422d65b91310bdf1f2a92dd3`), and all 16 B matrices are nonzero (combined L2 norm 5.520810). This supports that learned branch weights influence validation. The one-ID diagnostic should still complete as requested. Comparison report and contact sheet are on Vast in the previous run directory (`validation_change_000000_to_002000.json`, `comparison_000000_002000.jpg`) and locally under `runs/review_pilot_2000/`.

## 2026-10-01 03:09 UTC baseline and first updates

One-ID step-zero generation and both scoring passes completed. All 12 output masks are usable; the overlay grid was visually reviewed and selects the intended foreground face, including crowd scenes, without corrections. Baseline identity similarity is 0.310866, CLIP text similarity 28.510215, TOPIQ-Face 0.687094, MUSIQ 67.124743, MANIQA 0.615123, and TOPIQ 0.492962. The review image is stored remotely at `validation-000000/mask_review.jpg` and locally at `runs/review_one_id/baseline_mask_review.jpg`.

Training reached step 117 with all 16 B matrices updated at the first optimizer step; recent losses/gradients are finite, the latest B-gradient norm is 0.005645, and peak reserved CUDA memory is 9.52734 GiB. Mean update time over the last 100 steps is 10.58375 seconds. Queue PID 5913 and training parent PID 6851 remain alive. No trained one-ID validation is available before the first scheduled checkpoint at 500; no intervention was made.

## 2026-10-01 04:39 UTC step-500 comparison

One-ID checkpoint 500 and its complete twelve-image validation/scoring are saved; training resumed and reached step 572. Recent losses/gradients remain finite, latest B-gradient norm is 0.008103, and peak reserved memory is 9.52734 GiB. Queue and training parent are alive.

All 12 images changed versus step zero; per-image mean absolute pixel difference ranges from 10.436 to 47.001 on the 0–255 scale. The paired contact sheet shows visible changes in facial details, pose, framing, clothing, and background, especially prompts 04, 10 and 11. Step-zero and step-500 resolved configs are identical. The validation report records `branch_only` and `checkpoint-000500`, manifest SHA256 `a76ae60ceba10b1d08d07173e1c67da4cc829446b42c97647ef32a276dc4a260`. All 16 B matrices are nonzero; combined L2 norm is 5.788778. This satisfies the early check that trained branches affect validation, so no speculative wiring change or retry was made. Continue the scheduled run through 2,000 and review later validations before closing the diagnostic.

| Metric | Step 0 | Step 500 |
| --- | ---: | ---: |
| Identity similarity, mask matched | 0.310866 | 0.301947 |
| Identity similarity, best face | 0.310866 | 0.305427 |
| CLIP text similarity | 28.510215 | 28.073405 |
| TOPIQ-Face | 0.687094 | 0.682798 |
| MUSIQ | 67.124743 | 68.458800 |
| MANIQA | 0.615123 | 0.614962 |
| TOPIQ | 0.492962 | 0.488238 |

All twelve images retain detected/owned faces. Identity and text similarity decreased slightly, so output changes alone are not evidence of better personalization. Comparison report and contact sheet are stored in the remote one-ID run (`validation_change_000000_to_000500.json`, `comparison_000000_000500.jpg`), with local copies under `runs/review_one_id/`.

## 2026-10-01 06:09 UTC step-1,000 comparison

Checkpoint 1,000 and all validation/scoring completed; training resumed to step 1,028 with finite recent losses/gradients and peak reserved memory 9.52734 GiB. Both queue PID 5913 and training parent PID 6851 remain alive. All twelve images differ visibly from baseline (per-image mean absolute pixel difference 12.958–47.537/255); the contact sheet was reviewed. Changes include face details, action/pose and framing, including newly visible ski poles and changed motorcycle/chef compositions.

Resolved validation configs remain identical to baseline. The report records `branch_only`, `checkpoint-001000`, and checkpoint manifest SHA256 `9bd4455a3b6b835364e01614cad721dee372fb40babff0c483fde44a24de230b`. All sixteen B matrices are nonzero, with combined L2 norm 8.454246; their combined L2 change since checkpoint 500 is 4.409597, confirming continued parameter updates across the save/resume boundary.

| Metric | Step 0 | Step 1,000 |
| --- | ---: | ---: |
| Identity similarity, mask matched | 0.310866 | 0.302849 |
| Identity similarity, best face | 0.310866 | 0.307433 |
| CLIP text similarity | 28.510215 | 28.299676 |
| TOPIQ-Face | 0.687094 | 0.667383 |
| MUSIQ | 67.124743 | 68.069789 |
| MANIQA | 0.615123 | 0.614712 |
| TOPIQ | 0.492962 | 0.483583 |

All twelve images still have detected/owned faces. Identity similarity recovered slightly from checkpoint 500 but remains below baseline; output changes and continued parameter updates demonstrate branch influence without establishing improved identity fidelity. No evidence-backed wiring fix is indicated. Continue through the remaining scheduled validations. Reports `validation_change_000000_to_001000.json` and `comparison_000000_001000.jpg` are stored in the remote run and local `runs/review_one_id/`.

## 2026-10-01 08:09 UTC step-1,500 comparison

Checkpoint 1,500 and all validation/scoring completed; training resumed to step 1,654 with finite recent losses/gradients and peak reserved memory 9.52734 GiB. Queue and training parent remain alive. All twelve images changed versus baseline (per-image mean absolute pixel difference 18.147–42.608/255); the paired contact sheet was visually reviewed, showing changes in faces, poses and composition.

The validation configs match baseline. The report records `branch_only` with `checkpoint-001500`, manifest SHA256 `87b450984641b26c845591d10b3c568438fcf4e211073a45f0c2408f782ef934`. All sixteen B matrices remain nonzero; combined L2 norm is 10.430811, and L2 change since checkpoint 1,000 is 3.877858.

| Metric | Step 0 | Step 1,500 |
| --- | ---: | ---: |
| Identity similarity, mask matched | 0.310866 | 0.338955 |
| Identity similarity, best face | 0.310866 | 0.338955 |
| CLIP text similarity | 28.510215 | 28.673897 |
| TOPIQ-Face | 0.687094 | 0.713561 |
| MUSIQ | 67.124743 | 69.520918 |
| MANIQA | 0.615123 | 0.633927 |
| TOPIQ | 0.492962 | 0.494588 |

All twelve outputs retain detected/owned faces. This checkpoint improves identity similarity and all listed text/quality means on the diagnostic panel, supporting useful learning in addition to visible branch influence. These small-panel observations do not establish generalization. No wiring fix or retry is indicated. Continue to the planned final 2,000-step validation before pausing the follow-up. Reports `validation_change_000000_to_001500.json` and `comparison_000000_001500.jpg` are stored in the remote run and local `runs/review_one_id/`.

## 2026-10-01 08:40 UTC progress

The queue remains `one_id_running` (PID 5913), with parent PID 6851 and one training worker, PID 11431, resuming checkpoint 1,500 toward 2,000. At 08:41 UTC it reached 1,827/2,000 updates (91.35%). The last 100 updates have finite losses and gradient norms; B-gradient norms range from 0.007395 to 0.053983. Peak reserved memory remains 9.52734 GiB. Mean update time is 10.5976 seconds, estimating about 31 minutes to step 2,000 (approximately 09:12 UTC), plus final validation/scoring. No newly completed validation is available after step 1,500, so no duplicate comparison or intervention was performed. Evidence: `runs/review_one_id/status_20261001_0840.json`. The follow-up remains active until final validation is reviewed.

## 2026-10-01 09:12 UTC training complete; final validation running

All 2,000 optimizer updates completed, and checkpoint `checkpoint-002000` was saved in `branch_only` mode. Manifest SHA256: `592182ccf8b961bf015bf5240040e53afd122bb1eb36c5c0a08de7b9ea898f77`. All checkpoint tensors are finite and all sixteen B matrices are nonzero; combined B L2 norm is 11.996713, with L2 change 3.454949 since checkpoint 1,500. The last 100 training losses and gradient norms are finite. Peak reserved training memory is 9.52930 GiB (20.10% of device capacity).

The training worker exited and the parent launched serial final validation from checkpoint 2,000 (inference parent PID 11771, one GPU worker PID 11902). At 09:12:39 UTC one of twelve denoised samples was recorded; decoding and both scoring summaries are pending. Queue PID 5913 remains `one_id_running` while validation finishes. No duplicate GPU job or retry was started. Final comparison and follow-up pause must wait for complete images/scoring. Evidence: `runs/review_one_id/status_20261001_0910.json` and `status_20261001_0912.json`.

## 2026-10-01 09:41 UTC completed diagnostic and final review

The run completed all 2,000 updates, twelve final images and both scoring passes. The queue records `one_id_complete`; the queue, training and scoring processes have exited. At the completion check the GPU was idle (0% utilization, 15 MiB occupied). Instance `53574065` remains running. No additional GPU job was launched.

The comparison script was rerun for step 2,000. All twelve paired images were visually reviewed and show meaningful changes in faces, poses, framing and scene details; per-image mean absolute pixel differences are 23.482–47.143 on the 0–255 scale. Resolved validation configs, sample order, prompts, seeds and geometry match baseline. The final validation records `branch_only` and checkpoint `checkpoint-002000`; its recorded manifest SHA256 matches the file (`592182ccf8b961bf015bf5240040e53afd122bb1eb36c5c0a08de7b9ea898f77`).

All 2,000 logged training losses, gradient norms and B-gradient norms are finite. All final adapter tensors are finite, all sixteen B matrices are nonzero, and the first-update record confirms sixteen updated B matrices. Final combined B L2 norm is 11.996713; L2 parameter change since step 1,500 is 3.454949. Peak reserved CUDA memory was 9.52930 GiB (20.10% of device capacity).

| Metric | Step 0 | Step 1,500 | Step 2,000 |
| --- | ---: | ---: | ---: |
| Identity similarity, mask matched | 0.310866 | 0.338955 | 0.331836 |
| Identity similarity, best face | 0.310866 | 0.338955 | 0.331836 |
| CLIP text similarity | 28.510215 | 28.673897 | 28.009225 |
| TOPIQ-Face | 0.687094 | 0.713561 | 0.672174 |
| MUSIQ | 67.124743 | 69.520918 | 68.754972 |
| MANIQA | 0.615123 | 0.633927 | 0.625326 |
| TOPIQ | 0.492962 | 0.494588 | 0.487089 |

All twelve outputs retain detected/owned faces without missing masks or ambiguous ownership. Final identity similarity improves over baseline, while CLIP and TOPIQ-Face decline. Step 1,500 has stronger means than step 2,000 on every listed metric. These results establish that trained branches affect validation; they do not establish monotonic improvement or generalization. No inactive-branch or checkpoint-loading fault is indicated, so no wiring repair or retry was made.

The final audit and comparison JSON are saved both remotely in the run and locally under `runs/review_one_id/`. Local artifacts are `final_review.html`, `final_metric_curves.png` and `comparison_000000_002000.jpg`. The complete step-2,000 checkpoint and final validation images/metadata were copied to local `runs/flux48_one_id_diagnostic/`; the requested complete step-1,500 checkpoint is also preserved there. The earlier architecture report retains its dated step-1,500 evidence snapshot.

The scheduled follow-up `review-one-id-training-and-repair-branch-wiring` was paused after this successful completed review. The automation tool returned `PAUSED`, and its saved configuration was checked. The Vast instance was not stopped or terminated.
