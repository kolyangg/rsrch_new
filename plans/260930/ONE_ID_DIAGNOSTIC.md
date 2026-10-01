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
