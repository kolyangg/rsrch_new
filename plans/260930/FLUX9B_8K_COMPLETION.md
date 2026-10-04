# Automatic 8k completion and Vast stop

## Authorization and scope

On 2026-10-03 the user requested: finish the current run to8000 and its
validations, download checkpoint8000 locally, validate that checkpoint on the
original full96 panel, consolidate continuation results into the previous2k
Comet run, and **stop** Vast53994096 after successful completion. Also schedule
an agent check-in before stopping. This does not authorize termination or any
new rental. No further approval or agent turn is required by these scripts.

## Current ETA (measured 22:22UTC, 3 October)

- Current step4359; latest100 updates average5.11446seconds.
- Actual fixed12 validation at4000 took780.90seconds (inference/decode/scoring).
- Remaining updates plus fixed12 panels6000/8000 and5min loading reserve:
  **04:04UTC / 05:04BST on4October** (~5h41m remaining).
- Add approximately110min for full96, checkpoint/result transfer and Comet:
  **05:54UTC / 06:54BST** (~7h32m remaining). Allow roughly30min variation.
- Credit checked at22:12UTC:$5.9364; totalhourly$0.49829. Estimated remaining
  charge~$3.75–4.0, assuming measured throughput. Stop retains storage, so
  storage charges continue afterward.
- Extra heartbeat `check-8k-completion-before-vast-stop` scheduled every6hours;
  first check approximately04:19UTC /05:19BST. Pause it after confirmed stop.
- Absolute automatic-stop floor:05:00UTC /06:00BST, providing a window for that
  check-in. Completion gates still apply if work finishes later. The check-in
  is an audit, not a required manual release.

## Already installed and running

### Existing training (unchanged)

Remote `/workspace/rsrch_9b_8k`, supervisor `rsrch_9b_baseline_then_8k`, run
`runs/flux9b_large_qkvo_r128_to8k`. Comet source
`c2c484aa06f1488b92ce1d9795857b90`. Full optimizer/RNG/data cursor continuation
from2000; batch1, rank128,768px; fixed12 panels4000/6000/8000.

### Remote final96 worker

Supervisor **`rsrch_9b_final96`** runs `scripts.finalize_flux9b_remote`.
Run: `runs/flux9b_8000_fixed96_20261004`.

1. Wait for parent status completed8000 **and** last_completed_validation8000.
2. Wait for `local_checkpoint_verified.json`, written by the local controller.
3. Compare acknowledgement SHA256 against every remote checkpoint file.
4. Acquire the existing shared GPU lock; infer with trained8000 BA through
   `scripts.online_face_ba`, decode with exact native-background composition,
   then score original ID_sim/CLIP. No target photos are inference inputs.
5. Save completion and artifact-hash manifest to `ready.json`.

The full96 bundle was prepared and verified during live training using CPU
only. It retains original96 order/prompts/seeds/references and frozen masks
from `runs/flux9b_native_fixed96_20261003`. All96 masks are present. Config,
mask/native-image hashes and frozen runtime sources are recorded and checked.
The future checkpoint symlink points to the actual checkpoint008000; no
weights are fabricated or substituted. Separate stage receipts permit resume.

### Local completion controller

Systemd user service **`rsrch-flux9b-finish.service`**, enabled with user linger,
runs `scripts.finalize_flux9b_local` from this project. It is independent of
Codex remaining active. The **local computer/WSL must stay running and online**
for downloads and the final stop request.

State directory: `runs/flux9b_8k_completion/` (ignored).

1. Wait for completed8000 + scored fixed12; rsync full checkpoint to
   `runs/flux9b_large_qkvo_r128_to8k/checkpoint-008000`. Verify every SHA256,
   including adapters, optimizer/RNG state and manifests. Send acknowledgement.
2. Wait for remote full96 `ready.json`; download final96 and continuation
   results/source snapshots, excluding historical checkpoint directories.
   Verify all saved result hashes and exact exterior-preservation audit.
3. Consolidate metrics/images into original Comet
   **`cfbd6f879e3d4158959bbbef7fb8e894`**, renamed
   `flux9b_large_qkvo_r128_b1_0_to_8000`. Preserve original0–2000 data; import
   every update2001–8000, fixed12 panels4000/6000/8000 and paired-face charts.
   Full96 metrics use **`validation96/`** and images **`fixed96_8k/`**, so
   different panel sizes never share the same score curve. Native96 reference
   metrics use `validation96/native_*`. Source run stays preserved as provenance.
4. Verify desired metrics and image assets via Comet API read-back. Existing
   equal points/images are skipped on retries; conflicting metrics stop progress.
5. Require checkpoint/result/Comet receipts and configured not-before time.
   Verify shared GPU lock is free and no CUDA compute process remains.
   Use the local credential wrapper to request **stop53994096**, then poll
   actual instance state. Never terminate; no Vast key goes to the rented host.

Completion receipts: `checkpoint_verified.json`, `results_verified.json`,
`comet_verified.json`, `stop_requested.json`, `stopped.json`. Missing/failed
checks leave the machine running; local transport failures retry once/minute.
Historical errors are in `last_error.json`; compare timestamps with status.

## Verification performed before leaving it unattended

- Live parent process and contiguous continuation steps2001–4359 checked.
- Remote preparation verified full96 manifest, all masks, native hashes and
  immutable runtime; it is waiting and holds no GPU lock while training runs.
- Syntax checks on both new scripts; systemd unit verification; live local
  read-only probe succeeds. Stop gate rejects missing receipts, wrong instance,
  mismatched checkpoint hashes, missing authorization, and12-image-only results.
- Comet API has the original2000 loss points; metrics endpoint supports full
  histories needed for idempotent consolidation. Actual8k/full96 upload and
  download checks necessarily run after those artifacts exist.
- Local disk has~6.3GB free; final bundle currently151MB before trained outputs,
  full checkpoint~385MiB. Failed transfers cannot release the stop gate.

## Inspection / recovery

```bash
systemctl --user status rsrch-flux9b-finish
cat runs/flux9b_8k_completion/status.json
cat runs/flux9b_8k_completion/last_error.json
```

Use `scratch/deploy_53994096.py` for remote supervisor/status/receipts. Preserve
all checkpoints and completed receipts. Do not modify files listed in the
full96 `identity.json` source manifest after initialization. Do not start
another GPU job while either controller holds the shared lock.

To cancel the future automatic stop if the user changes their mind: stop the
local systemd service before it reaches its stop stage. The policy is loaded
at service startup, so a policy edit alone is not a live cancellation.

## 4 October 04:22UTC check-in

Training and fixed12 scoring completed at04:02UTC (05:02BST). Local checkpoint
verification receipt was written04:03:31UTC; all five checkpoint files were
independently rehashed again at this check-in. Saved Adam states all report
step8000/cursor8000 with finite moments, and all64 adapter tensors are finite
and differ from checkpoint2000. Torch/CUDA/Python RNG states are present.

Fixed12 ID_sim:2000=.43407627;4000=.42768707;6000=.41867339;
8000=.41646561. Thus2000 remains best; longer training did not improve this
panel. Step8000 CLIP28.07153845. Comet continuation API confirms all three
scored panels and14 images at8000 (12 samples plus2 paired-face sheets).
Original2k canonical run intentionally still ends2000 until consolidation.

Full96 generation is healthy at20/96,935seconds, on the sole GPU worker.
Frozen runtime hashes, all96 native-image/mask associations and loaded8000
adapter hash passed inspection. Estimated generation completion05:21UTC;
allow10–20minutes for decode/scoring/download/Comet, so automatic stop remains
approximately06:30–06:45BST, subject to actual completion checks.

Local controller and remote final96 supervisor are active. Intermittent SSH errors recovered earlier and recurred once at04:22UTC
after the CPU service restart; its bounded automatic retries remain active.
No failed completion gate was bypassed. Corrected the local status label from waiting_for_training to
waiting_for_full96 after parent completion; restarted only the CPU controller.
Vast instance is stillrunning, credit$2.87577 at$0.49829/hour; local disk5.8GB
free and remote129GB free. No stop request has been sent. Leave follow-up
active until final artifacts, Comet read-back and actual stopped state exist.

At04:23:26UTC the next automatic probe succeeded; local status now correctly
reads waiting_for_full96. The04:22 SSH failure required no manual retry.

## Final completion, 4 October 08:38UTC

Full96 results and checkpoint were downloaded and hash-verified. Consolidation
uploaded all expected metrics/images, but its read-back check originally compared
unsuffixed image names against Comet's filename form (`name.png`, then duplicate
suffixes). It retried and uploaded duplicate images. Fixed only the workstation
verification/presence check to normalize Comet PNG suffixes and disable API cache;
no training or validation data changed. Existing duplicate assets were retained.
The corrected check verified90048 metric points and138 intended image/step pairs.

Vast53994096 was then stopped by the existing authorized controller. Actual API
state `exited` confirmed at2026-10-04T08:38:26UTC (09:38BST). It was not terminated.
The extra check-in automation is paused. All receipts remain locally under
`runs/flux9b_8k_completion`; disk storage on Vast is retained and billed separately.
