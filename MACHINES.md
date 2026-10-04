# Research machines

Updated: 2026-10-03. Credentials stay in ignored local configuration.

| Machine | Role / authorization | Last known state |
|---|---|---|
| **Vast 53994096** | Current user-confirmed target for FLUX4B multi-ID setup, dataset transfer and training | SSH verified2026-10-03. GB10 compute12.1, ARM64,121GiB unified RAM,212GB filesystem. Driver580.178.04/CUDA13.0. CUDA matmul passed; no pre-existing GPU job. |
| Vast 53574065 | Historical 48GB one-ID/pilot machine; superseded | Not present in the account as of 2026-10-03. Do not use for current deployment. |
| Local workstation | Source, credentials, saved checkpoints and deployment tools | 16GB GPU; this deployment is remote. |

## Access and intended run

Use the credential wrapper in `SKILLS.MD`:

```bash
python3 scripts/vast_gpu.py status 53994096
python3 scripts/vast_gpu.py connect 53994096 --check
python3 scripts/vast_gpu.py connect 53994096
```

The current request authorizes environment setup, dataset transfer/download and
`runs/flux4b_large_qkvo_r128` on **53994096**. It does not authorize renting,
resizing or terminating instances. The launcher is
`scripts/run_flux4b_multi_id.sh`; the intended extracted image directory is
`/workspace/datasets/large_dataset`. The user explicitly rejected full training-cache preparation. Training now
encodes text/images on demand; only fixed96 validation inputs are cached.
Check actual filesystem capacity before transfer. Record measured compatibility and any
necessary deployment changes in `plans/260930/IMPLEMENTATION_NOTES.md`.

## Current Cosmic switch (3 October)

The user switched from Large to Cosmic and requested fast hardware confirmation
without long validation. `rsrch_native` and `rsrch_training` are STOPPED;
the local Large transfer/watcher were stopped too. Preserve their partial files.
Environment and locked weights are ready.

Both Cosmic Google Drive downloads currently return quota/access error pages.
`rsrch_cosmic_download` and `rsrch_cosmic_refs` exited unsuccessfully; no complete
archives were downloaded. The target link was verified as quota-exceeded.
Do not mistake a successful HTTP200 HTML response for dataset bytes.

Supervisor `rsrch_cosmic_smoke` runs the separate
`runs/flux4b_cosmic_hardware15_qkvo_r128` job using 15 complete original pairs
already available locally. This is a 100-update hardware probe, not full Cosmic
training. Entry: `scripts.run_cosmic_smoke`; config:
`configs/flux4b_cosmic_hardware15.yaml`. It uses online conditioning, rank128,
768px targets, microbatch1/accum8, short native/off and gradient checks, and a
real checkpoint/resume after2 updates. Image panels/scoring are explicitly
omitted for this named smoke at the user's request. Full-data training remains
blocked on the reference archive (and target download or local transfer).

Log: `/workspace/rsrch_new/runs/deploy_53994096_cosmic/training.log`.
Check the run's `status.json`, `metrics.jsonl`, checkpoint and actual process
before acting. All GPU children inherit `runs/face_flow_gpu.lock`.

Latest user steering authorizes whichever dataset transfers fastest. Resumed
the existing Large rsync with8 streams because14.27GB was already present.
Local PID/log: `scratch/deploy_53994096_large_resume.{pid,log}`. This is data
transfer only: the old completion watcher and long native/training queue remain
stopped. Cosmic hardware training continues independently on the GPU.

## 9B continuation (3 October; access granted)

The 4B Cosmic hardware probe completed100 updates successfully. The current
authorized continuation is fresh FLUX.2-klein-base-9B BA-only training on
full transferred Large, with a named fixed12 held-out pilot. Gated access
verified and pinned9B/text weights downloaded. Code checkout:
`/workspace/rsrch_9b`; main shared assets remain `/workspace/rsrch_new`.
Both use `/workspace/rsrch_new/runs/face_flow_gpu.lock`. Supervisor
`rsrch_9b_benchmark` tests microbatches1/2/4 and checkpointing before the
full-data pilot. The old native96/training queue remains stopped.
No new machine or rental change is authorized. Latest measurements/status
are in IMPLEMENTATION_NOTES.md and remote benchmark result JSON files.

Full Large transfer/import now complete:47,500 files,17.117GB;47,341 eligible
pairs after exclusions. Benchmark completed: selected microbatch1/accum8,
checkpointingOFF,40.97sec/update and51.79GiB peak reserved.
Supervisor **rsrch_9b_pilot** now runs startup for
`runs/flux9b_large_qkvo_r128_pilot12` in `/workspace/rsrch_9b` (12:05UTC).
Log:`/workspace/rsrch_9b/runs/pilot_controller.log`; startup status under
`runs/flux9b_large_qkvo_r128_pilot12_setup/status.json`. The target is2000
updates with native-generated9B masks and fixed12 checks at0/1000/2000.

Latest user override: **start training immediately**. Stopped remaining
startup replay checks and switched rsrch_9b_pilot to
`scripts.run_flux9b_training_first`. Main training workerPID30097, Comet
`7b7169cd61b64a2cace2fe1e52861afc`. Run remains
`/workspace/rsrch_9b/runs/flux9b_large_qkvo_r128_pilot12`; status/logs now
live inside that run. Train to1000 first; preserve checkpoint0 and evaluate
0/1000 afterward, then train/evaluate2000. Startup replay was incomplete;
pretrained parity/gradients/frozen checks passed. Old setup evidence remains.

At12:29UTC the main run completed updates1/2, ~41sec/update,51.63GiB peak
reserved and96% sampled GPU utilization. Checkpoint0 complete. Training
continues unattended; active monitoring stopped after two verified updates.

Five-hour budget supersedes the batch8 pilot. Old run stopped after9 logged
updates; checkpoint0/source/metrics preserved. Supervisor rsrch_9b_pilot now
runs `runs/flux9b_large_qkvo_r128_fast5h_b1` with
`configs/flux9b_gb10_large_qkvo_r128_fast5h.yaml --profile-validation`.
Same9B/768/r128, effectivebatch1; full Large pool,2000 examples total.
Runtime estimate including validation is saved in `five_hour_budget.json`.

Fast pilot confirmed training and inference timing at12:41UTC:5.831sec/update
including first-update warmup; projected4.47h for2000 updates plus all panels
and30min overhead reserve. Comet:cfbd6f879e3d4158959bbbef7fb8e894.

### 3 October — native fixed12 baseline, then 8k continuation

The batch1 fast pilot completed2000 and all scoring. Checkpoint002000
(including Adam/RNG state and its frozen source snapshot) is downloaded
locally; all five checkpoint files match remoteSHA256. The user requested a
fresh native-only12-image baseline before continuing. Supervisor
`rsrch_9b_baseline_then_8k` in `/workspace/rsrch_9b_8k` runs this serial queue:

1. `runs/flux9b_native_fixed12_20261003`: native9B, no BA/checkpoint; unchanged
   held-out12 prompts/references/seeds,768px,20steps/CFG4; separate Comet
   `1acaa7707e184bad9176008e78f97ea6`; original ID_sim/CLIP scoring.
2. `runs/flux9b_large_qkvo_r128_to8k`: resume full checkpoint002000 to8000,
   unchanged effectivebatch1/r128/optimizer/data; validation4000/6000/8000.

The shared GPU lock is unchanged. Parent checkout/run stay intact. Only the
stop step and validation cadence are allowed to change for exact state
restoration. Parent source/config/model/data hashes passed a fresh CPU check.
At16:56UTC credit was$3.5682 and hourly total$0.49829; projected continuation
9.5–10h costs~$4.7–5.0, plus fresh baseline. User authorized starting regardless
of balance and may top up. No rental lifecycle changes.

User then added a full96 baseline before training. The active supervised
queue now runs native12, then native96, then the8k continuation. Full96 uses
`data/validation/manual_val_96.jsonl` unchanged, limit96,768px/20steps/CFG4,
no BA or adapter checkpoint; Comet `6abec2cdaa104924a563e8b53f5a622c`.
Output:`/workspace/rsrch_9b_8k/runs/flux9b_native_fixed96_20261003`;
Comet record in sibling `_comet` directory. Failures stop the queue before
training (`set -euo pipefail`); both baselines include original ID/CLIP scoring.
Training has not yet resumed while the baseline queue is active.

### 3 October — authorized automatic finish and STOP

Current run is training toward8000, fixed12 validation4000 completed. User
now explicitly authorized checkpoint8000 download, trained full96 validation,
consolidation into original Comet cfbd6f879e3d4158959bbbef7fb8e894, and then
**STOP53994096**. Never terminate. Full procedure/recovery details are in
`plans/260930/FLUX9B_8K_COMPLETION.md`.

Remote supervisor `rsrch_9b_final96` is waiting for parent completion and a
verified local-checkpoint acknowledgement. Local enabled systemd user service
`rsrch-flux9b-finish` downloads/verifies, consolidates Comet, checks GPU idle,
and stops only after all receipts succeed. Check-in heartbeat first due around
05:19BST4October; automatic stop cannot occur before06:00BST. Measured ETA:
8k plus fixed12~05:04BST; final96/transfers/Comet~06:54BST4October. These are
estimates; completion gates determine the actual stop. Local machine must
remain powered/online. At22:12UTC credit$5.94 covers estimated~$3.75–4 remaining.

### 4 October — FLUX1 naming and completed Vast shutdown

User named the online masked Q/K/V/output BA setup **FLUX1**. Current fresh-run
configs and README distinguish Vast9B from HSE-cluster4B; exact frozen deployed
sources are archived on `FLUX1-vast-runtime` and `FLUX1-cluster-runtime`.
See `docs/FLUX1_DEPLOYMENTS.md` for commits, hashes and old/new config names.
Vast53994096 is now **stopped** (`actual_status=exited`,08:38:26UTC4October),
after full8000 checkpoint/result downloads, trained96 validation and verified
consolidation into original Comet. It has not been terminated; storage persists.
