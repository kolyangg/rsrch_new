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

### 4 October — user restarted GB10 for saved2k full96 validation

Vast53994096 is running again; SSH and the idle GPU were verified before launch.
Supervisor `rsrch_9b_2k_fixed96` in `/workspace/rsrch_9b_8k` runs the saved
checkpoint002000 on the exact completed8k fixed96 native bundle. Output is
`runs/FLUX1_vast9B_2000_fixed96_20261004`; the first4/96 latents completed
in175 seconds. Same768px/20steps/CFG4, batch2, prompts/order/references/seeds,
frozen masks, patched runtime and original metric definitions.

The controller automatically decodes, scores and verifies96 image uploads plus
metrics at step2000 in full96 Comet `2b3eda1b1f364ecfaa584ffe273102a1`.
Local one-off service `rsrch-flux1-2k-collect` downloads verified results and
mirrors them into original Comet `cfbd6f879e3d4158959bbbef7fb8e894` using
`validation96/*`, preserving pilot12 and8k metrics. The previous8k completion
service is inactive. This new job includes no training or automatic machine
stop. See `scripts/validate_flux1_2k_fixed96.py` and
`scripts/collect_flux1_2k_fixed96.py` for execution and receipts.

### 4 October — HSE cluster allocation repair

TaskMaster cancelled4374072 for unused allocated resources. The old setup had
one training worker and one frozen-encoder GPU. Its resumed allocation4374962
was stopped safely at checkpoint1000; historical runtime/run files remain at
`/home/nasilaev/rsrch_new`.

The isolated continuation is `/home/nasilaev/rsrch_new_staged`, run
`FLUX1_cluster_4b_ddp_20261004`. It uses two V100 DDP training workers (effective
batch2), one GPU for frozen-conditioning windows/serial validation, and zero
GPUs for hashing/scoring. Each stage has its own dependent Slurm allocation.
CPU preparation4375079 precedes real pretrained DDP admission4375081; production
training4375083 is gated on exact resume, gradients and per-device memory checks.
This record does not yet claim DDP admission success. See
`docs/FLUX1_CLUSTER_DDP.md` and the latest implementation notes for measurements.
Comet keeps key5d31de48010446248639e65a65236cbe; workstation service
`rsrch-clust-ddp-comet-4375151` publishes actual pipeline status and artifacts.

Cluster repair update10:42UTC: CPU verification4375079 and one-V100 exact
fixed96 conditioning check4375080 passed. Two-V100 admission4375081 remains
pending scheduler priority (estimated12:47London). DDP training is not yet
verified; downstream training stays gated on the real resume/memory checks.
Comet read-back confirms history through1000 and109 inherited images.

### 4 October — FLUX1a launch authorized

The user explicitly requested starting FLUX1a on the existing GB10 after the
2k validation. The old validation finished all96 images and scoring; its Comet
filename verification was reconciled without changing inference/scoring sources.
All96 images and metrics are verified in the dedicated and original training
Comet runs, and523 downloaded result files match their remote hashes.

New isolated checkout: `/workspace/rsrch_FLUX1abc`.
Supervisor: **rsrch_flux1a**; controller `scripts.launch_flux1a_gb10`.
Run: `runs/FLUX1a_vast9B_20261004`; Comet
**2018ec7a730243bc98d922178e58aa5c**, project `rsrch_new`.
It shares the existing GPU lock, verified weights and conditioning cache, but
has a separate pinned/patched Toolkit checkout and Python environment overlay.
The historical `/workspace/rsrch_9b_8k` runtime is preserved.

At11:26UTC the supervised launch was preparing identity labels (792/4,000
receipts). It then runs pretrained admission, full96 at step0, and training to
4,000 with full96 at2,000/4,000. Preparation covers the exact scheduled sample
IDs; the full47,341-pair training pool and seeded order remain unchanged.
This is startup progress, **not verified optimizer updates or GPU admission**.
Source hashes for218 deployed files were verified. The prior automatic Vast
stop service is inactive; no machine stop or rental change is part of this run.

Cluster repair update2026-10-04: admission4375081 failed in torchrun argument
parsing before either model worker started. The repaired executor has a tested
`--` separator, with an audited frozen-source update. First-window conditioning
4375267 completed successfully in26m48s. Job4375268 now combines the original
pretrained DDP admission and the first production segment through2000; it is
pending scheduler priority. The replacement chain ends at4375337. The publisher
service is now `rsrch-clust-ddp-comet-4375337`; the existing Comet key is unchanged
and its display label is `FLUX1_cluster_4b_2V100_DDP`. DDP admission/production
measurements are still pending. This supersedes the preceding current-job IDs.

FLUX1a GB10 health check12:00UTC: `rsrch_flux1a` remained healthy and advanced
through completed pretrained admission to full96 step0 inference (2/96 latents,
96% GPU utilization). Peak admission reserved60.67GiB/49.88%; exact resume and
all64 BA parameter updates passed. Production optimizer step is still0 pending
initial validation. Separate supervisor `rsrch_flux1a_progress` now publishes
stage/count/heartbeat every60seconds to the same FLUX1a Comet key. This fixes
stale preparation status without restarting training or changing frozen sources.

Cluster verified12:02UTC:4375268 is RUNNING oncn-004 with two V100s. Exact
pretrained DDP save/resume admission passed; production reached1041 with
finite loss, ~3.04s/update and17.34GiB peak reserved per-device maximum.
Measured GPU utilization89.6%/86.8% over60s. Comet continuation loss readback
confirmed1028; key5d31de48010446248639e65a65236cbe remains live. This supersedes
the preceding pending/admission-unverified status.

FLUX1a scheduling override: the user's subsequent request to start training ASAP
supersedes waiting for initial full96. `rsrch_flux1a` now invokes
`scripts.resume_flux1a_train_first`, training checkpoint0→2000 first, then
returning to the existing controller for deferred step0 and step2000 panels and
continuation to4000. Completed admission and partial step0 outputs are retained.

At12:06UTC production training was verified advancing through steps1–4 in the
main run (not admission). Step4 included the active identity auxiliary with
finite loss1.078796 and gradient norm0.103000; peak reserved59.416GiB/48.85%.
Observed updates took5.3–7.5seconds. Supervisor8576/trainer8593 remain running;
Comet keeps2018ec7a730243bc98d922178e58aa5c. The training-start evidence is
runs/FLUX1a_gb10_healthcheck_20261004/production_start.json locally.

### 4 October — FLUX1a cluster4B launch

The user requested an additional FLUX1a run on two V100s for20k updates, with
fixed96 validation every2k and clean console/Comet logging. Isolated checkout
`/home/nasilaev/rsrch_new_FLUX1a`, fresh run `FLUX1a_cluster_4b_20261004`,
Comet9a4c6ef2e160413dad596a3dea2dc451 (`FLUX1a_cluster_4b_2V100`).
CPU label preparation4375612 is running; plan4375658 depends on it. The finite
79-stage schedule submits one successor after success to fit the100-job QoS
limit; saved pipeline JSON includes the full remaining plan. GPU admission and
production updates remain pending. Existing FLUX1 continues independently.
Publisher: `rsrch-clust-FLUX1a-comet-4375612.service`. Recipe, commands and
limitations: `docs/FLUX1a_CLUSTER.md`. No new rental or Git push was performed.

FLUX1a GB10 at16:57UTC: checkpoint2000 saved and downloaded locally with all
five SHA256 hashes verified. Main supervisor is healthy, finishing deferred
step0 inference84/96 before2k validation and2000→4000 continuation. Progress
publisher now logs console stages and verifies/repairs missing Comet uploads
for2k/4k. Follow-up heartbeat verify-flux1a-validation-and-finish-4k runs every
30minutes until full4000 training and both fixed96 publications are verified.

FLUX1a live-logging update17:25UTC: main supervisor now invokes
scripts.flux1a_live, reusing10 existing2k latents; the progress supervisor owns a
continuous Comet SDK session and streams stage logs every5seconds. API confirmed
running=true/hasCrashed=false. Generation now decodes and uploads each pair before
advancing, using the original VAE/composition; no frozen model or training
sources were changed. See live_streaming_deployment_* receipts in the setup.


FLUX1a GB10 validation speed update17:54UTC: user-authorized real GPU probes
found batches6/12 slower per image than2, despite memory headroom. Retained2;
scripts.flux1a_live now installs scripts.flux1a_validation_speed's bounded exact
reference-bank cache across CFG and same-reference prompt batches. Real cached
predictions were bitwise equal; predicted denoising throughput1.228x (not yet
full-panel wall time). Checkpoint2000 validation resumed with28 images retained.
Only operational wrapper files were deployed/snapshotted; pinned scientific
sources and training remain unchanged. No new rental or machine stop occurred.

FLUX1a GB10 milestone19:05UTC: fixed96 checkpoint2000 validation and all Comet
images/metrics are verified; receipt and summary copied locally. Production
training resumed successfully beyond2000 (observed2006), with finite gradients
and59.785GiB/49.16% peak reserved. Controller continues to4000, then full96
validation; both supervisors and Comet remain active.


FLUX1a Vast9B interruption21:01UTC: instance53994096 reports actual_status=exited,
intended_status=stopped; SSH refuses connections. Comet is no longer running;
last logged update2980 at20:44:17UTC. The old8k stop service is inactive and its
last stop receipt is from this morning; no evidence it caused this interruption.
The stop's initiator is unknown. Asked the user whether to restart the existing
instance because its explicit stopped state may be intentional; do not override
that state until clarified. Fixed96 step2000 and local2k checkpoint remain
verified; no4k checkpoint/publication is verified. Remote checkpoint inventory
cannot currently be inspected. No machine operation was issued by this check.

The user confirmed the interruption was exhausted Vast credit. The saved2k
resolved config uses checkpoint_every=500 (validation remains every2000), so
checkpoint-002500 is expected to be the latest remote save after logged step2980.
This has not been directly inspected on the stopped host; only checkpoint2000
is downloaded and SHA256-verified locally. Inspect latest_checkpoint.txt and
complete checkpoint manifests on restoration before choosing a resume step.


FLUX1a restoration22:02UTC: existing Vast53994096 became running/accessibile
again; both experiment supervisors were stopped and no GPU compute process was
present. Verified checkpoint002500 has all five nonempty files, SHA256 hashes,
and matching config/data/training-source/parameter manifests against002000.
The last local remote-metrics row was2981 (Comet had2980); updates2501–2981 were
not checkpointed and must be replayed. The original controller already selects
the latest complete checkpoint and archives rolled-back local metric rows.
Started rsrch_flux1a and rsrch_flux1a_progress using their unchanged commands;
no source changes, new rental or machine lifecycle command. Confirmation of new
optimizer updates remains pending initialization. Remote checkpoint verification
receipt is local runs/FLUX1a_gb10_healthcheck_20261004/checkpoint_2500_remote_verified.json.

At22:06UTC resumed production updates2501/2502 were verified with finite loss
and gradients. Controller1407/trainer1550 and progress publisher1428 are active;
Comet Running=true. The original checkpoint restore completed successfully.
This supersedes the credit/stopped blocker and pending initialization status.


### FLUX1a6k continuation and automatic local checkpoint transfer —2026-10-04

The user requested4k and6k checkpoints locally after reviewing the remaining
credit budget. Main rsrch_flux1a continues uninterrupted through4k/full96.
New rsrch_flux1a_6k is a CPU-only waiting supervisor: after4k validation and main
exit, it acquires the shared GPU lock, resumes the latest complete checkpoint,
trains to6000 and runs full96 inference/decode/score/summarize. The separate
continuation config changes only training.steps; strict verify_extension passed
against checkpoint2500 and frozen source guards passed. Original root config,
model/training source hashes and checkpoint contents remain unchanged. Worker
config for6000 is explicit in resolved_config_6000.yaml when continuation starts;
continuation_6000.json records authorization and operational source hashes.

The existing live wrapper supports6000; Comet publisher follows both supervisors
and verifies2000/4000/6000. It remains the sole lifecycle owner and preserves
immediate per-batch images. Operational files were archived before/after under
setup/continuation6k_*; the main GPU trainer was not restarted. Production had
advanced to2635 while the queue and publisher were confirmed active.

Local user service rsrch-flux1a-checkpoint-download runs
scripts.mirror_flux1a_checkpoints every60s until both filesets are verified. It
starts downloads as soon as each complete atomic4k/6k checkpoint appears, before
validation finishes. Destination:
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004
(Windows C:\Users\ogure\FLUX1a_checkpoints\FLUX1a_vast9B_20261004).
This drive has~100GiB free; Linux has only~1.5GiB. Each checkpoint is~385MiB.
All five files are SHA256-verified against remote hashes, staged in a .partial
folder and renamed only after success. Receipts are mirrored beside checkpoints
and in runs/FLUX1a_checkpoint_transfers. Focused checks passed successful copy,
receipt-write restart recovery, corruption rejection and incomplete checkpoint
waiting. Live one-shot and daemon checks correctly show4000/6000 pending.

The heartbeat was updated to finish6000, verify4k/6k Comet publications and both
local download receipts. Latest budget forecast was~9.3 funded hours, versus
~8.9hours to6000 including both validations; credit margin remains small. No
machine stop/termination or extra rental was requested or performed.


FLUX1a Vast9B milestone2026-10-05 01:02UTC: training reached4000 and the complete
checkpoint was automatically downloaded at00:40:48UTC to
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-004000.
Recomputed all five local SHA256 hashes against the remote download receipt;
all match and manifest.step=4000. Receipt:
runs/FLUX1a_checkpoint_transfers/checkpoint_004000_download_verified.json.
Full96 step4000 validation is advancing (24generated,25Comet images confirmed
by the subsequent API read); Comet Running=true/hasCrashed=false. Full scoring
and publication verification are pending. The6k controller is still waiting for
completed4k validation; downloader remains active for6000. No repair needed.


FLUX1a Vast9B2026-10-05 02:32UTC: complete4k fixed96 scoring/publication verified,
receipt and quality_summary copied locally. Owner-ID0.2943213313592423,
CLIP29.38190931081772; all96 Comet images, metrics, provenance and backgrounds
passed. Main4k controller exited normally around02:09UTC.

The6k continuation failed before update4001 at02:13UTC: OnlineIdentityObjective
correctly rejected the4000-row label cache for the extended6000-row trajectory.
The prior continuation check covered config/source compatibility but missed this
derived-cache coverage requirement. Original4k labels/scientific sources and
all checkpoints remain unchanged. Repair adds a separate FLUX1_768_to6000 cache,
seeds all4000 original per-target receipts and parity fixture unchanged, and
runs the original frozen preparation code only for missing scheduled targets.
Preparation is serially supervised as prepare_identity_6000 before6k training;
Comet progress was restarted and logs this stage. Final guard requires full6000
coverage plus exact metadata/embedding equality for all original rows.

Runtime continuation now records the extra data.identity_supervision path in
its resolved config/checkpoints. Its scoped resume adapter permits only this
verified derived-label superset in addition to training.steps; optimizer,
model, sample order, input targets, RNG and all other settings remain guarded
by the original strict extension check. Original training/model/preparation
files and root config are not edited. Operational before/after source snapshots
and amended continuation_6000.json are recorded under setup/labels6k_repair_*.
Actual resumed optimizer updates remain pending label preparation.

Credit at02:36UTC was2.3939USD (~4.8h at0.4983/h), versus roughly5h remaining
including preparation, training and full96. User notified that1USD additional
credit would provide a buffer. No credit purchase or machine lifecycle action.


FLUX1a6k recovery confirmed2026-10-05 03:02UTC: extended identity labels cover
all6000 rows, accepted fraction98.5%; all4000 original records/embeddings are
exactly preserved. identity_extension_6000.json was copied locally. Training
resumed successfully from4000 and advanced to4156 with finite loss/gradients;
peak reserved59.994GiB/49.33%. Main6k controller16449/trainer17880 and publisher
16487 are healthy; original4k controller exited normally. The label-coverage
failure is resolved. Local checkpoint downloader remains active for6000.


FLUX1a Vast9B milestone2026-10-05 06:32UTC: training completed6000 updates and
full6k checkpoint was automatically downloaded at06:20:30UTC to
/mnt/c/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-006000.
Recomputed all five local SHA256 hashes against the transfer receipt; all match
and manifest.step=6000. Both requested4k/6k local copies are verified. Downloader
exited successfully after completing both, so inactive is now expected.
Full96 step6000 inference is advancing with18 images confirmed in Comet;
Running=true/hasCrashed=false. Final scoring/publication is not yet complete.
Credit is0.4474USD (~54min), versus roughly75min remaining validation/scoring;
previous top-up warning still applies. No running job was interrupted.


FLUX1a Vast9B interruption2026-10-05 07:32UTC: credit is0USD and instance53994096
again reports exited/intended stopped; SSH refuses connections. Comet is ended
(running=false,hasCrashed=false), with78/96 fixed96 step6000 images confirmed.
Training6000 and both full4k/6k local checkpoint downloads are complete and
SHA256-verified. Remaining work: resume incomplete6k inference, generate/upload
remaining18 images, score/summarize all96, then verify Comet publication. No6k
full-panel metric result is claimed. Await credit/machine restoration; do not
purchase credit or repeatedly attempt starts. On restoration inspect complete
latents/stream receipts, resume rsrch_flux1a_6k and progress owner, preserving all
completed work. Approximate remaining GPU/scoring/reload time25–30minutes.


FLUX1a Vast9B restoration 2026-10-05 12:05UTC: instance53994096 was restored externally;
credit4.373USD. Reboot left research supervisors stopped and GPU idle. Started
only rsrch_flux1a_6k (1359, inference1376) and rsrch_flux1a_progress (1393).
Completed training6000/preparation receipts were skipped; original78 outputs
and upload receipts were reused in2seconds. Worker is now generating missing
images with96% GPU utilization, and Comet Running=true/hasCrashed=false.
Final96 scoring/verification remains pending. Local4k/6k transfers remain done;
no download service restart, scientific source sync or machine lifecycle action.


FLUX1a Vast9B completion2026-10-05 12:31UTC: training6000, full96 validation
and Comet publication are complete. Supervisor6k exited normally12:24UTC;
progress owner exited12:25UTC after comet_verified_006000.json passed.
All96 step6000 images independently read back from Comet; Running=false and
hasCrashed=false correctly reflect completion. Owner-ID0.28458708553080214,
CLIP29.46402845780055;4k remains best measured owner-ID0.2943213313592423
(vs2k0.28100960698066046). No statistical significance claim.
Final verifier receipt, per-image CSV, quality summary, inference/background
audits and done receipts copied locally. Both4k/6k complete checkpoints were
already SHA256-verified on Windows; transfer receipts remain verified.
The completed follow-up automation is removed. No instance stop/termination
was performed; machine lifecycle remains under user control.


User-requested GB10 stop verified 2026-10-05 14:08UTC: Vast53994096 now actual_status=exited, intended_status=stopped. Instance was stopped, not terminated; files preserved. FLUX1a monitoring automation was already deleted after verified completion.


### 2026-10-05 — Local FLUX2 one-ID pilot

AICODE-NOTE: Local RTX 4090 Laptop GPU (16376 MiB), authorized for FLUX2 Lite
4B full768 training, 2,000 optimizer updates, fixed24 validation every 500
including step 0. No Vast/HSE job is part of this launch.
Run: `runs/FLUX2_local_4b_one_id_20261005` (ignored symlink to Windows storage).
Comet: `7c88a5c362164fdbad2df18c2f629153`, project `rsrch_new`.
Accepted accumulated-memory measurement: 11.561 GiB reserved (72.29%).
Requires `PYTORCH_ALLOC_CONF=expandable_segments:True` and frozen K/V buffer
CPU offload. Controller: `scripts.run_online_face_ba`; serial inference/decode/
scoring; stop after final step-2000 validation. Status and PID are in the run.
