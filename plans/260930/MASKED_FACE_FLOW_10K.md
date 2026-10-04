# BA prompted-generation continuation to 10k — 2026-10-01

## Request and unchanged experiment

Continue the verified 2,000-update masked-face experiment to 10,000 total
updates in a **new run**, carrying over both BA parameters and Adam state.
This adds 8,000 optimizer updates. There is no change to the architecture,
19-image one-ID dataset, cached inputs/noise, learning rate .001, batch 8,
reference, prompts, seeds, mask routing or native-background preservation.

- Parent: `runs/flux4b_masked_face_flow_24_stable_20261001`
- New run: `runs/flux4b_masked_face_flow_24_10k_20261001`
- [New Comet experiment](https://www.comet.com/nikolay-2104/rsrch-new/5b8b053fb6004f3094095d71c4d3bb21)
- Worker/orchestrator: `scripts/continue_masked_face_flow.py`
- Training loop: `scripts/train_masked_face_flow.py`

```bash
PYTHONUNBUFFERED=1 COMET_DISPLAY_SUMMARY_LEVEL=0 \
  envs/flux-toolkit/bin/python -m scripts.continue_masked_face_flow run \
  --run-dir runs/NEW_10K_RUN
```

The 2k image panel and metrics are inherited unchanged and logged at update
2000 in the new experiment. Fresh 24-image panels are generated and scored
serially at 4k, 6k, 8k and 10k (96 new images), using 20 Euler steps at 768 px.
The final comparison includes the untouched native image and all five trained
checkpoints. No target photographs are loaded for validation.

## Provenance and checks

`lineage.json` records the parent's identity/manifest/checkpoint/optimizer
hashes and immutable Comet key. Exact executed source snapshots are retained
for both runs. The only shared-loop change supports periodic checkpoint saves
beyond 2k and a resume check at any update; prediction/loss/optimizer arithmetic
is unchanged. The parent's files are preserved.

The extended random case sequence was checked to match the original first
2000 updates exactly. A separate process calculates update 2001 from the parent
Adam state; the resumed trainer reproduces every tensor exactly. Fixed source,
config, conditioning and cache hashes are checked before each worker stage.
Cached/live predictions, finite gradients, branch parameter changes and exact
exterior pixel preservation are checked at subsequent validations.

This remains cached-feature one-ID diagnostic training. The separate-noise
probe uses the same training photographs and was used for earlier learning-rate
selection. It is not an unbiased test set. The 24 prompted generations test the
requested task. Native ID similarity is 0.33140; the inherited 2k BA panel is
0.19661. Image quality must be inspected alongside aggregate scores.

## Progress

At 4k: fit flow MSE 0.74383; separate-noise MSE 1.05156 (2k: 0.83184/1.07052).
At 4k, mean generated-image ID similarity is 0.21165 versus 0.19661 at 2k;
CLIP is 26.7754 versus 27.0128. All 24 face regions remain owned/detected.
At 6k, fit/probe MSE is 0.70813/1.06053: fit continues to improve while the
noise probe has flattened. Later image metrics are pending. Final results will
distinguish the best observed checkpoint from the final 10k checkpoint if
later updates regress.

## Superseded at 6k

The user changed the target to 100k total updates and full validation every20k.
This run stopped at 6k, after completing that already-running image panel.
Its 6k ID/CLIP scores are 0.21402/27.1623. It is preserved as the parent of the
[new 100k continuation](MASKED_FACE_FLOW_100K.md); the old 10k target was not completed.
