# BA prompted-generation continuation to 100k — 2026-10-01

The user extended the target to **100,000 total optimizer updates** and changed
full image validation to **every 20,000 updates**. The run continues from the
verified 6,000-update checkpoint with Adam state intact: 94,000 new updates.

- Active run: `runs/flux4b_masked_face_flow_24_100k_r1_20261001`
- [Comet](https://www.comet.com/nikolay-2104/rsrch-new/fa43a8dd82274d1397cbf9befbbda9bf)
- Parent: `runs/flux4b_masked_face_flow_24_10k_20261001`, stopped at 6k when the request changed.
- Full 24-image prompted validation: **20k, 40k, 60k, 80k, 100k**.
- Cached fit/separate-noise probes: every 1,000 updates.
- Live training metrics: every 100 updates; full records remain in `metrics.jsonl`.
- Process/log pointers: `runs/masked_face_flow_100k_process.json`.

```bash
PYTHONUNBUFFERED=1 COMET_DISPLAY_SUMMARY_LEVEL=0 \
  envs/flux-toolkit/bin/python -m scripts.continue_masked_face_flow run \
  --run-dir runs/NEW_100K_RUN \
  --parent runs/flux4b_masked_face_flow_24_10k_20261001 \
  --parent-step 6000 --target-step 100000 --validate-every 20000
```

## Why training is fast

The frozen 4B backbone is absent from cached optimizer updates. Only the
2,408,448-parameter BA head trains on batches of eight cached cases, with 64
sampled face tokens per case. There are 228 fit cases: 19 full training images
× six noise levels × two noise seeds; 114 separate-noise probe cases. More
updates repeat this cache rather than generating fresh noising/backbone passes.
This is a one-ID cached-feature diagnostic, not end-to-end FLUX training.

Validation evaluates the full frozen backbone 20 times per new 768 px image,
with BA predicting face flow, and scores 24 prompt/seed pairs. It therefore
costs much more than an optimizer update. Architecture, learning rate .001,
batch size, cached inputs, prompts, reference, masks and native backgrounds
are unchanged. The new schedule is the user's explicit override of the former
2k image-validation interval.

## Inherited comparison

| Updates | Mean ID similarity | CLIP |
|---:|---:|---:|
|2,000|0.19661|27.0128|
|4,000|0.21165|26.7754|
|6,000|0.21402|27.1623|

All 24 intended faces were detected at these checkpoints. Untouched native
ID similarity is 0.33140. The 6k image sweep was already almost complete when
the request changed; it was finished and scored as the starting comparison.
The new run logs its inherited 6k images/metrics, then only the requested 20k
intervals. The prior 10k target was superseded, not completed.

## Admission and status

Exact source/checkpoint/Adam/cache provenance is recorded in `lineage.json` and
`identity.json`. The extended seeded case sequence preserves the previous 6k
prefix. A separate-process update-6001 check verifies optimizer continuation.
Finite gradients, memory limits, live/cached prediction parity and exact final
exterior pixel preservation remain enforced. Every scheduled image panel is
scored and logged automatically, and a final report compares all checkpoints
and names the best observed ID score alongside the final 100k result.

The first 100k initialization attempt failed schema validation before training:
`probe_every` and `live_metric_every` are experiment metadata, not core config
fields. The corrected runner validates the core config before creating Comet,
and stores cadence in the experiment identity. The failed attempt is retained
as `flux4b_masked_face_flow_24_100k_20261001` and is not an active training job.

The active continuation was launched as a detached process. Final 100k results
are pending; none are inferred from the number of optimizer updates.
