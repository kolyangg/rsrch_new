# Stronger reference-attention refiner — 1 October 2026

## Run and stopping rule

`runs/flux4b_reference_refiner1024_b128_20261001`,
[Comet](https://www.comet.com/nikolay-2104/rsrch-new/debb81df08cf4f5591730dffa86f704c).
The user requested stronger BA, a presentation, and continued training until
ID similarity starts converging. `scripts/run_reference_refiner.py` serializes
training, inference, decoding and scoring. It validates all 24 prompted images
at update 0, 500, 2,000, 5,000 and every 5,000 thereafter. The early500 check
was added when the user requested better GPU utilization and batch128 was
selected. It stops after two
successive checks improve the best ID score by less than .003, preserving a
hash-verified pointer to the best checkpoint. Falling flow loss is not the
stopping criterion. A 100k safety limit is distinct from convergence; reaching
it with continued gains requires a deliberate continuation, not a convergence
claim. State, command completion, child PID and scores are persisted locally.

## Actual architecture

`ba_dit/nn/reference_refiner_flow.py` is a separate module. Frozen FLUX.2-klein
Base 4B supplies native target Q and reference-face K/V from the final single
block, after native normalization/RoPE. Branch projections operate on these
captured features without another RoPE. Native velocity is not added to the
face prediction. Native queries already include pretrained reference and text
conditioning; this is an additional reference route, not exclusive ID input.

The best conditioned512 BA checkpoint (10k, ID .226596) becomes a frozen core.
A new **1024-wide, 16-head reference refiner** learns an additive correction:

1. Learned Q/K/V projections; normalized Q/K attention with a learned
   temperature initialized at 8 and clamped to [1,16].
2. Mean-pooled projected reference values condition bounded gain/shift of the
   query + sigma MLP + noisy-latent context.
3. Normalized attention read is multiplied by that context, then combined
   with the pooled reference and a nonlinear residual layer.
4. A zero-initialized output projection and a reference-gated zero-initialized
   noisy-latent projection produce the correction.

The new branch has **14,048,385 trainable parameters**; the core's **5,327,360**
parameters and all native FLUX weights are frozen. This is a projected attention
flow head, not native-attention LoRA. Initial total flow equals the trained
core exactly; only the newly added correction starts at zero. Step-zero images
therefore use the full 20-step sampler, not the old random-face shortcut.
The `refiner_off_mse` ablation retains the frozen core and its reference read.

## Data, objective and inference

Same 19 one-ID training photographs, same fixed reference. The verified cache
contains 1,140 fit cases and 114 separate-noise probe cases. It adds 456 fresh
full-backbone noise/sigma cases to the preceding 684-case fit cache. Cache
expansion took 270.4s and peaked at 8.389 GiB reserved. This finite cache is
reused by optimization; it is not fresh-noise online backbone training. The
probe uses the same photographs and is not an independent identity test.

AdamW batch128, lr .0001, weight decay .01, warmup100, clip1; native unweighted
flow MSE on 64 sampled face tokens per cached case. The actual optimizer weight
decay comes from `identity.head.weight_decay` and agrees with resolved config.
Batch size is in `identity.optimizer_batch_size`. Inherited generic config
LoRA and branch-rank fields do not describe this experimental head.
No ID-specific loss is used. Capacity, reference conditioning, data coverage
and optimizer changed together, so results cannot isolate one cause.

Validation remains 12 prompts × 2 fixed seeds, all ID51, 768², 20 Euler steps.
Native generated images supply reviewed masks and background; validation never
loads target photographs. Face CFG1, native background CFG4. Latent routing uses
a 16px feathered mask; final pixel compositing preserves the exterior exactly
after VAE decoding. Raw decoder outputs are retained. This named protocol was
explicitly requested by the user and differs from default reference-only masks.

## Admission evidence

The focused module check passes initial core equality, finite gradients,
reference-refiner ablation, frozen-core preservation and checkpoint reload.
Actual first50 updates changed all17 trainable tensors with finite gradients;
all11 frozen core tensors remained bit-exact. Peak cached-training reservation
was 2.170 GiB in the initial batch8 admission. Actual pretrained native-off
prediction is bit-exact. All24 step-zero images are pixel-identical to the
frozen parent10k panel and score .2265959578. This baseline is reused after
strict initial-weight/backbone/mask/panel hash checks in the batch128 run;
`baseline_reuse.json` records its generation provenance.

The initial batch8 run was superseded at50 updates to implement the user's
utilization request. It is retained at `runs/flux4b_reference_refiner1024_20261001`
(Comet `9b376fda8f3f4ad0a83f73b54d37713a`); it is not a rejected quality control.

## Throughput improvement requested by the user

The frozen core's prediction for each fixed feature case is computed once per
training process. Training gathers batches from stacked CUDA tensors instead
of concatenating Python lists repeatedly. A finite global gradient norm
replaces a separate GPU/CPU finite check for each parameter, while all gradients
and loss remain checked. Cached-core and live-core prediction parity is covered
by the focused test; live inference still runs the actual frozen core.

`scripts/benchmark_reference_refiner.py` measured10 warmup +40 optimizer updates
per batch size, with real cached cases and no saved training weights:

| Batch | ms/update | Cases/s | Peak reserved GiB |
| ---: | ---: | ---: | ---: |
| 8 | 8.36 | 957 | 1.95 |
| 32 | 10.43 | 3068 | 2.21 |
| 64 | 18.45 | 3468 | 2.54 |
| 128 | 36.20 | 3536 | 3.37 |
| 256 | 72.01 | 3555 | 4.77 |
| 512 | 143.80 | 3561 | 7.61 |

Selected128: post-startup GPU samples were97–98%; 256/512 added under1% cases/s.
Short benchmarks' average GPU utilization includes startup and is not a steady
utilization measurement. This is cached BA throughput, not full FLUX throughput.

The admitted batch128 run completed500 updates with all17 tensors changed,
all11 frozen tensors exact and exact process-resumed update51. Actual training
reservation was5.193 GiB (including cache preparation and batch-order storage),
and the last250 updates averaged35.99ms. Fit/probe MSE at500 is .45981/.71048;
refiner-off probe MSE remains .80847. Full prompted-image scoring is required
before claiming ID improvement. The larger batch changes optimization and
processes16× more cases per update; it is a separately named experiment.

## Reproduce

No additional dependencies are required. Use the existing FLUX, metrics and
report environments. Run only one GPU job at a time.

```bash
envs/flux-toolkit/bin/python -m scripts.conditioned_face_flow init \
  --run runs/flux4b_reference_refiner1024_b128_20261001 \
  --kind refiner --width 1024 --lr .0001 --weight-decay .01 --batch-size 128 --total 100000 \
  --cache runs/face_flow_refiner_cache_20261001 \
  --core-checkpoint runs/flux4b_face_conditioned512_diverse_20261001/checkpoint-010000/branch.safetensors
envs/flux-toolkit/bin/python -m scripts.run_reference_refiner \
  --run runs/flux4b_reference_refiner1024_b128_20261001
```

Initialization requires a new run directory. Do not start a second controller
while its PID is alive. Failed stages are preserved for diagnosis, not silently
restarted. `source_snapshot/`, checkpoint manifests, cache/mask hashes,
`execution_plan.json`, `convergence.json` and `best_checkpoint.json` record
the experiment and selection rule.

The thread follow-up `review-local-ba-refiner-convergence-and-report` runs
every30 minutes while this controller works. It checks actual processes and
completed panels, reviews paired images and audits, preserves the best model,
and refreshes the requested Dropbox PDF after stopping. It pauses once the
final result and report are delivered. Training itself is handled by the
serial local controller and does not wait for the follow-up schedule.

## First scored refiner result

At500 additional batch128 updates, full24-image ID similarity is **.2297139211**
versus the frozen core's **.2265959578**: a small increase of .0031179634
(+1.38%), with13/24 images improving. CLIP text similarity changes
26.1877139→26.1648177. All24 faces are owned under the unchanged scoring rules.
This is a modest development-panel gain, not evidence of broad image-quality
improvement or generalization. Training continues to2k and the later image
checks; the fixed stopping rule has not triggered.

Training reached2,000 updates with finite gradients and frozen-core equality.
Fit/probe MSE is .30232/.67729. Mean measured cached-update time across2k is
53.18ms; the last250 averaged70.14ms, so the short36ms benchmark is not a
sustained-speed guarantee. A5-second sample during real training measured96%
mean GPU utilization and6393 MiB total device memory. The2k image panel is
still being generated at this report cutoff.

## Presentation delivery

The16-page PDF `reports/261001_reference_refiner/flux4b_reference_refiner1024_architecture.pdf`
contains whole-model and BA diagrams, actual code excerpts, mask overlays,
batch-scaling and score charts, paired crops, CL14/CL39 comparison and limitations.
It was visually inspected and uploaded with verified Dropbox content hash to
`Apps/temp/rsrch_new/2026-10-01/flux4b_reference_refiner1024_architecture.pdf`.
This first delivered version includes scored panels0/500 and training evidence
through2k; its SHA256 is
`b3c362a37bfe1b8cfe054b346c3065e2677864ef83c1db24587d859766a5b47d`.
The PDF and source audit are also logged in the active Comet run. The ignored
run directory retains `dropbox_report_upload.json` as upload evidence. The
scheduled follow-up will refresh the report after the stopping rule triggers.
