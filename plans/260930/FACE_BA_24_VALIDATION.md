# FLUX 4B face BA: 24-image one-ID validation

The completed eight-site, rank-256 FLUX 4B run was evaluated at updates
0, 1,000 and 2,000. Checkpoint identity and branch-file checksums were verified
before inference. Training source, weights and native backend are unchanged.

The original one-ID validation manifest has 12 prompts, each with seed 0. The
new named panel keeps those 12 entries in order and adds the same prompts with
seed 1, giving 24 fixed images of identity 51. Reference image, reference face
box, 768 px target, 512 px reference, 20 denoising steps and CFG 4 are fixed
across the three evaluations. All 24 use the same patched FLUX backend and BA
in both CFG lanes. Generated-image boxes and masks are produced after step-zero
generation and are used only for scoring.

The first four seed-zero images and latent/gate records at each step are linked
byte for byte from the completed four-image validation. Twenty new images per
step are batched in pairs with the same reference-face support. This preserves
the original four images and cuts repeat inference while maintaining the same
paired panel at all three steps. Batch-two inference used 9.611 GiB peak CUDA
reserved memory. Initially it took about 47 seconds per pair, then slowed to
roughly 110–125 seconds per pair as the 16 GB laptop GPU reached 88°C and
thermally throttled. All 72 image/latent/gate records completed serially.

The 24-row manifest is `data/datasets/one_id/validation_24_seeds01.jsonl`.
Run artifacts are in `runs/flux4b_face_one_id_24_val_20261001`. The dedicated
new Comet experiment is
[65297da706af4cd6a0308b78fc18f9ca](https://www.comet.com/nikolay-2104/rsrch-new/65297da706af4cd6a0308b78fc18f9ca).
The local runner is `scripts/validate_face_suffix_24.py`; its exact versions,
including two corrected launch errors, are saved under the run's
`source_snapshot/` and `source_revision.json`.

Completed outputs: 72 PNGs and latent/gate records, frozen step-zero scoring
masks, original PhotoMaker ID/CLIP definitions and per-image CSVs, visual
comparisons with face versus background RGB changes, plus the original four
PyIQA face-quality models. All metrics and images were logged to the dedicated
Comet experiment. The original 96-image backbone panel remains separate.

| Updates | ID similarity | CLIP | TOPIQ-Face | TOPIQ | MUSIQ | MANIQA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.331399 | 28.0729 | 0.689587 | 0.488658 | 67.8222 | 0.615559 |
| 1,000 | 0.339098 | 27.6710 | 0.673590 | 0.473110 | 67.5330 | 0.600018 |
| 2,000 | 0.339220 | 26.8572 | 0.636220 | 0.477411 | 65.8247 | 0.594491 |

The mean ID change at 2,000 is +0.00782, but 10 of 24 samples improve and 14
decline; the median change is -0.00604. A paired bootstrap over the 12 prompt
clusters, keeping both seeds together and using 20,000 resamples, gives a 95%
interval of [-0.02296, +0.03788] for ID and [-2.3843, -0.0596] for CLIP. Thus
the identity gain is not reliable on this panel, while text alignment and face
quality deteriorate. All 24 images changed: mean RGB absolute difference from
baseline is 22.37/255 at 1,000 and 24.61/255 at 2,000. At 2,000 the difference
averages 41.33/255 in frozen face boxes versus 23.99/255 elsewhere, so effects
extend outside faces.

The actual cached reference mask was checked against runtime preprocessing:
88 of 1,024 reference tokens are selected on the intended face. All 24 frozen
step-zero output masks were automatically usable and visually reviewed on top
of the generated images; none was flagged as ambiguous or missing. They are
used for evaluation only. At 2,000, saved learned gate probabilities average
0.9168 inside those boxes and 0.0848 outside across all timesteps and eight
sites. Every image has higher activation in its face box, with per-image ratios
from 8.1× to 20.4×. The gate is face-focused, although image changes extend
outside the face. The branch is connected and active; this run does not yet
demonstrate a useful quality or identity improvement.

Local evidence: `runs/flux4b_face_one_id_24_val_20261001/report.md`,
`comparison_seed{0,1}.jpg`, `gate_review/gate_summary.json`,
`mask_visualization/index.html`, `paired_bootstrap.json`, and each
`validation-*/{quality_summary.json,face_quality_summary.json}`. The runner and
gate-review scripts are `scripts/validate_face_suffix_24.py` and
`scripts/review_face_suffix_24_gates.py`. Two launch/resume serialization fixes
affected only the validation runner, not model weights or checkpoints; exact
source hashes and explanations are in the run's `source_revision.json`.
