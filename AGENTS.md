# Project purpose

Research CL39-inspired branched reference attention on FLUX.2-klein Base 4B/9B and Qwen-Image-2.1 7B. Read `plans/260930/CL39_Qwen_FLUX_48GB_80GB_Implementation_Plan.md` for architecture and `plans/260930/IMPLEMENTATION_NOTES.md` for actual implementation status. Separate proposals from measured results.

# Working practices

- Start by checking branch, worktree status, and relevant `AICODE-NOTE:`, `AICODE-TODO:`, and `AICODE-QUESTION:` anchors. Preserve unrelated edits.
- Keep code concise and localized. Avoid broad refactors, speculative abstraction, and formatting churn.
- Preserve native backbone attention and conditioning exactly in the BA-off mode. Make target/reference token indices, Q/K/V projection points, masks, RoPE, branch gates, and trainable parameters explicit.
- Keep training and validation on the same patched backend. Do not claim branch validation using an unpatched stock pipeline.
- Keep the original fixed 96-item validation order, prompts, seeds, reference images, and metric definitions. A changed panel or resolution is a separate named experiment.
- Never feed a generated-image face box or target face mask to inference. It is evaluation or training supervision only.
- Use separate Qwen and FLUX environments and pinned upstream checkouts. Record exact commits, weight revisions, resolved configs, and deviations.
- Do not copy credentials into tracked files. `.env`, model weights, caches, generated images, and runs stay untracked.
- Do not commit or push unless the user explicitly asks.

# Verification and experiments

Use the smallest check that catches a real failure: configuration validation, import/syntax check, exact token-layout check, native/BA-off parity, first branch gradient, checkpoint reload, or one paired inference. Add focused tests only for critical architectural invariants; avoid tests that simply mirror implementation. Do not treat a tiny synthetic pass as pretrained-model validation.

Long training requires: same data/order and native loss across controls, finite gradients, branch parameter updates, measured peak CUDA reserved memory below 90% of device capacity, save/resume, and serial validation. Log startup and immutable Comet experiment key under project `rsrch_new`. Validate at step 0 and every 2,000 optimizer updates on the fixed 96-item panel; use a smaller explicitly named panel only for smoke work.

Keep machine operations within the currently requested scope. Do not submit 48 GB or 80 GB jobs without a specified host and access path. Report actual failures and memory measurements in `plans/260930/IMPLEMENTATION_NOTES.md`.
