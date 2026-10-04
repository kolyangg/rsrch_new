"""Plot fixed-input learning, actual inference gates and paired generated images."""

import argparse
import html
import json
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--log-comet", action="store_true")
    parser.add_argument("--score", action="store_true", help="Run the existing ID/CLIP scorer in its pinned CPU environment")
    args = parser.parse_args()
    run = args.run
    audit = json.loads((run / "identity.json").read_text())
    spec = audit["diagnostic"]
    inventory = json.loads((run / "optimizer_inventory.json").read_text())
    parameter_count = sum(int(np.prod(shape)) for shape in inventory.values())
    probes = json.loads((run / "probes.json").read_text())
    final = probes[-1]["step"]
    baseline, trained = run / "validation-000000", run / f"validation-{final:06d}"
    if args.score:
        root = Path(__file__).resolve().parents[1]
        metrics_python = Path(os.environ.get("BA_ENVS_DIR", str(root / "envs"))) / "metrics/bin/python"
        env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
        subprocess.run([str(metrics_python), str(root / "scripts/build_validation_output_masks.py"),
                        "--validation", str(baseline)], check=True, env=env)
        for folder in (baseline, trained):
            if not (folder / "quality_summary.json").exists():
                subprocess.run([str(metrics_python), str(root / "scripts/evaluate_metrics.py"),
                                "--validation", str(folder), "--no-comet"], check=True, env=env)
    report = json.loads((trained / "validation.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for ax, key, title in zip(axes, ("face_mse", "gate_iou", "delta_face"),
                             ("Face flow loss", "Face-gate IoU", "Velocity change inside face")):
        for split in ("fit", "probe"):
            ax.plot([p["step"] for p in probes], [p[f"probe/{split}/{key}"] for p in probes], marker="o", label=split)
        ax.set_title(title)
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.25)
        ax.legend()
    fig.savefig(run / "learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(report["samples"]), 4, figsize=(16, 4 * len(report["samples"])), squeeze=False, layout="constrained")
    measurements = []
    for axes_row, row in zip(axes, report["samples"]):
        before = np.asarray(Image.open(baseline / row["image"]).convert("RGB"))
        after = np.asarray(Image.open(trained / row["image"]).convert("RGB"))
        difference = np.abs(after.astype(float) - before.astype(float)).mean(-1)
        gates = load_file(trained / f"{row['sample_id']}.safetensors")["gates"].float().numpy()
        # The final quarter of denoising is easiest to align with the final image.
        gate = gates[-max(1, len(gates) // 4):].mean(0)
        overlay = np.asarray(Image.fromarray(gate).resize((after.shape[1], after.shape[0]), Image.Resampling.BILINEAR))
        axes_row[0].imshow(before); axes_row[0].set_title(f"{row['sample_id']} · native initialization")
        axes_row[1].imshow(after); axes_row[1].set_title(f"Face BA · update {final}")
        axes_row[2].imshow(after)
        axes_row[2].imshow(overlay, cmap="magma", vmin=0, vmax=1, alpha=.65)
        axes_row[2].set_title("Predicted gate · final quarter of denoising")
        axes_row[3].imshow(difference, cmap="inferno", vmin=0, vmax=128)
        axes_row[3].set_title(f"Pixel difference · mean {difference.mean():.2f}/255")
        for ax in axes_row: ax.axis("off")
        measurements.append({"sample_id": row["sample_id"], "mean_abs_pixel": float(difference.mean()),
                             "gate_mean": float(gate.mean()), "gate_peak": float(gate.max())})
    fig.savefig(run / "paired_images_and_gates.png", dpi=150)
    plt.close(fig)
    (run / "image_changes.json").write_text(json.dumps(measurements, indent=2) + "\n")
    start, end = probes[0], probes[-1]
    metrics = []
    for key in ("face_mse", "background_mse", "router_bce", "gate_iou", "gate_face", "gate_background", "delta_face", "delta_background"):
        values = [start[f"probe/{split}/{key}"] for split in ("fit", "probe")] + [end[f"probe/{split}/{key}"] for split in ("fit", "probe")]
        metrics.append("<tr><td>" + html.escape(key) + "</td>" + "".join(f"<td>{v:.5f}</td>" for v in values) + "</tr>")
    page = """<!doctype html><meta charset="utf-8"><title>Face BA diagnostic</title>
<style>body{font:17px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;background:#f4f6f7;color:#20313b}img{width:100%}p{max-width:1100px;line-height:1.5}table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #ccd3d8;text-align:right}td:first-child{text-align:left}</style>
<h1>FLUX one-ID face BA: fixed-input diagnostic</h1>
<p>DESCRIPTION</p>
<img src="learning_curves.png"><h2>Fit versus separate-noise probes</h2>
<table><tr><th>Metric</th><th>Fit initial</th><th>Probe initial</th><th>Fit final</th><th>Probe final</th></tr>"""
    description = (f"{spec.get('blocks', 1)} late reference-face reads, rank-{spec['rank']} output LoRA and learned face gates: "
        f"{parameter_count:,} trainable BA parameters. All native backbone, encoder and VAE weights are frozen. "
        "Target face boxes supervise training only; inference predicts gates from native queries. "
        f"{spec['train_images']} same-ID training pairs, {len(spec['sigmas'])} fixed noise levels and separate probe seeds. "
        f"{spec['validation_images']} generated validation prompts at {spec['resolution']} px / {spec['inference_steps']} steps. "
        "This tests finite-input learning, not identity generalization or exact face-only final-pixel changes.")
    if spec.get("blocks", 1) > 1:
        description += " Loss/gate probes cover all 19 pairs at sigma 0.5; training uses all six configured noise levels. Gates below average the eight sites."
    page = page.replace("DESCRIPTION", description)
    page += "".join(metrics) + "</table><h2>Generated images and actual inference gates</h2><img src='paired_images_and_gates.png'><p>Gate overlays use the final quarter of denoising. They are model predictions, not detector masks supplied to generation. Bright areas have higher gate probability.</p>"
    if (baseline / "quality_summary.json").exists() and (trained / "quality_summary.json").exists():
        initial_metrics = json.loads((baseline / "quality_summary.json").read_text())["metrics"]
        final_metrics = json.loads((trained / "quality_summary.json").read_text())["metrics"]
        page += "<h2>Validation identity/text metrics</h2><table><tr><th>Metric</th><th>Initial</th><th>Final</th></tr>"
        for key in ("id_sim", "text_sim"):
            page += f"<tr><td>{key}</td><td>{initial_metrics[key]:.5f}</td><td>{final_metrics[key]:.5f}</td></tr>"
        page += "</table>"
    (run / "review.html").write_text(page)
    if args.log_comet:
        from ba_dit.config import load_config, digest
        from ba_dit.logging import connect
        config = load_config(run / "resolved_config.yaml")
        audit = json.loads((run / "identity.json").read_text())
        spec = audit["diagnostic"]
        exp = connect(config, run)
        if exp is None:
            raise ValueError("--log-comet requires enabled Comet logging")
        try:
            # The separate diagnostic spec is authoritative for this custom loop.
            execution = json.loads((run / "execution.json").read_text()) if (run / "execution.json").exists() else {}
            exp.log_parameters({"model_identity": f"{audit['variant']}:{digest(audit)}",
                "training/seed": spec["seed"], "training/warmup": spec.get("warmup", 0),
                "training/checkpoint_every": spec["probe_every"], "training/validation_every": spec.get("validation_every", spec["steps"]),
                "training/batch_size": execution.get("batch_size", spec["batch_size"]), "training/grad_accum": 1,
                "native_attention_lora_enabled": False,
                "effective_training_spec": "identity.json diagnostic section"})
            for name in ("identity.json", "optimizer_inventory.json", "training_summary.json", "trained_cache_parity.json", "probes.json"):
                exp.log_asset(str(run / name), file_name=f"face_diagnostic_{name}")
            for name in ("paired_images_and_gates.png", "learning_curves.png"):
                exp.log_image(str(run / name), name=f"face_diagnostic/{name}", step=final)
            for step, folder in ((0, baseline), (final, trained)):
                if (folder / "quality_summary.json").exists():
                    values = json.loads((folder / "quality_summary.json").read_text())["metrics"]
                    exp.log_metrics({f"validation/{k}": v for k, v in values.items()}, step=step)
        finally:
            exp.end()
    print(run / "review.html")


if __name__ == "__main__":
    main()
