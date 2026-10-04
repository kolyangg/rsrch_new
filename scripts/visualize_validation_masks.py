"""Plot actual cached BA reference masks and optional frozen output scoring masks."""

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from safetensors import safe_open

from ba_dit.config import digest, load_config
from ba_dit.data.cache import cache_path, cache_spec
from ba_dit.data.geometry import face_mask, reference_geometry
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.nn.reference_read_delta import stratified_reference_indices


def overlay(ax, image, mask, color=(1., .55, .05), alpha=.35):
    ax.imshow(image)
    pixels = np.repeat(np.repeat(mask, image.height // mask.shape[0], axis=0), image.width // mask.shape[1], axis=1)
    rgba = np.zeros((*pixels.shape, 4))
    rgba[pixels.astype(bool)] = (*color, alpha)
    ax.imshow(rgba, interpolation="nearest")
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--learned-face-gate", action="store_true", help="Describe the face-suffix router used by this validation")
    args = parser.parse_args()
    config = load_config(args.config)
    rows = read_manifest(config["data"]["validation_manifest"])[:config["validation"]["limit"]]
    unique = {}
    for row in rows:
        unique.setdefault((row["reference_hash"], tuple(row["reference_box"])), []).append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    audit, figures = [], []
    native_prep = None
    if config["model"]["backend"] == "flux":
        from ba_dit.runtime import prepare_imports
        prepare_imports("flux")
        from extensions_built_in.diffusion_models.flux2.src.sampling import default_prep
        native_prep = default_prep
    items = list(unique.values())
    for start in range(0, len(items), 4):
        group = items[start:start+4]
        fig, axes = plt.subplots(len(group), 3, figsize=(13.5, 4.5 * len(group)), squeeze=False, layout="constrained")
        for axes_row, samples in zip(axes, group):
            row = samples[0]
            original = Image.open(row["reference"]).convert("RGB")
            processed, rebuilt, geometry = reference_geometry(original, row["reference_box"], config["model"]["backend"], config["data"]["reference_size"])
            path = cache_path(config, row, "vae")
            with safe_open(path, framework="pt", device="cpu") as cache:
                metadata = json.loads(cache.metadata()["record"])
                mask = cache.get_tensor("reference_mask").bool()
                token_count = cache.get_slice("reference_tokens").get_shape()[1]
            assert metadata["key"] == digest(cache_spec(config, row, "vae")), "Stale cache"
            assert np.array_equal(mask.numpy(), rebuilt), "Cached mask differs from runtime geometry"
            assert mask.numel() == token_count, "Reference token layout mismatch"
            assert json.loads(json.dumps(geometry)) == metadata["reference"], "Cached reference geometry differs"
            native_error = None
            if native_prep:
                native = native_prep(original, limit_pixels=config["data"]["reference_size"]**2).numpy()
                expected = np.array(processed).astype(np.float32).transpose(2, 0, 1) / 255 * 2 - 1
                assert native.shape == expected.shape, "Native preprocessing shape differs"
                native_error = float(np.abs(native - expected).max())
                assert native_error < 1e-6, "Native preprocessing pixels differ"
            selected = stratified_reference_indices(mask, config["branch"]["max_reference_keys"])
            active = np.zeros(mask.shape, dtype=bool)
            active.flat[selected.numpy()] = True
            box = row["reference_box"]
            overlay(axes_row[0], original, np.asarray(face_mask(original, box)) > 0, color=(0., .7, 1.), alpha=.2)
            axes_row[0].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="#00bfff", linewidth=1.5))
            axes_row[0].set_title(f"{row['identity_id']} — source face box\n{len(samples)} validation prompts share this reference")
            overlay(axes_row[1], processed, active)
            axes_row[1].set_title(f"Actual BA reference keys: {len(selected)}/{token_count}\n{processed.width}×{processed.height} pixels; {mask.shape[1]}×{mask.shape[0]} token grid")
            overlay(axes_row[2], processed, active, alpha=.2)
            yy, xx = np.where(active)
            for y, x in zip(yy, xx):
                axes_row[2].add_patch(Rectangle((x*16-.5, y*16-.5), 16, 16, fill=False, edgecolor="#ffad32", linewidth=.55))
            axes_row[2].set_xlim(max(0, (xx.min()-2)*16), min(processed.width, (xx.max()+3)*16))
            axes_row[2].set_ylim(min(processed.height, (yy.max()+3)*16), max(0, (yy.min()-2)*16))
            axes_row[2].set_title(f"Selected reference keys (zoom)\n{len(selected)} of {int(mask.sum())} face cells; 16×16 pixels each", fontsize=11)
            audit.append({"identity_id": row["identity_id"], "sample_ids": [s["sample_id"] for s in samples],
                "reference_sha256": row["reference_hash"], "source_box": box, "geometry": geometry,
                "cache_sha256": file_hash(path), "cached_mask_matches_geometry": True,
                "native_preprocessing_max_abs": native_error, "face_tokens": int(mask.sum()),
                "selected_keys": len(selected), "total_reference_tokens": token_count})
        fig.suptitle(f"{config['name']} | reference face masks used by branched attention", fontsize=15)
        name = f"reference_masks_{start//4+1}.png"
        fig.savefig(args.output / name, dpi=145, facecolor="white")
        plt.close(fig)
        figures.append(name)
    if args.validation_dir:
        from ba_dit.validation_masks import load_masks, mask_directory
        report = json.loads((args.validation_dir / "validation.json").read_text())
        assert report["panel_sha256"] == file_hash(config["data"]["validation_manifest"])
        frozen = load_masks(config, [row["sample_id"] for row in report["samples"]])
        for start in range(0, len(report["samples"]), 6):
            fig, axes = plt.subplots(2, 3, figsize=(12, 9.5), layout="constrained")
            for ax in axes.flat:
                ax.axis("off")
            for ax, row in zip(axes.flat, report["samples"][start:start+6]):
                im = Image.open(args.validation_dir / row["image"]).convert("RGB")
                mask = np.asarray(Image.open(mask_directory(config) / frozen["samples"][row["sample_id"]]["pixel_mask"])) > 0
                overlay(ax, im, mask, color=(.1, .85, .45), alpha=.35)
                ax.set_title(f"{row['sample_id']} — scoring mask only")
            fig.suptitle("Frozen generated-face masks: evaluation only\nThese masks are never supplied to branched attention", fontsize=13)
            name = f"scoring_masks_{start//6+1}.png"
            fig.savefig(args.output / name, dpi=145, facecolor="white")
            plt.close(fig)
            figures.append(name)
    query_scope = ("model-predicted gate on generated-image tokens; no external target-face mask is passed"
                   if args.learned_face_gate else "all generated-image tokens; no target-face gate is passed")
    gate_note = ("The face-suffix branch predicts a gate for each generated-image token from native queries. "
                 "Green masks on generated images are frozen evaluation masks and are never supplied to the branch."
                 if args.learned_face_gate else
                 "The branch currently updates all generated-image tokens. Green masks on generated images are frozen evaluation masks and do not constrain branch queries.")
    summary = {"profile": config["name"], "validation_items": len(rows), "references": audit,
        "target_query_scope": query_scope,
        "output_face_mask_role": "evaluation only"}
    (args.output / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>Validation face masks</title>'
        '<style>body{font:17px system-ui;max-width:1450px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#182230}img{width:100%;background:white;border-radius:8px;margin:12px 0}p{max-width:1000px;line-height:1.5}</style>'
        f'<h1>{html.escape(config["name"])}: actual validation masks</h1>'
        '<p>Blue: source face box. Orange: reference token cells selected by branched attention, read directly from the VAE cache and checked against runtime geometry. Each cell covers 16×16 preprocessed pixels, so mask edges extend slightly beyond the face box.</p>'
        '<p>A maximum of 512 reference keys is used. If a face occupies more cells, the model selects spatially spread cells within that face; the gaps in the orange overlay show this selection.</p>'
        f'<p>{html.escape(gate_note)}</p>'
        '<p><a href="audit.json">Download mask verification details</a></p>'
        + ''.join(f'<a href="{name}"><img src="{name}" alt="{name}"></a>' for name in figures))
    print(json.dumps({"output": str(args.output), "validation_items": len(rows), "unique_references": len(audit),
                      "cached_masks_verified": len(audit), "selected_keys": [r["selected_keys"] for r in audit]}))


if __name__ == "__main__":
    main()
