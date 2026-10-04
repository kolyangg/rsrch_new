"""Review saved inference gates against frozen step-zero face masks; never feeds masks to inference."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from ba_dit.config import load_config
from ba_dit.logging import connect
from ba_dit.validation_masks import load_masks, mask_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config = load_config(run / "resolved_config.yaml")
    panel = [json.loads(line) for line in (run / "panel.jsonl").read_text().splitlines()]
    masks = load_masks(config, [row["sample_id"] for row in panel])
    output = run / "gate_review"
    output.mkdir(exist_ok=True)
    records = []
    for step in (0, 1000, 2000):
        sheet = Image.new("RGB", (4 * 260, 6 * 280), "#e9edf0")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(panel):
            key = row["sample_id"]
            folder = run / f"validation-{step:06d}"
            gates = load_file(folder / f"{key}.safetensors")["gates"].float().numpy()
            if gates.shape != (20, 48, 48) or not np.isfinite(gates).all():
                raise ValueError(f"Unexpected saved gates: {step} {key} {gates.shape}")
            gate = gates.mean(0)
            record = masks["samples"][key]
            mask = np.asarray(Image.open(mask_directory(config) / record["token_mask"]).convert("L")) > 0
            if mask.shape != gate.shape:
                raise ValueError("Frozen scoring mask and gate token geometry differ")
            face = float(gate[mask].mean()) if mask.any() else None
            background = float(gate[~mask].mean()) if (~mask).any() else None
            records.append({"step": step, "sample_id": key, "seed": row["seed"], "face_gate": face,
                            "background_gate": background, "all_gate": float(gate.mean()),
                            "face_to_background": face / background if face is not None and background else None})
            image = Image.open(folder / f"{key}.png").convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            heat = Image.fromarray(np.clip(gate * 255, 0, 255).astype(np.uint8)).resize((256, 256), Image.Resampling.BILINEAR)
            red = Image.new("RGB", image.size, "#ff321f")
            image.paste(red, mask=heat.point(lambda value: round(value * .55)))
            if record["face_bbox"]:
                box = [round(value / 3) for value in record["face_bbox"]]
                ImageDraw.Draw(image).rectangle(box, outline="#27f59a", width=2)
            x, y = (index % 4) * 260, (index // 4) * 280
            sheet.paste(image, (x, y))
            draw.text((x + 3, y + 258), f"{key} gate {gate.mean():.3f}", fill="black")
        sheet.save(output / f"gate_overlays_{step:06d}.jpg", quality=90)
    with (output / "gate_per_image.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {}
    for step in (0, 1000, 2000):
        group = [r for r in records if r["step"] == step]
        summary[str(step)] = {field: float(np.mean([r[field] for r in group if r[field] is not None]))
                              for field in ("all_gate", "face_gate", "background_gate", "face_to_background")}
        summary[str(step)]["face_mask_coverage"] = sum(r["face_gate"] is not None for r in group) / len(group)
    (output / "gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    exp = connect(config, run)
    if not exp:
        raise RuntimeError("The validation Comet experiment is unavailable")
    try:
        for step in (0, 1000, 2000):
            exp.log_metrics({f"gate_review/{key}": value for key, value in summary[str(step)].items()}, step=step)
            exp.log_image(str(output / f"gate_overlays_{step:06d}.jpg"), name=f"one_id_24/gates_{step:06d}", step=step)
        for name in ("gate_per_image.csv", "gate_summary.json", "gate_overlays_000000.jpg",
                     "gate_overlays_001000.jpg", "gate_overlays_002000.jpg"):
            exp.log_asset(str(output / name), file_name=name)
        exp.log_asset(str(Path(__file__)), file_name=Path(__file__).name)
    finally:
        exp.end()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
