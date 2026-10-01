"""Import the original 19-image one-ID dataset and its fixed 12-prompt panel."""

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from ba_dit.config import ROOT
from ba_dit.data.geometry import face_mask, reference_geometry, target_geometry
from ba_dit.data.manifest import assert_disjoint, file_hash, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=ROOT / "data/validation/prompts_10.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "data/datasets/one_id")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    metadata = json.loads((source / "nm0005092_adj_train.json").read_text())
    validation_metadata = json.loads((source / "nm0005092_adj_test.json").read_text())
    classes = json.loads((source / "one_id_classes_ref.json").read_text())
    prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip()]
    if len(metadata) != 19 or len(prompts) != 12 or classes != {"51": "man"}:
        raise ValueError("Expected the original 19-image, 12-prompt one-ID dataset")
    output.mkdir(parents=True, exist_ok=True)
    sources = [source / "nm0005092_adj_train.json", source / "nm0005092_adj_test.json",
               source / "one_id_classes_ref.json", source / "id_embeds_one_id.pth"]
    copies = [(path, output / path.name) for path in sources]
    copies += [(source / "nm0005092_adj" / name, output / "images" / name) for name in metadata]
    copies += [(source / "ref/51.jpg", output / "ref/51.jpg"), (args.prompts, output / "prompts.txt")]
    for original, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_hash(original) != file_hash(destination):
            raise FileExistsError(f"Different existing data: {destination}")
        shutil.copy2(original, destination)
    names = sorted(metadata)
    common = {"identity_id": "51", "split_policy": "one_id_diagnostic", "seed": 0}
    train = []
    for index, name in enumerate(names):
        reference = names[(index + 1) % len(names)]
        for filename in (name, reference):
            with Image.open(output / "images" / filename) as image:
                face_mask(image, metadata[filename]["face_crop"])
                target_geometry(image, [768, 768], metadata[filename]["face_crop"])
                reference_geometry(image, metadata[filename]["face_crop"], "flux", 512)
        train.append({**common, "sample_id": f"oneid_train_{Path(name).stem}", "split": "train",
            "reference_image": f"images/{reference}", "reference_face_bbox": metadata[reference]["face_crop"],
            "target_image": f"images/{name}", "target_face_box": metadata[name]["face_crop"], "prompt": metadata[name]["text"]})
    box = validation_metadata["51.jpg"]["face_crop_new"]
    with Image.open(output / "ref/51.jpg") as image:
        face_mask(image, box)
    validation = [{**common, "sample_id": f"oneid_{index:02d}", "split": "validation",
        "reference_image": "ref/51.jpg", "reference_face_bbox": box,
        "reference_sha256": file_hash(output / "ref/51.jpg"), "prompt": prompt.replace("<class>", "man img"),
        "prompt_index": index} for index, prompt in enumerate(prompts)]
    for name, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path = output / name
        if path.exists() and path.read_text() != text:
            raise FileExistsError(path)
        path.write_text(text)
    assert_disjoint(read_manifest(output / "train.jsonl", training=True), read_manifest(output / "validation.jsonl"))
    audit = {"source": str(source), "source_identity": "nm0005092", "manifest_identity": "51",
        "train_pairs": len(train), "validation_items": len(validation),
        "pairing": "sorted cyclic distinct views; original target captions; no flips",
        "split_policy": "one_id_diagnostic: identity and validation-reference reuse are intentional",
        "files": {str(destination.relative_to(output)): file_hash(destination) for _, destination in copies},
        "train_sha256": file_hash(output / "train.jsonl"), "validation_sha256": file_hash(output / "validation.jsonl")}
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
