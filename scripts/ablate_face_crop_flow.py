"""One fixed-seed generation with only the added reference read switched off."""

import argparse
import gc
import json
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file, save_file

from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import file_hash, read_manifest
from ba_dit.nn.face_crop_flow import capture, install
from ba_dit.runtime import backend_module
from scripts.face_crop_flow import experiment, verify, write


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config, identity = verify(run)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    backend = backend_module(config)
    from extensions_built_in.diffusion_models.flux2.src.sampling import get_schedule

    checkpoint = run / "checkpoint-002000"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["identity"] == identity and manifest["sha256"] == file_hash(checkpoint / "branch.safetensors")
    model = backend.load_transformer(config)
    branch = install(model, identity["query_residual"], identity["noise_skip"], identity["timestep_scaling"])
    branch.load_state_dict(load_file(checkpoint / "branch.safetensors"), strict=True)
    row = read_manifest(config["data"]["validation_manifest"])[0]
    tensors, _ = load_pair(config, row, "cuda")
    latent = torch.randn((1,128,16,16), dtype=torch.bfloat16, generator=torch.Generator().manual_seed(row["seed"])).cuda()
    times = get_schedule(20, 256)
    for current, following in zip(times[:-1], times[1:]):
        sigma = torch.tensor([current], dtype=latent.dtype, device="cuda")
        features = capture(model, branch, backend, tensors, latent, sigma, config)
        velocity = branch.predict(**features, reference_read=False).transpose(1,2).reshape_as(latent).to(latent.dtype)
        latent = latent + (following-current) * velocity
    output = run / "reference_read_off"
    output.mkdir(exist_ok=False)
    save_file({"latent": latent.cpu()}, output / "crop_val_0.safetensors")
    write(output / "validation.json", {"samples": [{"sample_id": row["sample_id"], "identity_id": row["identity_id"],
          "prompt": row["prompt"], "seed": row["seed"], "image": "crop_val_0.png"}],
          "ablation": "added reference read off; BA query/noisy paths retained; native velocity disabled"})
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    del model, branch, features, tensors
    gc.collect()
    torch.cuda.empty_cache()
    vae = backend.load_vae(config)
    image = backend.decode(vae, latent.cpu(), config)
    image.save(output / "crop_val_0.png")
    exp = experiment(config, run)
    exp.log_image(image, name="face_crop/reference_read_off_seed0", step=2000)
    exp.log_asset(str(output / "validation.json"), file_name="reference_read_off.json")
    exp.log_asset(str(Path(__file__)))
    exp.end()


if __name__ == "__main__":
    main()
