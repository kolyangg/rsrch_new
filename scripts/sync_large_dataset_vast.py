#!/usr/bin/env python3
"""Resume transfer of the original adjusted Large Dataset and private bundles to a rented Vast instance."""

import argparse
import shlex
import subprocess
from pathlib import Path

from vast_gpu import DEFAULT_ENV, config, owned_instance, ssh_target

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ("validation-and-smoke-with-metrics.tar.gz", "training-metadata.tar.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--source", type=Path, default=Path("/home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir() or not any(source.iterdir()):
        parser.error(f"Missing local Large Dataset images: {source}")
    for name in BUNDLES:
        if not (ROOT / "data/bundles" / name).is_file():
            parser.error(f"Missing private transfer bundle: {name}")
    credentials, private, _ = config(argparse.Namespace(env_file=args.env_file, private_key=None, public_key=None))
    if not private or not private.is_file():
        parser.error("Set VAST_SSH_PRIVATE_KEY in the local .env")
    instance = owned_instance(credentials, args.instance_id)
    if instance.get("actual_status") != "running":
        parser.error(f"Instance {args.instance_id} is not running")
    user, host, port = ssh_target(credentials, args.instance_id)
    remote = f"{user}@{host}"
    options = ["-i", str(private), "-p", str(port), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
               "-o", "StrictHostKeyChecking=accept-new"]
    ssh = ["ssh", *options, remote]
    subprocess.run([*ssh, "mkdir -p /workspace/datasets/large_dataset"], check=True)
    ssh_transport = " ".join(shlex.quote(value) for value in ["ssh", *options])
    subprocess.run(["rsync", "-a", "--partial", "--info=progress2", "-e", ssh_transport,
                    str(source) + "/", f"{remote}:/workspace/datasets/large_dataset/"], check=True)
    subprocess.run(["rsync", "-a", "--partial", "-e", ssh_transport,
                    *(str(ROOT / "data/bundles" / name) for name in BUNDLES), f"{remote}:/workspace/"], check=True)
    print(f"Large images and private bundles transferred to Vast instance {args.instance_id}")


if __name__ == "__main__":
    main()
