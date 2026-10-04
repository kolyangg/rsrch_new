#!/usr/bin/env python3
"""Resume transfer of the original adjusted Large Dataset and private bundles to a rented Vast instance."""

import argparse
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
import tempfile
import time
from pathlib import Path

from vast_gpu import DEFAULT_ENV, config, owned_instance, ssh_target

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ("validation-and-smoke-with-metrics.tar.gz", "training-metadata.tar.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--source", type=Path, default=Path("/home/kolyangg/rsrch/dataset_full/large_dataset_adj/large_dataset"))
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--jobs", type=int, default=8, help="Parallel disjoint directory transfers for high-latency links")
    parser.add_argument("--balance-files", action="store_true", help="Distribute individual files across streams to avoid a slow final directory")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 16:
        parser.error('--jobs must be between 1 and 16')
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
               "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3", "-o", "IPQoS=none"]
    ssh = ["ssh", *options, remote]
    subprocess.run([*ssh, "mkdir -p /workspace/datasets/large_dataset"], check=True)
    ssh_transport = " ".join(shlex.quote(value) for value in ["ssh", *options])
    entries = sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file()) if args.balance_files else sorted(p.name for p in source.iterdir())
    def transfer(index, folder):
        listing = Path(folder)/f'files-{index}'
        listing.write_bytes(b'\0'.join(name.encode() for name in entries[index::args.jobs])+b'\0')
        command = ["rsync", "-ar", "--partial", "--stats", "--from0", "--files-from", str(listing),
                   "-e", ssh_transport, str(source)+"/", f"{remote}:/workspace/datasets/large_dataset/"]
        for attempt in range(3):
            result = subprocess.run(command)
            if result.returncode == 0:
                return
            if attempt == 2:
                result.check_returncode()
            time.sleep(3*(attempt+1))
    with tempfile.TemporaryDirectory() as folder, ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(lambda index: transfer(index, folder), range(args.jobs)))
    subprocess.run(["rsync", "-a", "--partial", "-e", ssh_transport,
                    *(str(ROOT / "data/bundles" / name) for name in BUNDLES), f"{remote}:/workspace/"], check=True)
    print(f"Large images and private bundles transferred to Vast instance {args.instance_id}")


if __name__ == "__main__":
    main()
