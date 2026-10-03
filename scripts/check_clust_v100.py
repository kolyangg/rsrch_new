"""Fail closed before model loading if the allocation, runtime or data is wrong."""
import hashlib
import json
import os
from pathlib import Path

import torch

from ba_dit.config import ROOT


def main():
    if not os.getenv('SLURM_JOB_ID'):
        raise RuntimeError('Run on the allocated compute node, not the login node')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Exactly two Slurm-visible GPUs are required')
    if torch.__version__ != '2.7.1+cu126' or torch.version.cuda != '12.6' or 'sm_70' not in torch.cuda.get_arch_list():
        raise RuntimeError('Expected the pinned CUDA 12.6 PyTorch build with sm_70 kernels')
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    if available < 64*2**30:
        raise RuntimeError('Need 64 GiB available host RAM for two FP32 text encoders and model loading')
    devices = []
    for index in range(2):
        gpu = torch.cuda.get_device_properties(index)
        if 'V100' not in gpu.name or (gpu.major, gpu.minor) != (7, 0) or gpu.total_memory < 30*10**9:
            raise RuntimeError(f'Unexpected device {index}: {gpu.name}, {gpu.total_memory} bytes')
        with torch.cuda.device(index):
            value = torch.ones((32,32), device='cuda', dtype=torch.float16)
            assert (value @ value).isfinite().all()
        devices.append({'rank':index,'name':gpu.name,'vram_gib':gpu.total_memory/2**30})
    manifest = ROOT/'data/train_pairs_large_clust.jsonl'
    receipt = json.loads((ROOT/'data/clust_large_transfer.json').read_text())
    if (receipt.get('verified') is not True or receipt.get('files') != 47500 or
            receipt.get('manifest_sha256') != hashlib.sha256(manifest.read_bytes()).hexdigest()):
        raise RuntimeError('Dataset transfer/checksum receipt is missing or does not match the manifest')
    # No credential values are read into logs.
    env = ROOT/'.env'
    if not os.getenv('COMET_API_KEY') and not (env.is_file() and any(
            line.startswith('COMET_API_KEY=') and line.split('=',1)[1].strip().strip("\"'")
            for line in env.read_text().splitlines())):
        raise RuntimeError('Configure COMET_API_KEY privately before submission')
    print(json.dumps({'slurm_job_id':os.environ['SLURM_JOB_ID'],'devices':devices,
                      'torch':torch.__version__,'cuda':torch.version.cuda,'dataset_verified':True,
                      'host_available_gib':available/2**30}), flush=True)


if __name__ == '__main__':
    main()
