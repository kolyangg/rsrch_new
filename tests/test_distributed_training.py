"""CPU/Gloo regression for global batching and exact per-rank checkpoint replay.

Run directly; no pretrained weights or CUDA allocation are used.
"""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from ba_dit.checkpoint import load_adapters, restore_training, save_training, trainable_parameters
from ba_dit.config import ROOT, load_config
from ba_dit.distributed_training import all_true, rank_rng, sample_offsets
from ba_dit.nn.sliced_native_lora import SlicedLoRALinear
from ba_dit.training import sample_at


def worker(rank, directory, resumed):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group('gloo', init_method='file://'+str(directory/('resume.store' if resumed else 'initial.store')),
                            rank=rank, world_size=2)
    try:
        assert not all_true(rank == 0, 'cpu')
        config = load_config(ROOT/'configs/clust/flux4b_2v100.yaml')
        torch.manual_seed(42)
        model = SlicedLoRALinear(torch.nn.Linear(8,8,bias=False), [(0,8,0,8)], 2,2)
        serial = copy.deepcopy(model)
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1/(s+1))
        scaler = torch.amp.GradScaler('cpu', init_scale=8.)
        torch.manual_seed(142+rank); random.seed(142+rank)
        folder = directory/('resumed' if resumed else 'continuous')
        folder.mkdir(exist_ok=True)
        step, cursor = 0, 0
        if resumed:
            checkpoint = directory/'continuous/checkpoint-000001'
            load_adapters(model,checkpoint,config,'branch_only')
            step,cursor = restore_training(optimizer,scheduler,checkpoint,config,'toy-data',rank,scaler)
            assert (step,cursor)==(1,8)
        while step < 2:
            optimizer.zero_grad(set_to_none=True)
            samples = []
            offsets = list(sample_offsets(cursor,rank,2,4))
            for micro,offset in enumerate(offsets):
                x = torch.randn(2,8)*random.random()
                target = sample_at(list(range(11)),offset,142)
                samples.append((x,target))
                from contextlib import nullcontext
                with ddp.no_sync() if micro < 3 else nullcontext():
                    loss = (ddp(x)-target).square().mean()
                    scaler.scale(loss/4).backward()
            scaler.unscale_(optimizer)
            if step == 0:
                gathered = [None,None]
                dist.all_gather_object(gathered,(offsets,samples))
                if rank == 0:
                    assert sorted(i for offsets,_ in gathered for i in offsets)==list(range(8))
                    for _,pairs in gathered:
                        for x,target in pairs:
                            ((serial(x)-target).square().mean()/8).backward()
                    actual,expected = trainable_parameters(model),trainable_parameters(serial)
                    for name in actual:
                        torch.testing.assert_close(actual[name].grad,expected[name].grad,rtol=1e-5,atol=1e-7)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            step+=1; cursor+=8
            states = [None,None]
            dist.all_gather_object(states,rank_rng())
            if rank == 0:
                assert not torch.equal(states[0]['torch_rng'],states[1]['torch_rng'])
                save_training(model,optimizer,scheduler,config,'branch_only',folder,step,cursor,'toy-data',
                              {'world_size':2,'ranks':states,'scaler':scaler.state_dict()})
            dist.barrier()
    finally:
        dist.destroy_process_group()


def equal(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def test_two_rank_global_batch_and_fresh_process_resume():
    from safetensors.torch import load_file
    with tempfile.TemporaryDirectory() as directory:
        for mode in ('continuous','resume'):
            subprocess.run([sys.executable,__file__,'--worker',directory,mode],check=True,
                           env={**os.environ,'CUDA_VISIBLE_DEVICES':''})
        first = Path(directory)/'continuous/checkpoint-000002'
        second = Path(directory)/'resumed/checkpoint-000002'
        assert equal(load_file(first/'adapters.safetensors'),load_file(second/'adapters.safetensors'))
        assert equal(torch.load(first/'training_state.pt',weights_only=True),
                     torch.load(second/'training_state.pt',weights_only=True))
        assert json.loads((first/'manifest.json').read_text())['step']==2
    print('PASS two-rank gradient averaging, disjoint global offsets and exact fresh-process resume')


if __name__ == '__main__':
    if len(sys.argv)>1:
        mp.spawn(worker,args=(sys.argv[2],sys.argv[3]=='resume'),nprocs=2,join=True)
    else:
        test_two_rank_global_batch_and_fresh_process_resume()
