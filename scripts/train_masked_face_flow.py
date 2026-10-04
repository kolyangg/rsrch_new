"""Cached full-scene BA training with a reproducible shuffled case order."""

import json
import time

import torch
from safetensors.torch import load_file, save_file

from ba_dit.data.manifest import file_hash
from ba_dit.logging import log_metrics
from ba_dit.nn.face_crop_flow import FaceCropFlow
from scripts.face_crop_flow import cases, evaluate, experiment, memory, prediction, write


def train(run, config, identity, resume=0, until=2000):
    branch=FaceCropFlow(query_residual=True,noise_skip=True,timestep_scaling=False).cuda()
    path=run/f'checkpoint-{resume:06d}/branch.safetensors' if resume else run/'initial.safetensors'
    branch.load_state_dict(load_file(path))
    optimizer=torch.optim.AdamW(branch.parameters(),lr=config['training']['lr'],weight_decay=0.,fused=True)
    if resume:
        record=json.loads((path.parent/'manifest.json').read_text())
        assert record['identity']==identity and record['sha256']==file_hash(path)
        optimizer.load_state_dict(torch.load(path.parent/'optimizer.pt',map_location='cpu',weights_only=True))
    groups=cases(run)
    # The same order is reconstructed in a new process; never reset it at resume.
    generator=torch.Generator().manual_seed(config['training']['seed'])
    order=torch.randint(len(groups['fit']),(config['training']['steps'],8),generator=generator).tolist()
    exp=experiment(config,run)
    exp.log_parameters({'live_metric_stride':identity.get('live_metric_every',25),'cached_probe_every':identity.get('probe_every',250)})
    probes=json.loads((run/'probes.json').read_text()) if resume else [{'step':0,**evaluate(branch,groups)}]
    torch.cuda.reset_peak_memory_stats()

    def update(step):
        selected=[groups['fit'][i] for i in order[step-1]]
        batch={k:torch.cat([c[k] for c in selected]) for k in selected[0]}
        for group in optimizer.param_groups:
            group['lr']=config['training']['lr']*min(1.,step/config['training']['warmup'])
        optimizer.zero_grad(set_to_none=True)
        loss=(prediction(branch,batch)-batch['flow_target'].float()).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is None or not p.grad.isfinite().all() for p in branch.parameters()):
            raise RuntimeError('Nonfinite or missing BA gradients')
        if step in (1,2):
            gradients={n:float(p.grad.norm()) for n,p in branch.named_parameters()}
            write(run/f'first_gradients_{step}.json',gradients)
            assert gradients['out.weight'] and gradients['noisy_out.weight']
            if step==2: assert all(gradients.values())
        norm=torch.nn.utils.clip_grad_norm_(branch.parameters(),1.)
        optimizer.step()
        return float(loss.detach()),float(norm)

    try:
        if not resume: exp.log_metrics({k:v for k,v in probes[0].items() if k!='step'},step=0)
        for step in range(resume+1,until+1):
            began=time.monotonic()
            loss,norm=update(step)
            torch.cuda.synchronize()
            expected_path=run/f'expected_update_{step}.safetensors'
            if expected_path.exists():
                expected=load_file(expected_path)
                assert all(torch.equal(p.cpu(),expected[n]) for n,p in branch.state_dict().items())
                write(run/'resume_check.json',{'step':step,'fresh_process_exact':True})
            log_metrics(exp if step==1 or step%identity.get('live_metric_every',25)==0 else None,run,
                        {'train/loss':loss,'train/gradient_norm':norm,'train/seconds':time.monotonic()-began,**memory()},step)
            if step%identity.get('probe_every',250)==0 or step==until:
                probes.append({'step':step,**evaluate(branch,groups)})
                write(run/'probes.json',probes)
                exp.log_metrics({k:v for k,v in probes[-1].items() if k!='step'},step=step)
            if step in (50,1000,2000) or step % config['training']['checkpoint_every']==0 or step==until:
                folder=run/f'checkpoint-{step:06d}'
                folder.mkdir()
                save_file({n:p.cpu().contiguous() for n,p in branch.state_dict().items()},folder/'branch.safetensors')
                torch.save(optimizer.state_dict(),folder/'optimizer.pt')
                write(folder/'manifest.json',{'step':step,'identity':identity,'sha256':file_hash(folder/'branch.safetensors')})
        write(run/'training_summary.json',{'steps':until,'all_gradients_finite':True,
              'parameters':sum(p.numel() for p in branch.parameters()),'lr':config['training']['lr'],
              'batch_order':'seed142 uniform cases with replacement, batch8',**memory()})
        if until==50:
            update(51)
            save_file({n:p.cpu().contiguous() for n,p in branch.state_dict().items()},run/'expected_update_51.safetensors')
    finally:
        exp.end()
