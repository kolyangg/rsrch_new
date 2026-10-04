"""Measure real BA update throughput and memory across batches, without saving weights."""
import argparse
import gc
import json
import subprocess
import threading
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from scripts.conditioned_face_flow import make
from scripts.face_crop_flow import cases, prediction, write


def main(run,destination):
    torch.manual_seed(142)
    identity=json.loads((run/'identity.json').read_text())
    b=make(identity).cuda()
    initial=load_file(run/'initial.safetensors')
    b.load_state_dict(initial)
    groups=cases(run)
    with torch.no_grad():
        for case in groups['fit']:
            case['core_velocity']=prediction(b,case,reference_read=False)
    fit={k:torch.cat([c[k] for c in groups['fit']]) for k in groups['fit'][0]}
    del groups
    gc.collect();torch.cuda.empty_cache()
    parameters=[p for p in b.parameters() if p.requires_grad]
    results=[]
    for batch_size in (8,32,64,128,256,512):
        b.load_state_dict(initial)
        optimizer=torch.optim.AdamW(parameters,lr=.0001,weight_decay=.01,fused=True)
        indices=torch.randint(len(fit['query']),(50,batch_size),generator=torch.Generator().manual_seed(142)).cuda()
        samples=[];done=threading.Event()
        def sample_gpu():
            while not done.is_set():
                values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader,nounits'],text=True)
                samples.append([float(x.strip()) for x in values.strip().split(',')])
                done.wait(.3)
        sampler=threading.Thread(target=sample_gpu,daemon=True)
        torch.cuda.reset_peak_memory_stats()
        sampler.start();seconds=[]
        try:
            for step in range(50):
                began=time.perf_counter()
                batch={k:v.index_select(0,indices[step]) for k,v in fit.items()}
                optimizer.zero_grad(set_to_none=True)
                loss=(prediction(b,batch)-batch['flow_target'].float()).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                if step>=10:seconds.append(time.perf_counter()-began)
            peak=torch.cuda.max_memory_reserved()/2**30
            record={'batch_size':batch_size,'seconds_per_update':sum(seconds)/len(seconds),
                'cases_per_second':batch_size/(sum(seconds)/len(seconds)),
                'peak_reserved_gib':peak,'final_loss':float(loss),'all_gradients_finite':True,
                'gpu_utilization_percent':sum(s[0] for s in samples)/len(samples),
                'gpu_power_watts':sum(s[2] for s in samples)/len(samples),'gpu_samples':samples}
            results.append(record);print(json.dumps(record),flush=True)
            write(destination,{'run':str(run),'protocol':'10 warmup + 40 measured updates; frozen core cached; gathered batches; no saved weights','results':results})
            if peak/16>=.88:break
        except torch.cuda.OutOfMemoryError:
            results.append({'batch_size':batch_size,'out_of_memory':True})
            write(destination,{'run':str(run),'results':results});break
        finally:
            done.set();sampler.join()
            del optimizer
            b.zero_grad(set_to_none=True)
            gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.run.resolve(),a.output.resolve())
