"""Measure inference cost in the loaded worker without advancing training RNG."""
import json
import time
import torch
from ba_dit.data.cache import load_pair
from ba_dit.data.manifest import read_manifest


@torch.no_grad()
def profile(model, backend, config, run, experiment):
    # Use the longest real reference layout and all target queries: conservative
    # for face-only inference. Timing only, not a scored generated-image panel.
    pairs=(load_pair(config,row,'cuda',negative=True)[0]
           for row in read_manifest(config['data']['validation_manifest']))
    tensors=max(pairs,key=lambda p:p['reference_tokens'].shape[1])
    h,w=config['data']['target_size']
    native_training=model.training
    model.eval()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            noisy=torch.zeros((1,128,h//16,w//16),device='cuda',dtype=torch.bfloat16)
            sigma=torch.tensor([.5],device='cuda',dtype=noisy.dtype)
            tensors['target_face_mask']=torch.ones((1,h*w//256),device='cuda')
            timings={}
            for branch,label in ((False,'native'),(True,'branch')):
                backend.predict(model,tensors,noisy,sigma,config,branch)
                torch.cuda.synchronize();start=time.monotonic()
                for _ in range(3):
                    backend.predict(model,tensors,noisy,sigma,config,branch)
                    backend.predict(model,tensors,noisy,sigma,config,branch,negative=True)
                torch.cuda.synchronize()
                timings[label+'_seconds_per_image']=(time.monotonic()-start)/3*config['validation']['steps']
    finally:model.train(native_training)
    metrics=[json.loads(x) for x in (run/'metrics.jsonl').read_text().splitlines()]
    update_seconds=sum(m['train/seconds'] for m in metrics[-2:])/len(metrics[-2:])
    generation=config['validation']['limit']*(timings['native_seconds_per_image']+3*timings['branch_seconds_per_image'])
    result={**timings,'train_seconds_per_update':update_seconds,
        'estimated_generation_seconds_all_panels':generation,
        'load_decode_scoring_reserve_seconds':1800,
        'estimated_total_hours':(2000*update_seconds+generation+1800)/3600,
        'method':'Three measured CFG steps extrapolated to20, batch1, largest reference, full target BA mask; excludes probe time from training ETA.'}
    (run/'five_hour_budget.json').write_text(json.dumps(result,indent=2)+'\n')
    experiment.log_metrics({'budget/'+k:v for k,v in result.items() if isinstance(v,(int,float))},step=2)
    experiment.log_asset(str(run/'five_hour_budget.json'))
    print('Runtime budget: '+json.dumps(result),flush=True)
