"""Compare actual prompted-panel scores and face crops across head checkpoints."""
import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
from safetensors import safe_open

from scripts.conditioned_face_flow import experiment
from scripts.masked_face_flow import verify
from scripts.face_crop_flow import write
from ba_dit.data.manifest import file_hash


def review(run):
    config,identity=verify(run)
    old=Path('runs/flux4b_masked_face_flow_24_100k_r1_20261001')
    baseline=json.loads((old/'metric_summary.json').read_text())
    original=dict(baseline)
    for step,source in [(0,'flux4b_masked_face_flow_24_stable_20261001'),
                        (2000,'flux4b_masked_face_flow_24_stable_20261001'),
                        (4000,'flux4b_masked_face_flow_24_10k_20261001')]:
        original[str(step)]=json.loads((Path('runs')/source/f'validation-{step:06d}/quality_summary.json').read_text())['metrics']
    current={int(p.parent.name.split('-')[-1]):json.loads(p.read_text())['metrics']
             for p in run.glob('validation-*/quality_summary.json')}
    assert current
    for step in current:
        folder=run/f'validation-{step:06d}'
        checkpoint=run/f'checkpoint-{step:06d}/branch.safetensors' if step else run/'initial.safetensors'
        expected=file_hash(checkpoint)
        validation=json.loads((folder/'validation.json').read_text())
        assert validation['checkpoint_sha256']==expected
        for sample in validation['samples']:
            assert sample['checkpoint_sha256']==expected and sample['target_photo_loaded'] is False
            with safe_open(str(folder/f"{sample['sample_id']}.safetensors"),framework='pt') as f:
                assert f.metadata()['checkpoint_sha256']==expected
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    for label,scores in [('Original width256', {int(k):v for k,v in original.items()}),('Conditioned 512 + diverse noise',current)]:
        steps=sorted(s for s in scores if s<=max(current))
        ax[0].plot(steps,[scores[s]['id_sim'] for s in steps],marker='o',label=label)
    ax[0].axhline(.3313986754,ls='--',color='gray',label='Native reference-conditioned backbone')
    ax[0].set(xlabel='Optimizer updates',ylabel='Mean ID similarity',title='Same 24 prompted images')
    ax[0].set_xlim(0,max(current)*1.05)
    ax[0].legend(fontsize=8)
    probes=json.loads((run/'probes.json').read_text())
    for split in ('fit','probe'):
        ax[1].plot([p['step'] for p in probes],[p[f'probe/{split}/mse'] for p in probes],label=split)
    ax[1].set(xlabel='Optimizer updates',ylabel='Flow MSE',title='Fixed feature cache; probe uses different noise')
    ax[1].legend();fig.tight_layout();fig.savefig(run/'comparison_metrics.png',dpi=160);plt.close(fig)
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    trained=sorted(s for s in current if s)
    columns=[('Native',run/'native'),('Old BA 6k',old/'validation-006000')]
    columns += [(f'New BA {s}',run/f'validation-{s:06d}') for s in trained]
    names=list(masks)
    for page in range(3):
        panel=Image.new('RGB',(224*len(columns),8*232+32),'white');draw=ImageDraw.Draw(panel)
        for col,(label,folder) in enumerate(columns):draw.text((col*224+8,8),label,fill='black')
        for row,key in enumerate(names[page*8:(page+1)*8]):
            box=masks[key]['face_bbox'];cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
            side=max(box[2]-box[0],box[3]-box[1])*1.35
            crop=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
            for col,(_,folder) in enumerate(columns):
                image=Image.open(folder/f'{key}.png').crop(crop).resize((224,208))
                panel.paste(image,(col*224,row*232+32));draw.text((col*224+4,row*232+242),key,fill='black')
        panel.save(run/f'paired_faces_{page+1}.png')
    audits={str(s):json.loads((run/f'inference_audit_{s}.json').read_text()) for s in trained}
    paired={}
    if len(trained)>1:
        by_step=[]
        for s in (trained[0],trained[-1]):
            with (run/f'validation-{s:06d}/quality_per_image.csv').open() as f:
                by_step.append({r['sample_id']:float(r['id_sim']) for r in csv.DictReader(f)})
        assert by_step[0].keys()==by_step[1].keys()
        differences={k:by_step[1][k]-v for k,v in by_step[0].items()}
        paired={'from_step':trained[0],'to_step':trained[-1],
            'mean_delta':sum(differences.values())/len(differences),
            'improved_images':sum(d>0 for d in differences.values()),'per_image':differences}
    backgrounds=[r for s in current for r in json.loads((run/f'background_audit_{s}.json').read_text())]
    assert all(r['background_max_abs']==0 for r in backgrounds)
    assert all(all(p['exact'] for p in v['parity']) for v in audits.values())
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,logs[-1]['step']+1))
    assert all(math.isfinite(r[k]) for r in logs for k in ('train/loss','train/gradient_norm'))
    write(run/'comparison_summary.json',{'metrics':current,'old':original,'paired_id_change':paired,'inference_audits':audits,
        'review_source_sha256':file_hash(Path(__file__)),
        'checkpoint_provenance_verified':True,'all_training_records_finite':True,
        'mean_cached_update_seconds':sum(r['train/seconds'] for r in logs)/len(logs),
        'peak_train_reserved_gib':max(r['hardware/peak_reserved_gib'] for r in logs),
        'exact_background_images':len(backgrounds),'training':json.loads((run/'training_summary.json').read_text()),
        'resume':json.loads((run/'resume_check.json').read_text())})
    exp=experiment(config,run)
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in (run/'source_snapshot').rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(run/'source_snapshot'))
        archive.write(Path(__file__),'scripts/review_conditioned_face_flow.py')
    exp.log_asset(str(run/'source_snapshot.zip'))
    for p in [run/'comparison_metrics.png',*sorted(run.glob('paired_faces_*.png'))]:exp.log_image(str(p),name=p.stem)
    for p in ('comparison_summary.json','probes.json','training_summary.json','resume_check.json'):exp.log_asset(str(run/p))
    if (run/'execution_plan.json').exists():exp.log_asset(str(run/'execution_plan.json'))
    exp.log_parameters({'reviewed_validation_steps':sorted(current),'best_id_step':max(current,key=lambda s:current[s]['id_sim'])})
    exp.end()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();review(a.run)
