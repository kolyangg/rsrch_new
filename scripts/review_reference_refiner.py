"""Audit scored refiner panels and log paired images, curves and source evidence."""
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

from ba_dit.data.manifest import file_hash
from scripts.conditioned_face_flow import experiment
from scripts.face_crop_flow import write
from scripts.masked_face_flow import verify


def review(run, experiment_factory=experiment, model_label='Refiner + frozen BA core', initial_label='Frozen core'):
    config,identity=verify(run)
    current={int(p.parent.name.split('-')[1]):json.loads(p.read_text())['metrics']
             for p in sorted(run.glob('validation-*/quality_summary.json'))}
    assert 0 in current
    backgrounds=[]
    for step in current:
        folder=run/f'validation-{step:06d}'
        checkpoint=run/f'checkpoint-{step:06d}/branch.safetensors' if step else run/'initial.safetensors'
        expected=file_hash(checkpoint)
        record=json.loads((folder/'validation.json').read_text())
        assert len(record['samples'])==24 and record['checkpoint_sha256']==expected
        assert record['panel_sha256']==identity['validation_manifest_sha256']
        for sample in record['samples']:
            assert sample['checkpoint_sha256']==expected and not sample['target_photo_loaded']
            with safe_open(str(folder/f"{sample['sample_id']}.safetensors"),framework='pt') as f:
                assert f.metadata()['checkpoint_sha256']==expected
        audit=json.loads((run/f'inference_audit_{step}.json').read_text())
        assert all(p['exact'] for p in audit['parity'])
        backgrounds.extend(json.loads((run/f'background_audit_{step}.json').read_text()))
    assert all(r['background_max_abs']==0 for r in backgrounds)
    latest=max(current)
    by_step=[]
    for step in (0,latest):
        with (run/f'validation-{step:06d}/quality_per_image.csv').open() as stream:
            by_step.append({r['sample_id']:float(r['id_sim']) for r in csv.DictReader(stream)})
    assert by_step[0].keys()==by_step[1].keys()
    changes={key:by_step[1][key]-value for key,value in by_step[0].items()}
    probes=json.loads((run/'probes.json').read_text())
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,logs[-1]['step']+1))
    assert all(math.isfinite(r[k]) for r in logs for k in ('train/loss','train/gradient_norm'))
    train=json.loads((run/'training_summary.json').read_text())
    assert all(train['updates'].values()) and all(train['frozen_parameters_exact'].values())
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    steps=sorted(current)
    axes[0].plot(steps,[current[s]['id_sim'] for s in steps],'o-',label=model_label)
    axes[0].axhline(current[0]['id_sim'],ls=':',label=initial_label);axes[0].axhline(.3313986754,ls='--',color='gray',label='Native')
    axes[0].set(ylabel='Mean ID similarity',title='24 fixed prompted images');axes[0].legend(fontsize=8)
    axes[1].plot(steps,[current[s]['text_sim'] for s in steps],'o-',color='darkorange');axes[1].set(ylabel='CLIP logit',title='Text alignment')
    for split in ('fit','probe'):
        axes[2].plot([p['step'] for p in probes],[p[f'probe/{split}/mse'] for p in probes],label=split)
    axes[2].set(ylabel='Flow MSE',title='Same-photo cached noise cases');axes[2].legend()
    for ax in axes:ax.set_xlabel('Additional BA updates');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(run/'comparison_metrics.png',dpi=160);plt.close(fig)
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    names=list(masks)
    best=max(current,key=lambda s:current[s]['id_sim'])
    columns=[('Native',run/'native'),(initial_label,run/'validation-000000')]
    for step in sorted(set((best,latest))-{0}):columns.append((f'BA +{step}',run/f'validation-{step:06d}'))
    for page in range(3):
        panel=Image.new('RGB',(224*len(columns),8*234+32),'white');draw=ImageDraw.Draw(panel)
        for col,(label,folder) in enumerate(columns):draw.text((col*224+8,8),label,fill='black')
        for row,key in enumerate(names[page*8:(page+1)*8]):
            x0,y0,x1,y1=masks[key]['face_bbox'];cx=(x0+x1)/2;cy=(y0+y1)/2;side=max(x1-x0,y1-y0)*1.35
            crop=tuple(map(int,(cx-side/2,cy-side/2,cx+side/2,cy+side/2)))
            for col,(_,folder) in enumerate(columns):
                image=Image.open(folder/f'{key}.png').crop(crop).resize((216,210))
                panel.paste(image,(col*224,row*234+32));draw.text((col*224+4,row*234+246),key,fill='black')
        panel.save(run/f'paired_faces_{page+1}.png')
    write(run/'comparison_summary.json',{'metrics':current,'latest_step':latest,'best_step':best,
        'paired_id_change':{'from_step':0,'to_step':latest,'mean_delta':sum(changes.values())/24,
                            'improved_images':sum(v>0 for v in changes.values()),'per_image':changes},
        'checkpoint_provenance_verified':True,'all_training_records_finite':True,
        'exact_background_images':len(backgrounds),'training':train,
        'mean_cached_update_seconds':sum(r['train/seconds'] for r in logs)/len(logs),
        'review_source_sha256':file_hash(Path(__file__))})
    with zipfile.ZipFile(run/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in (run/'source_snapshot').rglob('*'):
            if p.is_file():archive.write(p,p.relative_to(run/'source_snapshot'))
        archive.write(Path(__file__),'scripts/review_reference_refiner.py')
    exp=experiment_factory(config,run)
    for path in [run/'comparison_metrics.png',*sorted(run.glob('paired_faces_*.png'))]:exp.log_image(str(path),name=path.stem)
    for name in ('comparison_summary.json','source_snapshot.zip','training_summary.json','probes.json','resume_check.json','execution_plan.json'):
        if (run/name).exists():exp.log_asset(str(run/name))
    exp.log_parameters({'reviewed_image_steps':steps,'best_id_step':best})
    exp.end()
    print(json.dumps({'latest_step':latest,'best_step':best,'latest_id':current[latest]['id_sim'],'improved_images':sum(v>0 for v in changes.values())}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    review(p.parse_args().run.resolve())
