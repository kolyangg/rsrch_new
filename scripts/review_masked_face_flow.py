"""Audit completed prompted face-generation runs and make compact comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from ba_dit.config import load_config
from ba_dit.data.manifest import file_hash
from ba_dit.nn.masked_face_flow import face_alpha
from scripts.face_crop_flow import experiment, write


def main(run):
    config=load_config(run/'resolved_config.yaml')
    identity=json.loads((run/'identity.json').read_text())
    masks=json.loads((run/'routing_masks.json').read_text())['samples']
    assert len(masks)==24
    assert all(file_hash(run/'source_snapshot'/p)==h for p,h in identity['source_sha256'].items())
    assert file_hash(run/'routing_masks.json')==identity['routing_masks_sha256']
    initial=load_file(run/'initial.safetensors')
    final=load_file(run/'checkpoint-002000/branch.safetensors')
    assert sum(p.numel() for p in final.values())==identity['trainable_parameters']
    update={k:float((final[k]-initial[k]).norm()) for k in final}
    assert all(update.values()) and all(torch.isfinite(v).all() for v in final.values())
    logs=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in logs]==list(range(1,2001))
    assert all(np.isfinite(r['train/loss']) and np.isfinite(r['train/gradient_norm']) for r in logs)
    checks=[]
    for step in (0,1000,2000):
        folder=run/f'validation-{step:06d}'
        report=json.loads((folder/'validation.json').read_text())
        assert len(report['samples'])==24 and all(not r['target_photo_loaded'] for r in report['samples'])
        path=run/f'checkpoint-{step:06d}/branch.safetensors' if step else run/'initial.safetensors'
        assert file_hash(path)==report['checkpoint_sha256']
        for key,record in masks.items():
            native=np.array(Image.open(run/'native'/f'{key}.png')).astype(float)
            image=np.array(Image.open(folder/f'{key}.png')).astype(float)
            raw=np.array(Image.open(folder/f'{key}_raw.png')).astype(float)
            support=face_alpha((768,768),record['face_bbox']).squeeze().numpy()>0
            error=float(np.abs(image-native)[~support].max())
            assert error==0
            checks.append({'step':step,'sample_id':key,'outside_max_abs':error,
                           'raw_outside_mean_abs':float(np.abs(raw-native)[~support].mean()),
                           'inside_mean_abs':float(np.abs(image-native)[support].mean())})
    audit={'steps':2000,'all_2000_losses_gradients_finite':True,'parameter_update_l2':update,
           'optimizer_seconds':sum(r['train/seconds'] for r in logs),
           'peak_train_reserved_gib':max(r['hardware/peak_reserved_gib'] for r in logs),
           'peak_infer_reserved_gib':max(json.loads(p.read_text())['hardware/peak_reserved_gib'] for p in run.glob('inference_audit_*.json')),
           'resume':json.loads((run/'resume_check.json').read_text()),
           'native_off_parity':json.loads((run/'native_off_parity.json').read_text()),
           'images_checked':len(checks),'maximum_background_pixel_error':max(r['outside_max_abs'] for r in checks)}
    write(run/'final_audit.json',audit); write(run/'raw_and_final_pixel_audit.json',checks)
    sheet=Image.new('RGB',(4*256,4*280+30),'white'); draw=ImageDraw.Draw(sheet)
    selected=['oneid_00','oneid_01','oneid_03','oneid_08']
    for col,label in enumerate(('Native','BA 0','BA 1000','BA 2000')): draw.text((col*256+5,5),label,fill='black')
    for i,key in enumerate(selected):
        for col,folder in enumerate((run/'native',*(run/f'validation-{s:06d}' for s in (0,1000,2000)))):
            image=Image.open(folder/f'{key}.png'); image.thumbnail((256,256))
            sheet.paste(image,(col*256,i*280+30))
        draw.text((4,i*280+289),key,fill='black')
    sheet.save(run/'overview.png')
    for page in range(2):
        detail=Image.new('RGB',(4*192,12*216+26),'white'); draw=ImageDraw.Draw(detail)
        for col,label in enumerate(('Native','BA 0','BA 1000','BA 2000')): draw.text((col*192+4,4),label,fill='black')
        for i,(key,record) in enumerate(list(masks.items())[page*12:page*12+12]):
            box=record['face_bbox']; x0,y0,x1,y1=box
            side=max(x1-x0,y1-y0)+64; cx=(x0+x1)/2; cy=(y0+y1)/2
            region=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
            for col,folder in enumerate((run/'native',*(run/f'validation-{s:06d}' for s in (0,1000,2000)))):
                image=Image.open(folder/f'{key}.png').crop(region).resize((192,192))
                detail.paste(image,(col*192,i*216+26))
            draw.text((4,i*216+220),key,fill='black')
        detail.save(run/f'face_details_{page+1}.png')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probes=json.loads((run/'probes.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    steps=[p['step'] for p in probes]
    axes[0].plot(steps,[p['probe/fit/mse'] for p in probes],label='Fit noise')
    axes[0].plot(steps,[p['probe/probe/mse'] for p in probes],label='Separate noise')
    axes[0].axhline(probes[0]['probe/probe/zero_flow_mse'],ls=':',color='gray',label='Zero BA')
    axes[1].plot(steps,[p['probe/probe/mse'] for p in probes],label='Full BA')
    axes[1].plot(steps,[p['probe/probe/reference_read_off_mse'] for p in probes],label='Reference read disabled')
    for ax in axes:
        ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Face-token flow MSE'); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run/'learning_curves.png',dpi=150); plt.close(fig)
    exp=experiment(config,run)
    exp.log_metrics({'audit/maximum_background_pixel_error':0,'audit/all_parameters_updated':1,
                     'hardware/peak_inference_reserved_gib':audit['peak_infer_reserved_gib']},step=2000)
    for path in [run/'final_audit.json',run/'raw_and_final_pixel_audit.json',run/'native_off_parity.json',
                 *run.glob('inference_audit_*.json'),*run.glob('source_snapshot/scripts/*.py'),*run.glob('source_snapshot/ba_dit/nn/*.py')]:
        exp.log_asset(str(path),file_name=str(path.relative_to(run)).replace('/','_'))
    for name in ('overview.png','face_details_1.png','face_details_2.png','learning_curves.png'):
        exp.log_image(str(run/name),name=Path(name).stem,step=2000)
        exp.log_asset(str(run/name))
    exp.end()
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    main(parser.parse_args().run_dir.resolve())
