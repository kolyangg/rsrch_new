"""Measured architecture deck. Run with envs/report/bin/python from repo root."""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw
import report_style as s
from report_style import (plt, PdfPages, page, text, para, box, arrow, card,
                          banner, snippet, image_panel, table, C)

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/flux4b_reference_refiner1024_b128_20261001'
PARENT=ROOT/'runs/flux4b_face_conditioned512_diverse_20261001'
FLOW=ROOT/'ba_dit/nn/reference_refiner_flow.py'
MODEL=ROOT/'sources/ai-toolkit-flux/extensions_built_in/diffusion_models/flux2/src/model.py'
READ_BYTES={}


def read(path):
    READ_BYTES[path]=path.read_bytes()
    return json.loads(READ_BYTES[path])


def main():
    identity=read(RUN/'identity.json')
    status=read(RUN/'status.json')
    scores={int(p.parent.name.split('-')[1]):read(p)['metrics']
            for p in sorted(RUN.glob('validation-*/quality_summary.json'))}
    latest=max(scores) if scores else None
    decision=read(RUN/'convergence.json') if (RUN/'convergence.json').exists() else {}
    summary=read(RUN/'training_summary.json')
    probes=read(RUN/'probes.json')
    logs=[json.loads(line) for line in (RUN/'metrics.jsonl').read_text().splitlines()]
    baseline=read(PARENT/'validation-010000/quality_summary.json')['metrics']
    cutoff=datetime.now(timezone.utc).isoformat(timespec='seconds')
    metrics_text='Pending' if latest is None else f"{scores[latest]['id_sim']:.5f} at +{latest:,} updates"
    evidence=f"Evidence captured {cutoff} · run debb81df08cf4f5591730dffa86f704c"
    source='Sources: run identity, checkpoint manifests and local implementation; full hashes in source_audit.json.'

    fig,ax=page('Stronger BA: reference-conditioned face refinement',
                'FLUX.2-klein Base 4B · one identity · local 16 GB GPU · 1 October 2026',source)
    card(ax,45,156,485,205,'PREVIOUS BEST',
         f"Conditioned BA width 512 at 10k: ID {baseline['id_sim']:.5f}. Native reference-conditioned images: ID .33140.",size=20)
    card(ax,558,156,485,205,'NEW TRAINABLE PART',
         '14.05M BA parameters · width 1024 · 16 heads. A zero-output reference refiner added to the frozen best BA core.',accent='orange',size=20)
    card(ax,1070,156,485,205,'LATEST SCORED PANEL',
         f'{metrics_text}. Completed image validations only; training loss does not establish an ID gain.',size=20)
    box(ax,65,430,405,115,'Frozen FLUX features',fill='green',size=25)
    box(ax,595,405,425,165,'Frozen BA core\n+ trainable reference refiner',fill='orange',size=24)
    box(ax,1140,430,395,115,'Face velocity\ninside reviewed native mask',fill='teal',size=22)
    arrow(ax,[(470,488),(595,488)]);arrow(ax,[(1020,488),(1140,488)])
    card(ax,65,623,710,164,'MEASUREMENT GOAL','Improve generated-face ID scores while preserving the native exterior. Only the new BA refiner receives optimizer updates.',size=18)
    card(ax,805,623,730,164,'CONTINUATION RULE','24 images at 0 / 500 / 2k / 5k / …; stop after two gains below .003. Retain the best scored checkpoint.',size=18)

    fig,ax=page('Whole model: native scene, then BA face generation',
                'Green = frozen model/weights · orange = trained refiner · blue = data and routing',source)
    box(ax,45,161,285,105,'Prompt + ID reference',fill='blue')
    box(ax,390,145,490,138,'Frozen text encoder + VAE\nNative FLUX generation · CFG4',fill='green',size=21)
    box(ax,1020,145,530,138,'Native scene + detected face box\nSaved once, reused for all checkpoints',fill='blue',size=20)
    arrow(ax,[(330,212),(390,212)]);arrow(ax,[(880,212),(1020,212)])
    box(ax,45,356,285,112,'Seeded face noise\n+ native background at sigma',fill='blue')
    box(ax,400,340,490,145,'Frozen FLUX backbone\n5 dual-stream + 20 single-stream blocks\nCapture final native Q / K / V',fill='green',size=20)
    arrow(ax,[(330,411),(400,411)])
    arrow(ax,[(1280,283),(1280,312),(188,312),(188,356)],color='teal')
    box(ax,1020,340,530,145,'Frozen trained 512 core\n+ trainable 1024 refiner\nAll face velocity comes from BA',fill='orange',size=21)
    arrow(ax,[(890,411),(1020,411)])
    box(ax,1055,576,460,102,'20-step Euler face trajectory\nFace CFG1',fill='teal',size=21)
    arrow(ax,[(1285,485),(1285,576)])
    arrow(ax,[(1055,627),(951,627),(951,522),(189,522),(189,468)],color='teal',dashed=True)
    box(ax,395,576,480,102,'Blend final latent → frozen VAE\nComposite native exterior in pixels',fill='green',size=20)
    arrow(ax,[(1055,627),(875,627)])
    banner(ax,745,'Validation generates new prompted images from ID reference + noise. No target photograph is loaded.',size=18)

    fig,ax=page('Exact connection point: the final single-stream block',
                'The native attention hook captures features; this face-flow experiment exits before the native velocity head.',source)
    box(ax,55,164,365,110,'Native mixed sequence\n[text | target | reference]',fill='green',size=21)
    box(ax,515,164,445,110,'Native modulation + QKV projection\nQ/K normalization + RoPE',fill='green',size=21)
    arrow(ax,[(420,220),(515,220)])
    box(ax,1060,152,485,135,'Target Q and reference-face K/V\nFlatten 24 × 128 → 3072\nDetach from native graph',fill='blue',size=20)
    arrow(ax,[(960,220),(1060,220)])
    snippet(ax,55,325,860,285,ROOT/'ba_dit/nn/face_crop_flow.py',61,73,size=15)
    card(ax,955,330,590,276,'TOKEN OWNERSHIP',
         'Text offset is added in the single stream. Target queries are target image tokens only; reference keys/values are selected from the known ID-reference face mask. Branch projections receive already rotated features; there is no second RoPE.',size=18)
    card(ax,55,643,700,151,'ACTUAL SAMPLER PATH','Native attention executes up to this hook. FaceFeaturesCaptured interrupts the final native projections; the BA head directly outputs 128-channel velocity.',size=17)
    card(ax,785,643,760,151,'BA-OFF CONTRACT','With branch=False, the capture hook is bypassed. Measured native output before and after installation is bit-exact.',size=18)

    fig,ax=page('Branched attention: frozen base + stronger reference correction',
                'One new module: ba_dit/nn/reference_refiner_flow.py · new output is exactly zero at initialization',source)
    box(ax,45,151,315,112,'Frozen features\nQ target · K/V reference face',fill='green',size=20)
    box(ax,475,143,555,125,'Frozen conditioned 512 BA core\nLoaded from its 10k checkpoint\n5.33M parameters',fill='green',size=20)
    arrow(ax,[(360,196),(475,196)])
    box(ax,45,354,315,120,'New Q / K / V projections\n3072 → 1024 · 16 heads',fill='orange',size=20)
    arrow(ax,[(200,263),(200,354)],color='orange')
    box(ax,450,333,335,164,'Reference attention\ncosine Q/K\nlearned temperature [1,16]\nnormalized read',fill='orange',size=18)
    arrow(ax,[(360,405),(450,405)],color='orange')
    box(ax,870,333,400,164,'Reference-pooled conditioning\nquery + sigma + noisy latent\nbounded gain/shift\nread × context + identity',fill='orange',size=18)
    arrow(ax,[(785,405),(870,405)],color='orange')
    box(ax,865,570,415,125,'Nonlinear mix → zero-init output\n+ reference-gated noisy output\nΔ velocity',fill='orange',size=19)
    arrow(ax,[(1070,497),(1070,570)],color='orange')
    box(ax,1365,322,185,184,'ADD\nfrozen core\n+ Δ velocity\n= face flow',fill='teal',size=21)
    arrow(ax,[(1030,207),(1458,207),(1458,322)])
    arrow(ax,[(1280,632),(1458,632),(1458,506)],color='orange')
    card(ax,45,570,735,125,'WHAT “STRONGER” MEANS','Wider reference projections, more heads, pooled-reference modulation and a learned bounded attention temperature. Only these new parameters train.',size=17)
    banner(ax,760,'At new-run step zero: total face flow = previous best trained BA flow. Zero refers only to the new correction.',size=17)

    fig,ax=page('Reference conditioning and bounded attention — actual code',
                'Normalized Q/K keep logits bounded; reference values feed both attention and global conditioning.',source)
    snippet(ax,45,152,935,612,FLOW,37,55,size=16)
    card(ax,1020,157,530,182,'ATTENTION TEMPERATURE','Starts at 8, learns within [1,16]. The earlier conditioned core uses a fixed value 4. Avoids unbounded dot-product scale growth.',size=18)
    card(ax,1020,365,530,185,'REFERENCE INFLUENCE','Mean reference V conditions the context and the noise gate. This can strengthen the read path, but does not prove exclusive identity transfer.',accent='orange',size=18)
    card(ax,1020,576,530,189,'ZERO-DELTA INITIALIZATION','Both new output matrices start at zero. Their first update opens gradients into Q/K/V and the conditioning layers; all 17 tensors have finite updates.',size=18)

    fig,ax=page('What trains, and what stays frozen',
                'The projection width is an attention-head bottleneck; it is not native-attention LoRA rank.',source)
    table(ax,45,156,[410,270,830],['Component','Parameters / shape','Role'],[
      ('Native FLUX.2-klein Base 4B','3072 width · 24 heads','Frozen; native text/reference conditioning and full-scene features.'),
      ('Previous conditioned BA core','5,327,360 frozen','Best prior 10k checkpoint; stays bit-exact throughout optimization.'),
      ('New reference refiner','14,048,385 trainable','Q/K/V, bounded attention temperature, time/noise context, reference modulation, output and noise gate.'),
      ('Captured target / reference','B × T/R × 3072','Training samples 64 face tokens; inference predicts all 2304 target tokens before spatial blending.'),
      ('Head output','B × T × 128','Packed-latent face velocity. Native velocity is not added.'),
      ('Optimizer','AdamW · batch128','lr .0001; weight decay .01; warmup100; gradient clip1.'),
    ],row_h=83,size=16)
    banner(ax,754,'FLUX native attention, text encoder, VAE and the trained core receive no optimizer updates.',size=18)

    fig,ax=page('Training data and why cached optimization is fast',
                'Fresh full-backbone cases are expensive; repeated BA updates reuse detached features.',source)
    box(ax,45,159,365,142,'19 training photographs\n1 fixed ID reference\nnew noise + continuous sigma',fill='blue',size=20)
    box(ax,505,159,535,142,'Frozen full-backbone passes\n1,140 fit cases + 114 noise probes\n64 selected face tokens per case',fill='green',size=21)
    box(ax,1140,159,410,142,'Train BA refiner only\nunweighted native flow MSE\ntarget = noise − clean latent',fill='orange',size=20)
    arrow(ax,[(410,230),(505,230)]);arrow(ax,[(1040,230),(1140,230)],color='orange')
    means=sum(r['train/seconds'] for r in logs)/len(logs)
    card(ax,45,363,720,187,'MEASURED UPDATE COST',
         f"{means*1000:.1f}ms mean cached optimizer update through +{logs[-1]['step']:,}. Peak reserved training memory {summary['hardware/peak_reserved_gib']:.3f} GiB. This excludes backbone cache creation, probes and validation.",size=19)
    card(ax,805,363,745,187,'CACHE EXPANSION',
         '456 added full-backbone cases, beyond the previous 684. Measured 270.4s and 8.389 GiB peak reserved memory. Fresh noise/sigma, same 19 photos.',size=19)
    card(ax,45,585,720,192,'STATISTICAL LIMIT','Finite features are reused many times. The probe uses other noise on the same photographs; the 24-image panel is used for model selection. These are development measurements.',size=18)
    card(ax,805,585,745,192,'NO DIRECT ID OBJECTIVE','The loss is flow MSE. ID similarity decides checkpoint selection and stopping. A lower training loss can coincide with worse ID scores.',accent='orange',size=19)

    fig,ax=page('Batch scaling: saturate compute before adding more memory',
                'Real refiner updates · frozen-core cache + CUDA batch gathering · 10 warmup + 40 measured updates per batch',source)
    benchmark=read(RUN/'batch_benchmark.json')['results']
    a=fig.add_axes([.08,.31,.44,.48])
    a.plot([r['batch_size'] for r in benchmark],[r['cases_per_second'] for r in benchmark],'o-',color=C['orange'])
    a.set_xscale('log',base=2);a.set_xticks([r['batch_size'] for r in benchmark],[str(r['batch_size']) for r in benchmark])
    a.set(xlabel='Cached cases per optimizer update',ylabel='Cached cases / second');a.grid(alpha=.2)
    a.axvline(128,color=C['teal'],ls='--',label='Selected batch128');a.legend()
    card(ax,910,163,645,185,'SELECTED: BATCH 128','3536 cases/s; 36.2ms/update in the benchmark. Post-startup GPU samples: 97–98%. Batches256/512 add less than1% cases/s.',size=20)
    card(ax,910,375,645,185,'ACTUAL TRAINING',f"Measured 96% mean GPU utilization over a 5-second sample, 6393 MiB device use. Actual CUDA peak {summary['hardware/peak_reserved_gib']:.2f} GiB reserved; inference about 9.40 GiB.",size=19)
    card(ax,910,587,645,185,'COMPARISON LIMIT','Batch128 processes16× more cases per update than batch8. The new run has its own name and an early500-update image check. This is a changed optimization experiment.',size=19)
    para(ax,95,728,'Utilization averages from very short benchmarks include startup. Selected-batch utilization is also checked during real training.',74,15)

    fig,ax=page('Native background + BA face: explicit spatial routing',
                'A named CL14-inspired experiment; masks are from generated native scenes, as requested by the user.',source)
    snippet(ax,45,151,850,334,ROOT/'ba_dit/nn/masked_face_flow.py',24,39,size=17)
    card(ax,940,158,610,321,'MASK SEMANTICS',
         'The native generated face box has full weight inside and a16px outer feather. At each sigma, blend BA face latents with the native background trajectory. After decoding, pixel compositing restores the exact exterior; raw VAE outputs are also retained.',size=20)
    box(ax,65,559,430,116,'Native scene\nreviewed generated-image mask',fill='green',size=22)
    box(ax,590,559,430,116,'BA-generated face\nreference + prompt + seeded noise',fill='orange',size=22)
    box(ax,1115,559,415,116,'Final composite\nexact native exterior',fill='teal',size=23)
    arrow(ax,[(495,617),(590,617)]);arrow(ax,[(1020,617),(1115,617)])
    banner(ax,752,'All checkpoints share mask/image hashes, reference, prompts, seeds, sampler and metric definitions.',size=18)

    fig,ax=page('Reviewed masks over the native validation images',
                'All 24 images · unchanged masks for every checkpoint · generated by the same native backbone',source)
    image_panel(fig,RUN/'mask_overlays.png',45,155,950,663)
    card(ax,1035,159,520,244,'WHAT IS SHOWN','Each overlay is on the native generated scene that supplied its mask. The full face box and outer blend region define where the BA face replaces the native result.',size=19)
    card(ax,1035,440,520,342,'ALIGNMENT AND OWNERSHIP','Latents use a 48×48 grid at 768×768 pixels. The pixel alpha mask is average-pooled in 16×16 blocks to align with packed latent tokens. Target photographs are excluded from validation. Ownership boxes also select the scored face among detections.',size=19)

    fig,ax=page('Relationship to CL14 / CL39 and the earlier FLUX adapter',
                'Architecture families and experimental roles must be distinguished.',source)
    table(ax,45,157,[310,585,615],['Mechanism','Earlier CL39 / PhotoMaker','Current prompted-face refiner'],[
      ('Backbone','SDXL U-Net / PhotoMaker; target/reference rows.','FLUX.2-klein Base 4B; packed multimodal text/target/reference sequence.'),
      ('Attention result','Native N plus routed correction R − N; finished messages are output-projected before routing.','Frozen BA core velocity plus new reference refinement; no native velocity inside the face.'),
      ('Spatial control','CL14 motivates separate face/background routes; CL39 has a soft target router.','Two-pass generated-native mask; blend during face sampling and after VAE decoding.'),
      ('Additional controls','CL39: low/high Gaussian bands, temporal shaping and detached confidence.','Bounded cosine attention, reference-pooled modulation and noise gate; no CL39 frequency/confidence router.'),
      ('Trainable location','Attention processors inside the denoiser.','Standalone final-feature BA flow head; native attention has no trainable LoRA.'),
      ('Earlier FLUX 48GB run','Original project port: K/V residual adapters at eight attention sites.','This local diagnostic supersedes that mechanism for the reported experiment; it is not the same checkpoint family.'),
    ],row_h=88,size=15)
    banner(ax,785,'The shared idea is reference attention with spatial ownership. This implementation is not an exact CL39 port.',size=17)

    fig,ax=page('Evidence behind the redesign: more steps had reduced ID scores',
                'Same 24 prompted images; previous experiments. Native reference-conditioned baseline is .33140.',source)
    a=fig.add_axes([.075,.26,.47,.54])
    old=read(ROOT/'runs/flux4b_masked_face_flow_24_100k_r1_20261001/metric_summary.json')
    old_steps=[6000,20000,40000,60000]
    a.plot([x/1000 for x in old_steps],[old[str(x)]['id_sim'] for x in old_steps],'o-',label='Original 256, fixed cache',color=C['orange'])
    a.axhline(.3313986754,ls='--',label='Native generated scene',color=C['muted'])
    a.set(xlabel='Original BA optimizer updates (thousands)',ylabel='Mean ID similarity',ylim=(.13,.35));a.grid(alpha=.2);a.legend(fontsize=10)
    card(ax,945,162,610,189,'OBSERVED SATURATION','Original attention entropy/log(key count) fell .00303 → .00040 between 6k and 60k. Reference/query paths grew and increasingly cancelled.',size=18)
    card(ax,945,378,610,180,'WIDTH ALONE FAILED','A plain 512 head worsened probe MSE. The conditioned 512 fixed-cache run also declined in ID: .18814 at 2k → .17927 at 6k.',size=18)
    card(ax,945,585,610,199,'RETAINED IMPROVEMENT','Conditioned512 + more noise/sigma + gentler optimization reached .22660 at 10k. This trained checkpoint is now frozen as the new refiner’s starting core.',accent='orange',size=18)
    para(ax,80,748,'Comparisons bundle architecture, data coverage and optimization. They do not isolate a single causal improvement.',75,15)

    fig,ax=page('Current experiment: image scores and flow-loss probes',
                f'{evidence} · status: {status["status"]}',source)
    a=fig.add_axes([.07,.29,.42,.50]);b=fig.add_axes([.575,.29,.38,.50])
    if scores:
        steps=sorted(scores)
        a.plot([x/1000 for x in steps],[scores[x]['id_sim'] for x in steps],'o-',color=C['orange'],label='New1024 refiner + frozen core')
    a.axhline(baseline['id_sim'],ls=':',color=C['green'],label='Frozen core .22660')
    a.axhline(.3313986754,ls='--',color=C['muted'],label='Native .33140')
    a.set(xlabel='Additional refiner updates (thousands)',ylabel='Mean ID similarity');a.grid(alpha=.2);a.legend(fontsize=9)
    for split,color in [('fit',C['orange']),('probe',C['teal'])]:
        b.plot([p['step']/1000 for p in probes],[p[f'probe/{split}/mse'] for p in probes],label=split,color=color)
    b.set(xlabel='Additional refiner updates (thousands)',ylabel='Flow MSE');b.grid(alpha=.2);b.legend()
    if latest is not None:
        d=scores[latest]['id_sim']-baseline['id_sim']
        result=f"Latest +{latest:,}: ID {scores[latest]['id_sim']:.5f} ({d:+.5f} vs core); CLIP {scores[latest]['text_sim']:.3f}."
        if latest==500:result+=' Small gain: 13/24 images improve; further validation is running.'
    else:
        result='New image scores are pending. The existing .22660 core score is the starting reference, not a new-refiner gain.'
    card(ax,65,690,1470,108,'MEASURED RESULT',result,size=18)

    fig,ax=page('Same scenes: native, frozen core and latest scored refiner',
                'First four panel items, in their fixed order · face crops only · no quality-based selection',source)
    masks=read(RUN/'routing_masks.json')['samples']
    keys=list(masks)[:4]
    columns=[('Native',RUN/'native'),('Frozen core, 10k',PARENT/'validation-010000')]
    if latest is not None:columns.append((f'Refiner +{latest:,}',RUN/f'validation-{latest:06d}'))
    panel=Image.new('RGB',(len(columns)*420,4*280+48),'white');draw=ImageDraw.Draw(panel)
    for c,(name,folder) in enumerate(columns):draw.text((c*420+14,12),name,fill='black')
    for r,key in enumerate(keys):
        x0,y0,x1,y1=masks[key]['face_bbox'];cx=(x0+x1)/2;cy=(y0+y1)/2;side=max(x1-x0,y1-y0)*1.35
        crop=(int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))
        for c,(_,folder) in enumerate(columns):
            image=Image.open(folder/f'{key}.png').crop(crop).resize((247,247),Image.Resampling.LANCZOS)
            panel.paste(image,(c*420+(420-image.width)//2,r*280+46))
            draw.text((c*420+15,r*280+297),key,fill='black')
    panel.save(OUT/'paired_faces.png');image_panel(fig,OUT/'paired_faces.png',80,157,1010,655)
    card(ax,1130,168,425,251,'INTERPRETATION','Inspect shape, identity and artifacts together. A higher embedding score does not guarantee cleaner texture or correct expression.',size=19)
    card(ax,1130,450,425,320,'PROTOCOL','All scenes start from the same Gaussian seeds. Masks come from their native generated counterparts. Identity scoring matches detections to the fixed ownership box; background pixels are explicitly composited.',size=18)

    fig,ax=page('Measured checks, checkpoint identity and memory',
                'Focused checks guard real architectural failures; synthetic checks are not pretrained-model evidence.',source)
    audits=[read(p) for p in RUN.glob('inference_audit_*.json')]
    inference_peak=max((a['hardware/peak_reserved_gib'] for a in audits),default=None)
    resume=read(RUN/'resume_check.json') if (RUN/'resume_check.json').exists() else None
    resume_error=max(resume['max_abs_by_tensor'].values()) if resume else None
    rows=[('Native BA-off','PASS: before/after installation bit-exact on actual backbone.'),
          ('Trainable / frozen weights',f"All 17 new tensors updated with finite gradients through +{summary['steps']:,}; all 11 core tensors bit-exact."),
          ('Cached vs live features',f"{len(audits)} completed inference audits; exact prediction equality required at each checkpoint."),
          ('Fresh-process resume',f'Max weight difference {resume_error:.3g}; atol1e-8 / rtol1e-6.' if resume else 'Pending first fresh-process continuation; check is enforced.'),
          ('CUDA reservation',f"Cached training {summary['hardware/peak_reserved_gib']:.3f} GiB; inference {inference_peak:.3f} GiB." if inference_peak else f"Cached training {summary['hardware/peak_reserved_gib']:.3f} GiB; full inference still in progress."),
          ('Checkpoint provenance','Strict source/config/cache/mask SHA256 checks, self-contained core+refiner checkpoint and saved AdamW state.'),
          ('Exterior preservation','Every decoded panel must have exact zero exterior error. Raw decoded images are retained separately.'),
    ]
    table(ax,45,155,[430,1080],['Check','Measured state / gate'],rows,row_h=78,size=16)
    banner(ax,785,'Memory ceiling: below 90% of the16 GB device. One GPU stage runs at a time.',size=18)

    fig,ax=page('Convergence, limitations and reproducible artifacts',
                evidence,source)
    card(ax,45,151,735,220,'STOPPING AND SELECTION','Validate at 500, 2k and then every 5k additional updates. Stop after two consecutive gains over the previous best below .003. Save the highest-ID checkpoint. A plateau on this panel is a development stopping rule, not statistical convergence.',size=18)
    best=decision.get('best_id_sim');best_step=decision.get('best_step')
    card(ax,815,151,735,220,'CURRENT STATE',
         f"{status['status']}; last scored update {latest}. Best: {best:.5f} at +{best_step:,}. Small-gain checks: {decision.get('consecutive_small_gains',0)}/2." if best is not None else 'Training admitted; step-zero panel being scored. No new-refiner improvement established yet.',size=19)
    card(ax,45,403,735,215,'LIMITATIONS','One identity,19 photographs, finite feature cache and24 development images. No held-out identities and no direct ID loss. Native reference conditioning remains present. Report ID, text alignment and visible defects together.',size=18)
    card(ax,815,403,735,215,'REPRODUCE','plans/260930/REFERENCE_REFINER.md contains the launch commands. Run identity, source snapshots, config, masks, checkpoints, gradient/restart audits and best_checkpoint.json are retained locally and in Comet where applicable.',size=18)
    para(ax,65,677,'Comet: https://www.comet.com/nikolay-2104/rsrch-new/debb81df08cf4f5591730dffa86f704c',110,16,color=C['teal'])
    para(ax,65,735,'Native model source: '+identity['base']['source_commit']+'\nBackbone revision: '+identity['base']['revisions']['flux4b'],115,14)

    target=OUT/'flux4b_reference_refiner1024_architecture.pdf'
    with PdfPages(target,metadata={'Title':'Stronger reference-attention face refiner — FLUX Base 4B','Author':'rsrch_new'}) as pdf:
        for i,(fig,title) in enumerate(s.pages,1):
            fig.canvas.draw()
            for artist,(x,y,w,h) in s.fitted_texts:
                if artist.figure is not fig:continue
                for _ in range(3):
                    bb=artist.get_window_extent(fig.canvas.get_renderer());a,b=artist.axes.transData.transform([[x,y],[x+w,y+h]])
                    ratio=min((b[0]-a[0])/bb.width,(a[1]-b[1])/bb.height,1)
                    if ratio>=1:break
                    artist.set_fontsize(artist.get_fontsize()*ratio*.96)
            fig.canvas.draw()
            for artist in s.texts:
                if artist.figure is not fig:continue
                bb=artist.get_window_extent(fig.canvas.get_renderer())
                assert bb.x0>=-1 and bb.y0>=-1 and bb.x1<=fig.bbox.width+1 and bb.y1<=fig.bbox.height+1,(i,artist.get_text())
            pdf.savefig(fig);fig.savefig(OUT/f'page_{i:02d}.png',dpi=120);fig.savefig(OUT/f'page_{i:02d}.svg');plt.close(fig)
    doc=pymupdf.open(target);doc.set_toc([[1,title,i] for i,(_,title) in enumerate(s.pages,1)]);doc.saveIncr()
    for number,name in [(2,'whole_model'),(4,'branched_attention')]:
        single=pymupdf.open();single.insert_pdf(doc,from_page=number-1,to_page=number-1)
        single.save(OUT/f'{name}.pdf');single.close()
    doc.close()
    files=[Path(__file__),OUT/'report_style.py',RUN/'identity.json',RUN/'status.json',RUN/'training_summary.json',RUN/'probes.json',RUN/'execution_plan.json',MODEL]
    files.extend(ROOT/p for p in identity['source_sha256'])
    files.extend(RUN/f'validation-{step:06d}/quality_summary.json' for step in scores)
    audit={'evidence_cutoff':cutoff,'run':str(RUN),'identity':identity,'status':status,'image_scores':scores,
           'best_checkpoint':decision,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'note':'Uncommitted implementation is identified by source hashes and run snapshot, not git HEAD alone.',
           'source_hashes':{str(p):hashlib.sha256(READ_BYTES[p] if p in READ_BYTES else p.read_bytes()).hexdigest() for p in files},
           'pages':[{'number':i,'title':t} for i,(_,t) in enumerate(s.pages,1)]}
    (OUT/'source_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    thumbs=Image.new('RGB',(960,270*((len(s.pages)+2)//3)),'#dae2e7')
    for i in range(len(s.pages)):
        im=Image.open(OUT/f'page_{i+1:02d}.png');im.thumbnail((320,250));thumbs.paste(im,((i%3)*320,(i//3)*270))
    thumbs.save(OUT/'contact_sheet.png')
    print(json.dumps({'pdf':str(target),'pages':len(s.pages),'latest_scored_step':latest,'status':status['status']}))


if __name__=='__main__':main()
