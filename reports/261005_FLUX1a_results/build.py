"""Build measured FLUX1a results report from local and Comet-verified evidence."""
from pathlib import Path
import csv,json,hashlib,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageOps,ImageDraw,ImageFont
import pymupdf
O=Path(__file__).resolve().parent;R=O.parents[1];E=O/'evidence';RUN=R/'runs/FLUX1a_vast9B_20261004'
def read(p):return json.loads(p.read_text())
def scores(p):return {r['sample_id']:{k:v if k in ('sample_id','identity_id') else float(v) for k,v in r.items()} for r in csv.DictReader(p.open())}
S={s:scores(E/f'{s}_quality_per_image.csv') for s in (0,2000,4000,6000)}
S['native']=scores(R/'reports/261004_FLUX1_review/evidence/native96_quality_per_image.csv')
ids=list(S[6000]);owners=sorted({x['identity_id'] for x in S[6000].values()})
def mean(s,k='id_sim'):return float(np.mean([r[k] for r in S[s].values()]))
def paired(a,b):
 d=np.array([S[b][i]['id_sim']-S[a][i]['id_sim'] for i in ids]);g=np.array([np.mean([d[j] for j,i in enumerate(ids) if S[b][i]['identity_id']==owner]) for owner in owners]);rng=np.random.default_rng(142)
 ci=np.quantile(g[rng.integers(0,len(g),(100000,len(g)))].mean(1),[.025,.975])
 return {'delta':float(d.mean()),'wins':int((d>0).sum()),'identity_wins':int((g>0).sum()),'ci95':ci.tolist()}
A={'metrics':{str(s):{'id_sim':mean(s),'text_sim':mean(s,'text_sim')} for s in S},'paired':{f'{b}-{a}':paired(a,b) for a,b in [(2000,4000),(4000,6000),('native',4000),('native',6000)]},'identity':{owner:{str(s):float(np.mean([v['id_sim'] for v in S[s].values() if v['identity_id']==owner])) for s in S} for owner in owners}}
(O/'analysis.json').write_text(json.dumps(A,indent=2))
print(json.dumps(A,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(9.8,3.0),layout='constrained')
for k,label,j in [('id_sim','Owner-matched identity similarity',0),('text_sim','CLIP text alignment',1)]:
 ax[j].plot([2000,4000,6000],[mean(s,k) for s in (2000,4000,6000)],'o-',color='#087f8c',lw=2)
 ax[j].axhline(mean('native',k),color='#778899',ls='--',label='Native baseline')
 for s in (2000,4000,6000):ax[j].annotate(f'{mean(s,k):.4f}' if j==0 else f'{mean(s,k):.3f}',(s,mean(s,k)),xytext=(0,9),textcoords='offset points',ha='center',fontsize=9)
 ax[j].set_title(label);ax[j].set_xticks([2000,4000,6000]);ax[j].set_xlabel('Optimizer updates');ax[j].margins(y=.35);ax[j].grid(alpha=.18);ax[j].legend(fontsize=8,loc='lower left')
fig.savefig(O/'trajectory.png',dpi=180);fig.savefig(O/'trajectory.pdf');plt.close(fig)
rows=read(O/'selection.json');masks=read(E/'None_routing_masks.json')['samples']
native=R/'runs/FLUX1_vast9B_2000_fixed96_20261004/native'
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
def crop(im,box):
 x0,y0,x1,y1=box;cx=(x0+x1)/2;cy=(y0+y1)/2;w=max(x1-x0,y1-y0)*1.25
 return im.crop((max(0,int(cx-w/2)),max(0,int(cy-w/2)),min(im.width,int(cx+w/2)),min(im.height,int(cy+w/2))))
for row in rows:
 sid=row['sample_id'];p=native/f'{sid}.png';assert hashlib.sha256(p.read_bytes()).hexdigest()==masks[sid]['baseline_image_sha256']
 ims={'native':Image.open(p).convert('RGB'),**{s:Image.open(E/f'{s}_{sid}.png').convert('RGB') for s in (2000,4000,6000)}}
 ref=Image.open(R/'data/validation'/row['reference_image']).convert('RGB')
 canvas=Image.new('RGB',(1400,665),'white');d=ImageDraw.Draw(canvas)
 top=[('Reference',ref),('Native scene',ims['native']),('FLUX1a 4k',ims[4000]),('FLUX1a 6k',ims[6000])]
 for c,(label,im) in enumerate(top):
  d.text((c*350+8,4),label,font=font,fill='#163643');canvas.paste(ImageOps.contain(im,(338,330)),(c*350+6,33))
 for c,s in enumerate(('native',2000,4000,6000)):
  label=f'{"Native" if s=="native" else str(s//1000)+"k"} face | ID {S[s][sid]["id_sim"]:.3f}'
  d.text((c*350+8,373),label,font=small,fill='#163643');im=crop(ims[s],masks[sid]['face_bbox']);im=ImageOps.contain(im,(338,258));canvas.paste(im,(c*350+6,400))
 canvas.save(O/f'sample_{sid}.jpg',quality=93)
P=[]
def table(headers,rows):return '<table><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in rows)+'</table>'
def page(title,text):P.append(f'<h1>{title}</h1>'+text)
def fig(name,width=510):return f'<img src="{name}" width="{width}">'
def f(x):return f'{x:.4f}'
def ci(q):return f'[{q["ci95"][0]:+.4f}, {q["ci95"][1]:+.4f}]'
q=A['paired']['6000-4000'];n=A['paired']['4000-native']
page('FLUX1a: completed results',f'''<p class="kicker">Vast9B · fixed96 · 5 October 2026</p>
<div class="callout"><b>Use checkpoint 4k for identity-focused comparisons.</b> FLUX1a learns strongly from its untrained branch state, peaks at 4k on the measured identity metric, then declines at 6k. The 6k checkpoint slightly improves CLIP. This is one training run on eight held-out identities.</div>
{table(['Model / checkpoint','Owner ID ↑','CLIP ↑'],[['Native baseline',f(mean('native')),f'{mean("native","text_sim"):.3f}']]+[[f'FLUX1a {s:,}',f(mean(s)),f'{mean(s,"text_sim"):.3f}'] for s in (0,2000,4000,6000)])}
{fig('trajectory.png')}
<p>At 4k, owner-ID exceeds native by <b>{n['delta']:+.4f}</b>; {n['wins']}/96 images improve. From 4k to 6k, owner-ID changes by <b>{q['delta']:+.4f}</b> ({100*q['delta']/mean(4000):.1f}% relative), with {q['wins']}/96 improving. Neither difference alone establishes generalization beyond this panel; paired identity-level uncertainty is on page 3.</p>
<p><b>Completion:</b> 6,000 updates finished; all 96 images and metric uploads verified at 2k, 4k and 6k. Full 4k/6k checkpoints are hash-verified locally. GB10 is stopped, with files preserved; monitoring is removed.</p>
<p class="small">Scope: FLUX1a is this project's experiment name for FLUX.2-klein Base 9B, not BFL FLUX.1. The HSE 4B run is separate. Sources: immutable Comet run, checkpoint manifests, publication receipts and saved per-image scores.</p>''')
page('Architecture and learning signal','''<p><b>Backbone:</b> frozen FLUX.2-klein Base 9B, width 4096, 32 heads, 8 double-stream and 24 single-stream blocks. Text encoder: Qwen3-8B. VAE and original denoiser weights remain frozen. Only the BA projections train.</p>'''+table(['Component','FLUX1a implementation'],[
['BA sites (zero based)','Double blocks 2, 4, 6, 7; single blocks 4, 10, 16, 22.'],
['Trainable capacity','Rank128 / alpha128 Q, K, V and output deltas at 8 sites: 33,554,432 parameters in 64 tensors. No native-weight LoRA training.'],
['Reference features','Separate frozen, image-only pass through the backbone at the current noise level. Zero text or target tokens. Gather original-reference hidden states and pre-normalization K/V at branch sites.'],
['Attention read','Target-side native Q plus learned ΔQ; isolated-reference K/V plus learned ΔK/ΔV. Retain native Q/K normalization and original target/reference RoPE coordinates. Select at most 512 reference-face keys.'],
['Routing','Binary ownership of every latent token touched by the face mask. Owned query rows read reference attention; other rows retain native attention. Learned output delta enters before the native modulation gate.'],
['Objective','Face-masked native velocity MSE + 0.05 × identity cosine loss, enabled for valid training labels at σ ≤ 0.5. Identity uses the one-step clean estimate, frozen VAE and differentiable frozen ArcFace.']])+'''
<div class="callout"><b>Information flow:</b><br>Reference image → frozen image-only pass → isolated K/V bank<br>Current target scene → native query features → learned reference attention read<br>Face-owned token messages → unchanged native projection, residual and MLP computation</div>
<p><b>What changed from FLUX1:</b> the old branch read reference states from joint target/text/reference attention, permitting target information to enter its reference features. FLUX1a removes that direct path using a separate bank; binary ownership also removes fractional native-attention blending at owned latent tokens. Identity supervision is added. These changes were combined, so this run does not isolate their individual effects.</p>
<p><b>What isolation does not mean:</b> native queries, residuals, unmodified blocks and native reference conditioning still influence the target. Native backgrounds are injected by the spatial sampling contract. Therefore native-looking faces do not by themselves prove a reference-bank leak. No completed pretrained donor-reference causal ablation is included here.</p>
<p class="small">The native attention operation is preserved and BA-off parity passed; BA-on intentionally routes face-owned messages to the reference read. It is not identical to native attention on those rows. Zero-initialized deltas still change the BA-on path, explaining the weak step-0 baseline.</p>''')
page('Paired results and interpretation',table(['Comparison: later − earlier','Mean ΔID','Image wins','Identity wins','95% grouped interval'],[[label,f"{z['delta']:+.4f}",f"{z['wins']}/96",f"{z['identity_wins']}/8",ci(z)] for label,z in [(k,A['paired'][k]) for k in ('4000-2000','6000-4000','4000-native','6000-native')]])+'''
<p class="small">Intervals resample eight identity groups with replacement (100,000 draws, seed142), keeping their 12 prompts together. They describe this single panel/run, omit training-seed variance and checkpoint-selection uncertainty, and are not a confirmatory significance test. Eight groups provide limited precision.</p>'''+table(['Identity','Native','2k','4k','6k','6k − 4k'],[[owner]+[f(A['identity'][owner][str(s)]) for s in ('native',2000,4000,6000)]+[f"{A['identity'][owner]['6000']-A['identity'][owner]['4000']:+.4f}"] for owner in owners])+f'''
<p><b>Checkpoint selection:</b> 4k is the best observed FLUX1a identity checkpoint; 6k's CLIP gain is only {mean(6000,'text_sim')-mean(4000,'text_sim'):.3f}. Improving text alignment does not establish better identity. The 0→2k jump includes recovery from the disruptive untrained BA path, so it should not be presented as improvement over the native model.</p>
<p><b>Historical context:</b> earlier FLUX1 on the same fixed96 reported owner-ID 0.3174 at 2k and 0.2762 at 8k. FLUX1a's best 0.2943 is 0.0231 below that historical 2k result. Thus stronger isolation has not yet produced the highest measured identity score. This is a cross-run comparison of changed architecture/objective, not a controlled attribution of the cause.</p>
<p><b>Next priorities:</b> retain 4k; run FLUX1b (same isolated rank128 branch, flow-only) to measure the identity auxiliary's contribution; perform matched BA-off / own-bank / donor-bank inference to test causal reference use. Test rank256 FLUX1c after those controls. Avoid interpreting longer training alone as the solution.</p>''')
page('Validation contract and reproducibility','''<p><b>Fixed96:</b> 8 identities × 12 prompts, original order/references/seeds retained (seed0 for these rows); 768×768 output, reference preprocessing limited to 512² pixels, 20 Euler steps, CFG4, batch2. Masks come from frozen native generations. Training-target photos/masks are never loaded by validation.</p>
<p><b>Spatial contract:</b> each step replaces the scene outside the owned face support with the appropriately noised native latent. Final decoding composites the face into the frozen native image with a 16px feather. Native exterior pixels are preserved exactly. Background consistency is therefore enforced by the pipeline, not an independent learned-generalization result.</p>'''+table(['Metric / check','Meaning or result'],[
['Owner-ID (primary)','ArcFace cosine for the face assigned to the frozen owner box; IoU threshold0.05 and ambiguity margin0.02. Uses subject-aligned identity embeddings. Higher is better.'],
['Legacy ID','Separate historical best-face metric/embedding definition; do not substitute for owner-ID. At4k:0.2448; at6k:0.2437.'],
['CLIP','ViT-L/14@336 text-image logits, not a percentage. Higher is better on this unchanged protocol.'],
['Coverage at4k / 6k','No-face, unowned, ambiguous and missing-mask rates all0. Mean detected faces2.875/image. Owner mask IoU0.8887 →0.8628.'],
['Admission','Pretrained native/BA-off equality, zero-mask equality, frozen-weight preservation, branch updates and exact parameter/optimizer/scheduler/RNG/cursor resume checks passed.'],
['Publication','Verifier checks96 order/prompts/seeds, checkpoint and image hashes, per-image/aggregate metrics, exterior/background audits and all96 Comet image entries at each checkpoint.']])+'''
<p><b>Training:</b> effective batch1, seed142, LR5e−5, warmup100, Adam-style optimizer with zero weight decay, gradient clipping1.0; saves every500 updates. Fixed seeded trajectory draws from47,341 eligible Large pairs. Peak measured admission reservation60.67GiB of121.6GiB; resumed production peaked59.99GiB. These are GB10 measurements.</p>
<p><b>Operational deviations:</b> production reached2k before completing deferred step0 validation. Credit exhaustion caused restart from2.5k, replaying unsaved updates. Extension beyond4k initially lacked identity labels; a separate6k cache was verified with every original4k record/embedding bitwise preserved (98.5% accepted at6k). Final6k validation resumed after78 images; existing outputs were retained. These events do not constitute extra independent trials.</p>
<p><b>Local checkpoints:</b> C:/Users/ogure/FLUX1a_checkpoints/FLUX1a_vast9B_20261004/checkpoint-004000 and checkpoint-006000. All five files per checkpoint were SHA256-verified. The original2k copy remains in the Linux run folder.</p>
<p><a href="https://www.comet.com/nikolay-2104/rsrch-new/2018ec7a730243bc98d922178e58aa5c">Open the verified FLUX1a Comet experiment</a>. Evidence and reproducible report scripts are saved alongside this PDF in reports/261005_FLUX1a_results. The frozen Vast runtime, not the now-diverged local model sources, produced these measurements.</p>''')
for i in range(0,8,2):
 body='<p class="small">Deterministic examples: identity i uses prompt index i (0–7). Selected without using scores; not best/worst cases. Top: reference, native scene, 4k, 6k. Bottom: identical owner-box crops from native, 2k, 4k, 6k; labels show per-image owner-ID. Crops are display enlargements, not rescored.</p>'
 for row in rows[i:i+2]:
  body+=f'<h2>{html.escape(row["identity_id"])} · sample {row["sample_id"]}</h2><p class="caption">{html.escape(row["prompt"])}</p>'+fig(f'sample_{row["sample_id"]}.jpg')
 page(f'Sample images · {i+1}–{i+2} of 8 identities',body)
CSS='''body{font-family:sans-serif;font-size:10pt;line-height:1.25;color:#20323f}h1{font-size:21pt;color:#076775;margin:0 0 12pt}h2{font-size:12pt;color:#076775;margin:10pt 0 3pt}p{margin:7pt 0}.kicker{font-size:11pt;color:#58727a}.small,.caption{font-size:8pt;color:#546774}.caption{margin:3pt 0 5pt}.callout{background:#eaf4f4;padding:10pt;margin:10pt 0;border-left:4pt solid #087f8c}table{border-collapse:collapse;width:100%;font-size:8.7pt;margin:10pt 0}th{background:#076775;color:white;text-align:left;padding:5pt}td{padding:5pt;border-bottom:.5pt solid #d6e0e5;vertical-align:top}tr:nth-child(even){background:#f3f6f8}a{color:#087f8c}'''
(O/'FLUX1a_results_report.html').write_text('<html><head><meta charset="utf-8"><style>'+CSS+'</style></head><body>'+''.join('<section>'+p+'</section>' for p in P)+'</body></html>')
pdf=pymupdf.open();sections=[]
for i,p in enumerate(P):
 story=pymupdf.Story(html=p,user_css=CSS,archive=pymupdf.Archive(str(O)));doc=story.write_with_links(lambda _,__:(pymupdf.Rect(0,0,595,842),pymupdf.Rect(42,48,553,792),None));sections.append({'section':i+1,'start':len(pdf)+1,'pages':len(doc)});pdf.insert_pdf(doc)
for i,p in enumerate(pdf):
 p.insert_text((42,28),'FLUX1a | measured Vast9B results',fontsize=8,color=(.32,.4,.45));p.insert_text((42,817),'5 October 2026 | fixed96 | one training run, eight identities',fontsize=8,color=(.32,.4,.45));p.insert_text((517,817),f'{i+1}/{len(pdf)}',fontsize=8,color=(.32,.4,.45))
pdf.set_metadata({'title':'FLUX1a — Architecture, validation results and sample images','author':'rsrch_new','subject':'Verified Vast9B FLUX1a fixed96 results at 2k, 4k and 6k'})
pdf.save(O/'FLUX1a_results_report.pdf',garbage=4,deflate=True)
violations=[]
for i,p in enumerate(pdf):
 for b in p.get_text('dict')['blocks']:
  if b['type']!=0:continue
  for l in b['lines']:
   for s in l['spans']:
    x0,y0,x1,y1=s['bbox']
    if x0<0 or x1>595 or y0<0 or y1>842:violations.append([i+1,s['text']])
 for_preview=i in (0,1,2,4,7)
 if for_preview:p.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3)).save(O/f'preview_{i+1:02d}.png')
audit={'pages':len(pdf),'sections':sections,'bounds_violations':violations,'images':sum(len(p.get_images()) for p in pdf),'links':sum(len(p.get_links()) for p in pdf),'characters':[len(p.get_text()) for p in pdf]}
(O/'document_validation.json').write_text(json.dumps(audit,indent=2));print(audit)
