from pathlib import Path
import sys,json,hashlib,re
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from comet_ml.api import API
from scripts.vast_gpu import env_value,DEFAULT_ENV
OUT=Path(__file__).resolve().parent
api=API(api_key=env_value(DEFAULT_ENV,'COMET_API_KEY'),cache=False)
e=api.get_experiment_by_key('2018ec7a730243bc98d922178e58aa5c')
a=json.loads((OUT/'evidence/comet_assets.json').read_text())
# Keep asset identity evidence without expiring signed links.
(OUT/'evidence/comet_assets.json').write_text(json.dumps([{k:v for k,v in x.items() if 'link' not in k.lower() and 'url' not in k.lower()} for x in a],indent=2))
names={'quality_per_image.csv','quality_summary.json','routing_masks.json','native_checks.json','resume_parity.json','comparison_summary.json','best_checkpoint.json'}
work=[]
for x in a:
 if x['fileName'] in names:
  work.append((x,OUT/'evidence'/f"{x.get('step')}_{x['fileName']}"))
def get(item):
 x,p=item
 if not p.exists():p.write_bytes(e.get_asset(x['assetId']))
 return p.name
with ThreadPoolExecutor(max_workers=4) as pool:
 for name in pool.map(get,work):print(name,flush=True)
rows=[json.loads(x) for x in (ROOT/'data/validation/manual_val_96.jsonl').read_text().splitlines()]
selected=[rows[12*i+i] for i in range(8)]
(OUT/'selection.json').write_text(json.dumps(selected,indent=2))
work=[]
for row in selected:
 for step in (2000,4000,6000):
  x=next(x for x in a if x.get('step')==step and re.sub(r"(?: \(\d+\))?\.png$", "", x['fileName'])==f"fixed96/{row['sample_id']}")
  work.append((x,OUT/'evidence'/f"{step}_{row['sample_id']}.png"))
with ThreadPoolExecutor(max_workers=4) as pool:
 for name in pool.map(get,work):print(name,flush=True)
for step in (2000,4000,6000):
 r=json.loads((ROOT/f'runs/FLUX1a_vast9B_20261004/comet_verified_{step:06d}.json').read_text())
 for row in selected:
  p=OUT/'evidence'/f"{step}_{row['sample_id']}.png"
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['image_sha256'][row['sample_id']],str(p)
print('All 24 example images match verified publication SHA256.')
