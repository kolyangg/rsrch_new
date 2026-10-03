"""Download exact original metric assets without importing/loading any models."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]


def verify(path, record):
    if path.stat().st_size != record['bytes']:
        raise ValueError(f'Wrong size: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != record['sha256']:
        raise ValueError(f'Wrong SHA256: {path}')


def download(record):
    path = Path.home()/record['path']
    if not path.exists():
        if 'url' not in record:
            raise FileNotFoundError(f'Required existing InsightFace asset: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name+'.clust-download')
        with urlopen(record['url'], timeout=120) as response, partial.open('wb') as stream:
            shutil.copyfileobj(response, stream, length=1024*1024)
        verify(partial, record)
        partial.replace(path)
    verify(path, record)
    print(f'Verified {record["path"]}', flush=True)


def main():
    from huggingface_hub import hf_hub_download
    manifest = json.loads((ROOT/'locks/weights-clust-metrics.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, manifest['files']))
    for record in manifest['huggingface']:
        path = Path(hf_hub_download(record['repo'], record['filename'], revision=record['revision']))
        verify(path, record)
        # timm requests main; resolve it to the original locally measured revision
        # while jobs run with HF_HUB_OFFLINE=1.
        refs = path.parent.parent.parent/'refs'
        refs.mkdir(exist_ok=True)
        (refs/'main').write_text(record['revision'])
        print(f'Verified {record["repo"]}@{record["revision"]}', flush=True)
    receipt = ROOT/'scratch/clust-v100/metric-weights-verified.json'
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
