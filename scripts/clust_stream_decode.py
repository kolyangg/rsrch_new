"""Decode each completed cluster validation latent; all network I/O stays in the relay."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from safetensors import safe_open

from ba_dit.config import digest
from ba_dit.data.manifest import file_hash


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


class StreamingDecoder:
    def __init__(self, module, run, config, step):
        self.module,self.run,self.config,self.step=module,Path(run),config,step
        self.folder=self.run/f'validation-{step:06d}'
        self.path=self.run/f'stream_decode_{step}.json'
        self.identity={'config_sha256':digest(config),'decoder_code_sha256':file_hash(__file__),
                       'native_decoder_sha256':file_hash(module.__file__)}
        self.receipt=json.loads(self.path.read_text()) if self.path.exists() else {**self.identity,'samples':{}}
        assert all(self.receipt[k]==v for k,v in self.identity.items())
        self.masks=json.loads((self.run/'routing_masks.json').read_text())['samples']
        self.backend=module.backend_module(config)
        self.vae=None

    @torch.no_grad()
    def update(self, report):
        for row in report['samples']:
            key=row['sample_id'];path=self.folder/row['image']
            record=self.receipt['samples'].get(key)
            if record:
                assert record['checkpoint_sha256']==report['checkpoint_sha256']
                assert file_hash(path)==record['image_sha256']
                continue
            original=self.run/'native'/f'{key}.png'
            assert file_hash(original)==self.masks[key]['baseline_image_sha256']
            with safe_open(self.folder/f'{key}.safetensors',framework='pt') as latent_file:
                assert latent_file.metadata()['checkpoint_sha256']==report['checkpoint_sha256']
                latent=latent_file.get_tensor('latent')
            # VAE construction and decoding cannot perturb subsequent generation RNG.
            devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                if self.vae is None:self.vae=self.backend.load_vae(self.config)
                image=self.backend.decode(self.vae,latent,self.config)
            image.save(self.folder/f'{key}_raw.png')
            with Image.open(original) as source:native=source.convert('RGB')
            alpha=self.module.face_alpha(native.size,self.masks[key]['face_bbox'],self.config['branch']['mask_feather_pixels'])
            image=self.module.preserve_background(image,native,alpha)
            temporary=path.with_suffix('.png.tmp');image.save(temporary,format='PNG');temporary.replace(path)
            difference=np.abs(np.asarray(image).astype(float)-np.asarray(native).astype(float))
            support=alpha.squeeze().numpy()>0
            audit={'sample_id':key,'background_max_abs':float(difference[~support].max()) if (~support).any() else 0.,
                   'face_mean_abs':float(difference[support].mean())}
            self.receipt['samples'][key]={'checkpoint_sha256':report['checkpoint_sha256'],
                                          'image_sha256':file_hash(path),'audit':audit}
            write(self.path,self.receipt)
            self.module.memory()
        write(self.run/f'background_audit_{self.step}.json',
              [self.receipt['samples'][r['sample_id']]['audit'] for r in report['samples']])


def infer(module, run, config, step):
    """Observe the pinned sampler's completed reports without changing sampling."""
    decoder=StreamingDecoder(module,run,config,step)
    original=module.write
    def completed(path,value):
        original(path,value)
        if Path(path)==decoder.folder/'validation.json':decoder.update(value)
    module.write=completed
    try:module.infer(run,config,step)
    finally:module.write=original


def decoded(run, config, step):
    """The existing decode stage checks streamed outputs instead of generating twice."""
    run=Path(run);path=run/f'stream_decode_{step}.json'
    if not path.exists():return False
    receipt=json.loads(path.read_text())
    assert receipt['config_sha256']==digest(config) and receipt['decoder_code_sha256']==file_hash(__file__)
    report=json.loads((run/f'validation-{step:06d}/validation.json').read_text())
    assert len(report['samples'])==96 and len(receipt['samples'])==96
    for row in report['samples']:
        saved=receipt['samples'][row['sample_id']]
        assert saved['checkpoint_sha256']==report['checkpoint_sha256']
        assert file_hash(run/f'validation-{step:06d}'/row['image'])==saved['image_sha256']
    print(f'VALIDATION {step} | all96 PNGs decoded during generation and verified',flush=True)
    return True
