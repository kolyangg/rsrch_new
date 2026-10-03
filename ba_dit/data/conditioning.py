"""Frozen native conditioning, encoded per training pair without a dataset cache."""
import torch

from ba_dit.data.cache import load_pair
from ba_dit.precision import conditioning_devices


class TrainingConditioner:
    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.online = config['data'].get('conditioning', 'cached') == 'online'
        # Construct before restoring training RNG/optimizer state.
        if self.online:
            self.encoder = backend.load_encoder(config)
            self.vae = backend.load_vae(config)

    def __call__(self, row):
        if not self.online:
            return load_pair(self.config, row, device='cuda')
        # Encoding must not advance the RNG used by flow noise/timestep sampling.
        with torch.no_grad(), torch.random.fork_rng(devices=conditioning_devices(self.encoder)):
            text, text_info = self.backend.encode_text(self.encoder, self.config, row)
            images, image_info = self.backend.encode_images(self.vae, self.config, row)
        # Native geometry helpers create some IDs/masks on CPU. Match load_pair's
        # device placement for every tensor, including integer/boolean metadata.
        tensors = {name:value.to('cuda') for name,value in {**text, **images}.items()}
        return tensors, {'encoder':text_info, 'vae':image_info}
