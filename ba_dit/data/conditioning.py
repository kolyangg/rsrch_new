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
        if self.online or config.get('training', {}).get('identity_loss', {}).get('weight', 0) > 0:
            self.vae = backend.load_vae(config)

    def __call__(self, row):
        if not self.online:
            # Safetensors resolves bare "cuda" to cuda:0, regardless of the
            # process's current device. Each DDP rank must load its own inputs.
            pair = load_pair(self.config, row, device=f'cuda:{torch.cuda.current_device()}')
            if self.config.get('branch', {}).get('kind') == 'flux2_face':
                tensors, metadata = pair
                from ba_dit.data.flux2_memory import load_memory
                tensors.update(load_memory(self.config, row, device=f'cuda:{torch.cuda.current_device()}'))
            return pair
        # Encoding must not advance the RNG used by flow noise/timestep sampling.
        with torch.no_grad(), torch.random.fork_rng(devices=conditioning_devices(self.encoder)):
            text, text_info = self.backend.encode_text(self.encoder, self.config, row)
            images, image_info = self.backend.encode_images(self.vae, self.config, row)
        # Native geometry helpers create some IDs/masks on CPU. Match load_pair's
        # device placement for every tensor, including integer/boolean metadata.
        tensors = {name:value.to('cuda') for name,value in {**text, **images}.items()}
        return tensors, {'encoder':text_info, 'vae':image_info}
