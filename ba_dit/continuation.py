"""Allow a longer run without changing any optimizer, data or model setting."""
import copy
from ba_dit.config import config_digest, load_config


def verify_extension(checkpoint, config, original_digest):
    saved = load_config(checkpoint/'resume_config.yaml')
    if config_digest(saved) != original_digest:
        raise ValueError('Saved resume configuration differs from checkpoint manifest')
    if config['training']['steps'] <= saved['training']['steps']:
        raise ValueError('A continuation must increase the total step limit')
    expected = copy.deepcopy(saved)
    for key in ('steps', 'validation_every'):
        expected['training'][key] = config['training'][key]
    if config_digest(expected) != config_digest(config):
        raise ValueError('Continuation may change only total steps and validation interval; optimizer/data/model settings must match')
