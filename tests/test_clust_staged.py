"""Resource repair must retain model/loss settings and use two actual workers."""
import copy
import yaml
import pytest
from unittest.mock import patch
from ba_dit.config import ROOT,load_config
from scripts.clust_staged import execution_config


def test_torchrun_does_not_parse_worker_run_argument(tmp_path):
    import os
    import subprocess
    import sys
    from scripts.clust_staged import launch_ddp
    captured = []
    with patch('scripts.clust_staged.subprocess.run', side_effect=lambda command, **kw: captured.append(command)):
        launch_ddp(tmp_path, 'probe_worker', '--until', '1002')
    command = captured[0]
    # Exercise the real torchrun parser and two workers without CUDA/models.
    (tmp_path/'launch_probe.py').write_text(
        "import argparse,os\n"
        "p=argparse.ArgumentParser();p.add_argument('action');p.add_argument('--run');p.add_argument('--until');a=p.parse_args()\n"
        "assert a.action=='probe_worker' and a.until=='1002'\n"
        "from pathlib import Path\nPath(a.run, 'rank'+os.environ['RANK']).write_text('ok')\n")
    command[command.index('scripts.clust_staged')] = 'launch_probe'
    subprocess.run(command, check=True, timeout=45, env={**os.environ, 'PYTHONPATH':str(tmp_path),
        'CUDA_VISIBLE_DEVICES':''})
    assert (tmp_path/'rank0').read_text() == (tmp_path/'rank1').read_text() == 'ok'


def test_cached_conditioning_uses_the_rank_device():
    from ba_dit.data.conditioning import TrainingConditioner
    config = {'data': {'conditioning': 'cached'}}
    row = {'sample_id': 'rank1'}
    with patch('torch.cuda.current_device', return_value=1), \
            patch('ba_dit.data.conditioning.load_pair') as load:
        TrainingConditioner(config, None)(row)
        load.assert_called_once_with(config, row, device='cuda:1')


def test_cached_two_rank_config_preserves_scientific_settings(tmp_path):
    original=load_config(ROOT/'configs/clust/FLUX1_cluster_4b.yaml')
    changed=execution_config(original,'FLUX1_cluster_4b_ddp')
    assert original['training']['world_size']==1
    expected=copy.deepcopy(original)
    expected['name']='FLUX1_cluster_4b_ddp'
    expected['data'].update(conditioning='cached',encoder_device='cuda')
    expected['training']['world_size']=2
    assert changed==expected
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(changed))
    assert load_config(path)['training']['world_size']==2
    changed['data']['conditioning']='online'
    path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError,match='cached conditioning'):
        load_config(path)
