"""A completed stage may enqueue its successor exactly once."""
import json
from types import SimpleNamespace

from scripts import clust_flux1a as stage


def test_successor_submission_is_idempotent_and_preserves_remaining_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(stage,'ROOT',tmp_path)
    monkeypatch.setattr(stage,'SETUP',tmp_path/'setup')
    path=tmp_path/'scratch'/f'pipeline_{stage.NAME}.json'
    stage.write(path,{'jobs':[{'id':'10','action':'train','step':2000}],
                     'stream_decode_admission_job':'9',
                     'remaining_stages':[['infer',2000,1,2,'03:00:00'],['decode',2000,1,2,'00:30:00']]})
    calls=[]
    def submit(command,**kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0,stdout='11\n',stderr='')
    monkeypatch.setattr(stage.subprocess,'run',submit)
    stage.enqueue_next('train',2000)
    stage.enqueue_next('train',2000)
    record=json.loads(path.read_text())
    assert len(calls)==1 and '--dependency=afterok:10:9' in calls[0]
    assert '--gpus=v100:1' in calls[0]
    assert record['jobs'][-1]['id']=='11' and len(record['remaining_stages'])==1
