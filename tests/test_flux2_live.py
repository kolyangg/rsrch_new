"""Live publication never switches Comet keys, loses a batch, or wraps training twice."""
import json
from unittest.mock import Mock
from scripts.flux2_live import publish_images,wrap_command
from ba_dit.data.manifest import file_hash


def test_worker_wrap_preserves_interpreter_arguments():
    command=['metrics-python','-m','scripts.evaluate_metrics','--global-step','1000']
    wrapped=wrap_command(command)
    assert wrapped==['metrics-python','-m','scripts.flux2_live','worker',*command[2:]]
    assert wrap_command(wrapped)==wrapped
    assert wrap_command(['git','status'])==['git','status']


def test_each_completed_batch_uploads_once_and_retries_unacknowledged(tmp_path):
    folder=tmp_path/'validation-001000';folder.mkdir()
    (folder/'x.png').write_bytes(b'finished PNG')
    sha=file_hash(folder/'x.png')
    (folder/'validation.json').write_text(json.dumps({'checkpoint_sha256':'ckpt','samples':[
        {'sample_id':'x','image':'x.png','prompt':'test','seed':7}]}))
    (tmp_path/'stream_decode_1000.json').write_text(json.dumps({'samples':{'x':{
        'checkpoint_sha256':'ckpt','image_sha256':sha}}}))
    api=Mock();api.log_image.return_value=None;receipts={}
    try:publish_images(api,tmp_path,24,receipts)
    except RuntimeError:pass
    else:raise AssertionError('Unacknowledged image marked uploaded')
    assert not receipts
    api.log_image.return_value={'assetId':'accepted'}
    publish_images(api,tmp_path,24,receipts)
    publish_images(api,tmp_path,24,receipts)
    assert api.log_image.call_count==2
    assert api.log_image.call_args.kwargs['image_name']=='fixed24/x'
    assert api.log_image.call_args.kwargs['step']==1000
    assert json.loads((tmp_path/'live_uploaded_images.json').read_text())['1000/x']['image_sha256']==sha


def test_handoff_waits_for_saved_training_segment(tmp_path,monkeypatch):
    from scripts import flux2_live as live
    checkpoint=tmp_path/'checkpoint-001000';checkpoint.mkdir()
    (checkpoint/'manifest.json').write_text(json.dumps({'step':1000}))
    (checkpoint/'training_state.pt').write_bytes(b'optimizer state')
    (checkpoint/'adapters.safetensors').write_bytes(b'adapters')
    state=tmp_path/'status.json';done=tmp_path/'completed_commands.json'
    state.write_text(json.dumps({'status':'running','stage':'train','target_step':1000}))
    done.write_text('[]')
    events=[]
    def finish_training(_seconds):
        events.append('checkpoint completed')
        done.write_text('["train_1000"]')
        state.write_text(json.dumps({'status':'running','stage':'infer','target_step':1000}))
    def systemctl(command,**kwargs):
        if 'stop' in command:
            assert events==['checkpoint completed']
            events.append('old controller stopped')
        return Mock(returncode=0)
    monkeypatch.setattr(live,'verified_policy',lambda run:{})
    monkeypatch.setattr(live.time,'sleep',finish_training)
    monkeypatch.setattr(live.subprocess,'run',systemctl)
    monkeypatch.setattr(live,'controller',lambda run:events.append('live controller started'))
    live.handoff(tmp_path,'old.service')
    assert events==['checkpoint completed','old controller stopped','live controller started']
    assert json.loads((tmp_path/'live_handoff.json').read_text())['training_updates_discarded']==0
