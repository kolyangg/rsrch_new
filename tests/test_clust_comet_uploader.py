"""Regressions for expired Slurm IDs and Comet liveness during stalled uploads."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import Mock, patch

from scripts.upload_clust_comet import ArchiveUploader, complete_archive, keep_alive, scheduler_state
from scripts.sync_clust_comet import Publisher


class CometUploaderTest(unittest.TestCase):
    def test_pipeline_transport_failure_recovers_without_claiming_worker_failure(self):
        from scripts import sync_clust_comet as relay
        monitor, worker = Mock(), Mock()
        observed = {'job_id':'123', 'worker_job_id':'124', 'active':False,
                    'slurm_state':'COMPLETED', 'stage':'summarize_20000'}
        worker.result={'observation':observed,'pending_assets':0,'pending_metric_steps':0,'pending_summary_metrics':0}
        sdk=SimpleNamespace(API=Mock(return_value=SimpleNamespace(get_experiment_by_key=lambda key:monitor)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'.env').write_text('')
            with patch.object(relay,'ROOT',root), \
                    patch.object(relay,'PublicationWorker',return_value=worker), \
                    patch.object(relay,'pipeline_status',side_effect=[
                        subprocess.CalledProcessError(1,['ssh'],stderr='VPN unavailable'), observed]) as status, \
                    patch.object(relay.time,'sleep'), \
                    patch.dict(sys.modules,{'comet_ml':sdk,'comet_ml.config':SimpleNamespace(get_config=lambda:SimpleNamespace(override={}))}), \
                    patch('scripts.upload_clust_comet.keep_alive'), \
                    patch.object(sys,'argv',['relay','--run-name','run','--job-id','123',
                                           '--experiment-key','key','--pipeline']):
                relay.main()
        self.assertEqual(status.call_count,2)
        monitor.log_other.assert_any_call('cluster/monitoring_status',
                                         'unreachable; last worker observation is stale')
        monitor.log_other.assert_any_call('cluster/monitoring_status','connected')
        monitor.set_state.assert_called_once_with('finished')
        worker.submit.assert_called_once_with(observed)

    def test_resume_keeps_interrupted_metrics_out_of_new_curve(self):
        # The cancelled attempt has step 2 already, but its replay must still
        # publish and verify against the new attempt's curve.
        histories = {'train/loss': {1: 9.0, 2: 8.0}}
        api, experiment = Mock(), Mock()
        def metrics(name):
            return [{'step': step, 'metricValue': value}
                    for step, value in histories.get(name, {}).items()]
        def log(values, step):
            for name, value in values.items():
                histories.setdefault(name, {})[step] = value
        api.get_metrics.side_effect = metrics
        api.get_asset_list.return_value = []
        api.get_metrics_summary.return_value = []
        experiment.log_metrics.side_effect = log
        experiment.flush.return_value = True
        sdk = SimpleNamespace(API=Mock(return_value=SimpleNamespace(get_experiment_by_key=lambda key: api)),
                              ExistingExperiment=Mock(return_value=experiment))
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'comet_ml': sdk}):
            folder = Path(directory)
            (folder/'metrics.jsonl').write_text(''.join(json.dumps({'step': step, 'train/loss': step/10})+'\n'
                                                       for step in range(1, 5)))
            publisher = Publisher(folder, 'key', 'resume_123/')
            publisher.publish()
            self.assertEqual(histories['train/loss'], {1: 9.0, 2: 8.0})
            self.assertEqual(histories['resume_123/train/loss'], {1: 0.1, 2: 0.2, 4: 0.4})
            self.assertEqual(json.loads(publisher.receipt.read_text())['verified_metric_step'], 4)

    def test_open_empty_archive_is_not_ready_for_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'open.zip'
            with zipfile.ZipFile(path, 'w'):
                pass
            self.assertTrue(zipfile.is_zipfile(path))
            self.assertFalse(complete_archive(path))
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('experiment.json', '{}')
                archive.writestr('messages.json', '')
            self.assertTrue(complete_archive(path))

    def test_expired_job_uses_accounting(self):
        with patch('scripts.upload_clust_comet.subprocess.run', side_effect=[
            subprocess.CompletedProcess([], 1, '', 'Invalid job id specified'),
            subprocess.CompletedProcess([], 0, 'FAILED\n', ''),
        ]):
            self.assertEqual(scheduler_state('123'), ('FAILED', False))

    def test_upload_wait_does_not_block_live_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            uploader = ArchiveUploader((Path(directory),), Path(directory)/'uploaded.json')
            uploader.process = Mock()
            uploader.process.poll.return_value = None
            uploader.deadline = float('inf')
            uploader.poll()
            uploader.process.wait.assert_not_called()

    def test_heartbeat_obeys_server_interval_and_records_acknowledgements(self):
        experiment = Mock()
        experiment.update_status.return_value = {'isAliveBeatDurationMillis': 10000}
        api = Mock(return_value=SimpleNamespace(get_experiment_by_key=lambda key: experiment))
        active = threading.Event(); active.set()
        stop = threading.Event()
        waits = []
        def wait(interval):
            waits.append(interval)
            if len(waits) == 2:
                stop.set()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {'comet_ml': SimpleNamespace(API=api)}), \
                patch.object(stop, 'wait', side_effect=wait):
            receipt = Path(directory)/'heartbeat.json'
            keep_alive('immutable-key', active, stop, receipt)
            self.assertEqual(json.loads(receipt.read_text())['heartbeat_count'], 2)
        api.assert_called_once_with(cache=False)
        self.assertEqual(experiment.update_status.call_count, 2)
        self.assertTrue(all(0 < value <= 5 for value in waits))


if __name__ == '__main__':
    unittest.main()


def test_partial_panel_publishes_only_finished_pngs(tmp_path):
    from PIL import Image
    from scripts.sync_clust_comet import completed_outputs
    panel=tmp_path/'validation-002000';panel.mkdir()
    samples=[{'sample_id':str(i),'image':f'{i}.png','prompt':'same prompt','seed':i} for i in range(96)]
    (panel/'validation.json').write_text(json.dumps({'samples':samples}))
    Image.new('RGB',(8,8)).save(panel/'0.png')
    (panel/'1.png').write_bytes((panel/'0.png').read_bytes()[:30])
    images,assets,summaries=completed_outputs(tmp_path)
    assert [(r[1],r[2]) for r in images]==[('fixed96/0',2000)]
    assert not assets and not summaries
    Image.new('RGB',(8,8)).save(panel/'1.png')
    assert len(completed_outputs(tmp_path)[0])==2


def outbox(tmp_path):
    p=Publisher.__new__(Publisher)
    p.key='key';p.folder=tmp_path;p.delivery_path=tmp_path/'asset_delivery.json'
    p.delivery=json.loads(p.delivery_path.read_text()) if p.delivery_path.exists() else {}
    p.api=Mock();p.experiment=Mock()
    return p


def test_image_name_without_extension_is_confirmed_without_reupload(tmp_path):
    p=outbox(tmp_path)
    lookup=p.api._api._client.get_experiment_assets_list_by_name
    lookup.return_value=[{'userFileName':'fixed96/00','step':2000,'assetId':'existing'}]
    image=tmp_path/'00.png';image.write_bytes(b'unchanged')
    assert p.deliver([(image,'fixed96/00',2000,{})],[])==(1,0)
    lookup.assert_called_once_with('key','fixed96/00',asset_type='image',timeout=15)
    p.experiment.log_image.assert_not_called()
    restarted=outbox(tmp_path)
    assert restarted.deliver([(image,'fixed96/00',2000,{})],[])==(1,0)
    restarted.api._api._client.get_experiment_assets_list_by_name.assert_not_called()


def test_delayed_image_ack_survives_restart_without_duplicate_queue(tmp_path):
    p=outbox(tmp_path)
    lookup=p.api._api._client.get_experiment_assets_list_by_name
    lookup.return_value=[]
    image=tmp_path/'00.png';image.write_bytes(b'unchanged')
    items=[(image,'fixed96/00',2000,{})]
    assert p.deliver(items,[])==(0,1)
    p.experiment.log_image.assert_called_once()
    p=outbox(tmp_path);lookup=p.api._api._client.get_experiment_assets_list_by_name
    lookup.return_value=[]
    assert p.deliver(items,[])==(0,1)
    p.experiment.log_image.assert_not_called()
    lookup.return_value=[{'userFileName':'fixed96/00','step':2000,'assetId':'arrived'}]
    assert p.deliver(items,[])==(1,0)
    p.experiment.log_image.assert_not_called()


def test_metadata_with_implicit_zero_step_is_acknowledged(tmp_path):
    p=outbox(tmp_path)
    p.api._api._client.get_experiment_assets_list_by_name.return_value=[{'step':0,'assetId':'metadata'}]
    path=tmp_path/'admission.json';path.write_text('{}')
    assert p.deliver([],[(path,'run/admission.json',None)])==(0,0)
    p.experiment.log_asset.assert_not_called()


def test_changed_metadata_gets_its_own_remote_revision(tmp_path):
    p=outbox(tmp_path)
    lookup=p.api._api._client.get_experiment_assets_list_by_name
    lookup.return_value=[{'step':0,'assetId':'old'}]
    path=tmp_path/'deployment.json';path.write_text('{}')
    p.deliver([],[(path,'run/deployment.json',None)])
    path.write_text('{"revision":2}')
    lookup.return_value=[]
    assert p.deliver([],[(path,'run/deployment.json',None)])==(0,1)
    assert '.sha256-' in p.experiment.log_asset.call_args.kwargs['file_name']


def test_slow_upload_does_not_block_latest_scheduler_observation(tmp_path):
    from scripts.sync_clust_comet import PublicationWorker
    worker=PublicationWorker(SimpleNamespace(),tmp_path)
    started,release=threading.Event(),threading.Event()
    observed=[]
    def blocked(value):
        observed.append(value);started.set();release.wait(5)
    worker.publish_once=blocked
    worker.thread.start()
    try:
        worker.submit({'stage':'decode','completed':1})
        assert started.wait(2)
        worker.submit({'stage':'decode','completed':50})
        assert worker.observed['completed']==50 and len(observed)==1
    finally:
        worker.stop.set();release.set();worker.thread.join(2)
    assert not worker.thread.is_alive()
