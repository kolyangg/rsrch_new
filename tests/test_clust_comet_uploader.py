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
