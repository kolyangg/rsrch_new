"""Paid-instance guard checks using a fake Vast CLI, with no API calls."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/vast_gpu.py"


class VastGpuTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.log = root / "calls.jsonl"
        self.cli = root / "vastai"
        self.cli.write_text(f"#!{sys.executable}\n" + """\
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with Path(os.environ['MOCK_LOG']).open('a') as stream:
    stream.write(json.dumps({'args': args, 'key_set': bool(os.environ.get('VAST_API_KEY'))}) + '\\n')
if args[:2] == ['search', 'offers']:
    print(json.dumps([{'id': 42, 'gpu_name': 'A100', 'num_gpus': 1, 'gpu_ram': 81920,
      'cpu_ram': 131072, 'disk_space': 600, 'direct_port_count': 2, 'inet_down': 1500,
      'reliability': 0.99, 'dph_total': float(os.environ['MOCK_PRICE'])}]))
elif args[:2] == ['create', 'instance']:
    print(json.dumps({'success': True, 'new_contract': 9001}))
elif args[:2] == ['attach', 'ssh']:
    print({'success': os.environ.get('MOCK_ATTACH_FAIL') != '1'})
elif args[:2] == ['show', 'instance']:
    print(json.dumps({'id': 9001, 'actual_status': 'running'}))
elif args[:2] in (['stop', 'instance'], ['start', 'instance'], ['destroy', 'instance']):
    print({'stop': 'stopping', 'start': 'starting', 'destroy': 'destroying'}[args[0]] + ' instance 9001.')
else:
    print(json.dumps({'success': True}))
""")
        self.cli.chmod(0o700)
        self.private = root / "key"
        self.private.write_text("test-only")
        self.public = root / "key.pub"
        self.public.write_text("ssh-ed25519 AAAATEST test\n")
        self.env_file = root / ".env"
        self.env_file.write_text(f"VAST_API_KEY=test-secret\nVAST_SSH_PRIVATE_KEY={self.private}\nVAST_SSH_PUBLIC_KEY={self.public}\n")
        self.env = os.environ.copy()
        self.env.pop("VAST_API_KEY", None)
        self.env["VASTAI_BIN"] = str(self.cli)
        self.env["MOCK_LOG"] = str(self.log)
        self.env["MOCK_PRICE"] = "1.25"

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), "--env-file", str(self.env_file), *args],
                              env=self.env, text=True, capture_output=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_offer_search_and_confirmed_provision(self):
        found = self.run_script("offers", "--min-vram", "80", "--disk-gb", "250", "--max-hourly", "2")
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertIn("42", found.stdout)
        self.assertIn("80.0", found.stdout)
        args = ("provision", "--offer-id", "42", "--confirm-offer-id", "42",
                "--disk-gb", "250", "--max-hourly", "2")
        created = self.run_script(*args)
        self.assertEqual(created.returncode, 0, created.stderr)
        self.assertIn("instance 9001", created.stdout)
        calls = self.calls()
        self.assertEqual([call["args"][:2] for call in calls],
                         [["search", "offers"], ["search", "offers"], ["create", "instance"], ["attach", "ssh"]])
        self.assertNotIn("id=42", calls[1]["args"][2])
        self.assertTrue(all(call["key_set"] and "test-secret" not in str(call["args"]) for call in calls))

    def test_price_and_id_guards_prevent_mutation(self):
        wrong = self.run_script("provision", "--offer-id", "42", "--confirm-offer-id", "43",
                                "--disk-gb", "250", "--max-hourly", "2")
        self.assertNotEqual(wrong.returncode, 0)
        self.assertEqual(self.calls(), [])
        self.env["MOCK_PRICE"] = "2.50"
        expensive = self.run_script("provision", "--offer-id", "42", "--confirm-offer-id", "42",
                                    "--disk-gb", "250", "--max-hourly", "2")
        self.assertNotEqual(expensive.returncode, 0)
        self.assertEqual([call["args"][:2] for call in self.calls()], [["search", "offers"]])
        wrong_instance = self.run_script("terminate", "9001", "--confirm-instance-id", "9002")
        self.assertNotEqual(wrong_instance.returncode, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_stop_and_terminate_use_instance_id(self):
        stopped = self.run_script("stop", "9001")
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        terminated = self.run_script("terminate", "9001", "--confirm-instance-id", "9001")
        self.assertEqual(terminated.returncode, 0, terminated.stderr)
        calls = self.calls()
        self.assertEqual([call["args"][:2] for call in calls],
                         [["show", "instance"], ["stop", "instance"],
                          ["show", "instance"], ["destroy", "instance"]])
        self.assertIn("-y", calls[-1]["args"])

    def test_created_instance_id_is_reported_if_ssh_attachment_fails(self):
        self.env["MOCK_ATTACH_FAIL"] = "1"
        result = self.run_script("provision", "--offer-id", "42", "--confirm-offer-id", "42",
                                 "--disk-gb", "250", "--max-hourly", "2")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("instance 9001", result.stdout)
        self.assertIn("instance 9001 still exists", result.stderr)

    def test_download_requirement_blocks_changed_offer(self):
        self.env["MOCK_PRICE"] = "1.25"
        result = self.run_script("provision", "--offer-id", "42", "--confirm-offer-id", "42",
                                 "--disk-gb", "200", "--max-hourly", "2",
                                 "--min-vram", "48", "--min-download-mbps", "2000")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("download-speed", result.stderr)
        self.assertEqual([call["args"][:2] for call in self.calls()], [["search", "offers"]])

    def test_vram_guard_uses_raw_megabytes(self):
        self.env["MOCK_PRICE"] = "1.25"
        result = self.run_script("provision", "--offer-id", "42", "--confirm-offer-id", "42",
                                 "--disk-gb", "200", "--max-hourly", "2", "--min-vram", "96")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("VRAM", result.stderr)
        self.assertEqual([call["args"][:2] for call in self.calls()], [["search", "offers"]])


if __name__ == "__main__":
    unittest.main()
