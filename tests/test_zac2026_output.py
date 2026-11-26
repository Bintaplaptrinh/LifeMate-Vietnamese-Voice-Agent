import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lifemate.zac2026.run import confirm_output, write_results


class OutputTest(unittest.TestCase):
    def test_existing_result_keeps_its_inode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "submission.jsonl"
            path.write_text("stale", encoding="utf-8")
            inode = path.stat().st_ino
            record = {"id": "sample", "tool_calls": [], "state_hash": "0" * 64, "time": 12}
            write_results(lambda _: record, [record], path)
            confirm_output(path)
            self.assertEqual(path.stat().st_ino, inode)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), record)
            self.assertEqual(list(path.parent.iterdir()), [path])
