import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MINER_DIR = REPO_ROOT / "miner"
if str(MINER_DIR) not in sys.path:
    sys.path.insert(0, str(MINER_DIR))

import alice_miner  # noqa: E402
import plan_b  # noqa: E402


class MinerBatchConfigTests(unittest.TestCase):
    def test_batch_config_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "batch_config.json"
            saved = alice_miner.save_batch_config(
                8,
                "RTX 4090",
                48.0,
                path=config_path,
                selected_at="2026-04-06T12:00:00Z",
            )
            loaded = alice_miner.load_batch_config(config_path)

        self.assertEqual(saved["batch_size"], 8)
        self.assertEqual(loaded, saved)

    def test_batch_size_cli_override_beats_saved_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "batch_config.json"
            alice_miner.save_batch_config(4, "RTX 4090", 48.0, path=config_path)
            resolved = alice_miner.resolve_batch_size(
                16,
                "RTX 4090",
                48.0,
                config_path=config_path,
                input_func=lambda: "1",
                interactive=True,
            )
            saved = alice_miner.load_batch_config(config_path)

        self.assertEqual(resolved, 16)
        self.assertEqual(saved["batch_size"], 4)

    def test_tprint_includes_timestamp(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            alice_miner.tprint("hello")
        output = buffer.getvalue()
        self.assertRegex(output, r"^\[\d{2}:\d{2}:\d{2}\] hello\n$")

    def test_plan_b_log_keeps_timestamp_and_label(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            plan_b._plan_b_log("batch ready")
        output = buffer.getvalue()
        self.assertRegex(output, r"^\[\d{2}:\d{2}:\d{2}\] \[PLAN-B\] batch ready\n$")


if __name__ == "__main__":
    unittest.main()
