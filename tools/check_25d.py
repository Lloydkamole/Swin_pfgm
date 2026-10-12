"""Focused configuration and checkpoint checks; synthetic_demo covers numerical execution."""

import json
import tempfile
import unittest
from pathlib import Path

from train_accelerate_25d import config, load_config, save_config


class ConfigurationChecks(unittest.TestCase):
    def test_overrides_are_independent_and_stage_keys_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            original = load_config()
            save_config(original, path)
            restored = load_config(path)
            self.assertEqual(original, restored)
            restored["deformable_kwargs"]["kv_stride"] = 999
            self.assertEqual(config["deformable_kwargs"]["kv_stride"], 1)

    def test_unknown_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"batch_szie": 1}))
            with self.assertRaisesRegex(ValueError, "batch_szie"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
