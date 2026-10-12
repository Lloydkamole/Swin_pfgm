"""Focused configuration and checkpoint checks; synthetic_demo covers numerical execution."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from train_accelerate_25d import config, edm_sample, edm_sample_tta, load_config, mask_generated_ct, save_config


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


class GeneratedBackgroundChecks(unittest.TestCase):
    def test_mask_preserves_foreground_and_maps_background_to_zero(self):
        prediction = torch.tensor([-0.5, 0.25, 0.75, 1.5])
        mask = torch.tensor([0, 1, 0, 1])
        result = mask_generated_ct(prediction, mask)
        self.assertTrue(torch.equal(result[mask > 0], prediction[mask > 0]))
        self.assertTrue(torch.equal(((result + 1) / 2)[mask == 0], torch.zeros(2)))
        self.assertIs(mask_generated_ct(prediction, None), prediction)

    def test_sampler_outputs_and_tta_mask_background_without_changing_trajectory(self):
        class ZeroModel(torch.nn.Module):
            def forward(self, x, mri, r, mask=None):
                return torch.zeros_like(x)

        model = ZeroModel()
        mr = torch.zeros(1, 1, 2, 4, 4)
        mask = torch.ones_like(mr)
        mask[..., :2] = 0
        for with_stats in [False, True]:
            torch.manual_seed(42)
            unmasked = edm_sample(model, mr, num_steps=2, sigma_max=1,
                                  return_trajectory_stats=with_stats)
            torch.manual_seed(42)
            masked = edm_sample(model, mr, num_steps=2, sigma_max=1, mask=mask,
                                return_trajectory_stats=with_stats)
            if with_stats:
                self.assertEqual(unmasked[1], masked[1])
                unmasked, masked = unmasked[0], masked[0]
            self.assertTrue(torch.equal(masked[mask > 0], unmasked[mask > 0]))
            self.assertTrue(torch.equal(masked[mask == 0], torch.full_like(masked[mask == 0], -1)))
        tta = edm_sample_tta(model, mr, num_steps=2, sigma_max=1, mask=mask)
        self.assertTrue(torch.equal(tta[mask == 0], torch.full_like(tta[mask == 0], -1)))


if __name__ == "__main__":
    unittest.main()
