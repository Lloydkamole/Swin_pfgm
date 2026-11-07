"""Regression checks for foreground objectives, grouping and persisted EMA."""

import json
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate import Accelerator
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torchmetrics.functional.image import structural_similarity_index_measure

from train_accelerate_25d import EMA, edm_loss, save_training_checkpoint, compute_masked_metrics
from training_support_25d import (foreground_metrics, split_by_acquisition_group,
                                  collect_generation_batch, resolve_inference_checkpoint)


class ForegroundLossChecks(unittest.TestCase):
    def test_equal_patch_weight_despite_different_masks_and_empty_patch(self):
        prediction = torch.tensor([1., 3., 100.], requires_grad=True)
        clean = torch.zeros(3, 1, 2, 2, 2)
        mask = torch.zeros_like(clean)
        mask[0, 0, 0, 0, 0] = 1
        mask[1] = 1
        expected = {'l1': 16., 'l2': 40., 'huber': 12.}
        for loss_type in expected:
            with self.subTest(loss_type=loss_type), patch(
                'train_accelerate_25d.edm_precond',
                return_value=prediction[:, None, None, None, None].expand_as(clean),
            ):
                loss, metrics = edm_loss(torch.nn.Identity(), clean, clean,
                    torch.full((3,), .5), sigma_data=.5, body_mask=mask, loss_type=loss_type)
                self.assertAlmostEqual(loss.item(), expected[loss_type])
                self.assertEqual(metrics['valid_patches'], 2)
                if loss_type == 'l2':
                    loss.backward()
                    torch.testing.assert_close(prediction.grad, torch.tensor([8., 24., 0.]))

    def test_empty_batch_is_zero_with_zero_gradient(self):
        value = torch.tensor(2., requires_grad=True)
        clean = torch.zeros(2, 1, 2, 2, 2)
        with patch('train_accelerate_25d.edm_precond', return_value=value.expand_as(clean)):
            loss, metrics = edm_loss(torch.nn.Identity(), clean, clean, torch.ones(2),
                                     body_mask=torch.zeros_like(clean))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertEqual(value.grad.item(), 0.)
        self.assertEqual(metrics['valid_patches'], 0)

    def test_unmasked_loss_retains_full_grid_mean(self):
        clean = torch.zeros(2, 1, 2, 2, 2)
        predicted = torch.arange(clean.numel()).reshape_as(clean).float()
        with patch('train_accelerate_25d.edm_precond', return_value=predicted):
            loss, metrics = edm_loss(torch.nn.Identity(), clean, clean,
                                     torch.ones(2), sigma_data=.5)
        torch.testing.assert_close(loss, 5 * predicted.square().mean())
        self.assertEqual(metrics['valid_patches'], 2)


class ForegroundMetricChecks(unittest.TestCase):
    def test_background_and_padding_do_not_change_any_score(self):
        torch.manual_seed(5)
        target = torch.rand(1, 1, 32, 32) * .8
        prediction = target + .1
        mask = torch.zeros_like(target)
        mask[..., 6:26, 6:26] = 1
        baseline = foreground_metrics(prediction, target, mask)
        self.assertAlmostEqual(baseline['mae'], .1, places=6)
        self.assertAlmostEqual(baseline['psnr'], 20., places=4)
        prediction[mask == 0] = float('nan')
        target[mask == 0] = -1000
        changed = foreground_metrics(prediction, target, mask)
        padded = foreground_metrics(*[torch.nn.functional.pad(t, (8, 8, 8, 8))
                                      for t in (prediction, target, mask)])
        for key in baseline:
            self.assertAlmostEqual(baseline[key], changed[key], places=5)
            self.assertAlmostEqual(baseline[key], padded[key], places=5)

    def test_ssim_matches_standard_gaussian_ssim_on_full_foreground(self):
        torch.manual_seed(42)
        x, y = torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32)
        expected = structural_similarity_index_measure(x, y, data_range=1.)
        actual = foreground_metrics(x, y, torch.ones_like(x))['ssim']
        self.assertAlmostEqual(actual, expected.item(), places=5)

    def test_empty_samples_are_ignored_and_empty_rois_are_undefined(self):
        x, y = torch.ones(1, 1, 16, 16), torch.zeros(1, 1, 16, 16)
        base = foreground_metrics(x, y)
        combined = foreground_metrics(torch.cat([x, y]), torch.cat([y, y]), torch.cat([x, y]))
        self.assertEqual(base, combined)
        empty = foreground_metrics(x, y, y)
        self.assertTrue(all(math.isnan(empty[k]) for k in ['psnr', 'ssim', 'mae']))
        tiny_mask = torch.zeros_like(x); tiny_mask[..., 5:7, 5:7] = 1
        tiny = foreground_metrics(x, y, tiny_mask)
        self.assertEqual(tiny['mae'], 1.)
        self.assertEqual(tiny['psnr'], 0.)
        self.assertTrue(math.isnan(tiny['ssim']))
        self.assertEqual(tiny['ssim_windows'], 0)

    def test_existing_metric_api_uses_foreground_implementation(self):
        x = torch.ones(1, 1, 16, 16)
        scores = compute_masked_metrics(x, torch.zeros_like(x), x,
                                        psnr_metric=object(), ssim_metric=object(), mae_metric=object())
        self.assertEqual(scores['psnr'], 0.)
        self.assertEqual(scores['mae'], 1.)


class SplitChecks(unittest.TestCase):
    def test_provenance_groups_and_source_namespaces(self):
        with tempfile.TemporaryDirectory() as directory, warnings.catch_warnings():
            warnings.simplefilter('ignore')
            root = Path(directory)
            native = root / 'native' / 'training'; native.mkdir(parents=True)
            reports = native.parent / 'reports'; reports.mkdir()
            (reports / 'dataset_split_groups.csv').write_text(
                'patient,split_group\na,a\nb,a\nc,c\nd,d\n')
            normalized = root / 'normalized'; normalized.mkdir()
            (normalized / 'preprocessing_settings.json').write_text(json.dumps({'input_root': str(native)}))
            files = [{'id': p, 'mr': str(normalized / p / 'mr.nii.gz')} for p in 'abcd']
            files.append({'id': 'a', 'mr': str(root / 'other' / 'a' / 'mr.nii.gz')})
            for seed in range(10):
                train, val, metadata = split_by_acquisition_group(files, .6, seed)
                self.assertTrue(train and val)
                self.assertFalse({x['split_group'] for x in train} & {x['split_group'] for x in val})
                assignments = {x['id']: split for split, items in [('train', train), ('val', val)]
                               for x in items if str(normalized) in x['mr']}
                self.assertEqual(assignments['a'], assignments['b'])
                groups = [x['split_group'] for x in train + val if x['id'] == 'a']
                self.assertEqual(len(set(groups)), 2)
                self.assertEqual((train, val, metadata), split_by_acquisition_group(files, .6, seed))

    def test_manifest_omissions_fail_instead_of_leaking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'dataset_split_groups.csv'
            manifest.write_text('patient,split_group\na,a\n')
            with self.assertRaisesRegex(ValueError, 'missing'):
                split_by_acquisition_group([{'id': 'b', 'mr': str(root / 'b' / 'mr.nii')}], .9, 42)

    def test_explicit_manifest_links_two_preprocessed_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest = root / 'groups.csv'
            manifest.write_text('patient,split_group\na,a\nb,b\n')
            files = [{'id': p, 'mr': str(root / source / p / 'mr.nii')}
                     for source in ['one', 'two'] for p in ['a', 'b']]
            train, val, _ = split_by_acquisition_group(files, .5, 42,
                {str(root / source): str(manifest) for source in ['one', 'two']})
            self.assertEqual(len(train), 2)
            self.assertEqual(len(val), 2)
            self.assertEqual(train[0]['id'], train[1]['id'])

    def test_generation_count_spans_batches(self):
        loader = DataLoader([{'mr': torch.tensor([i]), 'ct': torch.tensor([i])}
                             for i in range(7)], batch_size=2)
        batch = collect_generation_batch(loader, 5)
        self.assertEqual(batch['mr'].flatten().tolist(), list(range(5)))
        self.assertEqual(len(collect_generation_batch(loader, 99)['mr']), 7)


class EMAChecks(unittest.TestCase):
    def test_checkpoint_exports_ema_and_restores_training_state(self):
        accelerator = Accelerator(cpu=True)
        model = torch.nn.Linear(2, 1)
        model.register_buffer('test_buffer', torch.tensor([7.]))
        optimizer = torch.optim.AdamW(model.parameters())
        model, optimizer = accelerator.prepare(model, optimizer)
        raw = accelerator.unwrap_model(model)
        ema = EMA(raw, decay=.9)
        accelerator.register_for_checkpointing(ema)
        optimizer.zero_grad(); accelerator.backward(model(torch.ones(1, 2)).sum()); optimizer.step()
        ema.update()
        expected_ema = ema.inference_state_dict()
        expected_raw = {k: v.detach().clone() for k, v in raw.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_training_checkpoint(accelerator, model, ema, directory, 2, 17,
                                     {'metric': 'val_gen/mae', 'value': .2})
            exported = load_file(str(directory / 'model_ema.safetensors'))
            for key in expected_ema:
                torch.testing.assert_close(exported[key], expected_ema[key])
            self.assertFalse(torch.equal(exported['weight'], expected_raw['weight']))
            self.assertEqual(json.loads((directory / 'training_state.json').read_text())['global_step'], 17)
            self.assertEqual(resolve_inference_checkpoint(directory)[1], 'ema')
            self.assertEqual(resolve_inference_checkpoint(directory / 'model.safetensors')[1], 'ema')
            self.assertEqual(resolve_inference_checkpoint(directory, 'raw')[1], 'raw')
            for parameter in raw.parameters():
                parameter.data.zero_()
            for value in ema.shadow.values():
                value.zero_()
            ema.num_updates = 100
            accelerator.load_state(directory)
            self.assertEqual(ema.num_updates, 1)
            for key, value in raw.state_dict().items():
                torch.testing.assert_close(value, expected_raw[key])
            for key in ema.shadow:
                torch.testing.assert_close(ema.shadow[key], expected_ema[key])

    def test_shadow_restored_after_error_and_legacy_weights_still_load(self):
        model = torch.nn.Linear(1, 1); ema = EMA(model)
        model.weight.data.add_(1)
        before = model.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'sampling'):
            with ema.average_parameters():
                raise RuntimeError('sampling failed')
        torch.testing.assert_close(model.weight, before)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.safetensors'; path.touch()
            self.assertEqual(resolve_inference_checkpoint(path), (path, 'raw'))
            with self.assertRaises(FileNotFoundError):
                resolve_inference_checkpoint(path, 'ema')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
