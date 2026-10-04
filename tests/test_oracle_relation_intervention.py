"""Synthetic CPU oracle tests; no training, dataset, checkpoint, or result files.

python -B -m unittest discover -s tests -p test_oracle_relation_intervention.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from config import build_config
from models import build_model_from_name
from models.mca import MultiHeadCrossAttention
from oracle_relation_intervention import (
    METRICS, Layer1Oracle, OracleDataset, VolumeComparison, denormalize_b0,
    replace_cross_distribution, unit_rows, validation_volume_counts,
)
from relation_distortion_analysis import make_mask
from data import transforms as T


class InterventionMathTests(unittest.TestCase):
    def test_joint_mass_preservation_not_full_mass_transplant(self):
        # Three self keys: .6 mass, four cross keys: .4 mass.
        row = torch.tensor([.1, .2, .3, .1, .1, .1, .1])
        attention = row.expand(2, 2, 3, 7).clone()
        saved = attention.clone()
        full_cross = torch.tensor([.08, .01, .005, .005]).expand(2, 2, 3, 4).clone()
        reference = unit_rows(full_cross)  # full mass .1 must NOT be used in oracle.
        result, error = replace_cross_distribution(attention, reference, query_chunk=2)
        self.assertTrue(torch.equal(attention, saved))
        self.assertTrue(torch.equal(result[..., :3], attention[..., :3]))
        torch.testing.assert_close(result[..., 3:].sum(-1), torch.full((2, 2, 3), .4))
        torch.testing.assert_close(result.sum(-1), torch.ones(2, 2, 3))
        torch.testing.assert_close(unit_rows(result[..., 3:]), reference)
        self.assertLess(error, 1e-6)
        self.assertFalse(torch.equal(result[..., 3:], full_cross))

    def test_own_relation_is_noop_within_roundoff(self):
        torch.manual_seed(6)
        attention = torch.randn(2, 2, 5, 17).softmax(-1)
        reference = unit_rows(attention[..., 5:])
        result, _ = replace_cross_distribution(attention, reference, 3)
        torch.testing.assert_close(result, attention, atol=1e-7, rtol=1e-6)

    def test_zero_degraded_mass_stays_zero_but_zero_reference_rejected(self):
        attention = torch.zeros(1, 1, 2, 5)
        attention[..., :2] = .5
        result, _ = replace_cross_distribution(attention, torch.ones(1, 1, 2, 3) / 3)
        self.assertTrue(torch.equal(result, attention))
        with self.assertRaises(ValueError):
            unit_rows(torch.zeros(1, 1, 2, 3))
        with self.assertRaises(ValueError):
            replace_cross_distribution(attention, torch.ones(1, 1, 3, 3))
        with self.assertRaises(ValueError):
            replace_cross_distribution(attention, torch.full((1, 1, 2, 3), float("nan")))
        with self.assertRaises(ValueError):
            replace_cross_distribution(attention * .5, torch.ones(1, 1, 2, 3))

    def test_value_aggregation_uses_current_v(self):
        torch.manual_seed(9)
        module = MultiHeadCrossAttention(8, 2).eval().requires_grad_(False)
        x, auxiliary = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
        reference = torch.zeros(1, 2, 3, 4)
        reference[..., 2] = 1
        with torch.no_grad():
            q = module.to_q(x).reshape(1, 3, 2, 4).permute(0, 2, 1, 3)
            kv = module.to_kv(torch.cat([x, auxiliary], dim=1)).reshape(1, 7, 2, 2, 4).permute(2, 0, 3, 1, 4)
            attention = ((q @ kv[0].transpose(-2, -1)) * module.scale).softmax(-1)
            replaced, _ = replace_cross_distribution(attention, reference)
            expected = module.proj((replaced @ kv[1]).transpose(1, 2).reshape(1, 3, 8) + x)
            handle = module.attn_drop.register_forward_pre_hook(
                lambda _, inputs: (replace_cross_distribution(inputs[0], reference)[0],))
            try:
                actual = module(x, auxiliary)
            finally:
                handle.remove()
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


class OracleModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        cfg = build_config("reconstruction_multi_cross").clone()
        cfg.INPUT_SIZE = 16
        cfg.MODEL.P1 = 2
        cfg.MODEL.P2 = 4
        cfg.MODEL.HEAD_HIDDEN_DIM = 2
        cfg.MODEL.TRANSFORMER_NUM_HEADS = 2
        self.model = build_model_from_name(cfg, "reconstruction_multi_cross").eval().requires_grad_(False)
        self.full, self.degraded, self.auxiliary = [torch.randn(1, 1, 16, 16) for _ in range(3)]

    def test_baseline_and_l1_kv_unchanged_and_no_persistent_hooks(self):
        state = {name: value.clone() for name, value in self.model.state_dict().items()}
        with torch.no_grad():
            expected_normal = self.model(self.degraded, self.auxiliary)[0]
            controller = Layer1Oracle(self.model, 13)
            kv_values = []
            handle = controller.attention.to_kv.register_forward_hook(
                lambda _, inputs, output: kv_values.append(output.clone()))
            try:
                normal, oracle = controller.compare(self.full, self.degraded, self.auxiliary)
            finally:
                handle.remove()
            self.assertTrue(torch.equal(normal, expected_normal))
            self.assertEqual(len(kv_values), 3)
            self.assertTrue(torch.equal(kv_values[1], kv_values[2]))
            self.assertFalse(torch.equal(normal, oracle))
            self.assertIsNone(controller.reference)
            for layer in self.model.cross_transformer.layers:
                for direction in layer:
                    self.assertFalse(direction.fn.attn.attn_drop._forward_pre_hooks)
                    self.assertIsNone(direction.fn.attn.last_attn)
            after = self.model(self.degraded, self.auxiliary)[0]
            self.assertTrue(torch.equal(after, expected_normal))
        self.assertEqual(set(state), set(self.model.state_dict()))
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(state[key], value))

    def test_full_equals_degraded_negative_control(self):
        with torch.no_grad():
            controller = Layer1Oracle(self.model)
            normal, oracle = controller.compare(self.degraded, self.degraded, self.auxiliary)
        torch.testing.assert_close(normal, oracle, atol=1e-6, rtol=1e-5)

    def test_exception_removes_hook_and_reference(self):
        controller = Layer1Oracle(self.model)
        with torch.no_grad(), patch("oracle_relation_intervention.replace_cross_distribution", side_effect=RuntimeError("test")):
            with self.assertRaises(RuntimeError):
                controller.compare(self.full, self.degraded, self.auxiliary)
        self.assertIsNone(controller.reference)
        self.assertFalse(controller.attention.attn_drop._forward_pre_hooks)


class EvaluationTests(unittest.TestCase):
    def test_denormalization_matches_engine_including_clamped_label(self):
        prediction = torch.arange(128, dtype=torch.float32).reshape(2, 1, 8, 8)
        normalized_target = (prediction[:, 0] * 2).clamp(-6, 6)
        mean, std = torch.tensor([.2, .3]), torch.tensor([.4, .5])
        output, target = denormalize_b0(prediction, normalized_target, mean, std)
        self.assertTrue(torch.equal(output, prediction[:, 0] * std[:, None, None] + mean[:, None, None]))
        self.assertTrue(torch.equal(target, normalized_target * std[:, None, None] + mean[:, None, None]))

    def test_volume_metrics_and_paired_deltas_match_original_functions(self):
        torch.manual_seed(18)
        target = torch.rand(4, 16, 16) + .1
        normal = target + .1 * torch.randn_like(target)
        oracle = target + .05 * torch.randn_like(target)
        rows = []
        comparison = VolumeComparison({"a": 3, "b": 2}, SimpleNamespace(writerow=rows.append))
        for i in range(3):
            comparison.add("a", i, target[i], normal[i], oracle[i])
        comparison.add("b", 0, target[3], normal[3], oracle[3])
        comparison.flush()
        self.assertEqual(comparison.volume_count, 2)
        self.assertEqual(comparison.partial_volume_count, 1)
        self.assertEqual(comparison.sample_count, 4)
        for metric, function in METRICS.items():
            expected = float(function(target[:3].numpy(), normal[:3].numpy()))
            self.assertAlmostEqual(rows[0]["normal_4x_" + metric], expected)
            self.assertAlmostEqual(rows[0]["oracle_minus_normal_" + metric],
                                   rows[0]["oracle_L1_" + metric] - expected)
            self.assertAlmostEqual(comparison.moments["normal_4x", metric].mean,
                                   .5 * (rows[0]["normal_4x_" + metric] + rows[1]["normal_4x_" + metric]))

    def test_dataset_uses_native_4x_statistics_and_auxiliary(self):
        rng = np.random.RandomState(20)
        kspace = (rng.randn(24, 20) + 1j * rng.randn(24, 20)).astype(np.complex64)
        target = np.abs(rng.randn(16, 16)).astype(np.float32)
        raw = (kspace, None, target, {"recon_size": (16, 16)}, "/fixed/pdfs.h5", 0)
        pd = (*raw[:4], "/fixed/pd.h5", 0)
        dataset = OracleDataset.__new__(OracleDataset)
        dataset.raw = [(pd, raw, 0)]
        dataset.input_size = 16
        dataset.masks = {"4x": make_mask(4, .08)}
        dataset.native = T.ReconstructionTransform("singlecoil", make_mask(4, .08))
        sample = dataset[0]
        expected = dataset.native(*raw)
        self.assertTrue(torch.equal(sample["4x"][0], expected[0]))
        self.assertTrue(torch.equal(sample["target"], expected[1]))
        self.assertTrue(torch.equal(sample["mean"], expected[2]))
        self.assertTrue(torch.equal(sample["std"], expected[3]))
        self.assertTrue(torch.equal(sample["auxiliary"][0], dataset.native(*pd)[1]))
        full = T.ReconstructionTransform("singlecoil")(*raw)[0]
        self.assertTrue(torch.equal(sample["full"][0], full))

    def test_duplicate_target_volumes_are_not_silently_overwritten(self):
        examples = [("pd", "pdfs", i, {}, {}, 0) for i in range(3)]
        self.assertEqual(validation_volume_counts(examples), {"pdfs": 3})
        with self.assertRaises(ValueError):
            validation_volume_counts(examples + [("different_pd", "pdfs", 0, {}, {}, 1)])


if __name__ == "__main__":
    unittest.main()
