"""Small synthetic CPU tests, not validation experiments or evidence for a hypothesis.

Run: python -m unittest discover -s tests -p test_relation_distortion_analysis.py -v
Requires the project's runtime dependencies; creates no checkpoints/results.
"""

import math
import unittest

import numpy as np
import torch

from config import build_config
from data import transforms as T
from models import build_model_from_name
from models.mca import MultiHeadCrossAttention
from relation_distortion_analysis import (
    AnalysisDataset, AttentionCollector, Moments, accumulate, conditional_relation, js_divergence,
    make_mask, magnitude_from_raw, relation_metrics, retention_from_indices,
    split_attention, stable_top_indices, target_conditions,
)


class MetricTests(unittest.TestCase):
    def test_js_known_distributions(self):
        p = torch.tensor([[[[1., 0.]]]])
        q = torch.tensor([[[[0., 1.]]]])
        self.assertEqual(js_divergence(p, p).item(), 0)
        self.assertAlmostEqual(js_divergence(p, q).item(), math.log(2), places=6)
        torch.testing.assert_close(js_divergence(p, q), js_divergence(q, p))

    def test_ties_and_overlap(self):
        ties = torch.ones(2, 3, 4, 12) / 12
        indices = stable_top_indices(ties, 10)
        self.assertTrue(torch.equal(indices[0, 0, 0], torch.arange(10)))
        for k in (1, 5, 10):
            self.assertTrue(torch.equal(retention_from_indices(indices, indices, k),
                                        torch.ones(2, 3, 4)))
        ref = torch.tensor([[[[0, 1, 2, 3, 4]]]])
        deg = torch.tensor([[[[3, 4, 5, 6, 7]]]])
        self.assertAlmostEqual(retention_from_indices(ref, deg, 5).item(), 0.4, places=6)
        with self.assertRaises(ValueError):
            stable_top_indices(ties, 13)

    def test_mass_and_conditional_are_distinct(self):
        reference = torch.full((2, 3, 7, 12), 0.02)
        degraded = reference * 2
        metrics = relation_metrics(degraded, reference, query_chunk=3)
        self.assertTrue(torch.allclose(metrics["attention_mass"], torch.full((2, 3), .48).double()))
        self.assertTrue(torch.allclose(metrics["js"], torch.zeros(2, 3).double(), atol=1e-9))
        self.assertTrue(torch.equal(metrics["retention_at_10"], torch.ones(2, 3).double()))
        for name, value in metrics.items():
            torch.testing.assert_close(value, relation_metrics(degraded, reference, 7)[name])

    def test_reject_invalid_attention(self):
        attn = torch.full((1, 2, 5, 17), 1 / 17)
        cross = split_attention(attn, 2)
        self.assertEqual(tuple(cross.shape), (1, 2, 5, 12))
        self.assertNotEqual(cross.untyped_storage().data_ptr(), attn.untyped_storage().data_ptr())
        with self.assertRaises(ValueError):
            split_attention(attn * .5)
        with self.assertRaises(ValueError):
            split_attention(attn * float("nan"))
        with self.assertRaises(ValueError):
            conditional_relation(torch.zeros_like(cross))

    def test_streaming_slice_weighting(self):
        moments = Moments()
        moments.update(torch.tensor([1., 2.]))
        moments.update(torch.tensor([6.]))  # Short final batch must not get extra weight.
        count, mean, std = moments.summary()
        self.assertEqual(count, 3)
        self.assertEqual(mean, 3)
        self.assertAlmostEqual(std, math.sqrt(14 / 3))
        stats = {}
        accumulate(stats, "A", "4x", {0: {"js": torch.tensor([[1., 3.], [3., 5.]])},
                                      1: {"js": torch.tensor([[5., 7.], [7., 9.]])}})
        self.assertEqual(stats[("global", "A", "4x", "all", "all", "js")].summary(), (2, 5., 1.))


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(17)
        kspace = (rng.randn(24, 20) + 1j * rng.randn(24, 20)).astype(np.complex64)
        self.raw = (kspace, None, np.ones((16, 16), dtype=np.float32),
                    {"recon_size": (16, 16)}, "/fixed/singlecoil_val/example.h5", 3)

    def test_native_matches_original_transform(self):
        masks = {"4x": make_mask(4, .08), "8x": make_mask(8, .05)}
        images = target_conditions(self.raw, masks, "A")
        for condition, mask in [("full", None)] + list(masks.items()):
            original = T.ReconstructionTransform("singlecoil", mask)(*self.raw)[0]
            self.assertTrue(torch.equal(images[condition], original))
        repeated = target_conditions(self.raw, masks, "A")
        self.assertTrue(torch.equal(images["4x"], repeated["4x"]))
        self.assertTrue(torch.equal(images["8x"], repeated["8x"]))

    def test_controlled_uses_unclamped_full_statistics(self):
        mask = make_mask(4, .08)
        images = target_conditions(self.raw, {"4x": mask}, "B")
        full = magnitude_from_raw(self.raw, None)
        degraded = magnitude_from_raw(self.raw, mask)
        expected = T.normalize(degraded, full.mean(), full.std(), eps=1e-11).clamp(-6, 6)
        self.assertTrue(torch.equal(images["4x"], expected))
        self.assertTrue(torch.equal(images["full"], target_conditions(self.raw, {"4x": mask}, "A")["full"]))

    def test_auxiliary_is_original_pd_target(self):
        dataset = AnalysisDataset.__new__(AnalysisDataset)
        pd = (*self.raw[:4], "/fixed/singlecoil_val/pd.h5", self.raw[5])
        dataset.raw = [(pd, self.raw, 0)]
        dataset.protocol = "A"
        dataset.input_size = 16
        dataset.masks = {"4x": make_mask(4, .08), "8x": make_mask(8, .05)}
        dataset.native = T.ReconstructionTransform("singlecoil", make_mask(4, .08))
        sample = dataset[0]
        self.assertTrue(torch.equal(sample["auxiliary"][0], dataset.native(*pd)[1]))
        self.assertFalse(torch.equal(sample["auxiliary"][0], dataset.native(*pd)[0]))
        self.assertEqual(sample["target_fname"], self.raw[4])
        self.assertEqual(sample["slice_num"], self.raw[5])


class CaptureTests(unittest.TestCase):
    def test_cache_does_not_change_dropout_or_state(self):
        module = MultiHeadCrossAttention(8, 2, attn_drop=.4, proj_drop=.3).train()
        x, aux = torch.randn(2, 5, 8), torch.randn(2, 12, 8)
        state = {k: v.clone() for k, v in module.state_dict().items()}
        params = sum(p.numel() for p in module.parameters())
        torch.manual_seed(71)
        original = module(x, aux)
        self.assertIsNone(module.last_attn)
        module.capture_attention = True
        torch.manual_seed(71)
        captured = module(x, aux)
        self.assertTrue(torch.equal(original, captured))
        self.assertFalse(module.last_attn.requires_grad)
        torch.testing.assert_close(module.last_attn.sum(-1), torch.ones(2, 2, 5))
        self.assertEqual(params, sum(p.numel() for p in module.parameters()))
        for key, tensor in module.state_dict().items():
            self.assertTrue(torch.equal(tensor, state[key]))
        module.load_state_dict(state, strict=True)

    def test_four_layer_collector_and_cleanup(self):
        cfg = build_config("reconstruction_multi_cross").clone()
        cfg.INPUT_SIZE = 16
        cfg.MODEL.P1 = 2
        cfg.MODEL.P2 = 4
        cfg.MODEL.HEAD_HIDDEN_DIM = 2
        cfg.MODEL.TRANSFORMER_NUM_HEADS = 2
        model = build_model_from_name(cfg, "reconstruction_multi_cross").eval()
        x, aux = torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16)
        with torch.no_grad():
            original = model(x, aux)
            collector = AttentionCollector(model, 4, 2, 13)
            try:
                full_metrics = collector.run(model, x, aux, "full")
                self.assertEqual(len(full_metrics), 4)
                self.assertEqual(collector.shape, (1, 2, 64, 16))
                deg_metrics = collector.run(model, x, aux, "4x")
                for layer in deg_metrics.values():
                    self.assertTrue(torch.equal(layer["js"], torch.zeros(1, 2).double()))
                    self.assertTrue(torch.equal(layer["retention_at_5"], torch.ones(1, 2).double()))
                self.assertTrue(all(m.last_attn is None for m in collector.modules))
                self.assertTrue(all(layer[1].fn.attn.last_attn is None for layer in model.cross_transformer.layers))
            finally:
                collector.close()
            after = model(x, aux)
            for a, b in zip(original, after):
                self.assertTrue(torch.equal(a, b))
            self.assertFalse(collector.reference)
            self.assertTrue(all(not m.capture_attention for m in collector.modules))


if __name__ == "__main__":
    unittest.main()
