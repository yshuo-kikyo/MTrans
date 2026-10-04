"""Analysis-only, mass-preserving Layer-1 oracle on native random4x B0 inputs.

For each query/head, A_4=[S_4,C_4], m_4=sum(C_4), P_full=C_full/sum(C_full).
Intervene with A_oracle=[S_4,m_4*P_full], keeping V from the current 4x forward.
Only L1 target-query attention is changed, via a temporary attn_drop pre-hook.
No training, feature transplantation, checkpoint writes, or other layer overrides.
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from config import build_config
from models import build_model_from_name
from models.mca import MultiHeadCrossAttention
from relation_distortion_analysis import (
    AnalysisDataset, EXPERIMENT, Moments, create_run_dir, finite, git_value,
    load_checkpoint, require, sha256_file, split_attention, target_conditions,
)
from util.metric import nmse, psnr, ssim


METRICS = {"PSNR": psnr, "SSIM": ssim, "NMSE": nmse}
CONDITIONS = ("normal_4x", "oracle_L1", "oracle_minus_normal")
LIMITATIONS = (
    "Oracle uses privileged fully sampled target information; it is not deployable inference "
    "or anatomical truth. Native normalization (Protocol A) is used for full and 4x. "
    "PD is the original pd[1], normalized with masked PD statistics. Pre-fusion remains. "
    "Only the L1 auxiliary conditional distribution is substituted; self weights and 4x cross "
    "mass are preserved. Later features, reverse attention, V and attention are recomputed "
    "by the original model and may change downstream of this intervention. Improvement "
    "supports utility of this substitution, not proof of anatomically incorrect relations; "
    "no improvement does not exclude relation distortion elsewhere or in cross mass."
)


class OracleDataset(AnalysisDataset):
    """Original pairing plus the exact target/mean/std used by engine.evaluate."""

    def __init__(self, root, input_size):
        super().__init__(root, "A", None, input_size)

    def __getitem__(self, index):
        pd, pdfs, pair_id = self.raw[index]
        require(pd[5] == pdfs[5], "PD/PDFS slice mismatch")
        require(pd[2] is not None and pdfs[2] is not None,
                "Both reconstruction_esc targets are required for B0 evaluation")
        pd_sample = self.native(*pd)
        target_sample = self.native(*pdfs)
        images = target_conditions(pdfs, self.masks, "A")
        require(torch.equal(images["4x"], target_sample[0]),
                "4x construction differs from original ReconstructionTransform")
        require(pd_sample[4:] == (pd[4], pd[5]) and target_sample[4:] == (pdfs[4], pdfs[5]),
                "Transform changed sample identity")
        auxiliary = pd_sample[1]
        for tensor in [auxiliary, target_sample[1], *images.values()]:
            finite(tensor, "Preprocessed image")
            require(tuple(tensor.shape) == (self.input_size, self.input_size),
                    "Image does not match B0 positional embedding size; no automatic resize")
        return {"full": images["full"].unsqueeze(0), "4x": images["4x"].unsqueeze(0),
                "auxiliary": auxiliary.unsqueeze(0), "target": target_sample[1],
                "mean": target_sample[2], "std": target_sample[3],
                "target_fname": pdfs[4], "pd_fname": pd[4], "slice_num": pdfs[5], "pair_id": pair_id}


def unit_rows(weights):
    """Exact conditional normalization, up to rounding; reject undefined rows.

    Unlike diagnostic C/(mass+eps), no additive epsilon steals cross mass here.
    Compute the division in float64, then return the original floating dtype.
    """
    require(weights.is_floating_point(), "Expected floating relation weights")
    finite(weights, "Relation weights")
    require(bool((weights >= 0).all()), "Negative relation weights")
    precise = weights.double()
    mass = precise.sum(-1, keepdim=True)
    require(bool((mass > 0).all()), "Full auxiliary mass is zero; conditional oracle undefined")
    result = (precise / mass).to(weights.dtype)
    require(torch.allclose(result.sum(-1), torch.ones_like(result[..., 0]), atol=1e-6, rtol=1e-6),
            "Conditional reference does not sum to one")
    return result


def replace_cross_distribution(attention, reference, query_chunk=128):
    """Return a new full attention, preserving every self weight and cross mass.

    The reference may be on CPU. Only one query chunk is transferred at a time.
    The original softmax tensor is never modified in-place, nor re-softmaxed.
    """
    require(attention.ndim == 4 and reference.ndim == 4, "Expected [B,H,Q,K] tensors")
    require(attention.dtype == torch.float32, "Oracle analysis requires FP32 attention (no AMP)")
    require(query_chunk > 0, "query_chunk must be positive")
    queries = attention.shape[-2]
    require(attention.shape[-1] > queries, "Missing auxiliary keys")
    require(tuple(reference.shape) == (*attention.shape[:-1], attention.shape[-1] - queries),
            "Reference does not match current batch/head/query/auxiliary-token shape")
    replaced = attention.clone()
    max_mass_error = 0.0
    for start in range(0, queries, query_chunk):
        original = attention[..., start:start + query_chunk, :]
        finite(original, "Original 4x attention")
        require(bool((original >= 0).all()), "Negative original attention")
        ones = torch.ones_like(original[..., 0])
        require(torch.allclose(original.sum(-1), ones, atol=1e-6, rtol=1e-6),
                "Original joint softmax does not sum to one")
        p = reference[..., start:start + query_chunk, :].to(attention.device, attention.dtype)
        p = unit_rows(p)
        mass_4x = original[..., queries:].sum(-1, keepdim=True)
        new_cross = mass_4x * p
        finite(new_cross, "Intervened cross attention")
        require(torch.allclose(new_cross.sum(-1), mass_4x[..., 0], atol=1e-7, rtol=1e-5),
                "Oracle changed the 4x cross-modal mass")
        replaced[..., start:start + query_chunk, queries:] = new_cross
        new = replaced[..., start:start + query_chunk, :]
        require(torch.equal(new[..., :queries], original[..., :queries]), "Oracle changed self weights")
        require(torch.allclose(new.sum(-1), original.sum(-1), atol=1e-6, rtol=1e-6),
                "Oracle changed total attention mass")
        max_mass_error = max(max_mass_error, float((new_cross.sum(-1) - mass_4x[..., 0]).abs().max()))
    return replaced, max_mass_error


class Layer1Oracle:
    """One-batch lifetime for a single L1 reference; all temporary hooks are removed."""

    def __init__(self, model, query_chunk=128):
        self.model = model
        self.attention = model.cross_transformer.layers[0][0].fn.attn
        require(isinstance(self.attention, MultiHeadCrossAttention), "Unexpected L1 attention module")
        require(not self.attention.attn_drop._forward_pre_hooks, "L1 dropout already has a pre-hook")
        require(all(not m.capture_attention for m in model.modules() if isinstance(m, MultiHeadCrossAttention)),
                "Disable distortion-analysis caches before oracle analysis")
        self.query_chunk = query_chunk
        self.reference = None
        self.last_audit = {}

    def _forward_with_hook(self, target, auxiliary, hook):
        calls = 0

        def checked_hook(module, inputs):
            nonlocal calls
            calls += 1
            require(calls == 1 and len(inputs) == 1, "L1 attention must execute exactly once")
            return hook(inputs[0])

        handle = self.attention.attn_drop.register_forward_pre_hook(checked_hook)
        try:
            output = self.model(target, auxiliary)
            require(calls == 1, "L1 attention hook did not execute")
            return output
        finally:
            handle.remove()

    def compare(self, full, degraded, auxiliary):
        require(not torch.is_grad_enabled(), "Oracle requires no_grad inference")
        require(all(not module.training for module in self.model.modules()), "Oracle requires model.eval()")
        require(all(not p.requires_grad for p in self.model.parameters()), "Oracle requires frozen model parameters")
        require(self.reference is None, "Stale reference from a previous batch")
        require(full.shape == degraded.shape == auxiliary.shape, "Input shapes differ")
        aux_original = auxiliary.clone()
        degraded_original = degraded.clone()

        def capture(attention):
            require(attention.dtype == torch.float32, "Full reference must use FP32 attention")
            cross = split_attention(attention.detach(), self.query_chunk)
            # Keep only conditional L1 cross attention on CPU, never full V/features.
            self.reference = unit_rows(cross)
            self.last_audit = {"reference_shape": list(self.reference.shape)}
            return None

        def intervene(attention):
            replaced, error = replace_cross_distribution(attention, self.reference, self.query_chunk)
            self.last_audit["max_cross_mass_error"] = error
            return (replaced,)

        try:
            # Full output is intentionally discarded; only its conditional relation is cached.
            self._forward_with_hook(full, auxiliary, capture)
            require(torch.equal(auxiliary, aux_original), "Full forward changed auxiliary input")
            # Baseline forward has NO intervention hook.
            normal = self.model(degraded, auxiliary)[0]
            require(torch.equal(auxiliary, aux_original) and torch.equal(degraded, degraded_original),
                    "Normal forward modified its inputs")
            normal_cpu = normal.detach().cpu()
            del normal
            oracle = self._forward_with_hook(degraded, auxiliary, intervene)[0]
            require(torch.equal(auxiliary, aux_original) and torch.equal(degraded, degraded_original),
                    "Oracle forward modified its inputs")
            finite(normal_cpu, "Normal prediction")
            finite(oracle, "Oracle prediction")
            return normal_cpu, oracle.detach().cpu()
        finally:
            self.reference = None


def denormalize_b0(prediction, target, mean, std):
    """Match engine.evaluate exactly: *std + mean, not *(std+eps)+mean.

    The target is original pdfs[1], already normalized and clamped; predictions
    receive no extra clamp. Both conditions use the SAME 4x mean/std and label.
    """
    require(prediction.ndim == 4 and prediction.shape[1] == 1, "Expected [B,1,H,W] output")
    require(prediction[:, 0].shape == target.shape, "Prediction/target shapes differ")
    require(mean.ndim == std.ndim == 1 and len(mean) == len(prediction), "Expected per-slice statistics")
    finite(mean, "4x mean")
    finite(std, "4x std")
    mean, std = mean[:, None, None], std[:, None, None]
    return prediction[:, 0] * std + mean, target * std + mean


def validation_volume_counts(examples):
    """Reject ambiguous duplicate target volumes rather than silently overwriting them."""
    counts = {}
    previous = None
    for pd_fname, fname, slice_num, _, _, pair_id in examples:
        identity = (fname, pd_fname, pair_id)
        if identity != previous:
            require(fname not in counts, "Repeated target volume in split; B0 overwrite semantics need review: " + fname)
            counts[fname] = 0
            previous = identity
        require(slice_num == counts[fname], "Expected ascending contiguous slices from zero")
        counts[fname] += 1
    return counts


class VolumeComparison:
    """Buffer one volume on CPU; use original metric functions and volume weighting."""

    def __init__(self, expected_counts, writer):
        self.expected_counts = expected_counts
        self.writer = writer
        self.current = None
        self.slices = []
        self.targets, self.normal, self.oracle = [], [], []
        self.moments = {(condition, metric): Moments() for condition in CONDITIONS for metric in METRICS}
        self.volume_count = self.partial_volume_count = self.sample_count = 0

    def add(self, fname, slice_num, target, normal, oracle):
        if self.current is not None and fname != self.current:
            self.flush()
        self.current = fname
        require(slice_num == len(self.slices), "Expected sequential, non-duplicate volume slices")
        self.slices.append(slice_num)
        # Copy one slice: views would retain an entire batch across volume boundaries.
        for buffer, value in ((self.targets, target), (self.normal, normal), (self.oracle, oracle)):
            require(value.ndim == 2, "Expected 2D reconstruction slice")
            finite(value, "Denormalized reconstruction")
            buffer.append(value.numpy().copy())

    def flush(self):
        if not self.slices:
            return
        target, normal, oracle = map(np.stack, (self.targets, self.normal, self.oracle))
        require(np.max(target) > 0 and np.linalg.norm(target) > 0,
                "Degenerate volume target: PSNR/SSIM/NMSE undefined")
        row = {"target_fname": self.current, "slice_count": len(self.slices),
               "expected_slice_count": self.expected_counts[self.current],
               "complete_volume": len(self.slices) == self.expected_counts[self.current]}
        for metric, function in METRICS.items():
            baseline, intervened = float(function(target, normal)), float(function(target, oracle))
            require(np.isfinite(baseline) and np.isfinite(intervened),
                    "Nonfinite " + metric + " (possibly perfect prediction); no volume silently omitted")
            delta = intervened - baseline
            for condition, value in zip(CONDITIONS, (baseline, intervened, delta)):
                row[condition + "_" + metric] = value
                self.moments[condition, metric].update(torch.tensor([value], dtype=torch.float64))
        self.writer.writerow(row)
        self.volume_count += 1
        self.partial_volume_count += int(not row["complete_volume"])
        self.sample_count += len(self.slices)
        self.current = None
        self.slices.clear()
        self.targets.clear()
        self.normal.clear()
        self.oracle.clear()


def write_summary(run, comparison, metadata):
    rows = []
    for condition in CONDITIONS:
        for metric in METRICS:
            count, mean, std = comparison.moments[condition, metric].summary()
            rows.append(dict(condition=condition, metric=metric, volume_count=count,
                             mean=mean, std=std, partial_volume_count=comparison.partial_volume_count))
    with open(run / "oracle_l1_summary.csv", "x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["Oracle L1, native random4x target", json.dumps(metadata, indent=2, default=str), "",
             "A_oracle=[S_4, sum(C_4)*P_full]. Full mass is NOT transplanted.",
             "Metrics use engine.evaluate target clamping/denormalization and util.metric functions.",
             "Volume-weighted mean and population std; delta = oracle minus normal per volume.",
             "Positive PSNR/SSIM delta and negative NMSE delta indicate improvement.", LIMITATIONS]
    if not metadata["full_validation"]:
        lines.append("FIXED PREFIX ONLY: metrics may use partial volumes and their own target max; not B0 full-validation scores.")
    lines.extend("{condition} {metric}: {mean:.9g} +/- {std:.9g} (volumes={volume_count})".format(**row) for row in rows)
    with open(run / "oracle_l1_summary.txt", "x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights_reconstruction_multi_cross_paper_random4x/best.pth")
    parser.add_argument("--data-root", help="B0 root containing singlecoil_val; preserve filename seed spelling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--output-dir", default="oracle_relation_outputs")
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.query_chunk <= 0 or args.num_workers < 0:
        parser.error("batch-size/query-chunk must be positive; num-workers must be nonnegative")
    if args.max_batches != -1 and args.max_batches <= 0:
        parser.error("max-batches must be -1 or a positive integer")
    return args


def main(args):
    cfg = build_config(EXPERIMENT).clone()
    require(cfg.DATASET.CHALLENGE == "singlecoil" and cfg.TRANSFORMS.MASKTYPE == "random"
            and list(cfg.TRANSFORMS.CENTER_FRACTIONS) == [0.08]
            and list(cfg.TRANSFORMS.ACCELERATIONS) == [4], "Expected unchanged B0 random4x config")
    checkpoint = Path(args.checkpoint).resolve()
    require(checkpoint.is_file(), "Checkpoint not found: " + str(checkpoint))
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable; no silent device fallback")
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_model_from_name(cfg, EXPERIMENT)
    checkpoint_info = load_checkpoint(model, checkpoint, cfg)
    model.requires_grad_(False)
    model.eval().to(device)
    # Version counters detect in-place changes without cloning B0's large state_dict.
    tensors = list(model.parameters()) + list(model.buffers())
    versions = [tensor._version for tensor in tensors]
    root = args.data_root if args.data_root is not None else cfg.DATASET.ROOT
    dataset = OracleDataset(root, cfg.INPUT_SIZE)
    expected_counts = validation_volume_counts(dataset.raw.examples)
    count = len(dataset) if args.max_batches == -1 else min(len(dataset), args.max_batches * args.batch_size)
    require(count > 0, "Empty validation selection")
    loader = DataLoader(Subset(dataset, range(count)), batch_size=args.batch_size, shuffle=False,
                        drop_last=False, num_workers=args.num_workers)
    run = create_run_dir(args.output_dir, checkpoint, cfg.OUTPUTDIR)
    metadata = dict(checkpoint_info, experiment=EXPERIMENT, timestamp=datetime.now(timezone.utc).isoformat(),
                    git_commit=git_value("rev-parse", "HEAD"), git_status=git_value("status", "--short"),
                    protocol="A (model-native only)", intervention_layer=1, acceleration=4, center_fraction=0.08,
                    mask_type="random", mask_seed="tuple(map(ord, full fname))", seed=cfg.SEED,
                    device=str(device), data_root=root, split_sha256=sha256_file(dataset.raw.csv_file),
                    config=cfg.dump(), torch_version=torch.__version__, numpy_version=np.__version__,
                    selected_samples=count, available_samples=len(dataset), full_validation=count == len(dataset),
                    batch_size=args.batch_size, max_batches=args.max_batches, num_workers=args.num_workers,
                    query_chunk=args.query_chunk, state_parameter_count=sum(p.numel() for p in model.parameters()),
                    intervention="[S_4, sum(C_4)*P_full], all L1 heads, post-softmax/pre-dropout",
                    subset_rule="Original split order, first max-batches*batch-size samples; no result filtering",
                    metric_target="Original pdfs[1] (clamped) * std_4x + mean_4x, exactly engine.evaluate",
                    limitation=LIMITATIONS, status="running")
    metadata_path = run / "metadata.json"
    with open(metadata_path, "x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    processed = batches = 0
    worst_mass_error = 0.0
    comparison = None
    try:
        oracle = Layer1Oracle(model, args.query_chunk)
        with open(run / "oracle_l1_per_volume.csv", "x", newline="", encoding="utf-8") as volume_file, \
                open(run / "samples.csv", "x", newline="", encoding="utf-8") as sample_file, torch.no_grad():
            fields = ["target_fname", "slice_count", "expected_slice_count", "complete_volume"]
            fields += [condition + "_" + metric for metric in METRICS for condition in CONDITIONS]
            writer = csv.DictWriter(volume_file, fieldnames=fields)
            writer.writeheader()
            manifest = csv.writer(sample_file)
            manifest.writerow(["target_fname", "pd_fname", "slice_num", "pair_id"])
            comparison = VolumeComparison(expected_counts, writer)
            for batch in loader:
                full, degraded, auxiliary = (batch[name].to(device) for name in ("full", "4x", "auxiliary"))
                normal, intervened = oracle.compare(full, degraded, auxiliary)
                normal, target = denormalize_b0(normal, batch["target"], batch["mean"], batch["std"])
                intervened, same_target = denormalize_b0(intervened, batch["target"], batch["mean"], batch["std"])
                require(torch.equal(target, same_target), "Evaluation labels differ between conditions")
                require([tensor._version for tensor in tensors] == versions, "Model parameters/buffers changed")
                worst_mass_error = max(worst_mass_error, oracle.last_audit["max_cross_mass_error"])
                metadata["reference_shape_excluding_batch"] = oracle.last_audit["reference_shape"][1:]
                for i, fname in enumerate(batch["target_fname"]):
                    slice_num = int(batch["slice_num"][i])
                    comparison.add(fname, slice_num, target[i], normal[i], intervened[i])
                    manifest.writerow([fname, batch["pd_fname"][i], slice_num, int(batch["pair_id"][i])])
                processed += len(normal)
                batches += 1
                del full, degraded, auxiliary, normal, intervened, target, same_target, batch
                print(f"Oracle L1 batch {batches}/{len(loader)}, slices={processed}, max mass error={worst_mass_error:.3g}", flush=True)
            comparison.flush()
        require(processed == count == comparison.sample_count, "Processed sample count mismatch")
        metadata.update(status="complete", validation_sample_count=processed, batch_count=batches,
                        volume_count=comparison.volume_count, partial_volume_count=comparison.partial_volume_count,
                        max_cross_mass_error=worst_mass_error, parameter_buffer_versions_unchanged=True)
        write_summary(run, comparison, metadata)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error), validation_sample_count=processed, batch_count=batches)
        raise
    finally:
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
    print("Oracle L1 comparison complete:", run, flush=True)


if __name__ == "__main__":
    main(parse_args())
