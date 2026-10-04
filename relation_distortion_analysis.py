"""Frozen B0 relation analysis. R_full is a learned reference, not anatomical truth.

No engine/train imports, optimizer, backward, checkpoint writes, or tensor dumps.
Statistics are slice-weighted: queries -> heads -> layers, then population mean/std
over slices. Only one batch's full-reference cross relations are retained on CPU.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from config import build_config
from data.fastmri import SliceDataset
from data import transforms as T
from data.subsample import create_mask_for_mask_type
from models import build_model_from_name
from models.mca import MultiHeadCrossAttention


EPS = 1e-11  # Same normalization epsilon as ReconstructionTransform.
REL_EPS = 1e-12
KS = (1, 5, 10)
EXPERIMENT = "reconstruction_multi_cross"
LIMITATION = (
    "R_full is MTrans attention under fully sampled target observations, not "
    "anatomical correspondence, registration, or clinical ground truth. JS measures "
    "deviation, not anatomical error or proven harm to reconstruction. Target queries "
    "already include original pre-fusion; deeper auxiliary features are also updated "
    "by the original bidirectional interaction. Input PD is fixed, not its hidden states."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(tensor, name):
    require(bool(torch.isfinite(tensor).all()), name + " contains NaN or Inf")


def make_mask(acceleration, center_fraction):
    require(0 < center_fraction <= 1 / acceleration,
            "Center fraction must be positive and <= 1/acceleration")
    return create_mask_for_mask_type("random", [center_fraction], [acceleration])


def magnitude_from_raw(raw, mask_func):
    """Reuse B0 FFT/crop/magnitude utilities, including its crop fallback.

    Do NOT add padding handling: ReconstructionTransform does not apply it.
    Do NOT reconstruct pre-clamp pixels from clamped normalized images.
    """
    kspace, _, target, attrs, fname, _ = raw
    kspace = T.to_tensor(kspace)
    require(kspace.ndim == 3 and kspace.shape[-1] == 2,
            "Analysis supports B0 singlecoil k-space only")
    if mask_func is not None:
        seed = tuple(map(ord, fname))  # Full path, exactly as B0 validation.
        cols = kspace.shape[-2]
        low = round(cols * mask_func.center_fractions[0])
        require(low < cols and low <= cols / mask_func.accelerations[0],
                "Rounded center size makes this random-mask probability invalid")
        kspace, mask = T.apply_mask(kspace, mask_func, seed)
        repeated = mask_func(kspace.shape, seed)
        require(torch.equal(mask, repeated), "Validation mask is not reproducible")
    image = T.ifft2c(kspace)
    crop = tuple(target.shape[-2:]) if target is not None else tuple(attrs["recon_size"][:2])
    if image.shape[-2] < crop[1]:
        crop = (image.shape[-2], image.shape[-2])
    image = T.complex_abs(T.complex_center_crop(image, crop))
    finite(image, "Magnitude image")
    return image


def target_conditions(raw, masks, protocol):
    """One raw PDFS slice; A = own stats, B = common full-image stats."""
    full = magnitude_from_raw(raw, None)
    full_norm, full_mean, full_std = T.normalize_instance(full, eps=EPS)
    images = {"full": full_norm.clamp(-6, 6)}
    for condition, mask in masks.items():
        image = magnitude_from_raw(raw, mask)
        if protocol == "A":
            norm, _, _ = T.normalize_instance(image, eps=EPS)
        elif protocol == "B":
            norm = T.normalize(image, full_mean, full_std, eps=EPS)
        else:
            raise ValueError("Unknown normalization protocol: " + protocol)
        images[condition] = norm.clamp(-6, 6)
    for condition, image in images.items():
        finite(image, condition + " normalized input")
        require(image.shape == full.shape, "Condition image shapes differ")
    return images


class AnalysisDataset(Dataset):
    """Use the original pairing/order, reading each raw pair once per sample."""

    def __init__(self, root, protocol, center_fraction_8x, input_size):
        # root deliberately not resolved: the filename spelling controls B0's mask.
        self.raw = SliceDataset(os.path.join(root, "singlecoil_val"), None,
                                "singlecoil", sample_rate=1, mode="val")
        self.protocol = protocol
        self.input_size = input_size
        self.masks = {"4x": make_mask(4, 0.08)}
        if center_fraction_8x is not None:
            self.masks["8x"] = make_mask(8, center_fraction_8x)
        self.native = T.ReconstructionTransform("singlecoil", make_mask(4, 0.08), use_seed=True)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        pd, pdfs, pair_id = self.raw[index]
        require(pd[5] == pdfs[5], "PD/PDFS slice numbers differ")
        require(pd[2] is not None, "PD reconstruction_esc is required to reproduce pd[1]")
        auxiliary_sample = self.native(*pd)
        auxiliary = auxiliary_sample[1]  # Exact B0 PD target, NOT pd[0].
        images = target_conditions(pdfs, self.masks, self.protocol)
        # Check new model-native preprocessing against the original implementation.
        if self.protocol == "A":
            native_target = self.native(*pdfs)
            require(native_target[4:] == (pdfs[4], pdfs[5]), "Target identity changed")
            require(torch.equal(images["4x"], native_target[0]),
                    "Analysis 4x input differs from original ReconstructionTransform")
        require(auxiliary_sample[4:] == (pd[4], pd[5]), "Auxiliary identity changed")
        finite(auxiliary, "PD auxiliary")
        for image in [auxiliary] + list(images.values()):
            require(tuple(image.shape) == (self.input_size, self.input_size),
                    "Image shape incompatible with fixed B0 positional embeddings; no resizing allowed")
        return {"images": {k: v.unsqueeze(0) for k, v in images.items()},
                "auxiliary": auxiliary.unsqueeze(0), "pd_fname": pd[4],
                "target_fname": pdfs[4], "slice_num": pdfs[5], "pair_id": pair_id}


def split_attention(attn, query_chunk=128):
    """Validate pre-dropout probabilities and copy only cross slice to CPU."""
    require(attn.ndim == 4, "Expected attention [B,H,Q,K]")
    n_query, n_keys = attn.shape[-2:]
    require(n_keys > n_query, "No auxiliary keys after self keys")
    # Chunk validation avoids allocating a boolean tensor for the entire attention.
    for start in range(0, n_query, query_chunk):
        part = attn[..., start:start + query_chunk, :]
        finite(part, "Attention")
        require(bool((part >= 0).all()), "Negative attention probability")
        mass = part[..., :n_query].sum(-1) + part[..., n_query:].sum(-1)
        require(torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=1e-5),
                "Self mass + cross mass does not equal 1; stop analysis")
    # copy=True prevents retaining the complete tensor when running on CPU.
    return attn[..., n_query:].to(device="cpu", dtype=torch.float32, copy=True)


def conditional_relation(cross):
    finite(cross, "Cross relation")
    require(bool((cross >= 0).all()), "Negative cross attention")
    mass = cross.sum(-1, keepdim=True)
    require(bool((mass > 0).all()), "Zero auxiliary mass: conditional relation is undefined")
    conditional = cross / (mass + REL_EPS)
    require(torch.allclose(conditional.sum(-1), torch.ones_like(mass[..., 0]),
                           atol=1e-5, rtol=1e-5),
            "Conditional relation does not sum to 1 (possibly near-zero auxiliary mass)")
    return conditional, mass[..., 0]


def js_divergence(p, q):
    """Natural-log Jensen-Shannon divergence, returning [B,H,Q]."""
    require(p.shape == q.shape, "JS distribution shapes differ")
    midpoint = 0.5 * (p + q)
    log_mid = midpoint.clamp_min(REL_EPS).log()
    return 0.5 * ((p * (p.clamp_min(REL_EPS).log() - log_mid)).sum(-1)
                  + (q * (q.clamp_min(REL_EPS).log() - log_mid)).sum(-1))


def stable_top_indices(probability, k):
    require(0 < k <= probability.shape[-1], "k must be <= number of auxiliary tokens")
    # Exact ties go to the lower auxiliary-token index. No perturbation of values.
    return torch.argsort(probability, dim=-1, descending=True, stable=True)[..., :k]


def retention_from_indices(reference, degraded, k):
    return (reference[..., :k, None] == degraded[..., None, :k]).any(-1).float().mean(-1)


def relation_metrics(cross, relation_full=None, query_chunk=128):
    """Compute metrics BEFORE averaging heads. Return per-slice/head [B,H]."""
    require(cross.ndim == 4 and cross.shape[-1] >= max(KS), "Need >=10 auxiliary tokens")
    if relation_full is not None:
        require(cross.shape == relation_full.shape, "Full/degraded relation shapes differ")
    totals = {}
    for start in range(0, cross.shape[-2], query_chunk):
        p, mass = conditional_relation(cross[..., start:start + query_chunk, :])
        values = {"attention_mass": mass}
        if relation_full is not None:
            ref, _ = conditional_relation(relation_full[..., start:start + query_chunk, :])
            values["js"] = js_divergence(ref, p)
            ref_top = stable_top_indices(ref, max(KS))
            top = stable_top_indices(p, max(KS))
            for k in KS:
                values["retention_at_" + str(k)] = retention_from_indices(ref_top, top, k)
        for name, value in values.items():
            finite(value, name)
            # Float64 sums keep results stable across different query chunk sizes.
            subtotal = value.double().sum(-1)
            totals[name] = totals.get(name, 0) + subtotal
    return {name: value / cross.shape[-2] for name, value in totals.items()}


class AttentionCollector:
    """Drain each enabled last_attn at module exit, retaining no GPU cache.

    Only layers[i][0].fn.attn is target-query attention. layers[i][1] is reverse.
    Hooks return None and never modify module inputs/outputs or attention values.
    """

    def __init__(self, model, depth, heads, query_chunk):
        self.modules = [layer[0].fn.attn for layer in model.cross_transformer.layers]
        require(len(self.modules) == depth, "Unexpected target-attention layer count")
        require(all(isinstance(m, MultiHeadCrossAttention) and m.num_heads == heads
                    for m in self.modules), "Unexpected target-attention module/head count")
        self.query_chunk = query_chunk
        self.reference = {}
        self.results = {}
        self.condition = None
        self.shape = None
        self.handles = []
        for index, module in enumerate(self.modules):
            module.capture_attention = True
            self.handles.append(module.register_forward_hook(self._hook(index)))

    def _hook(self, index):
        def capture(module, inputs, output):
            attn = module.last_attn
            module.last_attn = None
            require(attn is not None, "Missing pre-dropout attention cache")
            require(index not in self.results, "Attention module ran more than once")
            cross = split_attention(attn, self.query_chunk)
            del attn
            require(cross.shape[-2] == inputs[0].shape[1], "Query length mismatch")
            require(cross.shape[-1] == inputs[1].shape[1], "Auxiliary key length mismatch")
            if self.shape is None:
                self.shape = tuple(cross.shape)
            require(tuple(cross.shape) == self.shape, "Relation shapes differ across layers/conditions")
            if self.condition == "full":
                self.reference[index] = cross
                self.results[index] = relation_metrics(cross, query_chunk=self.query_chunk)
            else:
                self.results[index] = relation_metrics(cross, self.reference[index], self.query_chunk)
        return capture

    def run(self, model, target, auxiliary, condition):
        require(not model.training and not torch.is_grad_enabled(), "Expected eval/no_grad inference")
        require(condition in ("full", "4x", "8x"), "Unknown degradation condition")
        if condition != "full":
            require(len(self.reference) == len(self.modules), "Run full reference before degraded conditions")
        self.condition = condition
        self.results = {}
        model(target, auxiliary)
        require(len(self.results) == len(self.modules), "Not all target attention layers executed")
        return self.results

    def clear_batch(self):
        self.reference.clear()
        self.results = {}
        self.shape = None
        for module in self.modules:
            module.last_attn = None

    def close(self):
        self.clear_batch()
        for handle in self.handles:
            handle.remove()
        for module in self.modules:
            module.capture_attention = False


class Moments:
    """Online float64 Welford population moments across per-slice metric means."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values):
        for value in values.tolist():
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (value - self.mean)

    def summary(self):
        return self.n, self.mean, math.sqrt(max(0, self.m2 / self.n))


def accumulate(accumulators, protocol, condition, results):
    def add(level, layer, head, name, values):
        key = (level, protocol, condition, layer, head, name)
        accumulators.setdefault(key, Moments()).update(values)
    names = results[0].keys()
    for layer, metrics in sorted(results.items()):
        for name, values in metrics.items():
            for head in range(values.shape[1]):
                add("head", layer + 1, head + 1, name, values[:, head])
            add("layer", layer + 1, "all", name, values.mean(1))
    for name in names:
        values = torch.stack([results[layer][name].mean(1) for layer in sorted(results)], dim=1)
        add("global", "all", "all", name, values.mean(1))


def git_value(*arguments):
    try:
        return subprocess.check_output(["git", *arguments], cwd=Path(__file__).parent,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_dir(base, checkpoint, configured_weights):
    base = Path(base).resolve()
    for protected in (checkpoint.parent.resolve(), Path(configured_weights).resolve()):
        require(base != protected and protected not in base.parents,
                "Output directory must not be inside a checkpoint directory")
    run = base / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    run.mkdir(parents=True, exist_ok=False)
    return run


def load_checkpoint(model, checkpoint_path, cfg):
    require(checkpoint_path.is_file(), "Checkpoint not found: " + str(checkpoint_path))
    # B0 checkpoint contains a YACS config; load only the user's trusted checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, dict) and "model" in checkpoint, "Expected B0 checkpoint['model']")
    state = checkpoint["model"]
    keys = list(state)
    if keys and all(key.startswith("module.") for key in keys):
        state = {key[len("module."):]: value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    saved_cfg = checkpoint.get("args")
    if saved_cfg is not None:
        for section, fields in {
            "MODEL": list(cfg.MODEL.keys()),
            "TRANSFORMS": ["MASKTYPE", "CENTER_FRACTIONS", "ACCELERATIONS"],
            "DATASET": ["CHALLENGE"],
        }.items():
            for field in fields:
                require(saved_cfg[section][field] == cfg[section][field],
                        "Checkpoint configuration mismatch: " + section + "." + field)
        require(saved_cfg.INPUT_SIZE == cfg.INPUT_SIZE, "Checkpoint INPUT_SIZE mismatch")
    info = {"checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_best_epoch": checkpoint.get("best_epoch"),
            "checkpoint_best_status": checkpoint.get("best_status"),
            "checkpoint_config_verified": saved_cfg is not None}
    print("Strict checkpoint load succeeded:", json.dumps(info, default=str), flush=True)
    if saved_cfg is None:
        print("WARNING: checkpoint has no saved config; preprocessing provenance cannot be verified.", flush=True)
    if info["checkpoint_best_epoch"] is None or info["checkpoint_best_status"] is None:
        print("WARNING: checkpoint best metadata missing; recorded as null, not inferred.", flush=True)
    return info


def write_outputs(run, accumulators, metadata):
    file_levels = {"global": "summary", "layer": "per_layer", "head": "per_head"}
    all_rows = []
    for level, suffix in file_levels.items():
        rows = []
        for key, moments in sorted(accumulators.items()):
            scope, protocol, condition, layer, head, metric = key
            if scope != level:
                continue
            count, mean, std = moments.summary()
            rows.append(dict(protocol=protocol, condition=condition, layer=layer, head=head,
                             metric=metric, sample_count=count, mean=mean, std=std))
        require(bool(rows), "No analysis results; no summary files written")
        with open(run / ("relation_distortion_" + suffix + ".csv"), "x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        all_rows.extend(rows)
    text = ["Relation Distortion Analysis", json.dumps(metadata, indent=2, default=str), "", LIMITATION,
            "JS uses natural logarithms; retention@k = |TopK_full intersect TopK_deg| / k.",
            "Ties: descending stable sort, lower token index first.",
            "Cross mass uses raw auxiliary weights. JS/retention use conditional auxiliary probabilities.",
            "Means/std: average queries per slice/head; then average heads for layers,",
            "and layers for global. Population std across equally weighted slices (ddof=0).",
            "Slices within a volume are correlated; this std is descriptive, not a significance test.",
            "Full has mass only; its reference JS/retention entries are omitted, not missing data.", ""]
    if metadata["center_fraction_8x"] is None:
        text.append("8x explicitly skipped: the repository does not establish its experimental center fraction.")
    for row in all_rows:
        if row["head"] == "all":
            text.append("{protocol} {condition} layer={layer} {metric}: {mean:.9g} +/- {std:.9g} (n={sample_count})".format(**row))
    # Report both supported and unsupported directions, with no sample selection.
    lookup = {(r["protocol"], r["condition"], r["layer"], r["metric"]): r["mean"]
              for r in all_rows if r["head"] == "all"}
    for (protocol, condition, layer, metric), value in lookup.items():
        other = lookup.get((protocol, "4x", layer, metric))
        if condition == "8x" and other is not None:
            text.append(f"{protocol} layer={layer} {metric}: 8x minus 4x = {value - other:.9g}")
    with open(run / "relation_distortion_summary.txt", "x", encoding="utf-8") as stream:
        stream.write("\n".join(text) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="weights_reconstruction_multi_cross_paper_random4x/best.pth")
    parser.add_argument("--data-root", help="Parent of singlecoil_val; preserve B0 path spelling for masks")
    parser.add_argument("--protocol", choices=("A", "B"), default="A")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--center-fraction-8x", type=float,
                       help="Explicit research protocol value; repository does not establish it")
    group.add_argument("--skip-8x", action="store_true", help="Explicitly run full/4x only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--query-chunk", type=int, default=128)
    parser.add_argument("--output-dir", default="relation_analysis_outputs")
    args = parser.parse_args(argv)
    for name in ("batch_size", "query_chunk"):
        if getattr(args, name) <= 0:
            parser.error(name + " must be positive")
    if args.num_workers < 0 or any(v == 0 or v < -1 for v in (args.max_batches, args.max_samples)):
        parser.error("num-workers must be >=0; subset limits must be -1 or positive")
    if args.center_fraction_8x is not None and not 0 < args.center_fraction_8x <= 1 / 8:
        parser.error("8x center fraction must be >0 and <=0.125")
    return args


def main(args):
    cfg = build_config(EXPERIMENT).clone()
    require(cfg.DATASET.CHALLENGE == "singlecoil" and cfg.TRANSFORMS.MASKTYPE == "random"
            and list(cfg.TRANSFORMS.CENTER_FRACTIONS) == [0.08]
            and list(cfg.TRANSFORMS.ACCELERATIONS) == [4], "Expected frozen random4x B0 config")
    checkpoint_path = Path(args.checkpoint).resolve()
    require(checkpoint_path.is_file(), "Checkpoint not found: " + str(checkpoint_path))
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no silent CPU fallback")
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_model_from_name(cfg, EXPERIMENT)
    checkpoint_info = load_checkpoint(model, checkpoint_path, cfg)
    model.requires_grad_(False)
    model.eval().to(device)
    require(all(not m.training for m in model.modules()), "Model is not fully in eval mode")
    parameter_count = sum(p.numel() for p in model.parameters())
    state_keys = tuple(model.state_dict())
    root = args.data_root if args.data_root is not None else cfg.DATASET.ROOT
    dataset = AnalysisDataset(root, args.protocol, args.center_fraction_8x, cfg.INPUT_SIZE)
    available = len(dataset)
    limit = available
    if args.max_samples > 0:
        limit = min(limit, args.max_samples)
    if args.max_batches > 0:
        limit = min(limit, args.max_batches * args.batch_size)
    require(limit > 0, "Validation split is empty")
    selected = Subset(dataset, range(limit))
    loader = DataLoader(selected, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    run = create_run_dir(args.output_dir, checkpoint_path, cfg.OUTPUTDIR)
    split_file = Path(dataset.raw.csv_file)
    metadata = dict(checkpoint_info, experiment=EXPERIMENT, protocol=args.protocol,
                    timestamp=datetime.now(timezone.utc).isoformat(), git_commit=git_value("rev-parse", "HEAD"),
                    git_status=git_value("status", "--short"), mask_type="random", center_fraction_4x=0.08,
                    center_fraction_8x=args.center_fraction_8x, seed=cfg.SEED, device=str(device),
                    mask_seed="tuple(map(ord, full fname)) for both accelerations; masks not guaranteed nested",
                    data_root=root, split_file=str(split_file), split_sha256=sha256_file(split_file),
                    available_validation_samples=available, selected_samples=limit,
                    subset_rule="First N in original CSV pair order, then ascending slice; no shuffle/filter",
                    max_batches=args.max_batches, max_samples=args.max_samples,
                    batch_size=args.batch_size, num_workers=args.num_workers, query_chunk=args.query_chunk,
                    P1=cfg.MODEL.P1, P2=cfg.MODEL.P2, CTDEPTH=cfg.MODEL.CTDEPTH,
                    num_heads=cfg.MODEL.TRANSFORMER_NUM_HEADS, parameters=parameter_count,
                    torch_version=torch.__version__, numpy_version=np.__version__,
                    normalization="Own input mean/std" if args.protocol == "A" else "Shared full target mean/std",
                    auxiliary="B0 pd[1]: reconstruction_esc cropped, normalized with masked PD random4x stats, clamped",
                    limitation=LIMITATION, config=cfg.dump(), status="running")
    metadata_path = run / "metadata.json"
    with open(metadata_path, "x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    collector = AttentionCollector(model, cfg.MODEL.CTDEPTH, cfg.MODEL.TRANSFORMER_NUM_HEADS, args.query_chunk)
    require(tuple(model.state_dict()) == state_keys
            and sum(p.numel() for p in model.parameters()) == parameter_count,
            "Attention instrumentation changed state_dict or parameter count")
    accumulators = {}
    sample_count = batch_count = 0
    print("Output:", run, "Selected slices:", limit, flush=True)
    try:
        with open(run / "samples.csv", "x", newline="", encoding="utf-8") as stream, torch.no_grad():
            manifest = csv.writer(stream)
            manifest.writerow(["target_fname", "pd_fname", "slice_num", "pair_id"])
            for batch in loader:
                auxiliary = batch["auxiliary"].to(device)
                frozen_auxiliary = auxiliary.clone()
                # All conditions share this single identity record and auxiliary tensor.
                conditions = ("full", "4x") if args.skip_8x else ("full", "4x", "8x")
                require(set(batch["images"]) == set(conditions), "Missing or unexpected target condition")
                for condition in conditions:
                    images = batch["images"][condition]
                    require(torch.equal(auxiliary, frozen_auxiliary), "Auxiliary input changed")
                    target = images.to(device)
                    results = collector.run(model, target, auxiliary, condition)
                    require(torch.equal(auxiliary, frozen_auxiliary), "Forward modified auxiliary input")
                    accumulate(accumulators, args.protocol, condition, results)
                    if batch_count == 0:
                        print(condition, "cross shape per layer:", collector.shape, flush=True)
                        metadata["cross_relation_shape_excluding_batch"] = collector.shape[1:]
                    del target, results
                for i in range(auxiliary.shape[0]):
                    manifest.writerow([batch["target_fname"][i], batch["pd_fname"][i],
                                       int(batch["slice_num"][i]), int(batch["pair_id"][i])])
                sample_count += auxiliary.shape[0]
                batch_count += 1
                collector.clear_batch()
                del auxiliary, frozen_auxiliary, batch
                print(f"Completed batch {batch_count}/{len(loader)}, slices={sample_count}", flush=True)
        require(sample_count == limit, "Unexpected number of processed samples")
        metadata.update(validation_sample_count=sample_count, batch_count=batch_count,
                        full_validation=sample_count == available, status="complete")
        write_outputs(run, accumulators, metadata)
    except Exception as error:
        metadata.update(status="failed", error=str(error), validation_sample_count=sample_count,
                        batch_count=batch_count)
        raise
    finally:
        collector.close()
        with open(metadata_path, "w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, default=str)
    print("Analysis complete:", run, flush=True)


if __name__ == "__main__":
    main(parse_args())
