"""Localize mismatch damage with early addition ON/OFF in one frozen B0."""
import argparse
from collections import Counter
import inspect
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from config import build_config
from models import build_model_from_name
from models.cmmt_reconstruction_multi_cross import CrossCMMT
from cross_modal_dependency_audit import AuditDataset, run_condition, write_csv
from cross_modal_dependency_utils import (PILOT_VOLUMES, select_samples, VolumeAccumulator,
    paired_effects, descriptive)
from frozen_b0_pathway_audit import run_pathway, verify_forward_contract
from relation_distortion_analysis import (EXPERIMENT, require, finite, load_checkpoint,
    sha256_file, git_value, create_run_dir)

CONDITIONS = ("normal_matched", "normal_mismatched", "no_early_matched", "no_early_mismatched")
REGRESSION_PSNR = {"normal_matched": 20.83155125, "normal_mismatched": 20.60596273,
                   "no_early_matched": 20.20034065}
REGRESSION_ATOL = 1e-4  # dB, fixed before experiments; no relative tolerance.
LIMITATIONS = (
    "Same frozen B0 with inference-time early-add removal, not a retrained NoEarly model. "
    "Turning early fusion off changes feature scale/distribution and the matched baseline. "
    "Mismatch changes anatomy/content/correspondence and retains donor-native normalization. "
    "Attention remains fully active, including joint-softmax competition and bidirectional "
    "feature evolution. Localization is a descriptive difference of within-pathway damages, "
    "not a causal mediation proportion. Fractions are unstable for small positive ON damage. "
    "One cyclic donor per pair and ten preselected volumes limit generalization. Partial "
    "volumes are sanity only and cannot be compared to full-validation checkpoint scores."
)


def run_localization(model, target, matched, donor, condition):
    require(condition in CONDITIONS, "Unknown localization condition")
    if condition == "normal_matched":
        return run_condition(model, target, matched, donor, "normal")
    if condition == "normal_mismatched":
        return run_condition(model, target, matched, donor, "mismatched_aux")
    auxiliary = matched if condition == "no_early_matched" else donor
    return run_pathway(model, target, auxiliary, "no_early")


def run_four(model, target, matched, donor):
    """One donor tensor created by AuditDataset; never sampled inside conditions."""
    return {condition: run_localization(model, target, matched, donor, condition) for condition in CONDITIONS}


def damage_localization(values):
    on = paired_effects(values["normal_matched"], values["normal_mismatched"])
    off = paired_effects(values["no_early_matched"], values["no_early_mismatched"])
    result = []
    for metric, key in (("PSNR", "PSNR_drop"), ("SSIM", "SSIM_drop"), ("NMSE", "NMSE_increase")):
        d_on, d_off = on[key], off[key]
        localization = d_on - d_off
        if not all(math.isfinite(v) for v in (d_on, d_off, localization)):
            fraction, status = None, "undefined_nonfinite_damage"
        elif d_on <= 0:
            fraction, status = None, "undefined_nonpositive_on_damage"
        else:
            fraction, status = localization / d_on, "defined"
            if not math.isfinite(fraction):
                fraction, status = None, "undefined_nonfinite_fraction"
        result.append(dict(metric=metric, damage_on=d_on, damage_off=d_off, localization=localization,
                           reduction_fraction=fraction, fraction_status=status))
    return result


def mapping_record(dataset, index, donor_mean, donor_std):
    """Extend, never regenerate, the verified dependency mapping."""
    return dict(dataset.mapping(index), mapping_id=int(index),
                used_by_conditions="normal_mismatched;no_early_mismatched",
                mismatched_pd_normalization_mean=float(donor_mean),
                mismatched_pd_normalization_std=float(donor_std),
                normalization_source="donor own masked random4x magnitude; reconstruction_esc pd[1]")


def quick_regression(samples, volume_rows):
    applicable = (len(samples) == 1 and Path(samples[0]["fname"]).stem == "file1002538"
                  and int(samples[0]["slice_num"]) == 0)
    if not applicable:
        return "not_applicable_to_this_selection", []
    values = {row["condition"]: row["PSNR"] for row in volume_rows}
    checks = [dict(condition=c, expected_psnr=v, observed_psnr=values[c], absolute_tolerance=REGRESSION_ATOL,
                   absolute_error=abs(values[c]-v), passed=math.isfinite(values[c]) and abs(values[c]-v) <= REGRESSION_ATOL)
              for c, v in REGRESSION_PSNR.items()]
    return ("passed" if all(row["passed"] for row in checks) else "failed"), checks


def summarize(volume_rows, damage_rows, localization_rows, partial):
    rows = []
    def add(kind, label, metric, statistic, values, statuses=None):
        stats = descriptive(values)
        # Do not silently drop undefined fractions; summary stays blank if any missing.
        rows.append(dict(kind=kind, condition_or_early_state=label, metric=metric, statistic=statistic,
                         partial_volume=partial, volume_count=len(values),
                         defined_volume_count=sum(v is not None and math.isfinite(v) for v in values),
                         reason_counts=json.dumps(dict(Counter(statuses or [])), sort_keys=True), **stats))
    for c in CONDITIONS:
        for metric in ("PSNR", "SSIM", "NMSE"):
            add("raw", c, metric, "value", [r[metric] for r in volume_rows if r["condition"] == c])
    for early in ("on", "off"):
        for metric in ("PSNR", "SSIM", "NMSE"):
            add("mismatch_damage", early, metric, "damage",
                [r["damage"] for r in damage_rows if r["early_state"] == early and r["metric"] == metric])
    for metric in ("PSNR", "SSIM", "NMSE"):
        selected = [r for r in localization_rows if r["metric"] == metric]
        for statistic in ("localization", "reduction_fraction"):
            add("localization", "on_minus_off", metric, statistic, [r[statistic] for r in selected],
                [r["fraction_status"] for r in selected] if statistic == "reduction_fraction" else None)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target-volumes", nargs="+")
    selection.add_argument("--pilot-10-volumes", action="store_true")
    parser.add_argument("--output-dir", default="frozen_b0_mismatch_outputs")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_batches != -1 and args.max_batches < 1):
        parser.error("Invalid batch size/workers/limit")
    return args


def main(args):
    verify_forward_contract()
    cfg = build_config(EXPERIMENT).clone()
    require(cfg.DATASET.CHALLENGE == "singlecoil" and cfg.TRANSFORMS.MASKTYPE == "random"
            and list(cfg.TRANSFORMS.CENTER_FRACTIONS) == [.08] and list(cfg.TRANSFORMS.ACCELERATIONS) == [4], "Expected native B0")
    checkpoint = Path("weights_reconstruction_multi_cross_paper_random4x/best.pth").resolve()
    checksum = sha256_file(checkpoint)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA unavailable")
        torch.cuda.manual_seed_all(cfg.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model = build_model_from_name(cfg, EXPERIMENT)
    info = load_checkpoint(model, checkpoint, cfg)
    require(info["checkpoint_config_verified"], "Saved original B0 config required")
    model.eval().requires_grad_(False).to(device)
    state = list(model.parameters()) + list(model.buffers())
    versions = [value._version for value in state]
    root = args.data_root if args.data_root is not None else cfg.DATASET.ROOT
    dataset = AuditDataset(root, cfg.INPUT_SIZE)  # Exactly the existing mismatch implementation.
    indices = select_samples(dataset.pairs, PILOT_VOLUMES if args.pilot_10_volumes else args.target_volumes,
                             -1 if args.max_batches == -1 else args.max_batches * args.batch_size)
    require(bool(indices), "Empty selection")
    expected = Counter(str(dataset.raw.examples[i][1]) for i in indices)
    available = {p["target_fname"]: len(p["slices"]) for p in dataset.pairs.values()}
    partial = any(expected[f] != available[f] for f in expected)
    run = create_run_dir(args.output_dir, checkpoint, cfg.OUTPUTDIR)
    samples = [dict(dataset_index=i, fname=dataset.raw.examples[i][1], slice_num=dataset.raw.examples[i][2],
                    pd_fname=dataset.raw.examples[i][0], pair_id=dataset.raw.examples[i][5],
                    mismatch_mapping_id=i) for i in indices]
    write_csv(run / "samples.csv", samples)
    metadata = dict(info, checkpoint_sha256=checksum, git_commit=git_value("rev-parse", "HEAD"),
        git_status=git_value("status", "--short"), config=cfg.dump(), data_root=str(root),
        split_file=str(dataset.raw.csv_file), split_sha256=sha256_file(dataset.raw.csv_file),
        conditions=CONDITIONS, protocol="native B0", mask_type="random", acceleration=4, center_fraction=.08,
        mask_seed_rule="tuple(map(ord,full fname))", mismatch_implementation="unchanged cross_modal_dependency_audit.AuditDataset",
        donor_sharing="one AuditDataset item/tensor/mapping per target; same donor tensor supplied to both mismatch conditions",
        no_early_implementation="unchanged frozen_b0_pathway_audit.run_pathway(no_early); AST guard checked",
        cross_attention="fully active in ALL four conditions; no attention intervention hooks",
        forward_source_sha256=sha256_file(Path(inspect.getfile(CrossCMMT))),
        damage_definition="within each Early state: matched-mismatched for PSNR/SSIM, mismatched-matched for NMSE",
        localization_definition="damage_on-damage_off; positive means mismatch damage attenuated with Early OFF",
        fraction_definition="localization/damage_on only if damage_on>0 and finite; no epsilon, no clipping; descriptive not causal",
        fraction_summary="mean per-volume fractions, not ratio of means; blank if any undefined; counts/status retained",
        metric_definition="native GT/pred * std+mean, no pred clamp; existing VolumeAccumulator and util.metric on stacked volume slices",
        aggregation_unit="volume", summary_std_ddof=0, selected_volume_count=len(expected), selected_slice_count=len(indices),
        selected_volumes=list(expected), partial_volume=partial, regression_psnr=REGRESSION_PSNR,
        regression_absolute_tolerance_db=REGRESSION_ATOL, regression_rule="only file1002538 slice0 as sole sample; mismatch fails run, no pilot should proceed",
        torch_version=torch.__version__, numpy_version=np.__version__, seed=cfg.SEED,
        device=str(device), batch_size=args.batch_size, limitations=LIMITATIONS, status="running")
    meta_path = run / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    accumulator = VolumeAccumulator(expected, available, conditions=CONDITIONS)
    rows, mappings = [], []
    try:
        with torch.no_grad():
            loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            for batch in loader:
                target, matched, donor = (batch[k].to(device) for k in ("target", "auxiliary", "mismatched_auxiliary"))
                originals = [v.clone() for v in (target, matched, donor)]
                outputs = run_four(model, target, matched, donor)
                require("forward" not in model.__dict__, "Temporary forward leaked")
                restored = run_localization(model, target, matched, donor, "normal_matched")
                require(torch.equal(restored, outputs["normal_matched"]), "Normal output not restored")
                require(all(torch.equal(a,b) for a,b in zip(originals,(target,matched,donor)))
                        and [v._version for v in state] == versions, "Input/model state mutated")
                mean, std = batch["mean"][:,None,None], batch["std"][:,None,None]
                gt = batch["gt"] * std + mean
                predictions = {c: v.cpu().squeeze(1)*std+mean for c,v in outputs.items()}
                for value in (gt,*predictions.values()):
                    finite(value,"Physical-space reconstruction/GT")
                for b,fname in enumerate(batch["fname"]):
                    completed = accumulator.add(fname, int(batch["slice_num"][b]), gt[b].numpy(),
                                                {c:v[b].numpy() for c,v in predictions.items()})
                    if completed:
                        rows.extend(completed)
                    mappings.append(mapping_record(dataset,int(batch["dataset_index"][b]),batch["donor_mean"][b],batch["donor_std"][b]))
                print(f"Mismatch localization: {len(accumulator.completed)}/{len(expected)} volumes complete", flush=True)
        require(not accumulator.pending and len(accumulator.completed)==len(expected), "Incomplete aggregation")
        write_csv(run/"mismatch_mapping.csv",mappings)
        write_csv(run/"per_volume_metrics.csv",rows)
        damages,localizations=[],[]
        for fname in expected:
            values={r["condition"]:r for r in rows if r["fname"]==fname}
            for result in damage_localization(values):
                identity=dict(fname=fname,partial_volume=values["normal_matched"]["partial_volume"],metric=result["metric"])
                for early in ("on","off"):
                    damages.append(dict(identity,early_state=early,damage=result["damage_"+early]))
                localizations.append(dict(identity,**{k:v for k,v in result.items() if k!="metric"}))
        write_csv(run/"mismatch_damage_per_volume.csv",damages)
        write_csv(run/"localization_per_volume.csv",localizations)
        summary=summarize(rows,damages,localizations,partial)
        write_csv(run/"summary.csv",summary)
        regression_status,checks=quick_regression(samples,rows)
        metadata["regression_status"]=regression_status
        if checks:
            write_csv(run/"quick_regression.csv",checks)
        (run/"summary.txt").write_text(LIMITATIONS+"\nRegression: "+regression_status+"\n"+json.dumps(summary,indent=2),encoding="utf-8")
        require(regression_status!="failed", "Quick-run PSNR regression failed; do not proceed to formal pilot")
        require(sha256_file(checkpoint)==checksum,"Checkpoint changed")
        metadata.update(status="complete",checkpoint_unchanged=True)
    except BaseException as error:
        metadata.update(status="failed",error=repr(error))
        raise
    finally:
        meta_path.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
