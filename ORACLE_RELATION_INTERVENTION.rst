Oracle Relation Intervention: native 4x, Layer 1 only
===================================================

Design
------
For each slice, head and target query, original attention is a JOINT softmax over
[target keys; auxiliary keys]. Let A4=[S4,C4], m4=sum(C4), and let Cfull be the
auxiliary part of the first target-query attention under fully sampled target input.
The reference is Pfull=Cfull/sum(Cfull). The intervention is exactly:

    Aoracle = [S4, m4 * Pfull]

This preserves every self-key weight, total self mass and total cross mass from 4x.
The reference's own cross mass is NOT used. The full attention is NOT replaced by
a conditional distribution and is NOT passed through another softmax. There is no
averaging across heads or queries. All L1 target-query heads are intervened.

The full reference is divided by its positive mass with float64 intermediate
arithmetic and converted to FP32. It is renormalized after device transfer to
limit rounding drift. No additive denominator epsilon is used: that would shrink
the probability sum and then shrink cross mass. Undefined zero-mass reference rows
abort the run. A zero 4x cross mass stays zero. FP32 is required; there is no AMP.

Per query/head, assertions check finite/nonnegative original probabilities,
unit joint-softmax sum, identical self weights, unchanged cross mass, and unchanged
total mass within floating point tolerances. The maximum cross-mass error is saved.

Analysis-only hook
------------------
No changes are made to models/mca.py, engine.py, train.py or existing analysis code.
The standalone script temporarily registers a forward PRE-hook on:

    model.cross_transformer.layers[0][0].fn.attn.attn_drop

Its input is the joint-softmax attention before attention dropout. For the full
forward, the hook observes this tensor and caches only its conditional auxiliary
slice on CPU. For the normal 4x forward, no hook is installed. For oracle 4x, the
hook returns a NEW attention tensor with the mass-preserving cross replacement.
The model computes Q/K/V itself; full-forward V, output images and hidden features
are never used in the oracle forward. Residuals and projections run unchanged.
The original softmax tensor is not modified in-place. Hooks are removed in finally
blocks and the reference is cleared after each batch, including on exceptions.

Evaluation is frozen model.eval()/torch.no_grad(). Parameter/buffer version counters
are checked for in-place changes; no parameter, loss, optimizer or checkpoint output
is created. Only first-layer target-query attention is hooked. There is no flag to
extend this to later layers or to replace cross mass.

Three forwards per batch
------------------------
1. Full PDFS input + fixed PD: cache L1 conditional relation; discard predictions.
2. Native random4x PDFS input + same PD: record normal reconstruction.
3. Exactly the same 4x input + same PD: apply oracle L1 and record reconstruction.

Only the reference relation crosses from the full pass into the oracle pass.
Auxiliary and degraded input tensors are checked for mutation across passes.
Memory is bounded by one batch's L1 reference, one active attention replacement,
and CPU reconstruction buffers for one volume. No full-dataset relations, model
outputs or reconstruction images are dumped. Original B0 weights/activations may
still require substantial GPU memory; the default batch size is 1.

Preprocessing and metric target
-------------------------------
This first oracle experiment fixes Protocol A (model-native normalization), random4x
center fraction 0.08. There is no Protocol B or 8x option. Original SliceDataset
pairing and slice order are reused. Full/4x target inputs come from the SAME raw
PDFS k-space using the existing analysis functions. The 4x input is compared exactly
with ReconstructionTransform. Mask reproducibility checks remain enabled. Filename
seeds include the complete path, so preserve the original B0 path spelling.

PD is constructed once using the original validation transform, taking pd[1]: the
HDF5 reconstruction_esc target normalized/clamped with masked PD random4x statistics.
The reference full input uses its own mean/std; normal and oracle both use the
same native 4x input and its statistics. Full statistics never denormalize outputs.

To reproduce engine.evaluate, the reconstruction target is original pdfs[1], which
has already been normalized with 4x input statistics and clamped to [-6,6]. Both
predictions and that target are denormalized with:

    image = normalized_image * std_4x + mean_4x

The code intentionally follows the original use of std, NOT std+epsilon, for this
inverse step. No additional output clamp is applied. The evaluation target is NOT
raw unclamped reconstruction_esc and is NOT the independently normalized full input.
This choice preserves baseline comparability but inherits B0's clipped-label caveat.

Metrics and aggregation
-----------------------
The original util.metric.nmse, psnr and ssim functions are called on each processed
volume. NMSE is ||target-prediction||^2 / ||target||^2 over that volume. PSNR uses
volume MSE and data_range=target.max(). SSIM averages original skimage default SSIM
over slices with the SAME volume-level target.max() range. Both conditions share
the exact target and range. Summaries are volume-weighted mean and population std
(ddof=0), matching B0's mean weighting. Paired per-volume deltas are oracle-normal:
positive PSNR/SSIM and negative NMSE mean improvement. No direction is filtered.
The delta std is calculated from paired deltas, not by subtracting two std values.

Volumes are processed in original split order; one volume is buffered, evaluated,
and released. Duplicate target volumes (which original evaluation can overwrite)
are rejected for explicit review, not silently changed or collapsed. Degenerate
targets or nonfinite metrics, including infinite PSNR for an exact prediction,
abort with an error rather than omitting data or emitting invalid summary statistics.

--max-batches selects a fixed prefix before inference. A quick run can contain a
partial volume. Per-volume rows show processed versus expected slice counts, and
summary/metadata explicitly mark a subset/partial-volume run. Its PSNR and SSIM
range is the maximum over the processed portion, so it is NOT comparable with a
full-validation B0 score. Full mode uses the identical computation over all slices.

Commands (server Bash, from MTrans root)
---------------------------------------
First run the synthetic CPU tests, using the B0 environment plus scikit-image::

    python -B -m unittest discover -s tests -p test_oracle_relation_intervention.py -v

Quick sanity check::

    CUDA_VISIBLE_DEVICES=1 python oracle_relation_intervention.py --device cuda --batch-size 1 --num-workers 0 --max-batches 1

Full validation, when ready on the server::

    CUDA_VISIBLE_DEVICES=1 python oracle_relation_intervention.py --device cuda --batch-size 1 --num-workers 0 --max-batches -1

Optional --checkpoint defaults to
weights_reconstruction_multi_cross_paper_random4x/best.pth. Optional --data-root
defaults to the existing B0 DATASET.ROOT (parent of singlecoil_val). Physical GPU
selection remains outside the script. --output-dir defaults to oracle_relation_outputs;
every run gets a new timestamp directory outside the checkpoint directories.

Outputs
-------
* oracle_l1_per_volume.csv: normal/oracle PSNR, SSIM, NMSE, paired deltas, slice counts.
* oracle_l1_summary.csv: volume mean/std for both conditions and paired differences.
* oracle_l1_summary.txt: design, metadata, metric definitions, results and limitations.
* metadata.json: strict checkpoint load/best metadata, Git, config, split hash, mask,
  normalization, sample/batch/volume counts, partial-volume status and mass error.
* samples.csv: PDFS/PD paths, slice numbers and pair IDs.

Failed/interrupted runs are marked failed in metadata when caught. Their partial
CSV/manifest is not a completed experiment. No checkpoint or tensor files are written.

Interpretation and confounds
----------------------------
* Privileged full-target information makes this an oracle diagnostic, not an
  inference method or anatomical correspondence truth. No strict upper-bound claim.
* Full and 4x use different native normalization statistics and clipping patterns;
  full is outside the 4x training input distribution. Better/worse results cannot
  isolate missing k-space information from all preprocessing effects.
* The original target+auxiliary pre-fusion remains. After changing L1, downstream
  target/auxiliary features, reverse attention, and later V/attention can naturally
  change. "Only L1 intervention" means only one direct override, NOT identical
  downstream hidden tensors. Freezing them would define a different experiment.
* Cross mass and self attention stay at their 4x values, and V stays on the 4x
  computation path. Full correspondence can be incompatible with those features;
  a negative result does not rule out relation problems, other layers or mass effects.
* The analysis inherits PD normalization, paired-slice registration uncertainty,
  filename-dependent single mask realizations, clipped B0 labels, and correlated
  slices. Volume means/std and paired deltas are descriptive, not a significance test.

Validation status
-----------------
Local Python lacks torch/numpy/scikit-image. Numerical tests and real checkpoint/data
evaluation must be run on the server. The synthetic tests check joint-mass invariants,
self-weight identity, unchanged L1 K/V, negative-control identity, hook cleanup,
unchanged model state, native preprocessing and original volume metric aggregation.
No full experiment, reconstruction gains, or numerical-test pass is claimed locally.
