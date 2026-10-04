Relation Distortion Analysis, phase 1
====================================

Scope and interpretation
------------------------
This is frozen-checkpoint analysis, not training. No optimizer, backward, model
checkpoint output, loss change, or reconstruction evaluation is introduced.
``R_full`` is the learned reference under fully sampled PDFS observations, not an
anatomical annotation. Deviation does not by itself establish wrong correspondence
or harm to reconstruction. All layers/heads are reported, including contrary trends.

Actual B0 behavior
-----------------
``SliceDataset`` pairs volumes using the split CSV and takes the same slice index,
up to the smaller volume's slice count. It does not establish anatomical registration.
PD and PDFS share one ReconstructionTransform in the original build_dataset path.
Both have a random-masked input computed, but engine.py passes ``pd[1]`` to the
model. This is the HDF5 ``reconstruction_esc`` image, center cropped, normalized
using the MASKED PD input's mean/std (epsilon 1e-11), and clamped to [-6, 6].
Thus PD is a fully sampled reconstruction target in image content, with normalization
dependent on the 4x mask. It is not pd[0], the undersampled zero-filled input.
Analysis preserves this behavior and constructs PD once per sample for all conditions.

Both heads produce [B, 16, 320, 320] for B0. The original ``x = x + complement``
pre-fusion remains. The analyzed target-query representation is not purely target-only.
CrossTransformer also updates auxiliary features using target features in its reverse
branch at every layer: fixed auxiliary INPUT does not mean fixed auxiliary hidden
representations in deeper layers. We preserve both directions and analyze only the
four target-query attention modules ``cross_transformer.layers[i][0].fn.attn``.
Attention keys are [target; auxiliary], not auxiliary-only.

Implementation
--------------
``models/mca.py`` adds two non-buffer/non-parameter attributes: capture_attention
(default False) and last_attn (default None). When enabled, the cache is taken after
softmax and before attention dropout. No existing arithmetic, dropout, aggregation,
projection, residual, parameter, or state_dict key is changed. A module forward hook
immediately consumes last_attn and clears its GPU reference. Only the dynamic cross
slice ``attn[..., attn.shape[-2]:]`` is copied to CPU; the copy owns its storage.
The reverse-direction modules are not captured.

CPU storage is bounded by one batch's four full-reference cross relations, plus one
current degraded layer and query-chunk metric workspaces. The default batch size is
1, not the training batch size 4. At B0 sizes, the four float32 reference cross slices
alone occupy about 41 MB per sample. Original model weights, full attention, and
key/value activations still consume substantial GPU memory; batching may need care.
No relation tensors or representative images are written. Metric computation uses CPU
stable sorting and may be slow on a full validation set.

Data and masks
--------------
The analysis adapter uses unmodified SliceDataset(transform=None), then constructs
full/4x/8x from the SAME raw PDFS k-space. It reuses to_tensor, apply_mask, ifft2c,
complex_center_crop, complex_abs, normalize_instance, and normalize. Crop selection
and the original narrow-width fallback match ReconstructionTransform. No extra
padding, resizing, RSS, or alternative FFT is introduced for this singlecoil experiment.
Native 4x inputs are automatically compared bit-for-bit with ReconstructionTransform.

Full uses no mask. 4x uses RandomMaskFunc with center fraction 0.08. The repository
does NOT establish the original experimental 8x center fraction: subsample.py only
has a generic documentation example mentioning 0.04. Therefore --center-fraction-8x
is explicit, with no default. --skip-8x explicitly permits a full/4x-only run.

Both mask accelerations use ``tuple(map(ord, fname))``. This includes the COMPLETE
path string supplied by SliceDataset, not just a basename. Preserve B0's data-root
spelling to reproduce its validation masks. Changing mounts/paths changes masks.
Masks are checked for exact repeatability; shared seeds do not guarantee nested masks.
No claim of matching a paper 8x protocol is made solely from a user-provided value.
The random mask has nominal/expected acceleration; its realized sample count varies.

Normalization
-------------
Protocol A independently computes each condition's magnitude-image mean/std before
clamping, exactly as model-native preprocessing does. Protocol B computes full-image
mean/std before clamping and applies these SAME statistics to all target conditions.
Both use epsilon 1e-11 and clamp [-6, 6]. This is mathematically consistent with the
existing normalize utility and avoids recovering intensities from a clipped image.
PD preprocessing stays fixed in A and B. B controls normalization statistics, but
condition-dependent clipping and a shift from training-time input distributions remain.

Metrics and aggregation
-----------------------
For raw auxiliary slice C, mass = sum_j C_j. The conditional relation is
P_j = C_j / (mass + 1e-12), checked to sum approximately to 1. Near-zero/zero mass
that fails this check aborts the run; there is no silent uniform-distribution fallback.

JS(P,Q) = 0.5 sum_j P_j ln(P_j/M_j) + 0.5 sum_j Q_j ln(Q_j/M_j), M=(P+Q)/2.
Logs are protected with epsilon 1e-12; natural log units are used. JS and retention
are calculated per query/head before aggregation; attention probabilities are never
averaged across heads before metrics.

Retention@k = |TopK_full intersection TopK_degraded| / k, k in {1,5,10}.
Stable descending sort breaks exact ties by lower auxiliary token index. Values are
not perturbed. Fewer than 10 auxiliary tokens is an error, not an implicit k change.
Cross-modal mass always uses raw C, not its conditional normalization.

For each metric, average queries per slice/head. Per-layer statistics average these
head metrics; global statistics average all layer/head metrics per slice. Report
mean and population std (ddof=0) over equally weighted slices. Unequal final batches
do not change weighting. Std is between-slice variation of metric means, not pooled
query variance, head variance, a standard error, or a volume-level significance test.
Slices within a volume are correlated; original reconstruction metrics instead
aggregate volumes. Full-condition rows contain attention mass only.

Execution on the GPU server
---------------------------
Use the B0 Python environment with torch, numpy, einops, yacs, h5py, PyYAML and
matplotlib. The analysis uses stable torch.argsort and torch.load(weights_only=False),
so the README's old torch 1.7 pin is not a sufficient runtime specification.
Only load your trusted B0 checkpoint, which includes a serialized YACS config.

First run the small synthetic tests (no dataset/checkpoint required)::

    python -m unittest discover -s tests -p test_relation_distortion_analysis.py -v

Quick full/4x pipeline sanity check before an 8x center fraction is confirmed::

    CUDA_VISIBLE_DEVICES=1 python relation_distortion_analysis.py --device cuda --protocol A --skip-8x --max-batches 1

After explicitly setting CF8 to the approved numeric protocol value, check all
conditions (CF8 below is a shell variable, not a suggested center fraction)::

    CUDA_VISIBLE_DEVICES=1 python relation_distortion_analysis.py --device cuda --protocol A --center-fraction-8x "$CF8" --max-batches 1

Protocol A, entire validation split::

    CUDA_VISIBLE_DEVICES=1 python relation_distortion_analysis.py --device cuda --protocol A --center-fraction-8x "$CF8" --max-batches -1

Protocol B, entire validation split::

    CUDA_VISIBLE_DEVICES=1 python relation_distortion_analysis.py --device cuda --protocol B --center-fraction-8x "$CF8" --max-batches -1

Optionally pass --data-root and --checkpoint for server paths. Defaults come from
the existing B0 dataset configuration and
weights_reconstruction_multi_cross_paper_random4x/best.pth. No automatic GPU index
selection or CUDA_VISIBLE_DEVICES modification occurs. Default batch size 1,
num-workers 0; --max-samples N and --max-batches N select a deterministic prefix
of the original CSV/slice order, recorded as a subset rather than a full experiment.
Quick and full modes execute identical computation. No samples are filtered by result.

Outputs and safeguards
----------------------
Each invocation creates a fresh UTC timestamp directory under relation_analysis_outputs
(or --output-dir). Checkpoint directories are rejected as output destinations.

* relation_distortion_summary.csv: global metrics, long format.
* relation_distortion_per_layer.csv: all layer metrics, long format.
* relation_distortion_per_head.csv: all layer/head metrics, long format.
* relation_distortion_summary.txt: metadata, definitions, limitations, layer/global
  mean/std and 8x-minus-4x differences without assuming the hypothesis holds.
* metadata.json: checkpoint best epoch/status, strict-load/config-verification status,
  git commit/working-tree status, protocol, masks, seed, paths, split SHA256,
  sample/batch counts, model configuration, software versions, device and timestamp.
* samples.csv: processed PD/PDFS paths, slice indices and pair IDs for auditability.

Failure records status=failed and an error in metadata; partial manifests are not
completed experiments. Successful status is only recorded after all selected samples
and output writes finish. No result directories are created by the synthetic tests.
Checkpoint loads are strict, including parameter keys after an optional uniform
module. prefix removal. Available checkpoint config fields are compared with the
current B0 config; metadata missing from a checkpoint is reported, never fabricated.

Automatic checks include input identity by construction, matched slice indices,
unchanged auxiliary tensor before/after each forward, eval/no_grad, parameter/state
key invariance, layer/head counts, dynamic shapes, finite nonnegative attention,
self+cross mass near 1, conditional mass near 1, deterministic masks, exact native
4x preprocessing, image sizes, and no silent sample dropping.

Local verification limits
-------------------------
The implementation workstation's Python has neither torch nor numpy. Syntax/static checks can run,
but numerical tests, actual checkpoint loading, GPU memory use and data traversal
must be verified in the server environment. No B0 experiment result is claimed here.
No training, commit or push is part of this implementation.
