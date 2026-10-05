Frozen-B0 Mismatch Localization Audit
====================================

Question and scope
------------------
Does mismatch damage attenuate when original B0 unconditional early auxiliary
addition is disabled? Exactly four conditions, one original frozen checkpoint:
normal_matched, normal_mismatched, no_early_matched, no_early_mismatched.
No training/backward/optimizer, retrained NoEarly checkpoint, original forward edit,
NoCross, Full, 8x, Protocol B or automatic real experiment is introduced.

Fixed checkpoint: weights_reconstruction_multi_cross_paper_random4x/best.pth.
Existing strict loader validates state/config; saved config is required. Native
singlecoil/random4x CF=.08 is asserted. Existing dataset, volume metrics, pilot
selection and NoEarly guard are reused without modifications.

Reuse, not alternative implementations
--------------------------------------
* cross_modal_dependency_audit.AuditDataset is used directly, not subclassed.
  It constructs donor assignments over FULL validation pair order, before volume
  selection: cyclic next-pair PD, last wraps to first; donor != original PD.
* Its relative mapping, rational half-to-even rounding, native donor pd[1], own
  masked-PD mean/std and clamp are unchanged. Nt is available native paired target
  slices; Na is donor PD actual slice count. No new donor sampling is performed.
* normal_matched calls existing run_condition(...,"normal"); normal_mismatched
  calls existing run_condition(...,"mismatched_aux"). No intervention hooks.
* Both NoEarly conditions call existing run_pathway(...,"no_early") with matched
  or donor PD respectively. Its AST guard requires the wrapper body to be exactly
  CrossCMMT.forward minus x=x+complement. Both heads, embeddings, position additions,
  CrossTransformer and tails continue normally. No attention weights are zeroed.
* Existing temporary forward restores in finally, including exceptions. A final
  normal_matched replay must match its pre-intervention output exactly. Input
  tensors, parameter/buffer versions and checkpoint SHA256 are checked unchanged.

One AuditDataset item is read per selected target slice. It returns ONE donor image
and its normalization statistics. run_four passes the identical donor tensor object
to normal_mismatched and no_early_mismatched; neither condition samples or transforms
anything. mapping_record extends dataset.mapping(index) with the returned donor
mean/std and a shared mapping_id. samples.csv references that ID, while
mismatch_mapping.csv marks used_by_conditions=normal_mismatched;no_early_mismatched.
It contains original target/PD IDs, donor IDs, original/donor slices, counts,
relative position and normalization source. There is only one mapping row per slice.

Metrics, damage and localization
--------------------------------
All conditions use identical target4x/GT/target mean/std. Denormalization follows
the existing audit: prediction*std+mean, native already-clamped GT*std+mean, no
prediction clamp. Existing VolumeAccumulator receives the new condition labels,
stacks physical-space slices, calls original util.metric on the volume, and releases
the volume arrays. No mean of slice metric scores. Partial volume status is retained.

For each complete target volume and each Early state separately:

    D_ON_PSNR = PSNR_normal_matched - PSNR_normal_mismatched
    D_OFF_PSNR = PSNR_no_early_matched - PSNR_no_early_mismatched

SSIM uses the same direction. NMSE uses mismatched minus matched within each Early
state. All damage values >0 mean mismatch worsens performance. Never use
normal_matched versus no_early_mismatched to define mismatch damage.

    L_metric = D_ON_metric - D_OFF_metric
    F_metric = L_metric / D_ON_metric

L>0 means mismatch damage attenuates with Early OFF; L=0 means no attenuation on
that metric; L<0 means increased mismatch sensitivity after disabling early addition.
No threshold for "approximately zero" or significance is invented. F is defined
only for positive, finite D_ON with finite numerator/result. Otherwise it is blank
with explicit fraction_status: undefined_nonpositive_on_damage,
undefined_nonfinite_damage or undefined_nonfinite_fraction. No epsilon or clipping.
F may be negative or exceed1. It describes mismatch damage reduction, not causal
proportion. Small positive denominators remain mathematically defined but unstable.

Summaries use equal-volume means/population std (ddof=0)/volume counts, no significance
tests. Fraction summary is the mean of per-volume fractions, NOT ratio of mean
damages. If any fraction is undefined, aggregate mean/std remain blank rather than
silently selecting favorable/defined volumes; defined counts and reason counts are
reported. Nonfinite raw metrics/damages are not silently filtered.

Sampling, output and regression gate
------------------------------------
Default max-batches1/batch-size1 is sanity only. The existing pilot flag selects the
same preregistered ten volumes and ALL native available paired slices when
max-batches=-1:
file1002538, file1001566, file1001059, file1000196, file1000178,
file1001289, file1001191, file1000990, file1001834, file1001344.
No case replacement or result-dependent selection. Keep data-root spelling unchanged
to preserve native filename-based mask seeds. All conditions use the same slice set.

Independent frozen_b0_mismatch_outputs/<exclusive timestamp>/ contains:

* metadata.json, samples.csv, mismatch_mapping.csv
* per_volume_metrics.csv (four conditions)
* mismatch_damage_per_volume.csv (Early ON/OFF within-state paired damage)
* localization_per_volume.csv (D_ON, D_OFF, L, F and fraction status)
* summary.csv, summary.txt
* quick_regression.csv when the selection is exactly file1002538 slice0

For that one-slice selection, expected PSNR values are:

    normal_matched      20.83155125
    normal_mismatched   20.60596273
    no_early_matched    20.20034065

Fixed tolerance is absolute1e-4 dB, relative tolerance0. This is a numerical
regression tolerance, not a scientific effect threshold. Any mismatch writes the
observed errors, records regression_status=failed/status=failed and raises; DO NOT
proceed to formal pilot. Diagnose checkpoint/config/data/mask/runtime provenance
first, not loosen tolerance to fit results. Other selections record not_applicable,
not passed. A full-volume score is never compared to the slice0 references.
No reference is fabricated for no_early_mismatched; it is the new measured condition.

Real-data regression has NOT been executed during implementation. Pilot execution
is a separate user action after reviewing a passed quick run; no automatic pilot
launch or dependency on previously observed scientific outcomes is introduced.

Tests and commands
------------------
Synthetic contract/arithmetic tests (standard library only)::

    python -B -m unittest discover -s tests -p test_frozen_b0_mismatch_contract.py -v

Synthetic model/native-transform/aggregation tests in B0 environment::

    python -B -m unittest discover -s tests -p test_frozen_b0_mismatch_localization.py -v

Coverage includes exact three-condition regressions and new combination; identical
donor object/mapping/slice/stats; unchanged NoEarly AST contract; complement and
transformer execution; forbidden cross-deletion; parameter/buffer/input invariance;
normal and exceptional restoration; volume metric/batch partition invariance;
partial status; within-state damage, localization and undefined fractions; no
training/checkpoint writes; quick-regression pass/fail/applicability logic.

After review ONLY, first server quick run explicitly selects the reference case::

    CUDA_VISIBLE_DEVICES=1 python -B frozen_b0_mismatch_localization.py --target-volumes file1002538 --max-batches 1 --batch-size 1

After passed quick regression and separate review, complete ten-volume pilot::

    CUDA_VISIBLE_DEVICES=1 python -B frozen_b0_mismatch_localization.py --pilot-10-volumes --max-batches -1 --batch-size 1

Neither command is run by this implementation. One partial slice cannot support
scientific localization, and subset scores cannot directly reproduce full-validation
checkpoint scores.

Confounds and preregistered interpretation
-----------------------------------------
Turning off early addition is an off-distribution inference perturbation of one B0,
changes matched performance and downstream feature evolution, and is not a trained
NoEarly comparison. Mismatch jointly changes anatomy, content, correspondence and
donor normalization. Cross-attention and bidirectional updates remain fully active.
One deterministic donor per pair gives conditional evidence, not donor-population
robustness. L/F depend on metric scale (PSNR is logarithmic), and F is unstable for
small positive baseline damage. The ten-volume pilot is descriptive.

If D_OFF is much smaller than D_ON, support the wording "mismatch-induced negative
transfer is largely attenuated when unconditional early auxiliary addition is
disabled"; do not say all damage is caused by early fusion. Similar damages do not
support early-add localization; larger D_OFF means greater mismatch sensitivity
after removal and requires reconsidering pathway interpretation. No causal fraction,
relation-optimality or new architecture claim follows from this audit.
