# BPH Stage-2 → TFLite (superseded)

**Moved to [`../stage2/`](../stage2/).**

This directory held the BPH-only prototype of the Stage-2 TFLite conversion. It
has been replaced by the common pipeline in `tflite_conversion/stage2/`, which
covers all 8 pests — BPH included — with no per-pest duplication.

The prototype's scripts are **not tracked in git**; only this notice remains. Use
`../stage2/` for any conversion work, and see its README for the full setup and
run instructions.

The refactor was verified equivalent: the prototype's wrapper and the common
wrapper produce bit-identical `mu` for BPH across 18 (seed, alert) cases
(max |old − new| = 0.0).

Background and the original BPH findings — the `torch.export` blockers, the
tolerance rationale, the dynamic-range quantization failure — are recorded in
[`docs/bph_stage2_tflite_conversion_report.md`](../../docs/bph_stage2_tflite_conversion_report.md).
All-8 results are in
[`docs/all_pests_stage2_tflite_conversion_report.md`](../../docs/all_pests_stage2_tflite_conversion_report.md).
