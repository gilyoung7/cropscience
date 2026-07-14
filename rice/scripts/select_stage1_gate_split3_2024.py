"""Stage 1 best-gate selection on split3 (val=2023 / test=2024) — val-only.

For each pest, evaluates the 9 candidates (method × run) at target R>=0.88 and
picks ONE using validation metrics ONLY:

    1) prefer val_recall >= 0.875 (= 0.88 - 0.005)
    2) tie-break by min val_FAR
    3) tie-break by max val_F1  (val_precision is not stored in farmin CSV)
    4) tie-break by method priority: D_history > dispatch_group_tau > A_baseline
    5) tie-break by smallest run id
    Fallback (no candidate meets recall): max val_recall, then min val_FAR,
    then smallest run id.

Reports the selected gate plus its (val, test) metrics and the underlying
ckpt / summary path for downstream Stage 2 cohort construction.

The selected RUN drives Stage 2: build_dispatch_feature_table uses that
run's A_ckpt + D_ckpt to define the dispatch cohort. The selected METHOD
is reporting-only (lead_v3 always uses dispatch_group_tau as the gate
because the Stage 2 head reads dispatch alert_tstar + features).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


TARGET_NORMALIZED = "R>=0.88"
RECALL_TARGET = 0.88
RECALL_TOL = 0.005
EFF_THRESHOLD = RECALL_TARGET - RECALL_TOL
METHOD_PRIORITY = {"D_history": 0, "dispatch_group_tau": 1, "A_baseline": 2}

NOWCAST_WINDOW = 28
LEAD_MIN_LABEL = 14
LEAD_MAX_LABEL = 45


def _norm_target(t: str) -> str:
    if t in ("R>=0.9", "R>=0.90"):
        return "R>=0.90"
    return t


def pick(sub: pd.DataFrame) -> tuple[pd.Series, str]:
    """sub = candidate rows for one pest (method × run × split=split3)."""
    sub = sub.copy()
    meets = sub[sub["recall_val"] >= EFF_THRESHOLD]
    if len(meets):
        meets = meets.assign(
            _mp=meets["method"].map(METHOD_PRIORITY).fillna(99).astype(int))
        ranked = meets.sort_values(
            ["FAR_val", "F1_val", "_mp", "run"],
            ascending=[True, False, True, True],
        )
        return (ranked.iloc[0],
                f"val_recall >= {EFF_THRESHOLD:.3f} met; min FAR + tie-break")
    sub = sub.assign(_mp=sub["method"].map(METHOD_PRIORITY).fillna(99).astype(int))
    ranked = sub.sort_values(
        ["recall_val", "FAR_val", "_mp", "run"],
        ascending=[False, True, True, True],
    )
    return (ranked.iloc[0],
            f"no candidate meets recall>={EFF_THRESHOLD:.3f}; max-recall fallback")


def derive_paths(pest: str, run: int, method: str) -> dict:
    base = f"rice/outputs_stage1/batch_rolling/{pest}/run{run}/split3_v2023_t2024"
    paths = {
        "a_ckpt":  f"{base}/A/ckpt/event_xgb_w{NOWCAST_WINDOW}_lead{LEAD_MIN_LABEL}-{LEAD_MAX_LABEL}_A.pt",
        "d_ckpt":  f"{base}/D/ckpt/event_xgb_w{NOWCAST_WINDOW}_lead{LEAD_MIN_LABEL}-{LEAD_MAX_LABEL}_D.pt",
        "dispatch_summary": f"{base}/group_tau/group_tau_hybrid_summary.json",
        "a_sweep": f"{base}/A/useful_pareto/useful_sweep_A.csv",
        "d_sweep": f"{base}/D/useful_pareto/useful_sweep_D.csv",
    }
    if method == "A_baseline":
        primary = paths["a_sweep"]
    elif method == "D_history":
        primary = paths["d_sweep"]
    else:
        primary = paths["dispatch_summary"]
    return {**paths, "primary_source": primary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all_csv",
                    default="rice/outputs_stage1/batch_rolling/_summary/pest_batch_farmin_all.csv")
    ap.add_argument("--exclude_pests", default="BPH2")
    ap.add_argument("--out_csv",
                    default="rice/outputs_stage2_batch_2024/_summary/stage1_gate_selection_split3_2024.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.all_csv)
    df["target"] = df["target"].astype(str).map(_norm_target)
    df = df[(df["target"] == TARGET_NORMALIZED) & (df["split"] == "split3")].copy()
    excluded = {s.strip() for s in str(args.exclude_pests).split(",") if s.strip()}
    df = df[~df["pest"].isin(excluded)]
    pests = sorted(df["pest"].unique())
    print(f"[select] pests ({len(pests)}): {pests}  excluded={sorted(excluded)}")
    print(f"[select] rows={len(df)}  (expect 8 pests × 3 methods × 3 runs = 72)")

    rows = []
    for pest in pests:
        sub = df[df["pest"] == pest]
        if sub.empty:
            print(f"  [skip] {pest}: no rows")
            continue
        picked, reason = pick(sub)
        paths = derive_paths(pest, int(picked["run"]), str(picked["method"]))
        rows.append({
            "pest": pest,
            "selected_method": str(picked["method"]),
            "selected_run": int(picked["run"]),
            "split": "split3",
            "k": picked.get("k"),
            "tau": picked.get("tau"),
            "tau_no": picked.get("tau_no"),
            "tau_with": picked.get("tau_with"),
            "val_recall": float(picked["recall_val"]),
            "val_FAR": float(picked["FAR_val"]),
            "val_F1": float(picked["F1_val"]),
            "val_precision": float("nan"),  # not stored in farmin CSV
            "test_recall": float(picked["recall_test"]),
            "test_FAR": float(picked["FAR_test"]),
            "test_precision": float(picked["precision_test"]),
            "test_F1": float(picked["F1_test"]),
            "test_lead_median": float(picked["lead_median_test"]),
            "test_n_event": int(picked["n_event_test"]),
            "fallback": str(picked["fallback"]),
            "reason": reason,
            "selected_ckpt_or_summary_path": paths["primary_source"],
            "a_ckpt": paths["a_ckpt"],
            "d_ckpt": paths["d_ckpt"],
            "dispatch_summary": paths["dispatch_summary"],
        })

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[select] wrote {out_path}  rows={len(out)}")

    print("\n=== Per-pest gate selection (val-only) ===")
    show = ["pest", "selected_method", "selected_run",
            "val_recall", "val_FAR", "val_F1",
            "test_recall", "test_FAR", "test_F1"]
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                            "display.float_format", "{:.3f}".format):
        print(out[show].to_string(index=False))
    print()
    print("(test metrics shown for sanity only — selection used val only.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
