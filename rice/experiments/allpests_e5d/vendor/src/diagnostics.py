"""late-eval and actionable-lead diagnostics (post-hoc; test labels used only for scoring,
never for offset selection). Operates on a picked-rows table from selector_utils.picked_rows."""
from __future__ import annotations
import numpy as np
import pandas as pd

LEAD_BINS = ["<0", "0-2", "3-6", "7-13", ">=14"]


def _lead_bin(x: float) -> str:
    return "<0" if x < 0 else "0-2" if x <= 2 else "3-6" if x <= 6 else "7-13" if x <= 13 else ">=14"


def enrich(picked: pd.DataFrame) -> pd.DataFrame:
    d = picked.copy()
    d["issue_date"] = d["alert_tstar"] + d["offset"]
    d["true_start"] = d["L"] + 1
    d["true_end"] = d["R"]
    d["true_action_lead"] = d["true_start"] - d["issue_date"]
    d["pred_start"] = d["pred_L80"]; d["pred_end"] = d["pred_R80"]
    d["pred_action_lead"] = d["pred_start"] - d["issue_date"]
    d["late"] = d["issue_date"] > d["true_start"]
    d["lateness"] = d["issue_date"] - d["true_start"]
    return d


def late_split(d: pd.DataFrame, min_offset: int) -> dict:
    late = d[d["late"]]
    struct = late[(late["alert_tstar"] + min_offset) > late["true_start"]]
    sel = late[(late["alert_tstar"] + min_offset) <= late["true_start"]]
    return {"n_total": len(d), "late": int(len(late)),
            "structural": int(len(struct)), "selector_induced": int(len(sel)),
            "late_rate": round(len(late) / len(d), 4) if len(d) else float("nan")}


def true_action_lead_bins(d: pd.DataFrame) -> dict:
    tal = d["true_action_lead"]
    out = {"mean": round(float(tal.mean()), 2), "median": round(float(tal.median()), 1),
           "std": round(float(tal.std()), 2), "min": int(tal.min()), "max": int(tal.max())}
    for b in LEAD_BINS:
        out[f"n_{b}"] = int((tal.apply(_lead_bin) == b).sum())
    return out


def pred_action_lead_stats(d: pd.DataFrame) -> dict:
    pal = d["pred_action_lead"]
    out = {"mean": round(float(pal.mean()), 2), "median": round(float(pal.median()), 1),
           "std": round(float(pal.std()), 2), "min": int(pal.min()), "max": int(pal.max())}
    for b in LEAD_BINS:
        out[f"n_{b}"] = int((pal.apply(_lead_bin) == b).sum())
    return out


def non_actionable_interval(d: pd.DataFrame) -> dict:
    n = len(d)
    return {
        "frac_pred_start_before_issue": round(float((d["pred_start"] < d["issue_date"]).mean()), 4),
        "frac_issue_in_interval": round(float(((d["pred_start"] <= d["issue_date"]) &
                                               (d["issue_date"] <= d["pred_end"])).mean()), 4),
        "frac_issue_after_pred_end": round(float((d["issue_date"] > d["pred_end"]).mean()), 4),
        "n": n,
    }


def iou80_tol(d: pd.DataFrame, k: int, n_total: int) -> float:
    """coverage-weighted realized IoU80, forgiving lateness up to k days."""
    val = np.where(d["lateness"].values <= k, d["iou80_geom"].values, 0.0)
    return round(float(val.sum()) / n_total, 4)


def actionable_iou80(d: pd.DataFrame, L_min: int, n_total: int) -> float:
    """credit IoU80 only when true_action_lead >= L_min (else 0)."""
    val = np.where(d["true_action_lead"].values >= L_min, d["iou80_geom"].values, 0.0)
    return round(float(val.sum()) / n_total, 4)


def summarize(d: pd.DataFrame, min_offset: int, n_total: int,
              L_list=(0, 3, 7, 14)) -> dict:
    row = {}
    row.update({f"late_{k}": v for k, v in late_split(d, min_offset).items()})
    row["IoU80_overall_tol0"] = iou80_tol(d, 0, n_total)
    row["IoU80_overall_tol1"] = iou80_tol(d, 1, n_total)
    for L in L_list:
        row[f"IoU80_action_L{L}"] = actionable_iou80(d, L, n_total)
    row["mean_offset"] = round(float(d["offset"].mean()), 2)
    row["coverage"] = round(float((~d["late"]).mean()), 4)
    row["MAE_center"] = round(float(d.loc[~d["late"], "mae_center"].mean()), 3) if (~d["late"]).any() else float("nan")
    row["PI_hit"] = round(float(d.loc[~d["late"], "pi_hit80"].sum()) / n_total, 4)
    return row
