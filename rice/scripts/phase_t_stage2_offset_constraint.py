"""Stage-2 offset-constraint selector policies (selector/offset policy only — NO model/loss change).

Goal: reduce "late" Stage-2 predictions where eval_doy = alert_tstar + selected_offset
lands after the true onset (true_start_doy = true_L + 1).

Policies compared
-----------------
  original        : the V2 selector's chosen offset (unchanged baseline)
  oracle_m0       : ORACLE — drop offset candidates with eval_doy > true_start - 0   [ANALYSIS ONLY]
  oracle_m7       : ORACLE — drop offset candidates with eval_doy > true_start - 7   [ANALYSIS ONLY]
  deploy_q10      : DEPLOYABLE — cap offset by val lead_to_start q=0.10 (no test labels)
  deploy_q20      : DEPLOYABLE — cap offset by val lead_to_start q=0.20 (no test labels)

LEAKAGE RULES (enforced in code):
  * ORACLE policies use the TEST true_start_doy when *selecting* the offset. This is an
    upper-bound / diagnostic device ONLY and must NEVER be deployed at inference. It is
    flagged in the per-sample output (`uses_test_label=True`) and in the run log.
  * DEPLOYABLE policies select the offset using ONLY (pest, alert_tstar_doy). The offset
    caps come from VALIDATION data lead_to_start = (L+1) - alert_tstar quantiles
    (v2_per_candidate_val.csv). Test true_start_doy is never read during selection.
  * All policies use the test true_start ONLY afterwards to *score* metrics (offline eval),
    which is normal evaluation, not selection leakage.

Re-computation mirrors run_viz_interval_selector.build_per_sample exactly:
  mu(offset) = linear interp over coarse anchors {7,14,21,30,45,60} (clamped),
  PI = [round(mu - Z*sigma), round(mu + Z*sigma)],  IoU vs true [L+1, R].

Run:
    python -m rice.scripts.phase_t_stage2_offset_constraint
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT

# ---- constants mirrored from run_viz_interval_selector.py (do not drift) ----
SIGMA_DEFAULT = 5.0
Z = 1.96
ANCHORS = [7, 14, 21, 30, 45, 60]          # coarse offset grid the selector chooses from
PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]
# test sample_grid root per year (matches scripts/run_viz_selector_all_pests_years.sh)
YEAR_ROOT = {
    2022: "outputs/stage2/batch_2022_baseline",
    2023: "outputs/stage2/batch_2023_baseline",
    2024: "outputs/stage2/batch_2024_bestgate",
}
SEL_ROOT = "outputs/stage2/selector_cross_split"
VIZ_GLOB = "outputs/viz/viz_selector/**/per_sample.csv"

POLICIES = ["original", "oracle_m0", "oracle_m7", "deploy_q10", "deploy_q20"]
ORACLE = {"oracle_m0": 0, "oracle_m7": 7}
DEPLOY_Q = {"deploy_q10": 0.10, "deploy_q20": 0.20}

DIR_RE = re.compile(r"^(?P<pest>.+)_(?P<year>\d{4})_(?P<variant>v\d.*)$")


# --------------------- mirrored helpers (identical math) ---------------------
def mu_at_offset(off_mu: Dict[int, float], offset: float) -> float:
    valid = sorted((o, m) for o, m in off_mu.items() if not pd.isna(m))
    if not valid:
        return float("nan")
    xs = np.array([v[0] for v in valid], float)
    ys = np.array([v[1] for v in valid], float)
    if offset <= xs[0]:
        return float(ys[0])
    if offset >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(offset, xs, ys))


def iou_from_mu(mu: float, L: float, R: float, sigma: float) -> float:
    if pd.isna(mu):
        return 0.0
    pL = int(round(mu - Z * sigma)); pR = int(round(mu + Z * sigma))
    tL = int(L) + 1; tR = int(R)
    lo_hi = min(pR, tR); hi_lo = max(pL, tL)
    ov = (lo_hi - hi_lo + 1) if lo_hi >= hi_lo else 0
    un = max(pR, tR) - min(pL, tL) + 1
    return float(max(0.0, ov / un)) if un > 0 else 0.0


# ------------------------------- loading -------------------------------------
def parse_dir(p: str) -> dict:
    m = DIR_RE.match(Path(p).parent.name)
    return {"pest": m.group("pest"), "year": int(m.group("year")),
            "variant": m.group("variant")} if m else {}


def load_grid(pest: str, year: int) -> Dict[str, dict]:
    """sample_id -> {off_mu:{off:mu}, L, R, alert_tstar}."""
    gp = RICE_ROOT / YEAR_ROOT[year] / pest / "lead_v3_test_sample_grid.csv"
    g = pd.read_csv(gp)
    out = {}
    for sid, sub in g.groupby("sample_id"):
        h = sub.iloc[0]
        out[sid] = {
            "off_mu": {int(o): (float(m) if not pd.isna(m) else float("nan"))
                       for o, m in zip(sub["offset"], sub["mu"])},
            "L": float(h["L"]), "R": float(h["R"]),
            "alert_tstar": float(h["alert_tstar"]),
        }
    return out


def build_val_caps(qs, min_bin: int, min_pest: int, bin_w: int):
    """Leakage-free offset caps from VALIDATION lead_to_start=(L+1)-alert_tstar.
    Returns (caps dict, caps_dataframe). caps[(q, level, key)] = cap_value."""
    rows = []
    for vp in sorted(glob.glob(str(RICE_ROOT / SEL_ROOT / "*" / "v2_per_candidate_val.csv"))):
        cell = Path(vp).parent.name           # "<year>_<pest>"
        m = re.match(r"^(\d{4})_(.+)$", cell)
        pest = m.group(2) if m else cell
        d = pd.read_csv(vp, usecols=["sample_id", "L", "alert_tstar"]).drop_duplicates("sample_id")
        d["pest"] = pest
        d["lead_to_start"] = (d["L"] + 1) - d["alert_tstar"]
        d["alert_bin"] = (d["alert_tstar"] // bin_w).astype(int) * bin_w
        rows.append(d)
    val = pd.concat(rows, ignore_index=True)
    caps = {}
    cap_rows = []
    for q in qs:
        gbin = val.groupby(["pest", "alert_bin"])["lead_to_start"]
        for (pest, b), s in gbin:
            if len(s) >= min_bin:
                caps[(q, "bin", (pest, b))] = float(np.quantile(s, q))
        for pest, s in val.groupby("pest")["lead_to_start"]:
            if len(s) >= min_pest:
                caps[(q, "pest", pest)] = float(np.quantile(s, q))
        caps[(q, "global", None)] = float(np.quantile(val["lead_to_start"], q))
        # record table
        for (pest, b), s in gbin:
            cap_rows.append({"q": q, "level": "bin", "pest": pest, "alert_bin": b,
                             "n": len(s), "cap": float(np.quantile(s, q)),
                             "used": len(s) >= min_bin})
    return caps, val, pd.DataFrame(cap_rows)


def lookup_deploy_cap(caps, q, pest, alert_tstar, bin_w):
    b = int(alert_tstar // bin_w) * bin_w
    for key in [(q, "bin", (pest, b)), (q, "pest", pest), (q, "global", None)]:
        if key in caps:
            return caps[key], key[1]
    return float("inf"), "none"


# --------------------------- policy application ------------------------------
def choose_offset(selected: float, cap_offset: float) -> int:
    """Cap an offset to <= cap_offset using anchor candidates (never increase).
    cap_offset = max feasible offset (eval_doy<=bound). Keep selected if feasible,
    else snap to the LARGEST anchor <= cap, else the smallest anchor (best effort)."""
    if selected <= cap_offset:
        return int(round(selected))
    feasible = [a for a in ANCHORS if a <= cap_offset]
    return int(max(feasible)) if feasible else int(min(ANCHORS))


def recompute(off_mu, L, R, alert_tstar, offset, sigma) -> dict:
    mu = mu_at_offset(off_mu, offset)
    true_start = int(L) + 1
    true_mid = 0.5 * (L + R)
    pL = int(round(mu - Z * sigma)) if not pd.isna(mu) else float("nan")
    pR = int(round(mu + Z * sigma)) if not pd.isna(mu) else float("nan")
    iou = iou_from_mu(mu, L, R, sigma)
    eval_doy = alert_tstar + offset
    return {
        "selected_offset": int(offset), "eval_doy": eval_doy,
        "mu": mu, "pred_L": pL, "pred_R": pR,
        "true_L_plus_1": true_start, "true_mid": true_mid, "true_R": int(R),
        "iou": iou, "MAE_center": abs(mu - true_mid) if not pd.isna(mu) else float("nan"),
        "PI_hit": (not pd.isna(mu)) and (pL <= true_mid <= pR),
        "late_eval": eval_doy > true_start,
        "late_mu": (not pd.isna(mu)) and (mu > true_mid),
        "late_PI_start": (not pd.isna(mu)) and (pL > true_start),
        "no_overlap": iou == 0,
    }


def summarize(df, group_cols):
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("late_eval", "size"),
        late_eval_rate=("late_eval", "mean"),
        late_mu_rate=("late_mu", "mean"),
        late_PI_start_rate=("late_PI_start", "mean"),
        no_overlap_rate=("no_overlap", "mean"),
        mean_MAE_center=("MAE_center", "mean"),
        mean_IoU=("iou", "mean"),
        PI_hit_rate=("PI_hit", "mean"),
    ).reset_index()
    rc = ["late_eval_rate", "late_mu_rate", "late_PI_start_rate", "no_overlap_rate",
          "mean_MAE_center", "mean_IoU", "PI_hit_rate"]
    out[rc] = out[rc].round(4)
    return out


def write_no_overwrite(df, path: Path, force: bool):
    if path.exists() and not force:
        raise SystemExit(f"Refuse to overwrite: {path}  (use --force)")
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RICE_ROOT / "outputs/diag/stage2_offset_constraint"))
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT)
    ap.add_argument("--bin-width", type=int, default=20, help="alert_doy_bin width (days)")
    ap.add_argument("--min-bin", type=int, default=20, help="min val n for bin-level cap")
    ap.add_argument("--min-pest", type=int, default=30, help="min val n for pest-level cap")
    ap.add_argument("--focus-offsets", default="45,60")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    focus = [int(x) for x in args.focus_offsets.split(",") if x.strip()]
    print("[policy] selector/offset policy diagnostic — NO model/loss change.")
    print("[leakage] oracle_* use TEST true_start for SELECTION = ANALYSIS-ONLY upper bound, "
          "never deploy. deploy_* use ONLY (pest, alert_tstar) + VAL quantiles.\n")

    # 1) deployable caps from validation (leakage-free)
    caps, val, caps_df = build_val_caps(list(DEPLOY_Q.values()), args.min_bin, args.min_pest, args.bin_width)
    print(f"[caps] val pool n={len(val)}  bins used={int(caps_df['used'].sum())}/{len(caps_df)} "
          f"(min_bin={args.min_bin}); pest & global fallbacks ready.")

    # 2) original per-sample tables (one per pest-year-variant cell)
    files = sorted(glob.glob(str(RICE_ROOT / VIZ_GLOB), recursive=True))
    grids: Dict[tuple, dict] = {}
    all_rows = []
    sanity = []
    for f in files:
        meta = parse_dir(f)
        if not meta:
            continue
        pest, year, variant = meta["pest"], meta["year"], meta["variant"]
        ps = pd.read_csv(f)
        key = (pest, year)
        if key not in grids:
            grids[key] = load_grid(pest, year)
        grid = grids[key]
        for _, r in ps.iterrows():
            sid = r["sample_id"]
            if sid not in grid:
                continue
            gi = grid[sid]
            base = {"pest": pest, "year": year, "variant": variant, "sample_id": sid,
                    "site": r.get("site", ""), "alert_tstar": gi["alert_tstar"],
                    "orig_offset": int(round(r["selected_offset"]))}
            sel = float(r["selected_offset"])
            # caps per policy
            true_start = gi["L"] + 1
            for pol in POLICIES:
                if pol == "original":
                    cap = float("inf"); uses_label = False
                elif pol in ORACLE:
                    cap = true_start - ORACLE[pol] - gi["alert_tstar"]   # ANALYSIS-ONLY
                    uses_label = True
                else:  # deployable
                    safe, _lvl = lookup_deploy_cap(caps, DEPLOY_Q[pol], pest, gi["alert_tstar"], args.bin_width)
                    cap = safe; uses_label = False
                off = choose_offset(sel, cap)
                rec = recompute(gi["off_mu"], gi["L"], gi["R"], gi["alert_tstar"], off, args.sigma)
                row = {**base, "policy": pol, "uses_test_label": uses_label,
                       "orig_selected_offset": sel, "cap_offset": cap, **rec}
                all_rows.append(row)
                if pol == "original":
                    sanity.append(abs(rec["iou"] - float(r["iou"])))
    df = pd.DataFrame(all_rows)
    n_samples = (df["policy"] == "original").sum()
    print(f"[data] {n_samples} test samples x {len(POLICIES)} policies = {len(df)} rows. "
          f"original-vs-stored IoU mean|Δ|={np.mean(sanity):.4f} (recompute matches build_per_sample).\n")

    # 3) outputs
    cols = ["policy", "uses_test_label", "pest", "year", "variant", "sample_id", "site",
            "alert_tstar", "orig_offset", "orig_selected_offset", "cap_offset",
            "selected_offset", "eval_doy", "mu", "pred_L", "pred_R",
            "true_L_plus_1", "true_mid", "true_R", "iou", "MAE_center", "PI_hit",
            "late_eval", "late_mu", "late_PI_start", "no_overlap"]
    cols = [c for c in cols if c in df.columns]
    for pol in POLICIES:
        write_no_overwrite(df[df.policy == pol][cols], out_dir / f"per_sample_{pol}.csv", args.force)

    write_no_overwrite(caps_df, out_dir / "deployable_offset_caps.csv", args.force)
    write_no_overwrite(summarize(df, ["policy"]), out_dir / "summary_overall.csv", args.force)
    write_no_overwrite(summarize(df, ["policy", "pest"]), out_dir / "summary_by_pest.csv", args.force)
    write_no_overwrite(summarize(df, ["policy", "pest", "selected_offset"]),
                       out_dir / "summary_by_pest_offset.csv", args.force)

    # focused on ORIGINAL offset in {focus} (so we can read "blast off45" etc.)
    foc = df[df["orig_offset"].isin(focus)].copy()
    write_no_overwrite(summarize(foc, ["pest", "orig_offset", "policy"]),
                       out_dir / f"summary_off{'_'.join(map(str, focus))}.csv", args.force)

    late = df[df["late_eval"]].sort_values(["policy", "pest", "selected_offset", "sample_id"])
    write_no_overwrite(late[cols], out_dir / "late_eval_examples.csv", args.force)
    write_no_overwrite(late[late["orig_offset"].isin(focus)][cols],
                       out_dir / f"late_eval_examples_off{'_'.join(map(str, focus))}.csv", args.force)

    # 4) console digest — the requested comparisons
    piv = df.pivot_table(index="policy", values=["late_eval", "late_mu", "late_PI_start",
                         "no_overlap", "iou", "MAE_center"], aggfunc="mean").reindex(POLICIES).round(4)
    print("=== OVERALL by policy ===")
    print(piv.to_string(), "\n")

    def cell(pest, oo, metric):
        s = df[(df.pest == pest) & (df.orig_offset == oo)]
        return {p: round(s[s.policy == p][metric].mean(), 4) for p in POLICIES}, len(s) // len(POLICIES)

    print("=== requested checks (orig_offset group) ===")
    for pest, oo, metric, label in [
        ("blast", 45, "late_eval", "blast off45 late_eval"),
        ("bacterial_blight", 60, "late_eval", "bacterial_blight off60 late_eval"),
        ("brown_spot", 60, "late_eval", "brown_spot off60 late_eval"),
        ("BPH", 60, "no_overlap", "BPH off60 no_overlap"),
        ("rice_stem_borer_2", 60, "no_overlap", "rice_stem_borer_2 off60 no_overlap"),
    ]:
        vals, n = cell(pest, oo, metric)
        print(f"  {label:42s} (n={n:3d}): " +
              "  ".join(f"{p.split('_')[0][:4]}{('_'+p.split('_')[1]) if '_' in p else ''}={vals[p]}" for p in POLICIES))

    sb = df[df.pest == "sheath_blight"]
    print("\n  sheath_blight (all offsets) original: "
          f"late_eval={sb[sb.policy=='original'].late_eval.mean():.3f}  "
          f"late_mu={sb[sb.policy=='original'].late_mu.mean():.3f}  "
          "-> late_mu >> late_eval means the problem is point-estimate bias (mu late), "
          "which offset capping does NOT fix (needs model/calibration, deferred).")
    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
