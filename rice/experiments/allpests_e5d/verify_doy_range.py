#!/usr/bin/env python
"""READ-ONLY proof of the DOY range actually used by Stage-1 and Stage-2, per (pest, eval_year).

Deliberately does NOT trust rice/pests/<pest>/config.py. For each stage it cross-checks three
independent sources and flags any disagreement:

  A. checkpoint      Stage-1: ckpt['doy_start'/'doy_end'/'T'] (top-level keys)
                     Stage-2: lead_v3_final/hparams.json doy_start/doy_end
  B. execution log   Stage-1: 'DOY_START=..' / 'DOY_END=..' printed in train_A.log
                     Stage-2: --doy_start_override / --doy_end_override in the run_train cmd
  C. generated data  Stage-1: alert_tstar min/max in the gate CSV (issue DOY actually emitted)
                     Stage-2: t_star_doy min/max in lead_v3_{val,test}_sample_grid.csv
                              (the model grid points actually scored)

The Stage-1 model each Stage-2 cell consumed is resolved from that cell's own
dispatch_feature_table.log, not assumed from a run index.

No training, no forward, no dataset rebuild. Nothing is written outside --out.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
import pandas as pd, torch

CS = Path("/home/gpu4080/research/cropscience")
WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
PESTS = ["WBPH", "BPH", "blast", "sheath_blight", "brown_spot",
         "bacterial_blight", "rice_stem_borer_1", "rice_stem_borer_2"]
YEARS = [2022, 2023, 2024]


def batch_dir(y): return "batch_2024_bestgate" if y == 2024 else f"batch_{y}_baseline"
def cell(p, y): return CS / f"rice/outputs/stage2/{batch_dir(y)}/{p}"


def s1_ckpt_path(p, y):
    """Resolve the Stage-1 ckpt this cell actually used, from its dispatch build log."""
    lg = cell(p, y) / "logs/dispatch_feature_table.log"
    if lg.exists():
        m = re.findall(r"(\S*batch_rolling/\S+?/ckpt/\S+\.pt)", lg.read_text(errors="ignore"))
        if m:
            q = Path(m[0])
            if not q.is_absolute():
                q = CS / q
            if q.exists():
                return q, "dispatch_log"
    # fall back to the conventional forward-chained layout
    for run in (0, 1, 2):
        q = (CS / f"rice/outputs/stage1/batch_rolling/{p}/run{run}/"
                  f"split{y-2021}_v{y-1}_t{y}/A/ckpt/event_xgb_w28_lead14-45_A.pt")
        if q.exists():
            return q, f"convention_run{run}"
    return None, "MISSING"


def s1_row(p, y):
    r = {"stage": "Stage-1"}
    ck, how = s1_ckpt_path(p, y)
    r["src_ckpt_how"] = how
    if ck:
        c = torch.load(ck, map_location="cpu", weights_only=False)
        r["ckpt_doy"] = f"{c.get('doy_start')}-{c.get('doy_end')}"
        r["ckpt_T"] = c.get("T")
        r["ckpt_window"] = c.get("nowcast_window")
        r["ckpt_tstar_start"] = c.get("nowcast_tstar_start")
        lg = ck.parent.parent.parent / "logs/train_A.log"
        if lg.exists():
            t = lg.read_text(errors="ignore")
            a = re.search(r"DOY_START=(\d+)", t); b = re.search(r"DOY_END=(\d+)", t)
            r["log_doy"] = f"{a.group(1)}-{b.group(1)}" if a and b else "not_printed"
        else:
            r["log_doy"] = "no_log"
    else:
        r["ckpt_doy"] = r["log_doy"] = "MISSING"
    # generated data: the issue DOYs Stage-1 actually emitted
    g = sorted(cell(p, y).glob("gate_*_R088_features_per_sy.csv"))
    if g:
        a = pd.read_csv(g[0])["alert_tstar"].dropna()
        a = a[a > 0]
        r["data_min"] = int(a.min()) if len(a) else None
        r["data_max"] = int(a.max()) if len(a) else None
        r["data_field"] = "alert_tstar"
        r["data_n"] = int(len(a))
    return r


def s2_row(p, y):
    r = {"stage": "Stage-2"}
    hp = cell(p, y) / "lead_v3_final/hparams.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        r["ckpt_doy"] = f"{h.get('doy_start')}-{h.get('doy_end')}"
        r["ckpt_T"] = (h["doy_end"] - h["doy_start"] + 1) if h.get("doy_end") else None
        r["ckpt_window"] = h.get("stage2_nowcast_window")
        r["ckpt_tstar_start"] = h.get("stage2_nowcast_tstar_start")
    else:
        r["ckpt_doy"] = "MISSING"
    lg = cell(p, y) / "logs/stage2_lead_v3_train.log"
    if lg.exists() and lg.stat().st_size > 0:
        t = lg.read_text(errors="ignore")
        a = re.search(r"--doy_start_override (\d+)", t); b = re.search(r"--doy_end_override (\d+)", t)
        r["log_doy"] = f"{a.group(1)}-{b.group(1)}" if a and b else "no_override(pest_default)"
    else:
        r["log_doy"] = "log_empty"
    # generated data: the model grid points actually scored
    lo, hi, n = [], [], 0
    for sp in ("val", "test"):
        f = cell(p, y) / f"lead_v3_{sp}_sample_grid.csv"
        if f.exists():
            d = pd.read_csv(f, usecols=["t_star_doy"])["t_star_doy"].dropna()
            if len(d):
                lo.append(d.min()); hi.append(d.max()); n += len(d)
    r["data_min"] = int(min(lo)) if lo else None
    r["data_max"] = int(max(hi)) if hi else None
    r["data_field"] = "t_star_doy"
    r["data_n"] = n
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "outputs/allpests_e5d/_capacity"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in PESTS:
        for y in YEARS:
            for fn in (s1_row, s2_row):
                try:
                    r = fn(p, y)
                except Exception as e:                      # noqa: BLE001
                    r = {"stage": fn.__name__, "ckpt_doy": f"ERR:{type(e).__name__}"}
                r.update(pest=p, eval_year=y)
                # do the three sources agree on the window?
                ck = str(r.get("ckpt_doy", ""))
                lg = str(r.get("log_doy", ""))
                agree = (lg.startswith("no_override") or lg in ("not_printed", "no_log", "log_empty")
                         or lg == ck)
                r["ckpt_vs_log"] = "OK" if agree else f"MISMATCH({ck} vs {lg})"
                if r.get("data_min") is not None and "-" in ck:
                    s, e2 = (int(x) for x in ck.split("-"))
                    r["data_inside_ckpt_window"] = bool(s <= r["data_min"] and r["data_max"] <= e2)
                    r["data_offset_from_start"] = r["data_min"] - s
                rows.append(r)

    df = pd.DataFrame(rows)[
        ["pest", "eval_year", "stage", "ckpt_doy", "ckpt_T", "ckpt_window", "ckpt_tstar_start",
         "log_doy", "ckpt_vs_log", "data_field", "data_min", "data_max", "data_n",
         "data_inside_ckpt_window", "data_offset_from_start", "src_ckpt_how"]]
    df.to_csv(out / "doy_range_verification.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))
    bad = df[(df.ckpt_vs_log.astype(str).str.startswith("MISMATCH")) |
             (df.data_inside_ckpt_window == False)]                      # noqa: E712
    print(f"\n=== rows={len(df)}  mismatches/out-of-window={len(bad)} ===")
    if len(bad):
        print(bad.to_string(index=False))
    print(f"[verify] wrote {out/'doy_range_verification.csv'}")


if __name__ == "__main__":
    main()
