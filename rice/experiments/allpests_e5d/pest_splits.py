#!/usr/bin/env python
"""Clean-protocol 3-fold selection splits for one pest. Split generation + assertions only.

Generalises the WBPH make_splits.py. For each eval year y the y-1 season cohort is partitioned
by SAMPLE ID (md5, seed 42) into disjoint val_ckpt 40% / val_fit 30% / val_cal 30%:
    val_ckpt -> checkpoint selection (the trainer's val set is restricted to it)
    val_fit  -> coverage-aware LightGBM selector
    val_cal  -> that fold's additive mu shift, chosen on its OWN val_cal only
Stage-2 train = years <= y-2. Eval = year y, touched once.

The bucket function is a pure hash of the sample_id, so both servers derive identical splits
without exchanging any file -- that is what makes the two-server run reproducible.

Exits non-zero unless every isolation assertion passes.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d")); sys.path.insert(0, str(WS))
import pest_paths as PP

SEED = 42
FRAC = {"val_ckpt": 0.40, "val_fit": 0.30, "val_cal": 0.30}


def bucket(sample_id: str) -> str:
    h = hashlib.md5(f"{SEED}:{sample_id}".encode()).hexdigest()
    u = int(h[:12], 16) / float(16 ** 12)
    if u < FRAC["val_ckpt"]:
        return "val_ckpt"
    if u < FRAC["val_ckpt"] + FRAC["val_fit"]:
        return "val_fit"
    return "val_cal"


def sid_col(df):
    return df["site"].astype(str) + "-" + df["year"].astype(int).astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=PP.YEARS)
    a = ap.parse_args()

    out = PP.clean_root(a.pest); out.mkdir(parents=True, exist_ok=True)
    gpath = PP.dev_grid(a.pest)
    if not gpath.exists():
        raise SystemExit(f"[splits] dev grid missing: {gpath} (run pest_grid.py --mode dev first)")
    grid = pd.read_csv(gpath)
    grid = grid[(grid.variant == "E5d") & (grid.offset.isin(PP.OFFSETS))]

    rows, checks, amap = [], {}, {}
    for y in a.years:
        dsp = pd.read_csv(PP.dispatch_csv(a.pest, y))
        v = dsp[dsp.split == "val"].copy()
        v["sample_id"] = sid_col(v)
        if v["year"].nunique() != 1 or int(v["year"].iloc[0]) != y - 1:
            raise SystemExit(f"[splits] {a.pest}/{y}: dispatch val year != {y-1}")
        sids = sorted(v["sample_id"].unique())
        assign = {s: bucket(s) for s in sids}
        amap[str(y)] = assign

        tr = dsp[dsp.split == "train"]; te = dsp[dsp.split == "test"]
        tr_years = sorted(tr.year.unique())
        tr_sids = set(sid_col(tr).unique()); ev_sids = set(sid_col(te).unique())
        sets = {k: {s for s, b in assign.items() if b == k} for k in FRAC}

        gv = grid[(grid.year == y) & (grid.split == "val")].copy()
        gv["b"] = gv["sample_id"].map(assign)
        for k in FRAC:
            g = gv[gv.b == k]
            vo = g.groupby("sample_id").offset.nunique()
            rows.append(dict(pest=a.pest, eval_year=y, split=k, select_year=y - 1,
                             train_years=f"{min(tr_years)}-{max(tr_years)}",
                             n_cohort=len(sets[k]), n_grid_samples=int(g.sample_id.nunique()),
                             n_candidate_rows=int(len(g)),
                             mean_valid_offsets=float(vo.mean()) if len(vo) else float("nan")))
        gt = grid[(grid.year == y) & (grid.split == "test")]
        rows.append(dict(pest=a.pest, eval_year=y, split="eval(test)", select_year=y,
                         train_years=f"{min(tr_years)}-{max(tr_years)}",
                         n_cohort=len(ev_sids), n_grid_samples=int(gt.sample_id.nunique()),
                         n_candidate_rows=int(len(gt)),
                         mean_valid_offsets=float(gt.groupby("sample_id").offset.nunique().mean())
                         if len(gt) else float("nan")))

        c = {
            "val_ckpt&val_fit": len(sets["val_ckpt"] & sets["val_fit"]),
            "val_ckpt&val_cal": len(sets["val_ckpt"] & sets["val_cal"]),
            "val_fit&val_cal": len(sets["val_fit"] & sets["val_cal"]),
            "train&val_ckpt": len(tr_sids & sets["val_ckpt"]),
            "train&val_fit": len(tr_sids & sets["val_fit"]),
            "train&val_cal": len(tr_sids & sets["val_cal"]),
            "union_covers_cohort": int(len(set().union(*sets.values())) == len(sids)),
            "all_select_years_lt_eval": int(all(int(s.split("-")[-1]) < y for s in sids)),
            "train_years_le_eval_minus_2": int(max(tr_years) <= y - 2),
            "eval_not_in_any_select": int(len(ev_sids & set().union(*sets.values())) == 0),
            "eval_not_in_train": int(len(ev_sids & tr_sids) == 0),
            "rows_single_bucket": int(gv.groupby("sample_id")["b"].nunique().max() == 1) if len(gv) else 1,
            "n_cohort": len(sids),
            "frac": {k: round(len(sets[k]) / len(sids), 4) for k in FRAC} if sids else {},
        }
        if len(sets["val_ckpt"]) == 0:
            c["EMPTY_val_ckpt"] = 1
        checks[str(y)] = c

    man = pd.DataFrame(rows)
    man.to_csv(out / "split_manifest.csv", index=False)
    PP.split_assignment(a.pest).write_text(json.dumps(amap, indent=2))

    def fold_ok(c):
        zeros = ["val_ckpt&val_fit", "val_ckpt&val_cal", "val_fit&val_cal",
                 "train&val_ckpt", "train&val_fit", "train&val_cal"]
        ones = ["union_covers_cohort", "all_select_years_lt_eval", "train_years_le_eval_minus_2",
                "eval_not_in_any_select", "eval_not_in_train", "rows_single_bucket"]
        return all(c[k] == 0 for k in zeros) and all(c[k] == 1 for k in ones) and "EMPTY_val_ckpt" not in c

    allpass = all(fold_ok(c) for c in checks.values())
    checks["ALL_PASS"] = bool(allpass)
    (out / "split_overlap_checks.json").write_text(json.dumps(checks, indent=2))

    pd.set_option("display.width", 200)
    print(man.round(2).to_string(index=False))
    print(f"\n[splits] {a.pest} ALL_PASS = {allpass}")
    if not allpass:
        print(json.dumps(checks, indent=2))
    return 0 if allpass else 1


if __name__ == "__main__":
    sys.exit(main())
