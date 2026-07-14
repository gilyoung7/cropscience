"""3-way matched site-year eval:
  prod_prod : production Stage-1 + production Stage-2
  prod_s2n  : production Stage-1 + Stage-2 direct neighbor
  s1n_s2n   : Stage-1 neighbor   + Stage-2 direct neighbor

Each config's ckpt carries its own stage2_dispatch_feature_csv (production CSV for
the first two; the Stage-1-neighbor CSV for s1n_s2n), so matched_eval_one matches at
that config's own alert points. Offset is tuned on prod_prod and applied to all 3.

Reuses matched_eval_one from eval_s2n_matched_compare (no project file modified here).

Usage:
  PYTHONPATH=. .venv/bin/python rice/scripts/eval_s1n_s2n_matched.py \
      --pests WBPH sheath_blight BPH bacterial_blight brown_spot blast
Output:
  rice/outputs_stage2_compare_eval/all_pests_s1n_s2n_matched_compare.tsv
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from rice.scripts.eval_s2n_matched_compare import matched_eval_one, OFFSET_GRID

PROD = "rice/outputs_stage2_batch_2024_bestgate"
DN = "rice/outputs_stage2_direct_neighbor"
S1N = "rice/outputs_stage2_s1n_s2n"
OUT_BASE = "rice/outputs_stage2_compare_eval"

CONFIGS = [
    ("prod_prod", lambda p: f"{PROD}/{p}/lead_v3_final/ckpt/checkpoint_run4.pt"),
    ("prod_s2n",  lambda p: f"{DN}/{p}/lead_v3_final/ckpt/checkpoint_run4.pt"),
    ("s1n_s2n",   lambda p: f"{S1N}/{p}/lead_v3_final/ckpt/checkpoint_run4.pt"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="+",
                    default=["WBPH", "sheath_blight", "BPH", "bacterial_blight", "brown_spot", "blast"])
    ap.add_argument("--offsets", type=int, nargs="+", default=OFFSET_GRID)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    header = ["pest", "config", "matched_IoU80", "PI_hit", "MAE_center",
              "matched_event_count", "n_true", "alert_cohort", "offset"]
    rows = [header]
    for pest in args.pests:
        res = {}
        for tag, pathfn in CONFIGS:
            p = pathfn(pest)
            if not Path(p).exists():
                print(f"[skip] {pest}/{tag}: ckpt missing {p}"); continue
            try:
                res[tag] = matched_eval_one(p, pest, device, args.offsets)
            except Exception as e:
                print(f"[ERROR] {pest}/{tag}: {type(e).__name__}: {e}")
        if "prod_prod" not in res:
            print(f"[skip] {pest}: no prod_prod baseline"); continue
        bsw = res["prod_prod"]["sweep"]
        valid = [(o, v["matched_IoU80"]) for o, v in bsw.items() if np.isfinite(v["matched_IoU80"])]
        best_off = max(valid, key=lambda x: x[1])[0] if valid else args.offsets[0]
        for tag, _ in CONFIGS:
            if tag not in res:
                continue
            v = res[tag]["sweep"][best_off]
            rows.append([pest, tag, f"{v['matched_IoU80']:.4f}", f"{v['PI_hit_rate']:.4f}",
                         f"{v['MAE_center']:.4f}", str(v["matched_event_count"]), str(v["n_true"]),
                         str(res[tag]["alert_cohort"]), str(best_off)])
        print(f"[{pest}] off={best_off} | "
              + " | ".join(f"{t}={res[t]['sweep'][best_off]['matched_IoU80']:.4f}"
                           for t, _ in CONFIGS if t in res))

    out = Path(OUT_BASE) / "all_pests_s1n_s2n_matched_compare.tsv"
    out.write_text("\n".join("\t".join(r) for r in rows) + "\n")
    print("\n===== S1N+S2N 3-WAY MATCHED COMPARISON =====")
    print("\n".join("\t".join(r) for r in rows))
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
