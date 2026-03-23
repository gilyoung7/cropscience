import glob
import os
import re

import numpy as np
import torch

from rice.configs import config as C
from rice.scripts.common import make_loader
from rice.scripts.run_viz_interval import (
    EventTransformer,
    HazardTransformer,
    apply_temperature,
    best_tau_by_target,
    build_nowcast_samples,
    build_stage2_nowcast_samples,
    build_tabular_from_samples,
    build_samples_for_run,
    collect_interval_preds,
    compute_norm_stats,
    default_out_root,
    fit_temperature_grid,
    make_event_labels,
    overlap_metrics,
    predict_event_prob_event_model,
    resolve_pest,
    split_by_site,
    IntervalEventDataset,
)

# pest -> (stage1 window, tau_mode)
PEST_CFG = {
    "WBPH": (56, "f1"),
    "bacterial_blight": (56, "recall_target"),
    "blast": (84, "f1"),
    "brown_spot": (56, "recall_target"),
    "rice_stem_borer_1": (56, "recall_target"),
    "rice_stem_borer_2": (56, "recall_target"),
    "sheath_blight": (28, "f1"),
}

TAU_TARGET_PREC = 0.6
TAU_TARGET_REC = 0.6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_run_split(path: str):
    m = re.search(r"run(\d+).*_split(\d+)_ymin2002", os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def load_ckpt(path: str):
    return torch.load(path, map_location="cpu")


def mean_std(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return (float("nan"), float("nan"))
    if x.size == 1:
        return (float(x[0]), 0.0)
    return (float(x.mean()), float(x.std(ddof=1)))


def compute_metrics_for_pest(pest: str, w: int, tau_mode: str):
    out_root = default_out_root(pest)
    stage2_ckpts = sorted(glob.glob(f"{out_root}/stage2/ckpt/checkpoint_run*_stage2_nowcast_*_ymin2002.pt"))
    if not stage2_ckpts:
        raise FileNotFoundError(f"no ymin2002 stage2 ckpt for {pest}")

    rows_all = []

    for ckpt2_path in stage2_ckpts:
        parsed = parse_run_split(ckpt2_path)
        if not parsed:
            continue
        run, split_seed = parsed
        ckpt1_path = f"{out_root}/stage1/ckpt/event_classifier_{pest}_run{run}_stage1_xgb_nowcast_w{w}_s7_split{split_seed}_ymin2002.pt"
        if not os.path.exists(ckpt1_path):
            raise FileNotFoundError(ckpt1_path)

        ckpt1 = load_ckpt(ckpt1_path)
        ckpt2 = load_ckpt(ckpt2_path)

        _, get_feature_cols = resolve_pest(pest)
        _, _, T, samples = build_samples_for_run(run, get_feature_cols)

        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

        # Stage1 nowcast samples
        task_mode = str(ckpt1.get("task_mode", "season_complete"))
        nowcast_window = int(ckpt1.get("nowcast_window", 28))
        nowcast_stride = int(ckpt1.get("nowcast_stride", 7))
        nowcast_tstar_start = ckpt1.get("nowcast_tstar_start", None)
        nowcast_only_pre_event = int(ckpt1.get("nowcast_only_pre_event", 1))
        nowcast_event_time_proxy = str(ckpt1.get("nowcast_event_time_proxy", "r"))
        if task_mode != "nowcast":
            raise ValueError("stage1 task_mode not nowcast")
        val_s1 = build_nowcast_samples(
            val_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )
        test_s1 = build_nowcast_samples(
            test_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )

        # Stage2 nowcast samples
        stage2_nowcast = bool(ckpt2.get("stage2_nowcast", False))
        if stage2_nowcast:
            stage2_nowcast_window = int(ckpt2.get("stage2_nowcast_window", 56))
            stage2_nowcast_stride = int(ckpt2.get("stage2_nowcast_stride", 7))
            stage2_nowcast_tstar_start = ckpt2.get("stage2_nowcast_tstar_start", None)
            stage2_nowcast_only_pre_event = int(ckpt2.get("stage2_nowcast_only_pre_event", 1))
            stage2_nowcast_event_time_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "mid"))
            test_s2 = build_stage2_nowcast_samples(
                test_s,
                window=stage2_nowcast_window,
                stride=stage2_nowcast_stride,
                tstar_start=stage2_nowcast_tstar_start,
                only_pre_event=bool(stage2_nowcast_only_pre_event),
                event_time_proxy=stage2_nowcast_event_time_proxy,
            )
        else:
            test_s2 = test_s

        # site-year denominators
        site_year_all = {(s.get("site_id"), int(s.get("year"))) for s in test_s2}
        site_year_interval = {
            (s.get("site_id"), int(s.get("year")))
            for s in test_s2
            if str(s.get("censor_type", "")) == "interval"
        }
        n_total_site = len(site_year_all)
        n_interval_site = len(site_year_interval)

        # norm stats for stage2
        x_mean, x_std = compute_norm_stats(train_s)
        test_ds2 = IntervalEventDataset(test_s2, x_mean, x_std)
        test_loader2 = make_loader(test_ds2, C.BATCH_EVAL, shuffle=False)

        # stage1 labels
        y_val = make_event_labels(val_s1)

        for d2 in ckpt2["trained_states"]:
            seed = int(d2["seed"])

            # stage1 state
            d1 = None
            for s in ckpt1["trained_states"]:
                if int(s["seed"]) == seed:
                    d1 = s
                    break
            if d1 is None:
                continue

            # stage1 model
            model_kind = str(ckpt1.get("event_model", "transformer"))
            if model_kind == "transformer":
                model1 = EventTransformer(
                    d_in=int(test_ds2[0][0].shape[-1]),
                    d_model=C.D_MODEL,
                    nhead=C.N_HEAD,
                    num_layers=2,
                    dropout=C.DROPOUT,
                    max_len=C.MAX_LEN,
                ).to(DEVICE)
                model1.load_state_dict(d1["state_dict"])
                model1.eval()
                val_loader1 = make_loader(IntervalEventDataset(val_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
                test_loader1 = make_loader(IntervalEventDataset(test_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
                p_val_raw = predict_event_prob_event_model(model1, val_loader1, device=DEVICE)
                p_test_raw = predict_event_prob_event_model(model1, test_loader1, device=DEVICE)
            else:
                clf = d1.get("sk_model")
                if clf is None:
                    raise ValueError("stage1 checkpoint missing sk_model")
                X_val_tab = build_tabular_from_samples(val_s1)
                X_test_tab = build_tabular_from_samples(test_s1)
                if hasattr(clf, "predict_proba"):
                    p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
                    p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
                else:
                    p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
                    p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

            t_best, _ = fit_temperature_grid(y_val, p_val_raw)
            p_val_cal = apply_temperature(p_val_raw, t_best)
            p_test_cal = apply_temperature(p_test_raw, t_best)

            tau, _vf1, _vpr, _vrc = best_tau_by_target(
                y_val,
                p_val_cal,
                mode=tau_mode,
                target_precision=TAU_TARGET_PREC,
                target_recall=TAU_TARGET_REC,
            )

            # alert_map WITHOUT t* condition: first t* with p>=tau
            alert = {}
            for s, p in zip(test_s1, p_test_cal):
                tstar = int(s.get("tstar", -1))
                site = str(s.get("site_id", ""))
                year = int(s.get("year", -1))
                key = f"{site}-{year}"
                if p < float(tau):
                    continue
                prev = alert.get(key)
                if prev is None or tstar < int(prev):
                    alert[key] = tstar

            # stage2 model
            model2 = HazardTransformer(
                d_in=int(test_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(DEVICE)
            model2.load_state_dict(d2["state_dict"])
            model2.eval()

            rows = collect_interval_preds(
                model2,
                test_loader2,
                source_samples=test_s2,
                Tend=T,
                device=DEVICE,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=len(test_s2) + 5,
            )

            # row map
            row_map = {}
            for r in rows:
                tstar = r.get("tstar")
                if tstar is None:
                    continue
                row_map[(r["sample_id"], int(tstar))] = r

            # gate-before metrics: all rows
            ious_all = []
            recs_all = []
            precs_all = []
            for r in rows:
                iou, rec, prec = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
                ious_all.append(iou)
                recs_all.append(rec)
                precs_all.append(prec)

            # gate-after metrics: matched rows at alert t*
            ious = []
            recs = []
            precs = []
            for sid, alert_t in alert.items():
                r = row_map.get((sid, int(alert_t)))
                if r is None:
                    continue
                iou, rec, prec = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
                ious.append(iou)
                recs.append(rec)
                precs.append(prec)

            n_total_after = len(alert)
            n_interval_after = 0
            for sid in alert.keys():
                try:
                    site, year = sid.rsplit("-", 1)
                    if (site, int(year)) in site_year_interval:
                        n_interval_after += 1
                except Exception:
                    pass

            rows_all.append(
                {
                    "pest": pest,
                    "run": run,
                    "split_seed": split_seed,
                    "seed": seed,
                    "n_total_site": n_total_site,
                    "n_interval_site": n_interval_site,
                    "n_total_after": n_total_after,
                    "n_interval_after": n_interval_after,
                    "before_iou": float(np.mean(ious_all)) if ious_all else float("nan"),
                    "before_prec": float(np.mean(precs_all)) if precs_all else float("nan"),
                    "before_rec": float(np.mean(recs_all)) if recs_all else float("nan"),
                    "after_iou": float(np.mean(ious)) if ious else float("nan"),
                    "after_prec": float(np.mean(precs)) if precs else float("nan"),
                    "after_rec": float(np.mean(recs)) if recs else float("nan"),
                }
            )

    return rows_all


def main():
    all_rows = []
    for pest, (w, tau_mode) in PEST_CFG.items():
        print(f"processing {pest}", flush=True)
        all_rows.extend(compute_metrics_for_pest(pest, w, tau_mode))

    by_pest = {}
    for r in all_rows:
        by_pest.setdefault(r["pest"], []).append(r)

    headers = [
        "pest",
        "gate",
        "IoU80",
        "Precision80",
        "Recall80",
        "N_total(site-year)",
        "N_interval(site-year)",
        "Gate%",
    ]

    for pest, rows in by_pest.items():
        before_iou = [r["before_iou"] for r in rows]
        before_prec = [r["before_prec"] for r in rows]
        before_rec = [r["before_rec"] for r in rows]
        n_total = [r["n_total_site"] for r in rows]
        n_int = [r["n_interval_site"] for r in rows]

        after_iou = [r["after_iou"] for r in rows]
        after_prec = [r["after_prec"] for r in rows]
        after_rec = [r["after_rec"] for r in rows]
        n_total_a = [r["n_total_after"] for r in rows]
        n_int_a = [r["n_interval_after"] for r in rows]

        b_iou_m, b_iou_s = mean_std(before_iou)
        b_pr_m, b_pr_s = mean_std(before_prec)
        b_re_m, b_re_s = mean_std(before_rec)
        b_nt_m, b_nt_s = mean_std(n_total)
        b_ni_m, b_ni_s = mean_std(n_int)

        a_iou_m, a_iou_s = mean_std(after_iou)
        a_pr_m, a_pr_s = mean_std(after_prec)
        a_re_m, a_re_s = mean_std(after_rec)
        a_nt_m, a_nt_s = mean_std(n_total_a)
        a_ni_m, a_ni_s = mean_std(n_int_a)

        gate_rate = [r["n_total_after"] / max(r["n_total_site"], 1) * 100.0 for r in rows]
        g_m, g_s = mean_std(gate_rate)

        def fmt(m, s):
            return f"{m:.4f}±{s:.4f}"

        def fmtN(m, s):
            return f"{m:.2f}±{s:.2f}"

        def fmtG(m, s):
            return f"{m:.2f}%±{s:.2f}%"

        out_dir = os.path.join("rice", "outputs", pest, "gating")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "summary_gate_no_tstar_ymin2002.tsv")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\t".join(headers) + "\n")
            f.write(
                "\t".join(
                    [
                        pest,
                        "게이팅 전",
                        fmt(b_iou_m, b_iou_s),
                        fmt(b_pr_m, b_pr_s),
                        fmt(b_re_m, b_re_s),
                        fmtN(b_nt_m, b_nt_s),
                        fmtN(b_ni_m, b_ni_s),
                        fmtG(100.0, 0.0),
                    ]
                )
                + "\n"
            )
            f.write(
                "\t".join(
                    [
                        pest,
                        "게이팅 후",
                        fmt(a_iou_m, a_iou_s),
                        fmt(a_pr_m, a_pr_s),
                        fmt(a_re_m, a_re_s),
                        fmtN(a_nt_m, a_nt_s),
                        fmtN(a_ni_m, a_ni_s),
                        fmtG(g_m, g_s),
                    ]
                )
                + "\n"
            )
        print(out_path)


if __name__ == "__main__":
    main()
