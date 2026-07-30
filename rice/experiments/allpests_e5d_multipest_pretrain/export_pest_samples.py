#!/usr/bin/env python
"""Export ONE pest's pretraining shard for ONE (phase, eval_year), plus its leakage manifest.

ONE PROCESS = ONE PEST. This is not a style choice. `resolve_pest()` copies every UPPERCASE name
out of rice/pests/<pest>/config.py into the `rice.configs.config` module namespace
(rice/configs/config.py:42-50) and re-imports that pest's feature module
(rice/src/pest_resolver.py:35-46). Those globals -- PATH_OBS, DOY_START, THRESHOLD, SEEDS -- are
process singletons, so a second pest in the same process overwrites the first. Run this script
once per pest; pretrain.py then mixes the shards without ever touching a pest config.

THE YEAR CONTRACT, and why it is the same one the baseline uses.

The scratch baseline trains cell (pest, y) with --val_year y-1 --test_year_min y --test_year_max y
(allpests_e5d/common.sh:80-82). `split_by_year` (rice/src/dataset.py:209-225) then gives
train = {year < val_year} = {year <= y-2}, val = {y-1}, test = {y}. So this shard takes exactly
train, i.e. year <= y-2, and NOTHING else.

Applied uniformly across pests that yields the property the experiment needs: the backbone for
eval year y never sees ANY pest's year y (test) or year y-1 (selection). That matters more than
it looks, because the daily weather cache is shared across pests by design
(rice/configs/config.py:48-50) -- the X features of a given station-year are the SAME NUMBERS for
every pest. Only the labels differ. So admitting pest A's y-1 rows would put pest B's val_ckpt
inputs in front of the trunk even though pest B was never named. The uniform <= y-2 gate is what
closes that.

Normalization is computed here, from this shard's train rows only, exactly as the trainer does
(run_train.py:1374 -> rice/src/dataset.py:643-692), and stored with the shard. It stays per-pest:
see the rationale in multipest_sampler.PestShard.

  .venv/bin/python .../export_pest_samples.py --pest WBPH --phase dev --eval-year 2024 \
      --out <OUT_ROOT>/shards/dev/test2024/WBPH [--max-groups 64]
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

AP = Path(__file__).resolve().parent
sys.path.insert(0, str(AP))
import e5d_common as EC                                          # noqa: E402


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", required=True)
    ap.add_argument("--phase", choices=["dev", "clean"], required=True)
    ap.add_argument("--eval-year", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-groups", type=int, default=0, help="0 = all; >0 truncates (smoke)")
    ap.add_argument("--assign", default=None,
                    help="clean phase: split_assignment.json (only its val_ckpt ids may be "
                         "used for selection; train is unaffected)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    log = EC.RunLog(out / "export_log.jsonl")
    y = int(a.eval_year)
    val_year, train_max = y - 1, y - 2

    log("start", pest=a.pest, phase=a.phase, eval_year=y, val_year=val_year,
        train_years_max=train_max, out=str(out), gpu=EC.gpu_state())

    # --- required inputs, checked before any work ------------------------------------------
    prod = EC.prod_ckpt(a.pest, y)
    disp = EC.dispatch_csv(a.pest, y)
    need = {"production_ckpt": prod, "dispatch_csv": disp}
    missing = {k: str(v) for k, v in need.items() if v is None or not Path(v).is_file()}
    if missing:
        log("BLOCKED", reason="required Stage-2 inputs absent", missing=missing)
        log.close(); return 2

    ck = torch.load(prod, map_location="cpu", weights_only=False)
    feat_names = list(ck["feature_names"])
    fh = EC.feature_hash(feat_names)
    T = int(ck["T"])
    if fh != EC.feature_hash(list(torch.load(EC.prod_ckpt("WBPH", y), map_location="cpu",
                                             weights_only=False)["feature_names"])):
        log("BLOCKED", reason="feature contract differs from the reference pest; this pest "
                              "cannot share the backbone", pest=a.pest, feature_hash=fh)
        log.close(); return 2

    # --- build this pest's samples through the UNTOUCHED existing pipeline -------------------
    # reconstruct_samples is the same entry point the E5d grid/eval steps use, so the tensors
    # here are the ones the scratch baseline would have trained on -- not a reimplementation.
    try:
        from rice.configs import config as C
        from rice.src.pest_resolver import resolve_pest
        from rice.scripts.eval_s2n_direct_compare import reconstruct_samples
        _, gfc = resolve_pest(a.pest)
        C.DOY_START, C.DOY_END = int(ck["doy_start"]), int(ck["doy_end"])
        if ck.get("d_model"):
            C.D_MODEL = int(ck["d_model"])

        # reconstruct_samples reads the dispatch table from ckpt["stage2_dispatch_feature_csv"]
        # (rice/scripts/eval_s2n_direct_compare.py:64). That field records the path as it was on
        # the machine that trained the cell -- the older rice/outputs_stage2_<batch>/ layout --
        # while this host holds the same artifact under rice/outputs/stage2/<batch>/. Re-point it
        # at the file already located and sha256'd above.
        #
        # The BASENAME must match exactly. It encodes which Stage-1 gate this cell fired on
        # (one of gate_{A_baseline,D_history,dispatch_group_tau}_R088_*), chosen per (pest, year).
        # Silently accepting a different gate would change the model's inputs, so a mismatch is a
        # hard stop rather than a relocation.
        want = Path(str(ck.get("stage2_dispatch_feature_csv") or "")).name
        if want and want != Path(disp).name:
            log("BLOCKED", reason="dispatch CSV on disk is a different Stage-1 gate than the "
                                  "checkpoint was trained with",
                ckpt_expects=want, found=Path(disp).name)
            log.close(); return 2
        if str(ck.get("stage2_dispatch_feature_csv")) != str(disp):
            log("dispatch_repointed", ckpt_recorded=str(ck.get("stage2_dispatch_feature_csv")),
                resolved=str(disp), basename_match=True)
            ck["stage2_dispatch_feature_csv"] = str(disp)

        samples, _ = reconstruct_samples(ck, int(ck["run"]), gfc)
    except FileNotFoundError as e:
        log("BLOCKED", reason="input data file absent", error=str(e),
            hint="LONG_by_pest observation CSV (rice/configs/base.py:LONG_BY_PEST_DIR) and/or "
                 "the daily union table (RICE_DAILY_CSV) must be present")
        log.close(); return 2
    if not samples:
        log("BLOCKED", reason="reconstruct_samples returned 0 samples")
        log.close(); return 2

    # --- the year gate: train = year <= y-2, nothing else -----------------------------------
    all_years = sorted({int(s["year"]) for s in samples})
    train = [s for s in samples if int(s["year"]) <= train_max]
    if not train:
        log("BLOCKED", reason="no train rows at or below y-2", years_seen=all_years)
        log.close(); return 2
    kept_years = sorted({int(s["year"]) for s in train})
    excluded = [yy for yy in all_years if yy > train_max]
    if max(kept_years) > train_max:
        log("FAIL", reason="year gate leaked", kept_years=kept_years); log.close(); return 1

    # --- t* expansion, via the TRAINER'S OWN function with the cell's own recorded params ----
    # reconstruct_samples returns BASE site-year samples (X is (T,D), no t*). The trainer expands
    # them at run_train.py:1167-1175 with build_stage2_nowcast_samples; calling the same function
    # with the parameters stored in this cell's checkpoint keeps the rows identical to what the
    # scratch baseline trained on, instead of reimplementing the windowing.
    from rice.src.dataset import build_stage2_nowcast_samples, _mask_to_recent_window
    window = int(ck.get("stage2_nowcast_window", 28))
    expanded = build_stage2_nowcast_samples(
        train,
        window=window,
        stride=int(ck.get("stage2_nowcast_stride", 1)),
        tstar_start=ck.get("stage2_nowcast_tstar_start"),
        only_pre_event=bool(int(ck.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ck.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ck.get("stage2_nowcast_require_tstar_before_L", 1))),
    )
    if not expanded:
        log("BLOCKED", reason="nowcast expansion produced 0 rows", n_base=len(train))
        log.close(); return 2

    # --- group by (pest, site, year); pest is in the key so cross-pest station-years cannot
    # --- be merged by anything downstream (rice/src/dataset.py:740-757 groups on site+year only)
    groups: dict[str, list[dict]] = {}
    for s in expanded:
        groups.setdefault(f"{a.pest}|{s['site_id']}|{int(s['year'])}", []).append(s)
    gids = sorted(groups)
    n_groups_total = len(gids)
    if a.max_groups:
        gids = gids[: a.max_groups]

    # Store the BASE X once per group and the t* list beside it, exactly as the nowcast record
    # does (rice/src/dataset.py:616-624 keeps base X and masks on the fly in __getitem__). The
    # (K,T,D) view is rebuilt at load time with the same _mask_to_recent_window the trainer uses,
    # so the shard stays ~K times smaller and cannot drift from the trainer's masking.
    rows = []
    for gid in gids:
        g = sorted(groups[gid], key=lambda r: int(r["tstar"]))
        rows.append(dict(
            site_id=g[0]["site_id"], year=int(g[0]["year"]), pest=a.pest, group_id=gid,
            X_base=np.asarray(g[0]["X"], dtype=np.float32),                  # (T,D)
            window=window,
            tstar=np.array([int(r["tstar"]) for r in g], dtype=np.int64),
            valid_mask=np.ones(len(g), dtype=bool),
            L=np.array([float(r["L"]) for r in g], dtype=np.float32),
            R=np.array([float(r["R"]) for r in g], dtype=np.float32),
            censor=np.array([0 if r["censor_type"] == "interval" else 1 for r in g],
                            dtype=np.int64),
        ))

    # --- normalizer from THIS shard's masked train rows only ---------------------------------
    # Masked, not raw: the trainer computes stats after expansion (run_train.py:1374), so the
    # statistic is over windowed rows. Matching that keeps the input space identical.
    flat = np.concatenate([
        _mask_to_recent_window(r["X_base"], tstar=int(t), window=int(r["window"]))
        for r in rows for t in r["tstar"]], axis=0)
    mean = flat.mean(0)
    std = flat.std(0)
    std = np.where(std < 1e-6, 1.0, std)
    miss = np.arange(1, mean.shape[0], 2)          # odd indices are miss indicators: keep 0/1
    mean[miss], std[miss] = 0.0, 1.0
    n_disp = 15                                    # DISPATCH_TOTAL_CHANNELS; raw, not scaled
    mean[-n_disp:], std[-n_disp:] = 0.0, 1.0

    torch.save(rows, out / "rows.pt")
    d_in = int(rows[0]["X_base"].shape[-1])
    meta = dict(
        pest=a.pest, phase=a.phase, eval_year=y, d_in=d_in, T=T,
        doy_start=int(ck["doy_start"]), doy_end=int(ck["doy_end"]),
        feature_hash=fh, feature_names=feat_names,
        norm_mean=mean.tolist(), norm_std=std.tolist(),
        n_groups=len(rows), n_groups_available=n_groups_total,
        n_rows=int(sum(len(r["tstar"]) for r in rows)),
        train_years=kept_years, excluded_years=excluded,
        n_sites=len({r["site_id"] for r in rows}),
        source_files={"production_ckpt": {"path": str(prod), "sha256": sha256(prod)},
                      "dispatch_csv": {"path": str(disp), "sha256": sha256(Path(disp))}},
    )
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # --- leakage manifest: the assertions, evaluated, not asserted in prose -----------------
    manifest = dict(
        pest=a.pest, phase=a.phase, eval_year=y,
        val_year_excluded=val_year, test_year_excluded=y,
        train_years=kept_years, excluded_years=excluded,
        n_groups=len(rows), n_rows=meta["n_rows"], n_sites=meta["n_sites"],
        checks={
            "max_train_year_le_eval_minus_2": int(max(kept_years) <= train_max),
            "val_year_absent_from_train": int(val_year not in kept_years),
            "test_year_absent_from_train": int(y not in kept_years),
            "group_ids_pest_prefixed": int(all(r["group_id"].startswith(a.pest + "|")
                                               for r in rows)),
            "feature_contract_matches_reference": 1,
        },
        source_files=meta["source_files"],
    )
    (out / "leak_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    bad = [k for k, v in manifest["checks"].items() if v != 1]
    log("exported", n_groups=len(rows), n_rows=meta["n_rows"], n_sites=meta["n_sites"],
        d_in=d_in, T=T, train_years=kept_years, excluded_years=excluded,
        feature_hash=fh, leak_checks_failed=bad)
    log("DONE", verdict="PASS" if not bad else "FAIL", out=str(out))
    log.close()
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
