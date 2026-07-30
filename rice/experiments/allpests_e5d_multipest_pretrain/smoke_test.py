#!/usr/bin/env python
"""Smoke test for the multi-pest pretrain -> pest fine-tune transfer. Trains nothing real.

TWO MODES, and the difference matters when reading the result.

  --mode real       (default) Loads exported per-pest shards, mixes two pests in one batch, and
                    exercises the whole chain on real samples. REQUIRES export_pest_samples.py to
                    have run, which requires the LONG_by_pest observation CSVs. If the shards are
                    absent this mode REFUSES to fall back -- it prints the missing inputs and
                    exits non-zero, because a green smoke test that silently skipped the data is
                    worse than a red one.

  --mode structural Exercises only what is data-independent: model construction at the REAL
                    contract (T and d_in read from the production checkpoints), a mixed-pest
                    batch of synthetic tensors, forward/backward/step on the GPU, bundle save,
                    backbone export, load into a fresh pest model, key accounting, and one
                    fine-tune step. It verifies the TRANSFER MECHANICS. It does NOT verify that
                    real pest data mixes correctly, and it says so in its own output.

Neither mode writes anywhere except --out (default under the new multipest tree). The scratch
tree rice/outputs_allpests_e5d/ is checked for mtime changes and never opened for writing.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import torch

AP = Path(__file__).resolve().parent
sys.path.insert(0, str(AP))
import e5d_common as EC                                          # noqa: E402
from backbone_transfer import group_of, verify_load               # noqa: E402
from multipest_sampler import MixedPestDataset, PestShard, collate_mixed, make_sampler  # noqa: E402

SMOKE_PESTS = ["WBPH", "brown_spot"]     # both in the eligible 7; smallest contract risk


def scratch_fingerprint() -> dict:
    """Cheap proof we did not touch baseline A. Compared before and after."""
    root = EC.SCRATCH_TREE
    if not root.exists():
        return dict(exists=False)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(str(p.stat().st_mtime_ns).encode())
        h.update(str(p.stat().st_size).encode())
    return dict(exists=True, n_files=len(files), digest=h.hexdigest()[:16])


def read_contract() -> dict:
    """T / d_in / feature list from a REAL production checkpoint -- not invented numbers."""
    p = EC.prod_ckpt("WBPH", 2024)
    if not p.is_file():
        raise SystemExit(f"[smoke] no production ckpt to read the contract from: {p}")
    c = torch.load(p, map_location="cpu", weights_only=False)
    return dict(d_in=len(c["norm_mean"]), T=int(c["T"]),
                doy_start=int(c["doy_start"]), doy_end=int(c["doy_end"]),
                feature_hash=EC.feature_hash(c["feature_names"]),
                alert_idx=int(c.get("stage2_pmf_alert_tstar_feat_idx", -1)))


def gaussian_surrogate_loss(hazard, mu, L, R, valid_mask):
    """A stand-in objective for the mechanics test ONLY.

    The production objective lives in vendor/src/vendor/run_train.py and is NOT reimplemented
    here -- a second copy of the loss would be a silent fork of the thing under comparison. The
    real pretraining leg calls the vendored trainer's own loss. All this needs to do is produce a
    finite scalar that touches hazard and mu so backward exercises the whole graph.
    """
    mid = 0.5 * (L + R)
    m = valid_mask.float()
    mu_term = (((mu - mid) ** 2) * m).sum() / m.sum().clamp_min(1.0)
    haz_term = hazard.clamp(1e-6, 1 - 1e-6).log().mean().abs()
    return mu_term / 1e4 + haz_term


def build_batch_real(shard_dirs: list[Path], batch: int, log):
    shards = [PestShard(d) for d in shard_dirs]
    ds = MixedPestDataset(shards)
    sampler, probs = make_sampler(ds, "balanced", seed=0)
    log("data.real", pests=ds.pests, counts=ds.counts(), sampling="balanced",
        sampling_probs=probs, d_in=ds.d_in, T=ds.T, n_groups=len(ds))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, sampler=sampler,
                                     collate_fn=collate_mixed, num_workers=0)
    bt = next(iter(dl))
    # Log the REALIZED composition, not the sampling weights. "both pests are in one batch" is a
    # property of the drawn batch; the weights only make it likely.
    comp: dict[str, int] = {}
    for i in bt["pest_idx"].tolist():
        comp[ds.pests[i]] = comp.get(ds.pests[i], 0) + 1
    log("batch.composition", pests_in_batch=sorted(comp), counts_in_batch=comp,
        n_pests_in_batch=len(comp), mixed=len(comp) > 1,
        group_ids=bt["group_id"], X_shape=tuple(bt["X"].shape),
        pest_prefixed=all("|" in g and g.split("|")[0] in ds.pests for g in bt["group_id"]))
    if len(comp) < 2:
        log("FAIL", reason="batch did not mix pests", counts_in_batch=comp)
        raise SystemExit(1)
    return bt, ds.d_in, ds.T, ds.pests


def build_batch_structural(contract: dict, batch: int, log):
    """Synthetic tensors at the REAL shapes, tagged with two pest ids so the mixing path,
    the collate and the pest-aware group id are all exercised."""
    K, T, D = len(EC.OFFSETS), contract["T"], contract["d_in"]
    g = torch.Generator().manual_seed(0)
    items = []
    for b in range(batch):
        pi = b % len(SMOKE_PESTS)
        X = torch.randn(K, T, D, generator=g) * 0.5
        X[:, :, EC.ALERT_TSTAR_FEAT_IDX] = float(contract["doy_start"] + 100)  # alert_tstar: DOY
        items.append(dict(X=X, tstar=torch.full((K,), T // 2, dtype=torch.long),
                          valid_mask=torch.ones(K, dtype=torch.bool),
                          L=torch.full((K,), float(T // 2 - 10)),
                          R=torch.full((K,), float(T // 2 + 10)),
                          censor=torch.zeros(K, dtype=torch.long),
                          pest_idx=pi, group_id=f"{SMOKE_PESTS[pi]}|site{b}|2020"))
    bt = collate_mixed(items)
    log("data.structural", pests=SMOKE_PESTS, n_items=batch, d_in=D, T=T, K=K,
        pest_idx=bt["pest_idx"].tolist(), group_ids=bt["group_id"][:4],
        note="SYNTHETIC tensors at the real contract; does NOT validate real data mixing")
    return bt, D, T, SMOKE_PESTS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["real", "structural"], default="real")
    ap.add_argument("--shards", nargs="*", default=None, help="per-pest shard dirs (real mode)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--finetune-pest", default="WBPH")
    a = ap.parse_args()

    out = Path(a.out or EC.OUT_ROOT / "smoke")
    out.mkdir(parents=True, exist_ok=True)
    log = EC.RunLog(out / "smoke_log.jsonl")

    before = scratch_fingerprint()
    log("start", mode=a.mode, out=str(out), gpu=EC.gpu_state(),
        scratch_tree=str(EC.SCRATCH_TREE), scratch_before=before)

    contract = read_contract()
    log("contract", **contract, pretrain_pests=EC.PRETRAIN_PESTS,
        bph_excluded=EC.BPH_EXCLUDED_REASON)

    # ---- 1-2. data ------------------------------------------------------------------------
    if a.mode == "real":
        dirs = [Path(s) for s in (a.shards or [])]
        missing = [str(d) for d in dirs if not (d / "rows.pt").is_file()]
        if not dirs or missing:
            log("BLOCKED", reason="exported pest shards absent",
                missing=missing or ["--shards not given"],
                required_input=str(EC.CS / "rice/configs/base.py:LONG_BY_PEST_DIR"),
                next_step="run export_pest_samples.py per pest (needs the LONG_by_pest CSVs)")
            print("\n[smoke] REFUSING to run: real mode needs exported shards. "
                  "Use --mode structural to test transfer mechanics only.")
            log.close()
            return 2
        batch, d_in, T, pests = build_batch_real(dirs, a.batch, log)
    else:
        batch, d_in, T, pests = build_batch_structural(contract, a.batch, log)

    if d_in != contract["d_in"] or T != contract["T"]:
        log("FAIL", reason="batch contract != production contract",
            batch=dict(d_in=d_in, T=T), production=contract)
        log.close()
        return 1

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 3-5. shared backbone forward / loss / backward / step ----------------------------
    pre = EC.build_e5d_model(d_in=d_in, T=T, alert_idx=EC.ALERT_TSTAR_FEAT_IDX).to(dev)
    pr = EC.param_report(pre)
    log("pretrain.model", device=str(dev), d_in=d_in, T=T,
        backbone_params=pr.get("backbone"), head_params=pr.get("head"),
        dead_params=pr.get("dead"), trainable=pr["trainable"])

    opt = torch.optim.AdamW(pre.parameters(), lr=1e-4, weight_decay=1e-4)
    X = batch["X"].to(dev); ts = batch["tstar"].to(dev); vm = batch["valid_mask"].to(dev)
    L = batch["L"].to(dev); R = batch["R"].to(dev)
    pre.train()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True)
        hazard = pre(X, ts, vm, None)
        mu = getattr(pre, "_last_mu_BK", None)
        if mu is None:
            log("FAIL", reason="_last_mu_BK is None -- gaussian mu path did not run")
            log.close(); return 1
        loss = gaussian_surrogate_loss(hazard, mu, L, R, vm)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(pre.parameters(), 1.0)
        opt.step()
        finite_g = all(torch.isfinite(p.grad).all().item()
                       for p in pre.parameters() if p.grad is not None)
        log("pretrain.step", step=step, loss=float(loss),
            hazard_shape=tuple(hazard.shape), mu_shape=tuple(mu.shape),
            loss_finite=bool(torch.isfinite(loss)), grad_finite=finite_g,
            grad_norm=float(gnorm), gpu=EC.gpu_state())
        if not (torch.isfinite(loss) and finite_g):
            log("FAIL", reason="non-finite loss or gradient", step=step)
            log.close(); return 1

    # ---- 6. save the pretrain bundle in the trainer's own bundle format --------------------
    pt = out / "pretrain_ckpt.pt"
    torch.save({
        "trained_states": [{"seed": 0, "state_dict": {k: v.cpu()
                                                      for k, v in pre.state_dict().items()}}],
        "d_in": d_in, "d_model": EC.D_MODEL, "n_head": EC.N_HEAD, "n_layers": EC.N_LAYERS,
        "doy_start": contract["doy_start"], "doy_end": contract["doy_end"], "T": T,
        "feature_names": None, "feature_cols": None,
        "multipest_pests": pests, "multipest_train_years": "<=y-2",
        "multipest_excluded_years": ["y-1 (val)", "y (test)"],
        "smoke": True, "mode": a.mode,
    }, pt)
    log("pretrain.saved", path=str(pt), bytes=pt.stat().st_size)

    # ---- 7. export backbone only ----------------------------------------------------------
    bb = out / "backbone_only.pt"
    src = torch.load(pt, map_location="cpu", weights_only=False)
    sd = src["trained_states"][0]["state_dict"]
    keep = {k: v for k, v in sd.items() if group_of(k) == "backbone"}
    torch.save({"trained_states": [{"seed": 0, "state_dict": keep}],
                "d_in": d_in, "d_model": EC.D_MODEL, "T": T,
                "multipest_pretrain": {"exported_groups": ["backbone"], "source": str(pt)}}, bb)
    log("backbone.exported", path=str(bb), n_keys=len(keep),
        params=int(sum(v.numel() for v in keep.values())),
        groups_dropped=sorted({group_of(k) for k in sd if group_of(k) != "backbone"}))

    # ---- 8-9. load into a fresh pest-specific model, account for every key ------------------
    ft = EC.build_e5d_model(d_in=d_in, T=T, alert_idx=EC.ALERT_TSTAR_FEAT_IDX).to(dev)
    rep = verify_load(ft, bb, seed=0)
    log("finetune.backbone_loaded", pest=a.finetune_pest, n_loaded=rep["n_loaded"],
        loaded_params=rep["loaded_params"],
        n_missing=len(rep["missing"]), n_unexpected=len(rep["unexpected"]),
        missing_by_group={g: len(v) for g, v in rep["missing_by_group"].items() if v},
        missing_sample=rep["missing"][:4], unexpected=rep["unexpected"])
    if rep["unexpected"]:
        log("FAIL", reason="unexpected keys in backbone bundle", keys=rep["unexpected"])
        log.close(); return 1
    # the transferred tensors must actually equal the pretrained ones
    same = all(torch.allclose(ft.state_dict()[k].cpu(), keep[k]) for k in keep)
    log("finetune.transfer_verified", tensors_identical=same, n_checked=len(keep))
    if not same:
        log("FAIL", reason="transferred tensors differ from the exported backbone")
        log.close(); return 1

    # ---- 10. one pest-specific fine-tune step ---------------------------------------------
    fopt = torch.optim.AdamW(ft.parameters(), lr=1e-4, weight_decay=1e-4)
    ft.train()
    fopt.zero_grad(set_to_none=True)
    hz = ft(X, ts, vm, None)
    mu = getattr(ft, "_last_mu_BK", None)
    floss = gaussian_surrogate_loss(hz, mu, L, R, vm)
    floss.backward()
    fg = all(torch.isfinite(p.grad).all().item() for p in ft.parameters() if p.grad is not None)
    fopt.step()
    log("finetune.step", step=0, loss=float(floss), loss_finite=bool(torch.isfinite(floss)),
        grad_finite=fg, hazard_shape=tuple(hz.shape), gpu=EC.gpu_state())
    if not (torch.isfinite(floss) and fg):
        log("FAIL", reason="fine-tune loss/grad not finite"); log.close(); return 1

    # ---- 11-12. save and re-load forward -------------------------------------------------
    fp = out / f"finetune_{a.finetune_pest}_ckpt.pt"
    torch.save({"trained_states": [{"seed": 0, "state_dict":
                                    {k: v.cpu() for k, v in ft.state_dict().items()}}],
                "pest": a.finetune_pest, "d_in": d_in, "T": T, "smoke": True}, fp)
    chk = EC.build_e5d_model(d_in=d_in, T=T, alert_idx=EC.ALERT_TSTAR_FEAT_IDX).to(dev)
    chk.load_state_dict(torch.load(fp, map_location="cpu",
                                   weights_only=False)["trained_states"][0]["state_dict"])
    chk.eval()
    with torch.no_grad():
        h2 = chk(X, ts, vm, None)
    log("finetune.reload_forward", path=str(fp), hazard_shape=tuple(h2.shape),
        finite=bool(torch.isfinite(h2).all()))

    after = scratch_fingerprint()
    untouched = before == after
    log("scratch_tree_check", before=before, after=after, untouched=untouched)

    log("DONE", mode=a.mode, verdict="PASS" if untouched else "FAIL",
        caveat=None if a.mode == "real" else
        "structural mode: transfer mechanics verified on synthetic tensors; real multi-pest "
        "data mixing NOT verified -- use --mode real with exported shards for that")
    log.close()
    print(f"\n[smoke] {'PASS' if untouched else 'FAIL'} (mode={a.mode})  log={out/'smoke_log.jsonl'}")
    return 0 if untouched else 1


if __name__ == "__main__":
    sys.exit(main())
