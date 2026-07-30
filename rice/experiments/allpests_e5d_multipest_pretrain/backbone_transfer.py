#!/usr/bin/env python
"""Split an E5d checkpoint into backbone / head / dead groups and export a backbone-only bundle.

WHY THIS EXISTS -- and why it needs no change to vendor/run_train.py.

The vendored trainer already has a warm-start primitive: `--stage2_warm_start_ckpt` loads a
bundle and calls `load_state_dict(..., strict=False)`
(vendor/src/vendor/run_train.py:1668-1743). It transfers whatever keys the bundle happens to
contain. So "transfer only the backbone" does not require a new loading path -- it requires
writing a bundle that contains only the backbone keys. The filtering happens at WRITE time.

That keeps the fine-tuned model architecturally IDENTICAL to the scratch baseline: same class,
same flags, same state_dict layout. The only difference between baseline A and experiment B is
which tensors were non-random at step 0, which is precisely the variable under test.

GROUPS (verified against vendor/src/vendor/model.py; params for d_in=45, d_model=48, 12 offsets)

  backbone   in_proj.*        model.py:139  Linear(d_in, d_model)   2,208   the ONLY d_in-coupled
             time_encoder.*   model.py:149  3x TransformerEncoder  84,816   the real shared trunk
  head       offset_mu_heads.*  model.py:230  ModuleList of 12      28,812   pest-specific lead calib
  dead       head.*             model.py:169  per-time hazard head   2,401   unreachable in E5d
             head_mu.*          model.py:205  template mu head       2,401   bypassed
             tstar_encoder.*    model.py:160  per-tstar re-encode   28,272   bypassed

`in_proj` is included in the backbone ONLY because contract_check.py proves the participating
pests share one ordered 45-feature contract. If that ever stops holding, in_proj column i stops
meaning the same variable across pests and it must drop to the head group -- so the export
re-checks the contract hash and refuses rather than trusting the caller.

Buffers (pos.pe, candidate_offsets_buf) are deliberately NOT exported: pos.pe is sized from T
and candidate_offsets_buf is the offset-order guard (model.py:264-271). Both are rebuilt
correctly by the target model's own constructor, and shipping them would turn a season-length
difference into a silent shape clash.

  # inspect any checkpoint
  .venv/bin/python .../backbone_transfer.py inspect --ckpt <path>
  # write a backbone-only warm-start bundle
  .venv/bin/python .../backbone_transfer.py export --ckpt <pretrain.pt> --out <backbone.pt>
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import torch

BACKBONE_PREFIXES = ("in_proj.", "time_encoder.")
HEAD_PREFIXES = ("offset_mu_heads.",)
DEAD_PREFIXES = ("head.", "head_mu.", "tstar_encoder.", "tstar_pos.")
BUFFER_KEYS = ("pos.pe", "candidate_offsets_buf")


def group_of(key: str) -> str:
    if key in BUFFER_KEYS:
        return "buffer"
    # head_mu. must be tested before head. -- "head." is a prefix of "head_mu." only in the
    # other direction, but offset_mu_heads. also contains "head", so match on the full prefix.
    for g, pref in (("backbone", BACKBONE_PREFIXES), ("head", HEAD_PREFIXES),
                    ("dead", DEAD_PREFIXES)):
        if any(key.startswith(p) for p in pref):
            return g
    return "unknown"


def feature_hash(names) -> str | None:
    if not names:
        return None
    return hashlib.sha256("|".join(list(names)).encode()).hexdigest()[:16]


def select_state(bundle: dict, seed: int | None) -> tuple[dict, int]:
    """Mirror vendor/run_train.py:_select_stage_state (model.py-adjacent, run_train.py:184).

    A bundle holds one entry per training seed; the warm-start path picks by seed rather than
    position, so we do the same instead of assuming index 0.
    """
    states = bundle.get("trained_states") or []
    if not states:
        raise SystemExit("[transfer] bundle has no trained_states")
    if seed is None:
        st = states[0]
        return st["state_dict"], int(st.get("seed", 0))
    for st in states:
        if int(st.get("seed", -1)) == int(seed):
            return st["state_dict"], int(seed)
    have = [int(s.get("seed", -1)) for s in states]
    raise SystemExit(f"[transfer] no trained state for seed={seed}; bundle has {have}")


def summarize(sd: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for k, v in sd.items():
        g = group_of(k)
        e = out.setdefault(g, dict(keys=0, params=0, names=[]))
        e["keys"] += 1
        e["params"] += int(v.numel()) if hasattr(v, "numel") else 0
        e["names"].append(k)
    return out


def cmd_inspect(a) -> int:
    b = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd, seed = select_state(b, a.seed)
    print(f"=== {a.ckpt} ===")
    print(f"  pest={b.get('pest')}  seed={seed}  d_in={b.get('d_in')}  d_model={b.get('d_model')}  "
          f"doy={b.get('doy_start')}-{b.get('doy_end')}  T={b.get('T')}")
    print(f"  feature_hash={feature_hash(b.get('feature_names'))}  "
          f"n_feature_names={len(b.get('feature_names') or [])}")
    print(f"  seeds in bundle={[int(s.get('seed',-1)) for s in b.get('trained_states',[])]}")
    tot = 0
    for g in ("backbone", "head", "dead", "buffer", "unknown"):
        e = summarize(sd).get(g)
        if not e:
            continue
        tot += e["params"]
        print(f"\n  [{g}] {e['keys']} keys, {e['params']:,} params")
        for n in e["names"][:6]:
            print(f"      {n}  {tuple(sd[n].shape)}")
        if e["keys"] > 6:
            print(f"      ... +{e['keys']-6} more")
    print(f"\n  total in state_dict: {tot:,} params")
    if "unknown" in summarize(sd):
        print("  [!] 'unknown' keys mean model.py changed -- re-derive the prefix groups "
              "before trusting a transfer")
        return 1
    return 0


def cmd_export(a) -> int:
    b = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd, seed = select_state(b, a.seed)
    groups = summarize(sd)
    if "unknown" in groups:
        print(f"[transfer] ABORT: unclassified keys {groups['unknown']['names'][:5]}")
        return 1

    keep_groups = ["backbone"] + (["head"] if a.include_heads else [])
    keep = {k: v for k, v in sd.items() if group_of(k) in keep_groups}
    if not keep:
        print("[transfer] ABORT: nothing selected")
        return 1

    # Refuse to ship in_proj across a changed feature contract: column i would stop meaning the
    # same variable. --expect-feature-hash is how the caller asserts the contract it verified.
    fh = feature_hash(b.get("feature_names"))
    if a.expect_feature_hash and fh != a.expect_feature_hash:
        print(f"[transfer] ABORT: feature_hash {fh} != expected {a.expect_feature_hash}; "
              f"in_proj is not transferable across a different feature contract")
        return 1

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Same bundle shape the trainer expects, so this file can be handed straight to
    # --stage2_warm_start_ckpt with no new loading code.
    bundle = {
        "trained_states": [{"seed": seed, "state_dict": keep}],
        "d_in": b.get("d_in"), "d_model": b.get("d_model"),
        "n_head": b.get("n_head"), "n_layers": b.get("n_layers"),
        "doy_start": b.get("doy_start"), "doy_end": b.get("doy_end"), "T": b.get("T"),
        "feature_names": b.get("feature_names"), "feature_cols": b.get("feature_cols"),
        "norm_mean": b.get("norm_mean"), "norm_std": b.get("norm_std"),
        "multipest_pretrain": {
            "source_ckpt": str(a.ckpt), "exported_groups": keep_groups,
            "feature_hash": fh, "seed": seed,
            "source_pests": b.get("multipest_pests"),
            "source_train_years": b.get("multipest_train_years"),
            "excluded_years": b.get("multipest_excluded_years"),
        },
    }
    torch.save(bundle, out)
    n = sum(int(v.numel()) for v in keep.values())
    print(f"[transfer] wrote {out}")
    print(f"  groups={keep_groups}  keys={len(keep)}  params={n:,}  seed={seed}  feature_hash={fh}")
    for g in keep_groups:
        e = groups[g]
        print(f"    {g}: {e['keys']} keys / {e['params']:,} params")
    dropped = [g for g in ("head", "dead") if g in groups and g not in keep_groups]
    print(f"  dropped (left at target's random init): {dropped}")
    return 0


def verify_load(model, backbone_ckpt: str | Path, seed: int | None = None) -> dict:
    """Load a backbone bundle into `model` and return a report. Used by smoke_test.py.

    Raises on a shape mismatch inside the transferred set -- that is never benign here, because
    every participating pest is supposed to share one contract. Missing keys ARE expected: they
    are the head/dead groups the export deliberately dropped.
    """
    b = torch.load(backbone_ckpt, map_location="cpu", weights_only=False)
    sd, used_seed = select_state(b, seed)
    tgt = model.state_dict()

    mism = {k: (tuple(sd[k].shape), tuple(tgt[k].shape))
            for k in sd if k in tgt and tuple(sd[k].shape) != tuple(tgt[k].shape)}
    if mism:
        raise SystemExit(f"[transfer] shape mismatch in transferred tensors: {mism}")

    res = model.load_state_dict(sd, strict=False)
    loaded = sorted(set(sd) & set(tgt))
    report = dict(
        backbone_ckpt=str(backbone_ckpt), seed=used_seed,
        loaded=loaded, n_loaded=len(loaded),
        loaded_params=int(sum(tgt[k].numel() for k in loaded)),
        missing=list(res.missing_keys), unexpected=list(res.unexpected_keys),
        missing_by_group={g: [k for k in res.missing_keys if group_of(k) == g]
                          for g in ("backbone", "head", "dead", "buffer", "unknown")},
    )
    # A missing BACKBONE key means the transfer silently did less than intended.
    bad = report["missing_by_group"]["backbone"]
    if bad:
        raise SystemExit(f"[transfer] backbone keys absent from bundle: {bad}")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect"); i.add_argument("--ckpt", required=True)
    i.add_argument("--seed", type=int, default=None); i.set_defaults(fn=cmd_inspect)
    e = sub.add_parser("export"); e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True); e.add_argument("--seed", type=int, default=None)
    e.add_argument("--include-heads", action="store_true",
                   help="also ship offset_mu_heads.* (ablation; default is backbone only)")
    e.add_argument("--expect-feature-hash", default=None)
    e.set_defaults(fn=cmd_export)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
