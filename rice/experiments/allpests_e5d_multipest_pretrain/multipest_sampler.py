#!/usr/bin/env python
"""Mixed-pest batching for E5d pretraining, over PRE-EXPORTED per-pest shards.

WHY SHARDS AND NOT resolve_pest() IN-PROCESS.

`resolve_pest()` mutates module-level config globals -- rice/configs/config.py:42-50 copies every
UPPERCASE name out of rice/pests/<pest>/config.py into the `config` module namespace, and
rice/src/pest_resolver.py:35-46 then re-imports that pest's feature module. So `C.PATH_OBS`,
`C.DOY_START`, `C.THRESHOLD`, `C.SEEDS` are process-global singletons and a second
resolve_pest() call silently overwrites the first. Building 7 pests' samples in one process is
therefore not a matter of care -- it is structurally impossible without refactoring the config
layer, which would put the scratch baseline's reproducibility at risk.

The way out is to keep "one process = one pest" for sample CONSTRUCTION (export_pest_samples.py,
which reuses the untouched existing pipeline) and to make the mixing stage read plain tensors.
The pretraining process then needs no pest config at all, so nothing can be clobbered.

WHY THE PEST KEY IS PART OF THE GROUP ID.

`group_stage2_samples_by_site_year` (rice/src/dataset.py:740-757) groups on (site_id, year). The
same weather station-year exists in EVERY pest's corpus -- the daily cache is shared across pests
by design (rice/configs/config.py:48-50). Concatenating two pests without a pest key would make
that function merge two pests' rows into ONE causal group: different labels, same group. Every
id here is therefore (pest, site_id, year).
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PestShard:
    """One pest's exported samples for one (phase, eval_year), plus its own normalizer.

    Normalization stays PER PEST on purpose. The scratch baseline computes norm stats from that
    pest's own train split (run_train.py:1374 -> dataset.py:643-692), and fine-tuning must see the
    same input space it saw during pretraining. Using a pooled normalizer would instead change
    every fine-tuned model's input contract relative to the baseline, adding a second difference
    on top of the initialization -- and initialization is the only variable under test.
    """

    def __init__(self, shard_dir: str | Path):
        d = Path(shard_dir)
        self.dir = d
        self.meta = json.loads((d / "meta.json").read_text())
        self.pest: str = self.meta["pest"]
        self.mean = np.asarray(self.meta["norm_mean"], dtype=np.float32)
        self.std = np.asarray(self.meta["norm_std"], dtype=np.float32)
        self.rows: list[dict] = torch.load(d / "rows.pt", map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def d_in(self) -> int:
        return int(self.meta["d_in"])

    @property
    def T(self) -> int:
        return int(self.meta["T"])


class MixedPestDataset(Dataset):
    """Flat index over several PestShards. Returns the pest id alongside the tensors.

    __getitem__ yields (X, tstar, valid_mask, L, R, censor, pest_idx, group_id) where X is
    (K,T,D) for one site-year group -- the same shape the grouped collate produces per item
    (rice/scripts/common.py:46-76) -- so a batch is (B,K,T,D) exactly as model.forward expects
    (model.py:482-491).
    """

    def __init__(self, shards: list[PestShard]):
        if not shards:
            raise SystemExit("[sampler] no shards")
        d_in = {s.d_in for s in shards}
        T = {s.T for s in shards}
        if len(d_in) != 1 or len(T) != 1:
            raise SystemExit(f"[sampler] shards disagree on contract: d_in={d_in} T={T}. "
                             f"Mixing them would feed the shared in_proj different variables "
                             f"per pest; run contract_check.py first.")
        self.shards = shards
        self.d_in, self.T = d_in.pop(), T.pop()
        self.pests = [s.pest for s in shards]
        self.index: list[tuple[int, int]] = [(si, ri) for si, s in enumerate(shards)
                                             for ri in range(len(s))]

    def __len__(self) -> int:
        return len(self.index)

    def counts(self) -> dict[str, int]:
        return {s.pest: len(s) for s in self.shards}

    def __getitem__(self, i: int):
        si, ri = self.index[i]
        sh = self.shards[si]
        r = sh.rows[ri]
        # Rebuild the (K,T,D) view from the stored base season, masking per t* with the trainer's
        # own helper (rice/src/dataset.py:529) so the windowing cannot drift from the baseline's.
        from rice.src.dataset import _mask_to_recent_window
        base = np.asarray(r["X_base"], dtype=np.float32)
        w = int(r["window"])
        X = np.stack([_mask_to_recent_window(base, tstar=int(t), window=w)
                      for t in np.asarray(r["tstar"])])       # (K,T,D)
        X = (X - sh.mean) / sh.std                            # per-pest normalizer
        return dict(
            X=torch.from_numpy(X),
            tstar=torch.as_tensor(r["tstar"], dtype=torch.long),
            valid_mask=torch.as_tensor(r["valid_mask"], dtype=torch.bool),
            L=torch.as_tensor(r["L"], dtype=torch.float32),
            R=torch.as_tensor(r["R"], dtype=torch.float32),
            censor=torch.as_tensor(r["censor"], dtype=torch.long),
            pest_idx=si,
            # pest is part of the id so two pests' identical station-years stay distinct
            group_id=f"{sh.pest}|{r['site_id']}|{r['year']}",
        )


def collate_mixed(items: list[dict]) -> dict:
    """Pad the K axis across a batch that may mix pests. Mirrors collate_grouped_stage2."""
    Kmax = max(int(it["X"].shape[0]) for it in items)
    B = len(items)
    T, D = int(items[0]["X"].shape[1]), int(items[0]["X"].shape[2])
    X = torch.zeros(B, Kmax, T, D)
    tstar = torch.ones(B, Kmax, dtype=torch.long)
    vm = torch.zeros(B, Kmax, dtype=torch.bool)
    L = torch.zeros(B, Kmax); R = torch.zeros(B, Kmax)
    cen = torch.zeros(B, Kmax, dtype=torch.long)
    for b, it in enumerate(items):
        k = int(it["X"].shape[0])
        X[b, :k] = it["X"]; tstar[b, :k] = it["tstar"]; vm[b, :k] = it["valid_mask"]
        L[b, :k] = it["L"]; R[b, :k] = it["R"]; cen[b, :k] = it["censor"]
    return dict(X=X, tstar=tstar, valid_mask=vm, L=L, R=R, censor=cen,
                pest_idx=torch.tensor([it["pest_idx"] for it in items], dtype=torch.long),
                group_id=[it["group_id"] for it in items])


def make_sampler(ds: MixedPestDataset, policy: str, seed: int = 0):
    """Per-sample weights for a WeightedRandomSampler, so ONE batch mixes pests.

    Alternating pest-homogeneous batches was rejected: with BatchNorm-free LayerNorm the trunk
    would still see each pest in isolation per step, so the gradient direction at every step is
    single-pest and the trunk can oscillate between pests instead of finding shared structure.
    Mixing within the batch averages the pests' gradients at every step.

      natural  no reweighting; the largest pest dominates in proportion to its size
      balanced each pest contributes equal expected mass regardless of size
      sqrt     weight ~ 1/sqrt(n_p); between the two, keeps some of the size signal
    """
    n = ds.counts()
    if policy == "natural":
        w_pest = {p: 1.0 for p in n}
    elif policy == "balanced":
        w_pest = {p: 1.0 / max(1, n[p]) for p in n}
    elif policy == "sqrt":
        w_pest = {p: 1.0 / max(1.0, float(n[p]) ** 0.5) for p in n}
    else:
        raise SystemExit(f"[sampler] unknown policy {policy!r}")
    weights = torch.tensor([w_pest[ds.shards[si].pest] for si, _ in ds.index],
                           dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    probs = {p: 0.0 for p in n}
    tot = float(weights.sum())
    for (si, _), w in zip(ds.index, weights.tolist()):
        probs[ds.shards[si].pest] += w / tot
    return (torch.utils.data.WeightedRandomSampler(weights, num_samples=len(ds),
                                                   replacement=True, generator=g),
            {p: round(v, 5) for p, v in probs.items()})
