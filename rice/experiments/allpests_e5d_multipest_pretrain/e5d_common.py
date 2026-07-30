#!/usr/bin/env python
"""Shared setup for the multi-pest experiment: paths, the E5d model factory, and logging.

The model factory reproduces the E5d geometry the scratch baseline uses, so a pretrained
backbone and a fine-tuned pest model are the same architecture. The reference is
vendor/src/vendor/run_train.py:1576-1600 (constructor call) plus the three env vars read at
run_train.py:1526-1528 and exported by allpests_e5d/common.sh:export_e5d_env().

Nothing here writes into the scratch tree. OUT_ROOT is a separate directory by construction.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
from pathlib import Path

import torch

AP = Path(__file__).resolve().parent
E5D = AP.parent / "allpests_e5d"
sys.path.insert(0, str(E5D))
sys.path.insert(0, str(E5D / "vendor"))
from repo_paths import CS                                        # noqa: E402

# `rice.*` imports resolve against the repo root; the shells set PYTHONPATH but the scripts here
# are also run directly, so make the package importable either way.
if str(CS) not in sys.path:
    sys.path.insert(0, str(CS))

# --- paths: a NEW tree. The scratch tree rice/outputs_allpests_e5d/ is never written here. ----
OUT_ROOT = Path(os.environ.get("MULTIPEST_OUT_ROOT",
                               CS / "rice/outputs_allpests_e5d_multipest_pretrain"))
SCRATCH_TREE = CS / "rice/outputs_allpests_e5d"          # read-only reference (baseline A)

# --- the E5d contract, as verified by contract_check.py against the 24 production ckpts -----
OFFSETS = [3, 7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
YEARS = [2022, 2023, 2024]
# The 7 pests that share one ordered 45-feature contract at DOY 60-300 / T=241.
# BPH is excluded: T=131 and d_in=27 with only 15/45 feature names in common.
PRETRAIN_PESTS = ["WBPH", "brown_spot", "rice_stem_borer_1", "rice_stem_borer_2",
                  "sheath_blight", "blast", "bacterial_blight"]
BPH_EXCLUDED_REASON = ("DOY 140-270 (T=131) vs 60-300 (T=241), d_in=27 vs 45, and a different "
                       "feature generation (6 raw daily cols vs 15 engineered); only 15 of 45 "
                       "feature names overlap")
ALERT_TSTAR_FEAT_IDX = 30        # feature_names[30] == 'alert_tstar' in the shared contract
SHARED_BAND_WINDOW = 28
D_MODEL, N_HEAD, N_LAYERS, DROPOUT = 48, 4, 3, 0.2


def batch_dir(y: int) -> str:
    return "batch_2024_bestgate" if y == 2024 else f"batch_{y}_baseline"


def prod_ckpt(pest: str, year: int) -> Path:
    return (CS / f"rice/outputs/stage2/{batch_dir(year)}/{pest}"
            / "lead_v3_final/ckpt/checkpoint_run4.pt")


def dispatch_csv(pest: str, year: int) -> Path | None:
    d = CS / f"rice/outputs/stage2/{batch_dir(year)}/{pest}"
    hits = sorted(d.glob("gate_*_R088_features_per_sy.csv"))
    return hits[0] if len(hits) == 1 else None


def feature_hash(names) -> str:
    return hashlib.sha256("|".join(list(names)).encode()).hexdigest()[:16]


def build_e5d_model(d_in: int, T: int, *, d_model: int = D_MODEL, nhead: int = N_HEAD,
                    num_layers: int = N_LAYERS, dropout: float = DROPOUT,
                    offsets: list[int] | None = None,
                    alert_idx: int = ALERT_TSTAR_FEAT_IDX):
    """Construct the E5d model exactly as the trainer does for this experiment's config.

    Mirrors run_train.py:1576-1600 with the E5d env settings (shared multi-offset encoder +
    per-offset mu heads) and the post-construction attributes at run_train.py:1621-1623. Those
    two attributes are plain Python attrs, NOT in state_dict, so they must be re-set after every
    load -- forgetting alert_tstar_feat_idx makes the offset router silently read X[..., -1]
    (model.py:440-442 falls back to -1 with no validation).
    """
    from src.vendor.model import HierarchicalCausalHazardTransformer as M
    offsets = list(offsets or OFFSETS)
    m = M(d_in=d_in, d_model=d_model, nhead=nhead, num_layers=num_layers,
          num_tstar_layers=1, dropout=dropout,
          max_len=max(400, T + 8), max_tstar_len=512,
          use_shared_multi_offset=True, mu_head_mode="offset_specific",
          candidate_offsets=offsets, shared_band_window=SHARED_BAND_WINDOW,
          use_issue_doy_features=False)
    m.pmf_mode = "gaussian"
    m.gaussian_sigma = 5.0
    m.alert_tstar_feat_idx = int(alert_idx)
    m.doy_start = 60
    m.offset_cond_strict = True
    return m


def param_report(model) -> dict:
    from backbone_transfer import group_of
    out: dict[str, int] = {}
    for k, v in model.state_dict().items():
        out[group_of(k)] = out.get(group_of(k), 0) + int(v.numel())
    out["trainable"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return out


class RunLog:
    """Append-only JSONL + stdout. Every field requirement 9 asks for goes through here, so the
    log is machine-checkable rather than prose we have to trust."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = self.path.open("a", encoding="utf-8")

    def __call__(self, event: str, **kw):
        rec = dict(t=round(time.time(), 3), event=event, **kw)
        self.fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.fh.flush()
        detail = "  ".join(f"{k}={v}" for k, v in kw.items()
                           if not isinstance(v, (list, dict)) or len(str(v)) < 160)
        print(f"[{event}] {detail}", flush=True)

    def close(self):
        self.fh.close()


def gpu_state() -> dict:
    if not torch.cuda.is_available():
        return dict(cuda=False)
    i = torch.cuda.current_device()
    return dict(cuda=True, name=torch.cuda.get_device_name(i),
                torch=torch.__version__, cuda_version=torch.version.cuda,
                total_gb=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                allocated_mb=round(torch.cuda.memory_allocated(i) / 2**20, 1))
