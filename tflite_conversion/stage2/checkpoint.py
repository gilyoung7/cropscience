"""Load + validate a Stage-2 lead_v3 checkpoint for any of the 8 pests.

Loading is delegated to the production loader
`api_handoff_transformer/infer/ckpt.py::load_stage2_model`, which is imported
UNMODIFIED. It already raises on missing/unexpected state_dict keys, so weight
mapping errors cannot pass silently.

On top of that, `load_pest()` cross-checks the checkpoint's self-reported config
against `pest_configs.EXPECTED` and against internal consistency (feature_names
length, alert channel name, T == doy_end-doy_start+1, norm stat dims, and the
actual state_dict tensor shapes). Any disagreement raises `ConfigMismatch`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "api_handoff_transformer"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from infer.ckpt import LoadedModel, load_stage2_model  # noqa: E402

from pest_configs import EXPECTED, PESTS, SELECTED_OFFSET, ExpectedConfig  # noqa: E402

STAGE2_DIR = PKG_ROOT / "assets" / "stage2"
CKPT_NAME = "lead_v3_final_checkpoint_run4.pt"


class ConfigMismatch(RuntimeError):
    """A checkpoint disagrees with the expected contract."""


def ckpt_path(pest: str) -> Path:
    return STAGE2_DIR / pest / CKPT_NAME


def _eq(errs: list[str], name: str, got, want) -> None:
    if got != want:
        errs.append(f"{name}: ckpt={got!r} expected={want!r}")


def validate(loaded: LoadedModel, pest: str, cfg: ExpectedConfig) -> None:
    """Raise ConfigMismatch listing every discrepancy (not just the first)."""
    e: list[str] = []
    m = loaded.model

    # --- declared config vs expected -----------------------------------
    _eq(e, "d_in", loaded.d_in, cfg.d_in)
    _eq(e, "T", loaded.T, cfg.T)
    _eq(e, "doy_start", loaded.doy_start, cfg.doy_start)
    _eq(e, "doy_end", loaded.doy_end, cfg.doy_end)
    _eq(e, "alert_tstar_feat_idx", loaded.alert_tstar_feat_idx, cfg.alert_idx)
    _eq(e, "lead_min", loaded.lead_min, cfg.lead_min)
    _eq(e, "lead_max", loaded.lead_max, cfg.lead_max)
    _eq(e, "sigma", loaded.sigma, cfg.sigma)
    _eq(e, "pmf_mode", str(m.pmf_mode), cfg.pmf_mode)
    _eq(e, "mu_mode", str(getattr(m, "mu_mode", "")), cfg.mu_mode)
    _eq(e, "tstar_layers", int(m.num_tstar_layers), cfg.tstar_layers)

    # --- internal consistency ------------------------------------------
    if loaded.T != loaded.doy_end - loaded.doy_start + 1:
        e.append(
            f"T={loaded.T} != doy_end-doy_start+1="
            f"{loaded.doy_end - loaded.doy_start + 1}"
        )
    if len(loaded.feature_names) != loaded.d_in:
        e.append(f"len(feature_names)={len(loaded.feature_names)} != d_in={loaded.d_in}")
    if not (0 <= loaded.alert_tstar_feat_idx < loaded.d_in):
        e.append(f"alert_idx={loaded.alert_tstar_feat_idx} out of range for d_in={loaded.d_in}")
    else:
        nm = loaded.feature_names[loaded.alert_tstar_feat_idx]
        if nm != "alert_tstar":
            e.append(f"feature_names[{loaded.alert_tstar_feat_idx}]={nm!r} != 'alert_tstar'")
    for nm_, t in (("norm_mean", loaded.norm_mean), ("norm_std", loaded.norm_std)):
        if t.numel() != loaded.d_in:
            e.append(f"{nm_} dim {t.numel()} != d_in={loaded.d_in}")

    # --- weights actually match the declared shape ----------------------
    _eq(e, "in_proj.in_features", m.in_proj.in_features, cfg.d_in)
    _eq(e, "in_proj.out_features", m.in_proj.out_features, cfg.d_model)
    _eq(e, "head_mu[-1].out_features", m.head_mu[-1].out_features, 1)
    n_time_layers = len(m.time_encoder.layers)
    _eq(e, "time_encoder.num_layers", n_time_layers, cfg.n_layers)
    _eq(e, "tstar_encoder.num_layers", len(m.tstar_encoder.layers), cfg.tstar_layers)
    # nhead is not stored on the layer; infer it from in_proj_weight packing.
    attn = m.time_encoder.layers[0].self_attn
    _eq(e, "self_attn.num_heads", attn.num_heads, cfg.n_head)
    _eq(e, "self_attn.embed_dim", attn.embed_dim, cfg.d_model)

    # --- unsupported-by-wrapper features --------------------------------
    if m.phenology_bias_head or m.phen_head is not None:
        e.append("phenology_bias_head enabled — mu path would need phen_head bias")
    if m.tstar_scalar_proj is not None:
        e.append("use_tstar_scalar_pos enabled — wrapper omits tstar_scalar_proj")

    if e:
        raise ConfigMismatch(
            f"[{pest}] checkpoint disagrees with expected contract:\n  - "
            + "\n  - ".join(e)
        )


def load_pest(pest: str, device: str = "cpu") -> tuple[LoadedModel, ExpectedConfig]:
    if pest not in PESTS:
        raise KeyError(f"unknown pest {pest!r}; known: {list(PESTS)}")
    p = ckpt_path(pest)
    if not p.is_file():
        raise FileNotFoundError(f"[{pest}] checkpoint not found: {p}")
    loaded = load_stage2_model(p, device=device)
    cfg = EXPECTED[pest]
    validate(loaded, pest, cfg)
    return loaded, cfg


def selected_offset(pest: str) -> int:
    """Read selected_fixed_offset from the shipped policy (not from our copy)."""
    import yaml

    pol = yaml.safe_load((PKG_ROOT / "configs" / "fallback_policy.yaml").read_text())
    v = (pol.get("per_pest", {}).get(pest) or {}).get("selected_fixed_offset")
    if v is None:
        v = SELECTED_OFFSET[pest]
    return int(v)


def make_synthetic_input(
    loaded: LoadedModel,
    seed: int = 0,
    B: int = 1,
    K: int = 1,
    alert_doy: float | None = None,
    tstar_idx: int | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic synthetic input at the checkpoint's real shape/dtype.

    Base channels get seeded noise; the alert_tstar channel is filled causally
    (0 before t*, `alert_doy` from t* on), matching how the real pipeline builds
    it, so `X[..., alert_idx].amax(dim=2)` recovers `alert_doy`. Values are not
    physically meaningful — this exercises the graph, not the science.

    `alert_doy` defaults to the middle of the pest's own DOY range so it is
    always a valid post-doy_start alert.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    T, D = loaded.T, loaded.d_in
    X = torch.randn(B, K, T, D, generator=g, dtype=torch.float32)
    if tstar_idx is None:
        tstar_idx = T // 2
    if alert_doy is None:
        alert_doy = float(loaded.doy_start + T // 3)
    ai = loaded.alert_tstar_feat_idx
    X[:, :, :, ai] = 0.0
    X[:, :, tstar_idx - 1 :, ai] = float(alert_doy)
    tstar = torch.full((B, K), int(tstar_idx), dtype=torch.int64)
    valid_mask = torch.ones((B, K), dtype=torch.bool)
    return X.to(device), tstar.to(device), valid_mask.to(device)


def original_mu(loaded: LoadedModel, X, tstar, valid_mask) -> torch.Tensor:
    """Run the untouched production model, reading mu the way run_predict.py does
    (run_predict.py:468-471 discards forward()'s return and reads _last_mu_BK)."""
    with torch.no_grad():
        _ = loaded.model(X, tstar=tstar, valid_mask=valid_mask, pheno=None)
    mu = getattr(loaded.model, "_last_mu_BK", None)
    if mu is None:
        raise RuntimeError("_last_mu_BK was not set by the original model")
    return mu.detach()
