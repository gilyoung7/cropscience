"""Export-safe, inference-only wrapper around any Stage-2 lead_v3 checkpoint.

Generalizes the BPH-only wrapper in tflite_conversion/bph_stage2/ to all 8
pests. Every pest-specific value (d_in, T, doy_start, alert_idx, lead bounds)
is read from the checkpoint via checkpoint.py — nothing is hardcoded per pest.
All 8 share d_model=48, n_head=4, n_layers=3, tstar_layers=1,
pmf_mode=gaussian, mu_mode=lead_from_alert, and an identical state_dict key set,
so one wrapper covers them all.

Why this exists — what blocks torch.export in the original
----------------------------------------------------------
`api_handoff_transformer/infer/model.py::HierarchicalCausalHazardTransformer`
cannot be exported as-is. Two independent blockers:

1. Training/debug branches in the model:
     * `model.py:300`  `if not lead_loss_mask.any():` — bool() on a tensor forces
       .item() -> unbacked symint -> GuardOnDataDependentSymNode: Eq(u0, 1)
     * `model.py:293-299` — `strict and lead_loss_mask.any()` plus the
       boolean-mask index `alert_doy_abs[lead_loss_mask]` (data-dependent shape)
     * `model.py:310-349` — debug block built out of float(...)/.item()
     * `model.py:405-409` — Gaussian PMF -> hazard tail. The API discards the
       returned hazard (`run_predict.py:468-471` reads `_last_mu_BK`), so this is
       dead compute at inference.

2. Inside PyTorch itself: passing `mask=` to `nn.TransformerEncoder` routes
   through `_detect_is_causal_mask` (torch/nn/modules/transformer.py:535), which
   evaluates `bool((mask == causal_comparison).all())` — another .item(), and the
   same guard failure. Unreachable by editing our model.

   Resolved by dropping the masks at K=1, where both are provably no-ops:
   the causal mask is `triu(ones(1,1), diagonal=1) == [[False]]`, and an
   all-masked padding mask yields NaN that `nan_to_num` + `z * valid_mask`
   collapse to the same zero the unmasked path gives. K comes from X.shape and
   is static at export, so the branch is not data-dependent. Production is
   always B=K=1 (infer/preprocess.py:631-633). validate_all.py re-proves this
   per pest for valid=True and valid=False.

Weight safety
-------------
The wrapper re-implements no parameter. It is built from an already-loaded
model and holds references to the *same submodule objects*, so weight identity
is structural. `build_wrapper` additionally asserts every wrapper parameter is
the same object (by id) as one of the source model's.

Nothing under api_handoff_transformer/ is modified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from checkpoint import LoadedModel, load_pest
from pest_configs import ExpectedConfig


class Stage2InferenceModel(nn.Module):
    """mu-only forward for pmf_mode='gaussian', mu_mode='lead_from_alert'.

    forward(X, tstar, valid_mask) -> mu
        X          (B, K, T, d_in)  float32   normalized input tensor
        tstar      (B, K)           int64     1-based season index of t*
        valid_mask (B, K)           bool
        mu         (B, K)           float32   predicted centre, 1-based season index

    DOY = mu + doy_start - 1.

    Mirrors HierarchicalCausalHazardTransformer.forward lines 194-279 and 402
    exactly, minus the branches listed in the module docstring.
    """

    def __init__(self, src: nn.Module, pest: str):
        super().__init__()
        if str(getattr(src, "pmf_mode", "")) != "gaussian":
            raise ValueError(f"[{pest}] expected pmf_mode='gaussian', got {src.pmf_mode!r}")
        if str(getattr(src, "mu_mode", "")) != "lead_from_alert":
            raise ValueError(f"[{pest}] expected mu_mode='lead_from_alert', got {src.mu_mode!r}")
        if src.phenology_bias_head or src.phen_head is not None:
            raise ValueError(f"[{pest}] phenology_bias_head enabled; wrapper omits phen_head")
        if src.tstar_scalar_proj is not None:
            raise ValueError(f"[{pest}] use_tstar_scalar_pos enabled; wrapper omits it")

        # Shared by reference — identical parameter objects, no copy, no remap.
        self.in_proj = src.in_proj
        self.pos = src.pos
        self.time_encoder = src.time_encoder
        self.tstar_pos = src.tstar_pos
        self.tstar_encoder = src.tstar_encoder
        self.head_mu = src.head_mu

        self.pest = pest
        self.num_tstar_layers = int(src.num_tstar_layers)
        self.alert_idx = int(src.alert_tstar_feat_idx)
        self.doy_start = float(src.doy_start)
        self.lead_min = float(src.lead_min)
        self.lead_max = float(src.lead_max)
        self.eval()

    def forward(
        self, X: torch.Tensor, tstar: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        B, K, T, _ = X.shape
        BK = B * K

        # --- time encoder -------------------------------------------------
        # The original chunks this over BK (model.py:202-208) only to cap
        # activation memory; the encoder is independent along the batch axis, so
        # one call is numerically identical. validate_all.py asserts it per pest.
        x_flat = X.reshape(BK, T, -1)
        h = self.time_encoder(self.pos(self.in_proj(x_flat)))  # (BK, T, d_model)

        # --- gather the t* step -------------------------------------------
        t_idx = torch.clamp(tstar, 1, T).reshape(BK) - 1
        t_idx = t_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        z = h.gather(1, t_idx).squeeze(1).reshape(B, K, -1)  # (B, K, d_model)

        # --- causal t* encoder ---------------------------------------------
        z = self.tstar_pos(z)
        if self.num_tstar_layers > 0:
            if K == 1:
                # Masks dropped — provably a no-op at K=1, and passing mask=
                # would hit torch's _detect_is_causal_mask .item() guard.
                # See module docstring.
                z = self.tstar_encoder(z)
            else:
                causal_mask = torch.triu(
                    torch.ones(K, K, dtype=torch.bool, device=X.device), diagonal=1
                )
                z = self.tstar_encoder(
                    z, mask=causal_mask, src_key_padding_mask=~valid_mask
                )
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = z * valid_mask.unsqueeze(-1).to(z.dtype)

        # --- bounded-sigmoid lead head --------------------------------------
        mu_logit = self.head_mu(z).squeeze(-1)  # (B, K)
        alert_doy_abs = X[..., self.alert_idx].amax(dim=2)  # (B, K)
        alerted = alert_doy_abs > 0.0
        alert_safe = torch.where(
            alerted,
            alert_doy_abs,
            alert_doy_abs.new_full(alert_doy_abs.shape, self.doy_start),
        )
        alert_rel = alert_safe - self.doy_start + 1.0
        lead = self.lead_min + (self.lead_max - self.lead_min) * torch.sigmoid(mu_logit)
        return (alert_rel + lead).clamp(0.0, float(T) - 1.0)


def build_wrapper(loaded: LoadedModel, pest: str) -> Stage2InferenceModel:
    model = Stage2InferenceModel(loaded.model, pest)
    src_ids = {id(p) for p in loaded.model.parameters()}
    for name, p in model.named_parameters():
        if id(p) not in src_ids:
            raise RuntimeError(f"[{pest}] wrapper parameter {name!r} is not shared with the ckpt")
    return model


def load_and_wrap(
    pest: str, device: str = "cpu"
) -> tuple[LoadedModel, Stage2InferenceModel, ExpectedConfig]:
    """Load a pest's checkpoint (with full config validation) and wrap it."""
    loaded, cfg = load_pest(pest, device=device)
    return loaded, build_wrapper(loaded, pest).to(device), cfg
