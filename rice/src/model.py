import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=400):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class HazardTransformer(nn.Module):
    def __init__(self, d_in, d_model=64, nhead=4, num_layers=3, dropout=0.2, max_len=400):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, X):
        h = self.in_proj(X)
        h = self.pos(h)
        h = self.encoder(h)
        logits = self.head(h).squeeze(-1)
        return torch.sigmoid(logits).clamp(1e-6, 1-1e-6)


def _gaussian_pmf_from_mu(mu, T: int, sigma: float):
    """
    Build a discrete Gaussian PMF over t = 1..T for each row.

    mu:    (N,) predicted center in DOY units
    sigma: scalar std-dev (days)
    returns pmf: (N, T) with sum_t pmf == 1 (softmax-normalized).
    """
    device = mu.device
    dtype = mu.dtype
    t = torch.arange(1, int(T) + 1, device=device, dtype=dtype)
    delta = (t.view(1, -1) - mu.view(-1, 1)) / float(sigma)
    log_pdf = -0.5 * delta * delta
    return torch.softmax(log_pdf, dim=1)


def _pmf_to_hazard(pmf, eps: float = 1e-6):
    """
    Convert per-row PMF to discrete hazards: h_t = p_t / S_{t-1}.
    pmf: (N, T) summing to ~1.
    """
    cum = torch.cumsum(pmf, dim=1)
    S_prev = 1.0 - torch.cat([torch.zeros_like(cum[:, :1]), cum[:, :-1]], dim=1)
    S_prev = S_prev.clamp(min=eps)
    return (pmf / S_prev).clamp(eps, 1.0 - eps)


class HierarchicalCausalHazardTransformer(nn.Module):
    """
    Hierarchical Stage-2 model:
      1) Encode each t* sample along time axis T
      2) Encode ordered t* representations with causal self-attention
      3) Fuse t* context back to per-time hidden states, then predict hazard

    PMF modes (set via .pmf_mode attribute, default "hazard"):
      - "hazard":   per-time sigmoid hazard head (current behavior)
      - "gaussian": per-t* mu head; PMF = N(mu, sigma) on integer DOY support;
                    hazard derived from PMF for downstream compatibility.
                    The raw mu (B,K) is stashed as ._last_mu_BK for use by
                    a parametric loss in train_eval.
    """

    def __init__(
        self,
        d_in,
        d_model=64,
        nhead=4,
        num_layers=3,
        num_tstar_layers=1,
        dropout=0.2,
        max_len=400,
        max_tstar_len=512,
        use_tstar_scalar_pos=False,
        phenology_bias_head: bool = False,
        phenology_dim: int = 4,
        phenology_hidden: int = 8,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos = PositionalEncoding(d_model, max_len=max_len)
        time_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.time_encoder = nn.TransformerEncoder(time_enc_layer, num_layers=num_layers)

        self.tstar_pos = PositionalEncoding(d_model, max_len=max_tstar_len)
        tstar_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.tstar_encoder = nn.TransformerEncoder(tstar_enc_layer, num_layers=num_tstar_layers)
        self.num_tstar_layers = int(num_tstar_layers)
        self.use_tstar_scalar_pos = bool(use_tstar_scalar_pos)
        if self.use_tstar_scalar_pos:
            self.tstar_scalar_proj = nn.Linear(1, d_model)
        else:
            self.tstar_scalar_proj = None
        self.time_chunk_size = 64

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.head_mu = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.pmf_mode = "hazard"
        self.gaussian_sigma = 5.0
        self.gaussian_mu_max = 0.0  # 0 → use Tend at forward time
        self._last_mu_BK = None

        # Phenology bias head (Architecture A): mu = mu_temporal + phen_bias
        # phen_bias = Linear(4 → hidden) + ReLU + Linear(hidden → 1)
        # Last layer initialized to ~0 so phen_bias ≈ 0 at training start.
        self.phenology_bias_head = bool(phenology_bias_head)
        self.phenology_dim = int(phenology_dim)
        self.phenology_hidden = int(phenology_hidden)
        if self.phenology_bias_head:
            if self.phenology_hidden > 0:
                self.phen_head = nn.Sequential(
                    nn.Linear(self.phenology_dim, self.phenology_hidden),
                    nn.ReLU(),
                    nn.Linear(self.phenology_hidden, 1),
                )
                # zero-init the final Linear so phen_bias starts at 0
                with torch.no_grad():
                    self.phen_head[-1].weight.zero_()
                    self.phen_head[-1].bias.zero_()
            else:
                self.phen_head = nn.Linear(self.phenology_dim, 1)
                with torch.no_grad():
                    self.phen_head.weight.zero_()
                    self.phen_head.bias.zero_()
        else:
            self.phen_head = None
        self._last_phen_bias = None
        self._last_mu_temporal = None

    def _causal_tstar_mask(self, K: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)

    def _encode_time_chunk(self, x):
        h = self.in_proj(x)
        h = self.pos(h)
        return self.time_encoder(h)

    def forward(self, X, tstar, valid_mask, pheno=None):
        """
        X:          (B,K,T,D)
        tstar:      (B,K) in 1..T
        valid_mask: (B,K) bool
        pheno:      (B, phenology_dim) optional, used iff phenology_bias_head=True
                    and pmf_mode='gaussian'. Adds learnable scalar bias to mu.
        returns:
          hazard:   (B,K,T)
        """
        B, K, T, _ = X.shape
        BK = B * K

        x_flat = X.reshape(BK, T, -1)
        chunk_size = int(getattr(self, "time_chunk_size", 32) or BK)
        if chunk_size <= 0:
            chunk_size = BK
        h_chunks = []
        for start in range(0, BK, chunk_size):
            x_chunk = x_flat[start:start + chunk_size]
            if self.training and torch.is_grad_enabled():
                h_chunk = checkpoint(self._encode_time_chunk, x_chunk, use_reentrant=False)
            else:
                h_chunk = self._encode_time_chunk(x_chunk)
            h_chunks.append(h_chunk)
        h = torch.cat(h_chunks, dim=0)  # (BK,T,d_model)

        t_idx = torch.clamp(tstar, 1, T).reshape(BK) - 1
        t_idx = t_idx.view(-1, 1, 1).expand(-1, 1, h.size(-1))
        z = h.gather(1, t_idx).squeeze(1).reshape(B, K, -1)  # (B,K,d_model)

        z = self.tstar_pos(z)
        if self.tstar_scalar_proj is not None:
            t_rel = torch.clamp(tstar.float() / float(max(T, 1)), 0.0, 1.0).unsqueeze(-1)
            z = z + self.tstar_scalar_proj(t_rel)
        causal_mask = self._causal_tstar_mask(K, X.device)
        if self.num_tstar_layers > 0:
            z = self.tstar_encoder(z, mask=causal_mask, src_key_padding_mask=~valid_mask)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = z * valid_mask.unsqueeze(-1).to(z.dtype)
        # Phase B diagnostic: stash post-encoder, pre-head representation for
        # linear-probe analysis. Detached so it does not affect gradients.
        self._last_z_BKD = z.detach()

        mode = str(getattr(self, "pmf_mode", "hazard"))
        if mode == "gaussian":
            mu_logit = self.head_mu(z).squeeze(-1)  # (B, K)
            mu_max_attr = float(getattr(self, "gaussian_mu_max", 0.0) or 0.0)
            mu_max = mu_max_attr if mu_max_attr > 0 else float(T)

            mu_mode = str(getattr(self, "mu_mode", "absolute"))
            if mu_mode == "lead_from_alert":
                # Bounded-sigmoid lead head reusing self.head_mu(z) as raw_lead.
                # mu_DOY = alert_rel + lead, where alert_rel is the absolute-DOY
                # alert_tstar (carried in X via the dispatch feature channel) shifted
                # to the same 1-based season-index coords as L. Window aggregation
                # is MAX over the time axis. X is (B,K,T,D); the T axis is the
                # nowcast window (per causal-tstar group K). With causal fill,
                # pre-alert rows hold 0 at the alert_tstar channel and post-alert
                # rows hold the true absolute DOY, so max yields the true alert
                # when the window contains at least one post-alert row.
                lead_min = float(getattr(self, "lead_min", 7.0))
                lead_max = float(getattr(self, "lead_max", 75.0))
                doy_start = float(getattr(self, "doy_start", 1.0))
                alert_idx = int(getattr(self, "alert_tstar_feat_idx", -1))
                if alert_idx < 0 or alert_idx >= X.shape[-1]:
                    raise ValueError(
                        f"mu_mode='lead_from_alert' requires alert_tstar_feat_idx "
                        f"in [0, X.shape[-1]={X.shape[-1]}); got {alert_idx}"
                    )
                # X: (B, K, T, D); reduce along the window (T) axis.
                alert_doy_abs = X[..., alert_idx].amax(dim=2)        # (B, K)

                # Pre-alert cells (alert_abs <= 0) stay in the attention/context
                # path (valid_mask unchanged) but must NOT participate in the
                # lead-target loss/eval. Build lead_loss_mask = valid & alerted.
                # Loss callers (train_eval.asymmetric_mu_loss) and eval callers
                # (phase_r) read self._last_lead_loss_mask to skip pre-alert
                # cells. mu is still computed for all cells using a SAFE alert
                # value so the forward graph and downstream PMF code never see
                # negative alert_rel; those mu values are dummies and excluded
                # by lead_loss_mask.
                vm = (valid_mask.to(torch.bool)
                      if valid_mask is not None
                      else torch.ones_like(alert_doy_abs, dtype=torch.bool))
                alerted = alert_doy_abs > 0.0                        # (B, K)
                lead_loss_mask = vm & alerted                        # (B, K)
                alert_safe = torch.where(
                    alerted, alert_doy_abs,
                    alert_doy_abs.new_full(alert_doy_abs.shape, float(doy_start)),
                )
                alert_rel = alert_safe - doy_start + 1.0             # 1-based season idx
                raw_lead = mu_logit                                  # rename for clarity
                lead = lead_min + (lead_max - lead_min) * torch.sigmoid(raw_lead)
                mu_pre = alert_rel + lead
                mu_temporal = mu_pre.clamp(0.0, float(T) - 1.0)

                # ---- Safety: only check cells that will actually contribute to
                # loss. Pre-alert cells with alert_abs<=0 are legitimate context;
                # they are excluded by lead_loss_mask.
                #
                # A batch with zero alerted cells is NOT fatal. During row_map
                # construction in phase_r_oracle_iou, group batches can contain
                # only no-alert / pre-alert sy; the forward should still produce
                # valid (dummy) mu so downstream code can skip those cells via
                # lead_loss_mask. We only print a warning. Strict RuntimeError
                # is reserved for the impossible-by-construction invariant
                # (lead_loss_mask True but alert_abs <= 0).
                strict = bool(getattr(self, "lead_strict_alert_check", True))
                if strict and lead_loss_mask.any():
                    loss_cells = alert_doy_abs[lead_loss_mask]
                    if (loss_cells <= 0).any():
                        raise RuntimeError(
                            "[lead_from_alert] internal invariant violated: "
                            "lead_loss_mask True but alert_abs<=0."
                        )
                if not lead_loss_mask.any():
                    if bool(getattr(self, "_warned_empty_lead_batch", False)) is False:
                        n_valid = int(vm.long().sum().item())
                        print(f"[lead_from_alert] WARNING: batch has 0 alerted "
                              f"(B,K) cells out of {n_valid} valid; loss/eval "
                              f"will skip this batch's lead targets. (suppressing "
                              f"further per-batch warnings)")
                        self._warned_empty_lead_batch = True

                # ---- One-shot debug print (set self.lead_debug_once_pending=True)
                if bool(getattr(self, "lead_debug_once_pending", False)):
                    with torch.no_grad():
                        n_valid = int(vm.long().sum().item())
                        n_loss = int(lead_loss_mask.long().sum().item())
                        n_drop = n_valid - n_loss
                        frac_drop = (n_drop / max(n_valid, 1))
                        def _stat(t, m=None):
                            tt = t[m] if (m is not None and t.shape == m.shape) else t
                            return (float(tt.min()), float(tt.mean()), float(tt.max())) \
                                if tt.numel() > 0 else (float("nan"),)*3
                        print(f"[lead_debug] X.shape={tuple(X.shape)}  "
                              f"z.shape={tuple(z.shape)}  "
                              f"alert_idx={alert_idx}  "
                              f"alert_chan.shape={tuple(X[..., alert_idx].shape)}")
                        print(f"[lead_from_alert] valid cells {n_valid} -> {n_loss} "
                              f"after lead_loss_mask; dropped_pre_alert={n_drop}  "
                              f"frac_dropped={frac_drop:.4f}")
                        s = _stat(alert_doy_abs, vm)
                        print(f"[lead_debug] alert_abs_raw [valid]      min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(alert_doy_abs, lead_loss_mask)
                        print(f"[lead_debug] alert_abs_raw [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(alert_rel, lead_loss_mask)
                        print(f"[lead_debug] alert_rel     [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(raw_lead, lead_loss_mask)
                        print(f"[lead_debug] raw_lead      [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(lead, lead_loss_mask)
                        print(f"[lead_debug] lead          [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(mu_pre, lead_loss_mask)
                        print(f"[lead_debug] mu_pre_clamp  [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        s = _stat(mu_temporal, lead_loss_mask)
                        print(f"[lead_debug] mu_post_clamp [loss_only]  min={s[0]:.3f} mean={s[1]:.3f} max={s[2]:.3f}")
                        if lead_loss_mask.any():
                            mp = mu_pre[lead_loss_mask]; mc = mu_temporal[lead_loss_mask]
                            f0 = float((mc == 0).float().mean())
                            print(f"[lead_debug] frac(mu_pre<0) [loss]={float((mp<0).float().mean()):.4f}  "
                                  f"frac(mu_post==0) [loss]={f0:.4f}  T={int(T)}")
                            if f0 > 0.05:
                                print(f"[lead_debug] WARNING: {f0:.1%} of loss-target cells "
                                      f"collapsed to mu=0 (clamp boundary). Check lead bounds.")
                    self.lead_debug_once_pending = False

                self._last_lead_BK = lead.detach()
                self._last_alert_rel_BK = alert_rel.detach()
                self._last_lead_loss_mask = lead_loss_mask.detach()
            elif mu_mode == "residual_clim":
                # mu_DOY = clim_mid + delta_max * tanh(raw_delta)
                # clim_mid_rel is in 1-based season-index coords (matching L/R).
                # When the head outputs raw_delta=0 (e.g. right after
                # --stage2_reset_head_mu), delta=0 and mu = clim_mid exactly,
                # giving the climatology-baseline starting point. tanh bounds
                # the per-sample shift to [-delta_max, +delta_max].
                clim_mid_rel = float(getattr(self, "clim_mid_rel", float(T) / 2.0))
                delta_max = float(getattr(self, "delta_max", 60.0))
                raw_delta = mu_logit
                delta = delta_max * torch.tanh(raw_delta)
                mu_pre = mu_logit.new_full(mu_logit.shape, clim_mid_rel) + delta
                mu_temporal = mu_pre.clamp(0.0, float(T) - 1.0)
                # residual_clim has no pre-alert masking: clim_mid is defined
                # for every (sample, group) cell, so all valid cells participate.
                self._last_lead_BK = delta.detach()       # repurposed for logging
                self._last_alert_rel_BK = None
                self._last_lead_loss_mask = None
                if bool(getattr(self, "lead_debug_once_pending", False)):
                    with torch.no_grad():
                        vm_bool = (valid_mask.to(torch.bool)
                                   if valid_mask is not None
                                   else torch.ones_like(delta, dtype=torch.bool))
                        d_v = delta[vm_bool] if vm_bool.any() else delta.flatten()
                        mu_v = mu_temporal[vm_bool] if vm_bool.any() else mu_temporal.flatten()
                        print(f"[residual_clim] clim_mid_rel={clim_mid_rel:.2f}  "
                              f"delta_max={delta_max:.1f}  "
                              f"delta[min/mean/max]="
                              f"{float(d_v.min()):.2f}/{float(d_v.mean()):.2f}/{float(d_v.max()):.2f}  "
                              f"mu_temporal[min/mean/max]="
                              f"{float(mu_v.min()):.2f}/{float(mu_v.mean()):.2f}/{float(mu_v.max()):.2f}")
                    self.lead_debug_once_pending = False
            else:
                mu_temporal = torch.sigmoid(mu_logit) * mu_max  # (B, K)
                self._last_lead_BK = None
                self._last_alert_rel_BK = None
                self._last_lead_loss_mask = None
            self._last_mu_temporal = mu_temporal.detach()

            if (self.phenology_bias_head and self.phen_head is not None
                    and pheno is not None and pheno.numel() > 0
                    and pheno.shape[-1] == self.phenology_dim):
                pheno_f = pheno.to(mu_temporal.dtype)               # (B, phenology_dim)
                phen_bias = self.phen_head(pheno_f).squeeze(-1)     # (B,)
                self._last_phen_bias = phen_bias.detach()
                mu = (mu_temporal + phen_bias.unsqueeze(-1)).clamp(0.0, mu_max)  # (B, K)
            else:
                self._last_phen_bias = None
                mu = mu_temporal
            self._last_mu_BK = mu

            sigma = float(getattr(self, "gaussian_sigma", 5.0))
            mu_flat = mu.reshape(BK).to(torch.float32)
            pmf_flat = _gaussian_pmf_from_mu(mu_flat, T=T, sigma=sigma)
            hazard_flat = _pmf_to_hazard(pmf_flat).to(h.dtype)
            return hazard_flat.reshape(B, K, T)

        self._last_mu_BK = None
        z_flat = z.reshape(BK, 1, -1)
        h_fused = h + z_flat

        logits = self.head(h_fused).squeeze(-1)
        hazard = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6).reshape(B, K, T)
        return hazard
