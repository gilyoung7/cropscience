import copy
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
        # ---- Offset-aware conditioning (2026-07; OFF by default -> baseline-identical).
        # Explicitly feed each t* sample's offset (and optionally issue_doy sin/cos)
        # into the mu head. Both are DERIVED inside forward() from the already-passed
        # `tstar` (1-based season index) and the alert_tstar feature channel, so NO new
        # forward inputs and NO dataloader change are required. When all flags are
        # False, cond_dim==0 and head_mu keeps its baseline (d_model->d_model) first
        # layer, so parameter shapes / state_dict keys are byte-identical to baseline.
        use_offset_embedding: bool = False,
        offset_embedding_dim: int = 8,
        offset_max: int = 240,   # = doy_end-doy_start (max possible issue-alert gap in-season);
        offset_min: int = 0,     #   covers the full training offset range, eval offsets are 1..75
        use_issue_doy_features: bool = False,
        doy_period: float = 365.0,
        # ---- Shared-encoder + offset-specific-head family (D1/D2; 2026-07; OFF by
        # default -> A/B/C paths untouched). When use_shared_multi_offset=True the
        # site-year base sequence is encoded ONCE with a band-causal mask (width =
        # shared_band_window, matching the baseline 28-day recent-window receptive
        # field) and each candidate offset's issue hidden state is gathered from that
        # single H. mu_head_mode='offset_specific' routes each (sample,offset) slot to
        # its own small mu head; all heads start from one shared template (identical
        # init) and diverge during training. issue_doy sin/cos optionally concatenated
        # (D2). NOTHING about the gaussian PMF / mu scaling / lead_from_alert / loss is
        # changed -- mu_logit produced here feeds the SAME downstream block.
        use_shared_multi_offset: bool = False,
        mu_head_mode: str = "shared",              # 'shared' | 'offset_specific'
        candidate_offsets=None,                    # list[int]; required for offset_specific/residual
        shared_band_window: int = 28,
        # ---- D4: shared base mu head + small per-offset residual head (OFF by default).
        # mu_logit = shared head_mu(z) + residual_scale * offset_residual_head[k](z).
        # The residual's final layer is zero-initialized so at init D4 == D3 exactly.
        # (No neural score head exists in gaussian mode -> residual corrects mu only;
        # the "score" is the downstream LightGBM selector, unchanged.)
        use_offset_residual: bool = False,
        residual_hidden_dim: int = 16,
        residual_scale: float = 1.0,
        zero_init_residual: bool = True,
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

        # ---- Offset-aware conditioning setup -------------------------------------
        # cond_dim = extra features concatenated to the per-t* representation `z`
        # (B,K,d_model) before it enters head_mu. Only head_mu is conditioned:
        # in gaussian/lead_from_alert mode `self.head` (per-time hazard head) is
        # never reached (forward returns early from the mu-PMF), and the downstream
        # "score" is the LightGBM coverage-aware selector, not a network head.
        self.use_offset_embedding = bool(use_offset_embedding)
        self.use_issue_doy_features = bool(use_issue_doy_features)
        self.offset_embedding_dim = int(offset_embedding_dim)
        self.offset_max = int(offset_max)
        self.offset_min = int(offset_min)
        self.doy_period = float(doy_period)
        cond_dim = 0
        if self.use_offset_embedding:
            # index mapping (explicit & unambiguous): idx = offset_days + 1;
            # idx 0 is a reserved sentinel for non-alerted / pre-alert (offset<0)
            # cells (excluded from the lead loss anyway). Table spans offset in
            # [offset_min, offset_max] -> indices [offset_min+1, offset_max+1].
            self.offset_embedding = nn.Embedding(self.offset_max + 2, self.offset_embedding_dim)
            cond_dim += self.offset_embedding_dim
        else:
            self.offset_embedding = None
        if self.use_issue_doy_features:
            cond_dim += 2   # sin, cos of issue_doy
        # In the shared-multi-offset family (D1/D2) the A/B/C "concat-to-head_mu"
        # conditioning is bypassed: issue_doy (D2) is added inside the offset head,
        # and head_mu itself is unused. Force cond_dim=0 so head_mu stays d_model->d_model.
        self.offset_cond_dim = 0 if bool(use_shared_multi_offset) else int(cond_dim)

        self.head_mu = nn.Sequential(
            nn.Linear(d_model + self.offset_cond_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # ---- D1/D2: shared-encode + offset-specific mu heads ---------------------
        self.use_shared_multi_offset = bool(use_shared_multi_offset)
        self.mu_head_mode = str(mu_head_mode)
        self.candidate_offsets = [int(o) for o in candidate_offsets] if candidate_offsets else []
        self.shared_band_window = int(shared_band_window)
        if self.use_shared_multi_offset and self.mu_head_mode == "offset_specific":
            if not self.candidate_offsets:
                raise ValueError("mu_head_mode='offset_specific' requires a non-empty candidate_offsets")
            # each offset head mirrors head_mu's structure/activation; input is z
            # (d_model) plus issue_doy sin/cos (D2). One shared template -> deepcopy
            # into every head so all heads start IDENTICAL and diverge in training.
            d_in_head = d_model + (2 if self.use_issue_doy_features else 0)
            template = nn.Sequential(
                nn.Linear(d_in_head, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            self.offset_mu_heads = nn.ModuleList(
                [copy.deepcopy(template) for _ in self.candidate_offsets]
            )
        else:
            self.offset_mu_heads = None

        # ---- D4: per-offset residual heads (shared base head_mu + small residual).
        self.use_offset_residual = bool(use_offset_residual)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.residual_scale = float(residual_scale)
        if self.use_offset_residual:
            if self.mu_head_mode != "shared":
                raise ValueError("use_offset_residual requires mu_head_mode='shared' "
                                 "(D4 = shared base head + per-offset residual)")
            if not self.candidate_offsets:
                raise ValueError("use_offset_residual requires a non-empty candidate_offsets")

            def _res_block():
                blk = nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, self.residual_hidden_dim),
                    nn.GELU(),
                    nn.Linear(self.residual_hidden_dim, 1),   # delta_mu (raw_lead correction)
                )
                if bool(zero_init_residual):
                    with torch.no_grad():
                        blk[-1].weight.zero_()
                        blk[-1].bias.zero_()
                return blk

            self.offset_residual_heads = nn.ModuleList([_res_block() for _ in self.candidate_offsets])
        else:
            self.offset_residual_heads = None

        # Persist the offset list/order whenever heads are indexed by offset (D1/D2/D4):
        # a length mismatch vs a checkpoint -> strict-load shape error; the exact-order
        # guard lives in assert_offsets_match() (called by the grid rebuild path).
        if self.offset_mu_heads is not None or self.offset_residual_heads is not None:
            self.register_buffer(
                "candidate_offsets_buf",
                torch.tensor([int(o) for o in self.candidate_offsets], dtype=torch.long),
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

    def _build_offset_conditioning(self, X, tstar):
        """Build the (B,K,offset_cond_dim) conditioning tensor for the mu head.

        Derived entirely from `tstar` (1-based season index of each t* sample) and
        the alert_tstar feature channel already present in X — no new inputs.

          issue_doy_abs = doy_start + tstar - 1          (t* / issue absolute DOY)
          alert_doy_abs = max_t X[..., alert_idx]        (0 pre-alert via causal fill)
          offset        = issue_doy_abs - alert_doy_abs  ( == grid `offset` )

        Conditioning features (flag-gated):
          - offset_embedding : learnable embedding indexed by the integer `offset`.
          - issue_doy sin/cos: seasonal encoding of the issue (t*) absolute DOY.
        (time_since_alert is NOT a model feature — it is identically equal to `offset`
        by construction, so the offset embedding already carries it.)

        For a t* placed by the eval grid at eval_tstar = alert + off, `offset == off`
        exactly, so train/eval semantics match. Non-alerted (pre-alert, alert<=t*)
        cells get alert_doy_abs==0 and are mapped to the sentinel offset index 0
        (they are excluded from the lead loss).
        """
        alert_idx = int(getattr(self, "alert_tstar_feat_idx", -1))
        if alert_idx < 0 or alert_idx >= X.shape[-1]:
            raise ValueError(
                "offset conditioning requires a valid alert_tstar_feat_idx in "
                f"[0, X.shape[-1]={X.shape[-1]}); got {alert_idx}"
            )
        doy_start = float(getattr(self, "doy_start", 1.0))
        tstar_f = tstar.to(torch.float32)                                    # (B,K)
        issue_doy_abs = tstar_f + (doy_start - 1.0)                          # (B,K)
        alert_doy_abs = X[..., alert_idx].amax(dim=2).to(torch.float32)      # (B,K)
        alerted = alert_doy_abs > 0.0
        offset_val = issue_doy_abs - alert_doy_abs                           # (B,K) == grid offset
        offset_safe = torch.where(alerted, offset_val, torch.zeros_like(offset_val))

        parts = []
        if self.use_offset_embedding:
            off_round = torch.round(offset_safe)
            if bool(getattr(self, "offset_cond_strict", True)) and alerted.any():
                oa = off_round[alerted]
                lo = int(oa.min().item()); hi = int(oa.max().item())
                if lo < self.offset_min or hi > self.offset_max:
                    raise ValueError(
                        f"[offset_cond] observed alerted offset range [{lo},{hi}] "
                        f"outside configured [{self.offset_min},{self.offset_max}]. "
                        f"Raise offset_max / adjust offset_min for this data."
                    )
            # idx = offset + 1; sentinel 0 for non-alerted; clamp guards dummy cells.
            idx = (off_round + 1.0).clamp(0, self.offset_max + 1).to(torch.long)
            idx = torch.where(alerted, idx, torch.zeros_like(idx))
            parts.append(self.offset_embedding(idx))                        # (B,K,emb)
        if self.use_issue_doy_features:
            ang = (2.0 * np.pi / self.doy_period) * issue_doy_abs
            parts.append(torch.sin(ang).unsqueeze(-1))                      # (B,K,1)
            parts.append(torch.cos(ang).unsqueeze(-1))                      # (B,K,1)
        return torch.cat(parts, dim=-1)                                     # (B,K,cond_dim)

    # ==================== D1/D2 shared-encode helpers =========================
    def assert_offsets_match(self, expected_offsets):
        """Raise if the model's offset heads do not match `expected_offsets` exactly
        (same values AND same order). Length mismatch is already caught by strict
        state_dict load (buffer shape); this catches an order/value mismatch that
        strict load would silently accept."""
        got = [int(o) for o in self.candidate_offsets]
        exp = [int(o) for o in expected_offsets]
        if got != exp:
            raise ValueError(
                f"[shared_multi_offset] candidate_offsets mismatch between config and "
                f"checkpoint: model={got} vs expected={exp}. Offset list and order must "
                f"be identical (each head is bound to a fixed offset by position)."
            )

    def _band_causal_mask(self, T: int, window: int, device) -> torch.Tensor:
        """(T,T) bool attention mask (True = blocked). Position i may attend only to
        j in [i-window+1, i] -> causal AND limited to the same `window`-day recent
        receptive field as the baseline per-t* input window."""
        i = torch.arange(T, device=device).view(T, 1)
        j = torch.arange(T, device=device).view(1, T)
        allowed = (j <= i) & (j > i - int(window))
        return ~allowed

    def _reconstruct_base_X(self, X, tstar, window: int):
        """Rebuild the site-year base sequence (B,T,D) from the grouped, per-t*
        window-masked batch X:(B,K,T,D). Each slot k carries the SAME base masked to
        its recent window [tstar_k-window+1, tstar_k] (see dataset._mask_to_recent_window);
        overlapping windows hold identical (normalized) values, so averaging the
        covered views recovers the base. Positions covered by no window stay 0 but are
        never attended: every gathered issue at tstar attends only to [tstar-window+1,
        tstar], which is exactly that slot's own coverage."""
        B, K, T, D = X.shape
        dev = X.device
        t0 = torch.arange(T, device=dev).view(1, 1, T)                 # 0-based positions
        lo = (tstar - int(window)).view(B, K, 1)                       # inclusive (0-based)
        hi = (tstar - 1).view(B, K, 1)                                 # inclusive (0-based)
        cov = ((t0 >= lo) & (t0 <= hi)).to(X.dtype).unsqueeze(-1)      # (B,K,T,1)
        denom = cov.sum(dim=1).clamp(min=1.0)                          # (B,T,1)
        return (X * cov).sum(dim=1) / denom                           # (B,T,D)

    def _shared_encode(self, X, tstar, valid_mask):
        """Encode the base site-year sequence ONCE (band-causal) and gather each
        offset's issue hidden state. Returns z:(B,K,d_model)."""
        B, K, T, _ = X.shape
        window = int(getattr(self, "shared_band_window", 28))
        base = self._reconstruct_base_X(X, tstar, window)              # (B,T,D)
        h = self.pos(self.in_proj(base))                              # (B,T,d_model)
        mask = self._band_causal_mask(T, window, X.device)
        H = self.time_encoder(h, mask=mask)                          # (B,T,d_model)  <- ONE encode
        t_idx = (torch.clamp(tstar, 1, T) - 1)                        # (B,K) 0-based issue positions
        idx = t_idx.unsqueeze(-1).expand(-1, -1, H.size(-1))
        z = H.gather(1, idx)                                          # (B,K,d_model)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        return z * valid_mask.unsqueeze(-1).to(z.dtype)

    def _offset_specific_mu_logit(self, z, X, tstar):
        """Route each (sample,offset) slot to its nearest candidate-offset head.
        offset = tstar - alert_rel (== grid offset at inference, where issue = alert +
        offset). Training slots have dense offsets; each is routed to the nearest
        configured offset's head (ties -> smaller offset via argmin). Loss stays
        per-slot (identical reduction). Returns mu_logit (B,K)."""
        B, K, _ = z.shape
        if self.use_issue_doy_features:
            doy_start = float(getattr(self, "doy_start", 1.0))
            issue = tstar.to(z.dtype) + (doy_start - 1.0)
            ang = (2.0 * np.pi / self.doy_period) * issue
            zc = torch.cat([z, torch.sin(ang).unsqueeze(-1), torch.cos(ang).unsqueeze(-1)], dim=-1)
        else:
            zc = z
        alert_idx = int(getattr(self, "alert_tstar_feat_idx", -1))
        doy_start = float(getattr(self, "doy_start", 1.0))
        alert_abs = X[..., alert_idx].amax(dim=2).to(z.dtype)         # (B,K)
        alert_rel = alert_abs - doy_start + 1.0
        offset = tstar.to(z.dtype) - alert_rel                        # (B,K)
        cand = torch.tensor(self.candidate_offsets, device=z.device, dtype=z.dtype)  # (Q,)
        route = (offset.unsqueeze(-1) - cand.view(1, 1, -1)).abs().argmin(dim=-1)    # (B,K), tie->smaller
        mu_logit = z.new_zeros(B, K)
        for q in range(len(self.candidate_offsets)):
            sel = route == q
            if sel.any():
                out_q = self.offset_mu_heads[q](zc[sel]).squeeze(-1)
                # under AMP autocast the head output may be bf16 while mu_logit is
                # float32 (TransformerEncoder ends in a float32 LayerNorm); cast so the
                # scatter dtypes match.
                mu_logit[sel] = out_q.to(mu_logit.dtype)
        return mu_logit

    def _offset_route(self, z, X, tstar):
        """(B,K) index of the nearest candidate offset for each slot (ties -> smaller
        offset). Shared by the offset-specific head (D1/D2) and the residual head (D4)."""
        alert_idx = int(getattr(self, "alert_tstar_feat_idx", -1))
        doy_start = float(getattr(self, "doy_start", 1.0))
        alert_abs = X[..., alert_idx].amax(dim=2).to(z.dtype)         # (B,K)
        offset = tstar.to(z.dtype) - (alert_abs - doy_start + 1.0)    # (B,K)
        cand = torch.tensor(self.candidate_offsets, device=z.device, dtype=z.dtype)
        return (offset.unsqueeze(-1) - cand.view(1, 1, -1)).abs().argmin(dim=-1)

    def _offset_residual_delta(self, z, X, tstar):
        """D4: per-offset small residual on mu_logit (raw_lead). Routed like the
        offset-specific head; each slot uses its nearest candidate offset's residual
        head. Zero at init (zero_init_residual) so D4 starts == D3. Returns (B,K)."""
        B, K, _ = z.shape
        route = self._offset_route(z, X, tstar)
        delta = z.new_zeros(B, K)
        for q in range(len(self.candidate_offsets)):
            sel = route == q
            if sel.any():
                out_q = self.offset_residual_heads[q](z[sel]).squeeze(-1)
                delta[sel] = out_q.to(delta.dtype)
        return delta

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

        if self.use_shared_multi_offset:
            # D1/D2: encode the base site-year sequence ONCE (band-causal) and gather
            # each offset's issue hidden state. No per-t* re-encode, no K-axis cross
            # encoder (z_k is exactly H[:, issue_k] per the target structure).
            h = None
            z = self._shared_encode(X, tstar, valid_mask)  # (B,K,d_model)
            self._last_z_BKD = z.detach()
        else:
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

        # ---- Offset-aware conditioning: concat derived (offset embedding, optional
        # issue_doy sin/cos) features to z ONLY for the mu head. `z` itself is left
        # unchanged so the hazard path (h + z_flat) stays byte-identical. When all
        # flags are off, offset_cond_dim==0 and z_cond is z (no-op).
        if self.offset_cond_dim > 0 and not self.use_shared_multi_offset:
            cond = self._build_offset_conditioning(X, tstar).to(z.dtype)  # (B,K,cond_dim)
            self._last_offset_cond = cond.detach()
            z_cond = torch.cat([z, cond], dim=-1)
        else:
            z_cond = z

        mode = str(getattr(self, "pmf_mode", "hazard"))
        if mode == "gaussian":
            if self.use_shared_multi_offset:
                if self.offset_mu_heads is not None:
                    # D1/D2: fully-independent offset-specific heads.
                    mu_logit = self._offset_specific_mu_logit(z, X, tstar)  # (B, K)
                else:
                    # D3: single shared head on the gathered issue hidden state.
                    # D4: + small per-offset residual (zero at init -> D4==D3 initially).
                    mu_logit = self.head_mu(z).squeeze(-1)                 # (B, K)
                    if self.offset_residual_heads is not None:
                        delta = self._offset_residual_delta(z, X, tstar)   # (B, K)
                        mu_logit = mu_logit + self.residual_scale * delta
            else:
                mu_logit = self.head_mu(z_cond).squeeze(-1)  # (B, K)
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
            elif mu_mode == "prior_residual_alert_bin":
                # PRIOR-RESIDUAL (vendored addition, 2026-07): mu = prior_mu + delta_max*tanh(raw_delta).
                # prior_mu (alert_bin prior, DOY) is carried in a DEDICATED input channel
                # (prior_mu_feat_idx), broadcast-constant over the season window so amax(time)
                # recovers it. The head learns ONLY the residual delta. Existing dispatch
                # alert_tstar channel is left intact. delta_max default 30.
                doy_start = float(getattr(self, "doy_start", 1.0))
                delta_max = float(getattr(self, "delta_max", 30.0))
                pidx = int(getattr(self, "prior_mu_feat_idx", -1))
                if pidx < 0 or pidx >= X.shape[-1]:
                    raise ValueError(
                        f"mu_mode='prior_residual_alert_bin' requires prior_mu_feat_idx "
                        f"in [0, X.shape[-1]={X.shape[-1]}); got {pidx}")
                prior_doy_abs = X[..., pidx].amax(dim=2)             # (B, K) DOY
                prior_rel = prior_doy_abs - doy_start + 1.0          # 1-based season idx
                raw_delta = mu_logit
                delta = delta_max * torch.tanh(raw_delta)
                mu_pre = prior_rel + delta
                mu_temporal = mu_pre.clamp(0.0, float(T) - 1.0)
                self._last_lead_BK = delta.detach()
                self._last_alert_rel_BK = prior_rel.detach()
                self._last_lead_loss_mask = None
                if bool(getattr(self, "lead_debug_once_pending", False)):
                    with torch.no_grad():
                        print(f"[prior_residual_alert_bin] prior_idx={pidx} delta_max={delta_max:.1f} "
                              f"prior_rel[min/mean/max]={float(prior_rel.min()):.1f}/{float(prior_rel.mean()):.1f}/{float(prior_rel.max()):.1f} "
                              f"delta[min/mean/max]={float(delta.min()):.2f}/{float(delta.mean()):.2f}/{float(delta.max()):.2f}")
                    self.lead_debug_once_pending = False
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
            # dtype anchor: `h` is None in the shared-encode path -> fall back to z.
            out_dtype = h.dtype if h is not None else z.dtype
            hazard_flat = _pmf_to_hazard(pmf_flat).to(out_dtype)
            return hazard_flat.reshape(B, K, T)

        self._last_mu_BK = None
        z_flat = z.reshape(BK, 1, -1)
        h_fused = h + z_flat

        logits = self.head(h_fused).squeeze(-1)
        hazard = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6).reshape(B, K, T)
        return hazard
