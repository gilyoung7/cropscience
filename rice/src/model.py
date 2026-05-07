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

    def _causal_tstar_mask(self, K: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(K, K, dtype=torch.bool, device=device), diagonal=1)

    def _encode_time_chunk(self, x):
        h = self.in_proj(x)
        h = self.pos(h)
        return self.time_encoder(h)

    def forward(self, X, tstar, valid_mask):
        """
        X:          (B,K,T,D)
        tstar:      (B,K) in 1..T
        valid_mask: (B,K) bool
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

        mode = str(getattr(self, "pmf_mode", "hazard"))
        if mode == "gaussian":
            mu_logit = self.head_mu(z).squeeze(-1)  # (B, K)
            mu_max_attr = float(getattr(self, "gaussian_mu_max", 0.0) or 0.0)
            mu_max = mu_max_attr if mu_max_attr > 0 else float(T)
            mu = torch.sigmoid(mu_logit) * mu_max  # (B, K)
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
