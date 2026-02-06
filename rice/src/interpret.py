from __future__ import annotations

import numpy as np
import torch

from rice.src.train_eval import eval_nll_model, interval_nll_per_sample, weighted_loss_from_ctype


@torch.no_grad()
def permutation_importance_features(
    model,
    val_loader,
    Tend: int,
    device,
    feature_names: list[str] | None = None,
    n_repeats: int = 3,
    seed: int = 0,
):
    """
    Permutation Importance on val NLL:
      - baseline val NLL
      - for each feature j, shuffle across batch dimension and measure delta NLL
    Returns:
      base_val_nll, importances (D,), ranked [(name, delta), ...]
    """
    model.eval()
    base_val_nll = float(eval_nll_model(model, val_loader, Tend=Tend, device=device))

    # infer D
    X0, *_ = next(iter(val_loader))
    D = int(X0.shape[-1])
    if feature_names is None:
        feature_names = [f"f{j}" for j in range(D)]
    assert len(feature_names) == D

    importances = np.zeros(D, dtype=np.float64)

    for j in range(D):
        deltas = []
        for r in range(n_repeats):
            # different permutation each repeat (deterministic)
            g = torch.Generator(device="cpu").manual_seed(int(seed) * 1000 + j * 10 + r)

            total, n = 0.0, 0
            for X, L, R, ctype in val_loader:
                Xp = X.clone()
                idx = torch.randperm(Xp.size(0), generator=g)

                # Xp: (B,T,D)
                Xp[:, :, j] = Xp[idx, :, j]

                Xp = Xp.to(device, non_blocking=True)
                L = L.to(device, non_blocking=True)
                R = R.to(device, non_blocking=True)
                ctype = ctype.to(device, non_blocking=True)

                hazard = model(Xp)
                nll_vec = interval_nll_per_sample(hazard, L, R, ctype, Tend=Tend)
                loss = weighted_loss_from_ctype(nll_vec, ctype)

                total += float(loss.item()) * Xp.size(0)
                n += Xp.size(0)

            shuffled_val_nll = total / max(n, 1)
            deltas.append(shuffled_val_nll - base_val_nll)

        importances[j] = float(np.mean(deltas))

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return base_val_nll, importances, ranked


def summarize_importance(all_imps: list[np.ndarray], feature_names: list[str], topk: int = 20):
    """
    all_imps: list of (D,) arrays
    Returns ranked_mean list and TOPK feature names
    """
    M = np.stack(all_imps, axis=0)  # (S,D)
    mean_imps = M.mean(axis=0)
    ranked_mean = sorted(zip(feature_names, mean_imps), key=lambda x: x[1], reverse=True)
    top_names = [name for name, _ in ranked_mean[:topk]]
    return ranked_mean, top_names
