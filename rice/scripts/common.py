from __future__ import annotations

from torch.utils.data import DataLoader
import torch

from rice.configs import config as C


def make_loader(
    ds,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    sampler=None,
    multiprocessing_context: str | None = None,
    collate_fn=None,
):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS,
        prefetch_factor=C.PREFETCH_FACTOR,
    )
    if multiprocessing_context is not None and kwargs["num_workers"] > 0:
        kwargs["multiprocessing_context"] = multiprocessing_context
    if kwargs["num_workers"] <= 0:
        kwargs.pop("persistent_workers", None)
        kwargs.pop("prefetch_factor", None)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn

    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
        return DataLoader(ds, **kwargs)

    if shuffle and seed is not None:
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, generator=gen, **kwargs)
    return DataLoader(ds, **kwargs)


def collate_grouped_stage2(batch):
    """
    Batch items:
      X_seq: (K,T,D), L/R/c/tstar: (K,)
    Output:
      X_pad: (B,Kmax,T,D)
      L/R/c/tstar_pad: (B,Kmax)
      valid_mask: (B,Kmax) bool
    """
    if not batch:
        raise ValueError("collate_grouped_stage2: empty batch")

    B = len(batch)
    Kmax = max(int(x[0].shape[0]) for x in batch)
    T = int(batch[0][0].shape[1])
    D = int(batch[0][0].shape[2])

    X_pad = torch.zeros((B, Kmax, T, D), dtype=batch[0][0].dtype)
    L_pad = torch.ones((B, Kmax), dtype=torch.long)
    R_pad = torch.ones((B, Kmax), dtype=torch.long)
    c_pad = torch.ones((B, Kmax), dtype=torch.long)
    tstar_pad = torch.ones((B, Kmax), dtype=torch.long)
    valid_mask = torch.zeros((B, Kmax), dtype=torch.bool)

    for i, (X_seq, L_seq, R_seq, c_seq, tstar_seq) in enumerate(batch):
        k = int(X_seq.shape[0])
        X_pad[i, :k] = X_seq
        L_pad[i, :k] = L_seq
        R_pad[i, :k] = R_seq
        c_pad[i, :k] = c_seq
        tstar_pad[i, :k] = tstar_seq
        valid_mask[i, :k] = True

    return X_pad, L_pad, R_pad, c_pad, tstar_pad, valid_mask


def parse_seed_candidates(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    if ":" in raw:
        start_s, end_s = raw.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def init_wandb_run(
    use_wandb: bool,
    project: str | None,
    entity: str | None,
    run_name: str | None,
    group: str | None,
    job_type: str | None,
    tags: list[str] | None,
    config: dict | None,
):
    if not use_wandb:
        return None
    try:
        import wandb
    except ImportError as e:
        raise RuntimeError(
            "W&B logging requested but `wandb` is not installed. "
            "Install with `pip install wandb`."
        ) from e
    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        job_type=job_type,
        tags=tags or None,
        config=config or None,
    )


def finish_wandb_run(wandb_run):
    if wandb_run is not None:
        wandb_run.finish()
