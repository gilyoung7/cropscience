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

    if sampler is not None:
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False
        return DataLoader(ds, **kwargs)

    if shuffle and seed is not None:
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, generator=gen, **kwargs)
    return DataLoader(ds, **kwargs)


def parse_seed_candidates(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    if ":" in raw:
        start_s, end_s = raw.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x) for x in raw.split(",") if x.strip()]
