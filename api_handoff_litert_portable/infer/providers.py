"""Data-supply interfaces — the seam where a weather API attaches later.

Inference never reaches for a file itself; it asks a provider. Today the only
implementations are CSV/DataFrame-backed, matching the deployed API's inputs
exactly. When the 880-representative-site <-> 105-ASOS mapping and the live
weather API land, they become new implementations of these same protocols and
nothing in stage1_*/stage2_* has to change.

Deliberately NOT implemented here (out of scope for this step):
  * network calls of any kind
  * the representative-site <-> ASOS mapping
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class ProviderError(ValueError):
    """A provider could not supply required data. Never silent, never zero-filled."""


class WeatherProvider(Protocol):
    """Daily weather for one (site, year), in the deployed Korean schema."""

    def daily(self, site_id: str, year: int) -> pd.DataFrame: ...


class SiteMetadataProvider(Protocol):
    """Static per-site metadata (coordinates)."""

    def latlon(self, site_id: str) -> tuple[float, float]: ...


class PhenologyProvider(Protocol):
    """Per-(site, year) phenology observation rows (LONG-style step data)."""

    def rows(self, site_id: str, year: int) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# CSV / DataFrame implementations (this step)
# ---------------------------------------------------------------------------
class FrameWeatherProvider:
    """Serves daily weather from a preloaded frame (or a CSV path).

    Mirrors the deployed `load_input_daily` + `_daily_year_from_frame` contract:
    the frame may hold many sites/years; it is filtered per request.
    """

    def __init__(self, daily: pd.DataFrame | str | Path):
        if isinstance(daily, pd.DataFrame):
            self._df = daily
        else:
            p = Path(daily)
            if not p.is_file():
                raise ProviderError(f"daily weather CSV not found: {p}")
            self._df = pd.read_csv(p, encoding="utf-8-sig")
        self._df = self._df.rename(
            columns=lambda c: c.strip() if isinstance(c, str) else c
        )

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def daily(self, site_id: str, year: int) -> pd.DataFrame:
        col = "지점ID" if "지점ID" in self._df.columns else "site_id"
        if col not in self._df.columns:
            return self._df
        sub = self._df[self._df[col].astype(str) == str(site_id)]
        if sub.empty:
            raise ProviderError(
                f"no daily weather rows for site_id={site_id!r} in the supplied data"
            )
        return sub


class LongObsProvider:
    """Serves coordinates + phenology from a LONG observation frame/CSV.

    This is the deployed API's own source for both, so the values match by
    construction. Implements SiteMetadataProvider and PhenologyProvider.
    """

    def __init__(self, obs: pd.DataFrame | str | Path):
        if isinstance(obs, pd.DataFrame):
            df = obs
        else:
            p = Path(obs)
            if not p.is_file():
                raise ProviderError(f"LONG observation CSV not found: {p}")
            df = pd.read_csv(p, encoding="utf-8-sig")
        df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        for c in ("site_id", "year", "obs_doy"):
            if c not in df.columns:
                raise ProviderError(f"LONG observation data missing column {c!r}")
        df["site_id"] = df["site_id"].astype(str)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        self._df = df

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def latlon(self, site_id: str) -> tuple[float, float]:
        """Port of stage1.py::_site_latlon — first non-null for the site, else
        the global mean (the deployed fallback, kept verbatim)."""
        for c in ("좌표-위도", "좌표-경도"):
            if c not in self._df.columns:
                raise ProviderError(f"LONG observation data missing coordinate column {c!r}")
        lat = pd.to_numeric(self._df["좌표-위도"], errors="coerce")
        lon = pd.to_numeric(self._df["좌표-경도"], errors="coerce")
        sl = lat[self._df["site_id"] == str(site_id)].dropna()
        so = lon[self._df["site_id"] == str(site_id)].dropna()
        return (
            float(sl.iloc[0]) if not sl.empty else float(lat.mean()),
            float(so.iloc[0]) if not so.empty else float(lon.mean()),
        )

    def rows(self, site_id: str, year: int) -> pd.DataFrame:
        return self._df[
            (self._df["site_id"] == str(site_id)) & (self._df["year"] == int(year))
        ]
