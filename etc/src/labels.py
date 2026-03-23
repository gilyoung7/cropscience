from __future__ import annotations

import numpy as np
import pandas as pd

from etc.configs import config as C


def build_interval_labels_from_doy(
    obs2: pd.DataFrame,
    threshold: float,
    season_start_doy: int = 1,
    season_end_doy: int = 365,
) -> pd.DataFrame:
    """
    obs2: [site_id, year, obs_doy, COUNT_COL] (+ optional LABEL_COL)
    threshold: y >= threshold -> "above" (LABEL_COL이 없을 때만 사용)
    returns labels: [site_id, year, censor_type, L_doy, R_doy, threshold]
      - interval: 마지막 below 시점 L_doy, 첫 above 시점 R_doy
      - left-like early hit is converted to interval:
          L_doy=max(season_start_doy, R_doy-LEFT_WINDOW_DAYS), R_doy=첫 above 시점
      - right: 시즌 끝까지 (이 구현은 L_doy=season_start, R_doy=season_end로 둠)
    """
    rows = []
    count_col = C.COUNT_COL
    label_col = C.LABEL_COL
    left_window_days = int(getattr(C, "LEFT_WINDOW_DAYS", 15))

    for (site, year), sub in obs2.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("obs_doy")

        t = sub["obs_doy"].to_numpy().astype(int)
        if label_col in sub.columns:
            y = pd.to_numeric(sub[label_col], errors="coerce").fillna(0.0).to_numpy()
            above = y > 0
        else:
            y = sub[count_col].to_numpy()
            above = y > threshold

        if above.any():
            idx_R = int(np.argmax(above))
            R_doy = int(t[idx_R])
            left_L = max(int(season_start_doy), int(R_doy) - left_window_days)

            if idx_R == 0:
                censor_type = "interval"
                L_doy = left_L
            else:
                idx_Ls = np.where(~above[:idx_R])[0]
                if len(idx_Ls) == 0:
                    censor_type = "interval"
                    L_doy = left_L
                else:
                    censor_type = "interval"
                    L_doy = int(t[idx_Ls[-1]])

            rows.append(
                {
                    "site_id": site,
                    "year": int(year),
                    "censor_type": censor_type,
                    "L_doy": int(L_doy),
                    "R_doy": int(R_doy),
                    "threshold": float(threshold),
                }
            )
        else:
            rows.append(
                {
                    "site_id": site,
                    "year": int(year),
                    "censor_type": "right",
                    "L_doy": int(season_start_doy),
                    "R_doy": int(season_end_doy),
                    "threshold": float(threshold),
                }
            )

    return pd.DataFrame(rows)


def filter_labels_by_gap(
    labels: pd.DataFrame,
    doy_start: int,
    doy_end: int,
    max_gap: int,
) -> pd.DataFrame:
    """
    interval 라벨만: 시즌 범위로 clamp한 후 gap = R_cl - L_cl <= max_gap 인 것만 유지
    left/right는 그대로 유지
    """
    labels_f = labels.copy()
    labels_f["L_cl"] = labels_f["L_doy"].clip(doy_start, doy_end)
    labels_f["R_cl"] = labels_f["R_doy"].clip(doy_start, doy_end)
    labels_f["gap"] = labels_f["R_cl"] - labels_f["L_cl"]

    # Interval labels must keep strictly positive width after seasonal clamp.
    # gap <= 0 causes log(0) in interval likelihood and can explode training.
    keep = (labels_f["censor_type"] != "interval") | (
        (labels_f["gap"] >= 1) & (labels_f["gap"] <= max_gap)
    )
    out = labels_f.loc[keep].copy()
    out = out.drop(columns=["L_cl", "R_cl", "gap"], errors="ignore")
    return out
