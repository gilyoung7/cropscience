"""Climatology + fallback policy — behaviour copied from the deployed API verbatim.

POLICY IS NOT CHANGED HERE. BPH/WBPH keep learned_stage2 as the official
final_prediction; the other 6 keep climatology with the learned output as an
experimental auxiliary field. Newer research findings are deliberately ignored.

Ports api_handoff_transformer/run_predict.py:246-263 (compute_climatology) and
:514-590 (the final-block selection).
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

# run_predict.py:67-68 — module constants, used ONLY for climatology.
# (The learned block uses the ckpt's own sigma with `half` computed live.)
SIGMA_DAYS_DEFAULT = 5.0
PI95_HALFWIDTH = 9.8  # round(1.96 * 5.0, 1)


class PolicyError(RuntimeError):
    """Policy/climatology asset missing or malformed."""


def load_policy(path: Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise PolicyError(f"fallback_policy.yaml not found: {p}")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def per_pest_policy(policy: dict, pest: str) -> dict:
    return (policy.get("per_pest", {}) or {}).get(pest, {}) or {}


def compute_climatology(climatology_dir: Path, pest: str, variant: str) -> dict:
    """Port of run_predict.py:246-263.

    The CSV is authoritative; fallback_policy.yaml's inline mean_L/mean_mid/mean_R
    are dead in the deployed code and are ignored here too. Only row 0 is used.

    Read with the stdlib csv module rather than pandas: the value is a single
    scalar from row 0, so no pandas semantics are involved and this keeps the
    dependency off the climatology path. float() of the CSV text reproduces
    pandas' parse for these values (asserted in tests/test_end_to_end.py).
    """
    csv_path = Path(climatology_dir) / f"{pest}_climatology_train_stats.csv"
    if not csv_path.is_file():
        raise PolicyError(f"climatology stats not found for pest={pest}: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise PolicyError(f"climatology CSV has no rows: {csv_path}")
    if variant not in rows[0]:
        raise PolicyError(f"climatology variant '{variant}' not in {csv_path.name}")
    mu = float(rows[0][variant])
    return {
        "mu_doy": round(mu, 2),
        "pi_95": {
            "lower_doy": int(round(mu - PI95_HALFWIDTH)),
            "upper_doy": int(round(mu + PI95_HALFWIDTH)),
            "sigma_days": SIGMA_DAYS_DEFAULT,
        },
        "variant": variant,
        "_source_csv": csv_path.name,
    }


def climatology_variant(policy: dict, pest: str) -> str:
    """run_predict.py:805 — default 'mean_mid'."""
    return (per_pest_policy(policy, pest).get("climatology") or {}).get("variant", "mean_mid")


def select_final(learned: dict | None, climatology: dict, recommended: str) -> dict:
    """EXACT port of the final-block selection, run_predict.py:540-566.

    Note the deployed quirk, reproduced verbatim: the `learned is None` branch
    emits "climatology_no_alert" for ANY Stage-2 failure (not just no-alert), and
    only when recommended == learned_stage2; for climatology-recommended pests a
    Stage-2 failure is indistinguishable from a healthy run in final_prediction.
    """
    if learned is not None:
        if recommended == "learned_stage2":
            return {
                "source": "learned_stage2",
                "mu_doy": learned["mu_doy"],
                "pi_95": learned["pi_95"],
                "selected_offset": learned["selected_offset"],
                "fallback_triggered": False,
            }
        return {
            "source": "climatology",
            "mu_doy": climatology["mu_doy"],
            "pi_95": climatology["pi_95"],
            "selected_offset": None,
            "fallback_triggered": False,
        }
    return {
        "source": "climatology" if recommended == "climatology" else "climatology_no_alert",
        "mu_doy": climatology["mu_doy"],
        "pi_95": climatology["pi_95"],
        "selected_offset": None,
        "fallback_triggered": recommended == "learned_stage2",
    }
