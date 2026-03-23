from __future__ import annotations


def build_ckpt_meta(
    run: int,
    pest: str,
    d_in: int,
    feature_cols: list[str],
    feature_names: list[str],
    year_max: int | None,
) -> dict:
    return {
        "run": int(run),
        "pest": pest,
        "d_in": int(d_in),
        "feature_cols": list(feature_cols),
        "feature_names": list(feature_names),
        "year_max": year_max,
    }


def _feature_name_diff(expected: list[str], actual: list[str]) -> tuple[list[str], list[str], list[int]]:
    expected_set = set(expected)
    actual_set = set(actual)
    missing_in_eval = [x for x in expected if x not in actual_set]
    extra_in_eval = [x for x in actual if x not in expected_set]
    order_mismatch = [i for i, (a, b) in enumerate(zip(expected, actual)) if a != b]
    return missing_in_eval, extra_in_eval, order_mismatch


def validate_ckpt_meta(
    ckpt: dict,
    *,
    pest: str,
    run: int,
    d_in: int,
    feature_names: list[str],
    allow_run_mismatch: bool = False,
) -> None:
    ckpt_run = ckpt.get("run")
    if ckpt_run is not None and int(ckpt_run) != int(run):
        if not allow_run_mismatch:
            raise ValueError(f"Checkpoint run mismatch: ckpt={ckpt_run} eval={run}")

    ckpt_pest = ckpt.get("pest")
    if ckpt_pest is not None and ckpt_pest != pest:
        raise ValueError(f"Checkpoint pest mismatch: ckpt={ckpt_pest} eval={pest}")

    ckpt_d_in = ckpt.get("d_in")
    if ckpt_d_in is not None and int(ckpt_d_in) != int(d_in):
        raise ValueError(f"Checkpoint d_in mismatch: ckpt={ckpt_d_in} eval={d_in}")

    ckpt_feature_names = ckpt.get("feature_names")
    if ckpt_feature_names is None:
        print("[feature_check] ckpt has no feature_names; strict check skipped")
        return
    if ckpt_feature_names != feature_names:
        missing_in_eval, extra_in_eval, order_mismatch = _feature_name_diff(ckpt_feature_names, feature_names)
        print("[feature_check] mismatch detected")
        print(f"[feature_check] missing_in_eval={missing_in_eval}")
        print(f"[feature_check] extra_in_eval={extra_in_eval}")
        print(f"[feature_check] order_mismatch_indices_head={order_mismatch[:20]}")
        raise ValueError("Checkpoint feature_names and eval feature_names are not identical")
