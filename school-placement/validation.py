"""Проверки целостности данных, сценариев и результатов."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

import config
import provision_engine


def check_integrity(blocks, pc: pd.DataFrame, baseline: pd.DataFrame, acc: pd.DataFrame) -> Dict[str, Any]:
    ids = list(range(config.N_BLOCKS))
    res: Dict[str, Any] = {}
    res["n_blocks_ok"] = len(blocks) == config.N_BLOCKS
    res["ids_blocks_ok"] = blocks["block_id"].astype(int).tolist() == ids if "block_id" in blocks.columns else False
    res["ids_pc_ok"] = pc.index.astype(int).tolist() == ids
    res["ids_baseline_ok"] = baseline.index.astype(int).tolist() == ids
    res["matrix_shape_ok"] = acc.shape == (config.N_BLOCKS, config.N_BLOCKS)
    res["matrix_ids_ok"] = acc.index.astype(int).tolist() == ids and acc.columns.astype(int).tolist() == ids
    res["matrix_float32"] = str(acc.dtypes.iloc[0]).startswith("float32") or acc.dtypes.iloc[0] == np.float32
    for col in ["population", "capacity"]:
        if col in pc.columns:
            res[f"non_negative_{col}"] = bool((pd.to_numeric(pc[col], errors="coerce").fillna(0) >= 0).all())
    for col in ["demand", "capacity", "provision_strong", "demand_left"]:
        res[f"baseline_has_{col}"] = col in baseline.columns
    res["ok"] = all(bool(v) for v in res.values())
    return res


def check_scenario(placements: List[Dict[str, Any]]) -> Dict[str, Any]:
    errors = []
    normed = []
    agg = {}
    for p in placements or []:
        try:
            bid = int(p.get("block_id"))
        except Exception:
            errors.append(f"Некорректный block_id: {p}")
            continue
        if bid < 0 or bid >= config.N_BLOCKS:
            errors.append(f"block_id вне диапазона 0..{config.N_BLOCKS-1}: {bid}")
            continue
        try:
            cap = int(p.get("added_capacity", 0))
        except Exception:
            errors.append(f"Некорректная вместимость для block_id={bid}: {p}")
            continue
        if cap < 0:
            errors.append(f"added_capacity < 0 для block_id={bid}")
            continue
        agg[bid] = agg.get(bid, 0) + cap
        normed.append({"block_id": bid, "added_capacity": cap, "label": p.get("label", ""), "source": p.get("source", "manual")})
    if errors:
        return {"ok": False, "errors": errors, "placements": normed, "aggregated": agg}
    return {"ok": True, "errors": [], "placements": normed, "aggregated": agg}


def compare_to_baseline(
    scenario_prov: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    eps: float = config.EPS,
) -> Dict[str, Any]:
    idx = pd.Index(range(config.N_BLOCKS))
    s = scenario_prov.reindex(idx)
    b = baseline_prov.reindex(idx)
    # сравниваем там, где у baseline определён provision_strong (demand>0)
    b_dem = pd.to_numeric(b.get("demand", 0), errors="coerce").fillna(0)
    s_dem = pd.to_numeric(s.get("demand", 0), errors="coerce").fillna(0)
    mask = (b_dem > 0) & (s_dem > 0)
    b_ps = pd.to_numeric(b.get("provision_strong", np.nan), errors="coerce")
    s_ps = pd.to_numeric(s.get("provision_strong", np.nan), errors="coerce")
    if mask.any():
        diff = (s_ps[mask] - b_ps[mask]).abs()
        max_diff = float(diff.max())
        mean_diff = float(diff.mean())
    else:
        max_diff = 0.0
        mean_diff = 0.0
    # также сравним ключевые балансы
    for col in ["demand_left", "demand_within", "capacity"]:
        if col in b.columns and col in s.columns:
            d = (pd.to_numeric(s[col], errors="coerce").fillna(0) - pd.to_numeric(b[col], errors="coerce").fillna(0)).abs()
            # не включаем в ok провижн, но вернём
            pass
    ok = max_diff < eps
    return {"ok": ok, "max_abs_diff_provision_strong": max_diff, "mean_abs_diff_provision_strong": mean_diff, "eps": eps, "compared_blocks": int(mask.sum())}


def check_no_new_schools_equals_baseline(
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    eps: float = config.EPS,
) -> Dict[str, Any]:
    scenario = provision_engine.run_scenario(base_pc, acc, placements=[], accessibility=config.ACCESSIBILITY_MIN)
    prov = scenario["prov"]
    cmp_res = compare_to_baseline(prov, baseline_prov, eps=eps)
    provision_engine.cleanup(scenario)
    cmp_res["check"] = "no_new_schools_equals_baseline"
    return cmp_res


def check_manual_equals_autopick(
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    autopick_row: Dict[str, Any],
    accessibility: int = config.ACCESSIBILITY_MIN,
) -> Dict[str, Any]:
    """Ручное размещение одной школы в найденном квартале = точный результат автоподбора."""
    bid = int(autopick_row["block_id"])
    cap_added = int(autopick_row.get("added_capacity", config.DEFAULT_NEW_SCHOOL_CAPACITY))
    placements = [{"block_id": bid, "added_capacity": cap_added, "source": "manual"}]
    scenario = provision_engine.run_scenario(base_pc, acc, placements=placements, accessibility=accessibility)
    prov = scenario["prov"]
    metrics = provision_engine.compute_metrics(prov, baseline_prov=baseline_prov)
    # сравнение с записанными точными метриками автоподбора (если есть)
    diffs = {}
    for key in ["sum_demand_left", "capacity_increase", "deficit_reduction_sum_demand_left"]:
        if key in autopick_row and key in metrics:
            try:
                diffs[key] = float(metrics[key]) - float(autopick_row[key])
            except Exception:
                pass
    # ключевое: тот же sum_demand_left (с точностью вычислений)
    ok = True
    if "exact_sum_demand_left" in autopick_row:
        ok = abs(float(metrics["sum_demand_left"]) - float(autopick_row["exact_sum_demand_left"])) < 1e-6
    elif "sum_demand_left" in autopick_row:
        ok = abs(float(metrics["sum_demand_left"]) - float(autopick_row["sum_demand_left"])) < 1e-6
    provision_engine.cleanup(scenario)
    return {"ok": ok, "block_id": bid, "metrics": metrics, "diffs": diffs}


def find_equal_best(table: pd.DataFrame, criterion_col: str = "exact_deficit_reduction", eps: float = 1e-9) -> pd.DataFrame:
    """Вернуть все строки с лучшим (максимальным) значением критерия среди точно оценённых."""
    if criterion_col not in table.columns:
        return table.iloc[0:0]
    exact = table[table.get("mode", "") == "точно"].copy()
    if exact.empty:
        exact = table.copy()
    vals = pd.to_numeric(exact[criterion_col], errors="coerce")
    if vals.notna().sum() == 0:
        return exact.iloc[0:0]
    best = vals.max()
    return exact[np.abs(vals - best) <= eps]
