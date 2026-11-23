"""Расчёт обеспеченности (competitive_provision), метрики и дельты."""

from __future__ import annotations

import gc
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from blocksnet.analysis.provision import competitive_provision


def compute_demand(population: pd.Series | np.ndarray) -> np.ndarray:
    pop = pd.to_numeric(pd.Series(population), errors="coerce").fillna(0).to_numpy(dtype=float)
    return np.ceil(pop * config.DEMAND_PER_1000 / 1000.0).astype(int)


def build_blocks_df(
    base_pc: pd.DataFrame,
    placements: Optional[List[Dict[str, Any]]] = None,
    capacity_override: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Формирует вход competitive_provision: index 0..5642, колонки demand, capacity (int)."""
    idx = pd.Index(range(config.N_BLOCKS), name="block_id")
    if capacity_override is not None:
        cap = pd.to_numeric(capacity_override, errors="coerce").reindex(idx).fillna(0).astype(int)
    else:
        cap = pd.to_numeric(base_pc["capacity"], errors="coerce").reindex(idx).fillna(0).astype(int)
    if placements:
        add = pd.Series(0, index=idx, dtype=int)
        for p in placements:
            bid = int(p.get("block_id"))
            if bid not in add.index:
                continue
            val = int(p.get("added_capacity", 0))
            if val < 0:
                raise ValueError("added_capacity должен быть >= 0")
            add.loc[bid] += val
        cap = (cap + add).astype(int)
    demand = compute_demand(base_pc.reindex(idx)["population"])
    df = pd.DataFrame({"demand": demand.astype(int), "capacity": cap.astype(int)}, index=idx)
    return df


def run_competitive(
    blocks_df: pd.DataFrame,
    acc: pd.DataFrame,
    accessibility: int = config.ACCESSIBILITY_MIN,
    self_supply: bool = config.SELF_SUPPLY,
    max_depth: int = config.MAX_DEPTH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prov_df, links_df = competitive_provision(
        blocks_df.copy(),
        acc,
        accessibility=int(accessibility),
        self_supply=bool(self_supply),
        max_depth=int(max_depth),
    )
    return prov_df, links_df


def run_scenario(
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    placements: Optional[List[Dict[str, Any]]] = None,
    capacity_override: Optional[pd.Series] = None,
    accessibility: int = config.ACCESSIBILITY_MIN,
    self_supply: bool = config.SELF_SUPPLY,
    max_depth: int = config.MAX_DEPTH,
) -> Dict[str, Any]:
    bdf = build_blocks_df(base_pc, placements=placements, capacity_override=capacity_override)
    prov_df, links_df = run_competitive(
        bdf, acc, accessibility=accessibility, self_supply=self_supply, max_depth=max_depth
    )
    # нормализация индекса
    prov_df = prov_df.copy()
    prov_df.index = pd.to_numeric(prov_df.index, errors="coerce").astype(int)
    prov_df = prov_df.reindex(range(config.N_BLOCKS))
    # для блоков с demand==0 provision может быть NaN/inf в raw, но competitive ставит 0/0 -> NaN
    # оставим как есть; доп. безопасная обработка
    result = {
        "blocks_df": bdf,
        "prov": prov_df,
        "links": links_df,
        "accessibility": accessibility,
        "self_supply": self_supply,
        "max_depth": max_depth,
    }
    # освобождение памяти вызывающим по необходимости, но links/prov возвращаем; gc здесь мягко
    gc.collect()
    return result


def _safe_series(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def compute_metrics(prov_df: pd.DataFrame, baseline_prov: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    demand = _safe_series(prov_df, "demand", 0).fillna(0)
    capacity = _safe_series(prov_df, "capacity", 0).fillna(0)
    d_left = _safe_series(prov_df, "demand_left", 0).fillna(0)
    d_within = _safe_series(prov_df, "demand_within", 0).fillna(0)
    ps = _safe_series(prov_df, "provision_strong", np.nan)

    mask_d = demand > 0
    w = demand[mask_d]
    ps_d = ps[mask_d].fillna(0)
    weighted_ps = float((ps_d * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    mean_ps = float(ps[mask_d].mean()) if mask_d.any() else float("nan")
    share_ge1 = float(((ps[mask_d] >= 1.0 - 1e-12)).mean()) if mask_d.any() else float("nan")
    share_lt1 = float(((ps[mask_d] < 1.0 - 1e-12)).mean()) if mask_d.any() else float("nan")
    cnt_lt1 = int(((ps[mask_d] < 1.0 - 1e-12)).sum()) if mask_d.any() else 0

    metrics = {
        "sum_demand": float(demand.sum()),
        "sum_capacity": float(capacity.sum()),
        "sum_demand_left": float(d_left.sum()),
        "sum_demand_within": float(d_within.sum()),
        "mean_provision_strong_demand_gt0": mean_ps,
        "weighted_provision_strong": weighted_ps,
        "share_provision_ge1_demand_gt0": share_ge1,
        "share_provision_lt1_demand_gt0": share_lt1,
        "count_provision_lt1_demand_gt0": cnt_lt1,
    }

    if baseline_prov is not None:
        b = baseline_prov.reindex(prov_df.index)
        b_d_left = _safe_series(b, "demand_left", 0).fillna(0)
        b_cap = _safe_series(b, "capacity", 0).fillna(0)
        b_ps = _safe_series(b, "provision_strong", np.nan)
        metrics["capacity_increase"] = float(capacity.sum() - b_cap.sum())
        metrics["deficit_reduction_sum_demand_left"] = float(b_d_left.sum() - d_left.sum())
        # сравнение provision_strong
        common = mask_d | (_safe_series(b, "demand", 0).fillna(0) > 0)
        diff = (ps.fillna(-1) - b_ps.reindex(ps.index).fillna(-1)).abs()
        # для честного сравнения там где оба demand>0 или оба NaN-подобны
        both_d = (demand > 0) & (_safe_series(b, "demand", 0).fillna(0) > 0)
        if both_d.any():
            diff2 = (ps[both_d].fillna(0) - b_ps.reindex(ps.index)[both_d].fillna(0)).abs()
            metrics["provision_strong_max_abs_diff_vs_baseline"] = float(diff2.max())
            metrics["provision_strong_mean_abs_diff_vs_baseline"] = float(diff2.mean())
        else:
            metrics["provision_strong_max_abs_diff_vs_baseline"] = float("nan")
            metrics["provision_strong_mean_abs_diff_vs_baseline"] = float("nan")
        metrics["baseline_sum_demand_left"] = float(b_d_left.sum())
        metrics["baseline_sum_capacity"] = float(b_cap.sum())
    return metrics


def per_block_comparison(
    base_pc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    scenario_prov: pd.DataFrame,
    placements: Optional[List[Dict[str, Any]]] = None,
) -> pd.DataFrame:
    idx = pd.Index(range(config.N_BLOCKS), name="block_id")
    add = pd.Series(0, index=idx, dtype=int)
    if placements:
        for p in placements:
            bid = int(p.get("block_id"))
            if bid in add.index:
                add.loc[bid] += int(p.get("added_capacity", 0))
    base_cap = pd.to_numeric(base_pc["capacity"], errors="coerce").reindex(idx).fillna(0).astype(int)
    pop = pd.to_numeric(base_pc["population"], errors="coerce").reindex(idx).fillna(0).astype(int)
    sap = pd.to_numeric(base_pc.get("school_age_population", 0), errors="coerce").reindex(idx).fillna(0).astype(int)

    b = baseline_prov.reindex(idx)
    s = scenario_prov.reindex(idx)
    df = pd.DataFrame(index=idx)
    df["block_id"] = idx.astype(int)
    df["population"] = pop.values
    df["school_age_population"] = sap.values
    df["capacity_base"] = base_cap.values
    df["capacity_added"] = add.values
    df["capacity_scenario"] = (base_cap + add).values
    df["demand"] = _safe_series(s, "demand", 0).reindex(idx).fillna(0).values
    df["provision_strong_base"] = _safe_series(b, "provision_strong", np.nan).reindex(idx).values
    df["provision_strong_scenario"] = _safe_series(s, "provision_strong", np.nan).reindex(idx).values
    df["demand_left_base"] = _safe_series(b, "demand_left", 0).reindex(idx).fillna(0).values
    df["demand_left_scenario"] = _safe_series(s, "demand_left", 0).reindex(idx).fillna(0).values
    df["delta_provision_strong"] = df["provision_strong_scenario"].fillna(0) - df["provision_strong_base"].fillna(0)
    df["delta_demand_left"] = df["demand_left_scenario"] - df["demand_left_base"]
    return df


def cleanup(*objs: Any) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
