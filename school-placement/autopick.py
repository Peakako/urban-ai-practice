"""Автоподбор размещения одной школы: прокси по всем 5643 + точная проверка топ-K / полный перебор."""

from __future__ import annotations

import csv
import gc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
import provision_engine

ProgressCB = Optional[Callable[[int, int, str], None]]


def _report(cb: ProgressCB, done: int, total: int, phase: str) -> None:
    if cb:
        try:
            cb(done, total, phase)
        except Exception:
            pass


def compute_proxy_all(
    acc: np.ndarray | pd.DataFrame,
    baseline_prov: pd.DataFrame,
    added_capacity: int,
    accessibility: int = config.ACCESSIBILITY_MIN,
) -> pd.DataFrame:
    """Векторный прокси для всех 5643 кандидатов j: покрытие unmet demand в зоне доступности.
    acc: матрица [i,j] = время от i (спрос/население) к j (объект/школа)."""
    if isinstance(acc, pd.DataFrame):
        acc_arr = acc.to_numpy(dtype=np.float32, copy=False)
    else:
        acc_arr = np.asarray(acc, dtype=np.float32)
    demand_left = pd.to_numeric(baseline_prov["demand_left"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    # reachable[i,j] = True если население i может достичь объекта j
    reachable = acc_arr <= float(accessibility)
    reachable_unmet = (reachable * demand_left[:, None]).sum(axis=0)
    proxy_gain = np.minimum(float(added_capacity), reachable_unmet)
    df = pd.DataFrame({
        "block_id": np.arange(config.N_BLOCKS, dtype=int),
        "proxy_reachable_unmet": reachable_unmet.astype(np.float64),
        "proxy_gain": proxy_gain.astype(np.float64),
    })
    df["rank_proxy"] = df["proxy_gain"].rank(method="min", ascending=False).astype(int)
    df = df.sort_values(["proxy_gain", "proxy_reachable_unmet", "block_id"], ascending=[False, False, True]).reset_index(drop=True)
    # восстановим rank по block_id позже удобно иметь
    del reachable
    gc.collect()
    return df


def _exact_for_candidate(
    block_id: int,
    added_capacity: int,
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    accessibility: int,
) -> Dict[str, Any]:
    placements = [{"block_id": int(block_id), "added_capacity": int(added_capacity), "source": "auto"}]
    scenario = provision_engine.run_scenario(
        base_pc, acc, placements=placements, accessibility=accessibility,
        self_supply=config.SELF_SUPPLY, max_depth=config.MAX_DEPTH,
    )
    prov = scenario["prov"]
    metrics = provision_engine.compute_metrics(prov, baseline_prov=baseline_prov)
    row = {
        "block_id": int(block_id),
        "added_capacity": int(added_capacity),
        "mode": "точно",
        "exact_sum_demand_left": float(metrics.get("sum_demand_left", np.nan)),
        "exact_deficit_reduction": float(metrics.get("deficit_reduction_sum_demand_left", np.nan)),
        "exact_weighted_provision_strong": float(metrics.get("weighted_provision_strong", np.nan)),
        "exact_mean_provision_strong": float(metrics.get("mean_provision_strong_demand_gt0", np.nan)),
        "exact_count_lt1": float(metrics.get("count_provision_lt1_demand_gt0", np.nan)),
        "exact_capacity_increase": float(metrics.get("capacity_increase", np.nan)),
    }
    provision_engine.cleanup(scenario, prov)
    return row


def _save_partial(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_autopick_topk(
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    added_capacity: int = config.DEFAULT_NEW_SCHOOL_CAPACITY,
    k: int = config.K_TOP_DEFAULT,
    accessibility: int = config.ACCESSIBILITY_MIN,
    progress_cb: ProgressCB = None,
    save_intermediate: bool = True,
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    partial_path = config.EXPORTS_DIR / f"autopick_topk_partial_{ts}.csv"
    final_path = config.EXPORTS_DIR / f"autopick_topk_{ts}.csv"

    _report(progress_cb, config.N_BLOCKS, config.N_BLOCKS, "Прокси-оценка: проверено 5643 из 5643")
    proxy_df = compute_proxy_all(acc, baseline_prov, added_capacity=added_capacity, accessibility=accessibility)
    _report(progress_cb, 0, config.N_BLOCKS, "Прокси завершён: проверено 5643 из 5643; точная проверка топ-K")

    k_eff = max(1, min(int(k), config.N_BLOCKS))
    top_ids = proxy_df.head(k_eff)["block_id"].astype(int).tolist()

    exact_rows: List[Dict[str, Any]] = []
    for i, bid in enumerate(top_ids, start=1):
        _report(progress_cb, i, k_eff, f"Точная проверка топ-K: {i} из {k_eff} (прокси по всем 5643 выполнен)")
        row = _exact_for_candidate(bid, added_capacity, base_pc, acc, baseline_prov, accessibility)
        exact_rows.append(row)
        if save_intermediate:
            _save_partial(exact_rows, partial_path)

    exact_df = pd.DataFrame(exact_rows)
    table = proxy_df.merge(exact_df, on="block_id", how="left")
    table["mode"] = table["mode"].fillna("прокси")
    if "added_capacity_y" in table.columns:
        table["added_capacity"] = table["added_capacity_y"].fillna(added_capacity)
        table.drop(columns=[c for c in ["added_capacity_x", "added_capacity_y"] if c in table.columns], inplace=True, errors="ignore")
    else:
        table["added_capacity"] = table.get("added_capacity", added_capacity).fillna(added_capacity)

    # сортировка для просмотра: точно по убыванию сокращения дефицита, затем прокси
    sort_cols = ["mode", "exact_deficit_reduction", "proxy_gain", "proxy_reachable_unmet"]
    table["_mode_rank"] = (table["mode"] != "точно").astype(int)
    table = table.sort_values(
        ["_mode_rank", "exact_deficit_reduction", "proxy_gain", "proxy_reachable_unmet", "block_id"],
        ascending=[True, False, False, False, True],
    ).drop(columns=["_mode_rank"]).reset_index(drop=True)

    table.to_csv(final_path, index=False, encoding="utf-8-sig")

    # лучшие среди точно оценённых (все равные)
    best_df = pd.DataFrame()
    if "exact_deficit_reduction" in table.columns:
        exact_only = table[table["mode"] == "точно"].copy()
        vals = pd.to_numeric(exact_only["exact_deficit_reduction"], errors="coerce")
        if vals.notna().any():
            best_val = vals.max()
            best_df = exact_only[np.abs(vals - best_val) <= 1e-9].copy()

    result = {
        "table": table,
        "best": best_df,
        "k": k_eff,
        "total": config.N_BLOCKS,
        "added_capacity": added_capacity,
        "accessibility": accessibility,
        "partial_path": str(partial_path),
        "result_path": str(final_path),
        "mode_info": f"Прокси по всем {config.N_BLOCKS}, точная проверка топ-{k_eff}.",
    }
    _report(progress_cb, k_eff, k_eff, "Готово")
    return result


def run_full_exhaustive(
    base_pc: pd.DataFrame,
    acc: pd.DataFrame,
    baseline_prov: pd.DataFrame,
    added_capacity: int = config.DEFAULT_NEW_SCHOOL_CAPACITY,
    accessibility: int = config.ACCESSIBILITY_MIN,
    progress_cb: ProgressCB = None,
    save_path: Optional[Path] = None,
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_path is None:
        save_path = config.EXPORTS_DIR / f"autopick_full_{ts}.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # предварительный прокси для справки
    proxy_df = compute_proxy_all(acc, baseline_prov, added_capacity=added_capacity, accessibility=accessibility)
    rows: List[Dict[str, Any]] = []
    total = config.N_BLOCKS
    for i, bid in enumerate(range(total), start=1):
        _report(progress_cb, i, total, f"проверено {i} из {total}")
        row = _exact_for_candidate(bid, added_capacity, base_pc, acc, baseline_prov, accessibility)
        rows.append(row)
        # промежуточное сохранение после каждого
        _save_partial(rows, save_path)

    exact_df = pd.DataFrame(rows)
    table = proxy_df.merge(exact_df, on="block_id", how="left", suffixes=("_proxy", ""))
    table["mode"] = "точно"
    if "added_capacity_x" in table.columns or "added_capacity_y" in table.columns:
        table["added_capacity"] = int(added_capacity)
        table.drop(columns=[c for c in ["added_capacity_x", "added_capacity_y"] if c in table.columns], inplace=True, errors="ignore")
    table = table.sort_values(["exact_deficit_reduction", "proxy_gain", "block_id"], ascending=[False, False, True]).reset_index(drop=True)
    table.to_csv(save_path, index=False, encoding="utf-8-sig")

    vals = pd.to_numeric(table["exact_deficit_reduction"], errors="coerce")
    best_val = vals.max() if vals.notna().any() else np.nan
    best_df = table[np.abs(vals - best_val) <= 1e-9].copy() if pd.notna(best_val) else table.iloc[0:0]

    _report(progress_cb, total, total, "Полный перебор завершён")
    return {
        "table": table,
        "best": best_df,
        "k": total,
        "total": total,
        "added_capacity": added_capacity,
        "accessibility": accessibility,
        "result_path": str(save_path),
        "mode_info": f"Полный точный перебор: {total} из {total}.",
    }
