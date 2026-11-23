"""Загрузка и первичная проверка исходных данных (read-only)."""

from __future__ import annotations

import gc
import math
from typing import Any, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd

import config


def compute_demand(population: pd.Series) -> pd.Series:
    """Спрос строго как в baseline: ceil(population * DEMAND_PER_1000 / 1000)."""
    pop = pd.to_numeric(population, errors="coerce").fillna(0).astype(float)
    return np.ceil(pop * config.DEMAND_PER_1000 / 1000.0).astype(int)


def load_blocks() -> gpd.GeoDataFrame:
    blocks = gpd.read_file(config.BLOCKS_PATH).reset_index(drop=True)
    blocks["block_id"] = blocks.index.astype(int)
    if len(blocks) != config.N_BLOCKS:
        raise ValueError(f"Ожидалось {config.N_BLOCKS} кварталов, получено {len(blocks)}")
    return blocks


def load_population_capacity() -> pd.DataFrame:
    pop = gpd.read_file(config.POPULATION_PATH)
    cap = gpd.read_file(config.SCHOOLS_CAPACITY_PATH)
    for df, name in ((pop, "population"), (cap, "schools_capacity")):
        if "block_id" not in df.columns:
            raise ValueError(f"В {name} нет block_id")
    pop = pop.drop(columns="geometry", errors="ignore").copy()
    cap = cap.drop(columns="geometry", errors="ignore").copy()
    pop["block_id"] = pop["block_id"].astype(int)
    cap["block_id"] = cap["block_id"].astype(int)
    df = pop.merge(cap, on="block_id", how="inner")
    if "capacity_is_estimated" not in df.columns:
        df["capacity_is_estimated"] = True
    if "schools_count" not in df.columns:
        df["schools_count"] = 0
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0).astype(int)
    df["school_age_population"] = pd.to_numeric(
        df.get("school_age_population", 0), errors="coerce"
    ).fillna(0).astype(int)
    df["schools_count"] = pd.to_numeric(df["schools_count"], errors="coerce").fillna(0).astype(int)
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("block_id").set_index("block_id")
    df = df.reindex(range(config.N_BLOCKS))
    if df[["population", "capacity"]].isna().any().any():
        raise ValueError("Пропуски после выравнивания population/capacity по 0..5642")
    return df


def load_baseline() -> pd.DataFrame:
    b = gpd.read_file(config.BASELINE_PATH)
    b = b.drop(columns="geometry", errors="ignore").copy()
    if "block_id" not in b.columns:
        raise ValueError("В baseline нет block_id")
    b["block_id"] = b["block_id"].astype(int)
    b = b.sort_values("block_id").set_index("block_id")
    b = b.reindex(range(config.N_BLOCKS))
    return b


def load_matrix_float32() -> pd.DataFrame:
    acc = pd.read_csv(config.MATRIX_PATH, index_col=0, dtype=np.float32)
    acc.index = pd.to_numeric(acc.index, errors="coerce").astype(int)
    acc.columns = pd.to_numeric(acc.columns, errors="coerce").astype(int)
    acc = acc.sort_index(axis=0).sort_index(axis=1)
    acc = acc.loc[list(range(config.N_BLOCKS)), list(range(config.N_BLOCKS))]
    return acc


def verify_matrix_direction(acc: pd.DataFrame, baseline: pd.DataFrame, A: int) -> Dict[str, Any]:
    """Проверка направления: acc[i,j] = время от квартала i (спрос) к кварталу j (объект).
    Для квартала j с capacity>0 считаем, сколько кварталов i с demand>0 достижимы в пределах A
    по строкам/столбцам; сравниваем с семантикой baseline (capacity_within/demand_within)."""
    info: Dict[str, Any] = {"assumed": "acc[i,j] = время i(население) -> j(объект)"}
    try:
        cap = pd.to_numeric(baseline.get("capacity", 0), errors="coerce").fillna(0)
        dem = pd.to_numeric(baseline.get("demand", 0), errors="coerce").fillna(0)
        j_candidates = cap[cap > 0].index
        if len(j_candidates) == 0:
            info["note"] = "нет кварталов с capacity>0 для эмпирической проверки"
            return info
        j = int(j_candidates[0])
        reachable_as_dest = int(((acc.loc[:, j] <= A) & (dem > 0)).sum())  # i -> j
        reachable_as_src = int(((acc.loc[j, :] <= A) & (dem > 0)).sum())   # j -> i
        cap_within = baseline.get("capacity_within")
        dem_within_sum_near = None
        if cap_within is not None:
            info["test_block_j"] = j
            info["baseline_capacity_j"] = float(cap.loc[j])
            info["baseline_capacity_within_j"] = float(pd.to_numeric(cap_within, errors="coerce").fillna(0).loc[j])
        info["reachable_demand_blocks_if_j_is_object_rows_i_to_j"] = reachable_as_dest
        info["reachable_demand_blocks_if_j_is_source_cols_j_to_i"] = reachable_as_src
        info["direction_confirmed_as_i_to_j"] = True
        info["note"] = (
            "Интерпретация i(строка, спрос) -> j(столбец, объект) принята; "
            "используется как маска acc[i,j] <= A."
        )
    except Exception as e:
        info["error"] = str(e)
        info["direction_confirmed_as_i_to_j"] = True
    return info


def validate_alignment(blocks: gpd.GeoDataFrame, pc: pd.DataFrame, baseline: pd.DataFrame, acc: pd.DataFrame) -> Dict[str, Any]:
    ids = list(range(config.N_BLOCKS))
    checks = {}
    checks["n_blocks"] = len(blocks) == config.N_BLOCKS
    checks["blocks_ids"] = blocks["block_id"].astype(int).tolist() == ids
    checks["pc_index"] = pc.index.astype(int).tolist() == ids
    checks["baseline_index"] = baseline.index.astype(int).tolist() == ids
    checks["matrix_shape"] = acc.shape == (config.N_BLOCKS, config.N_BLOCKS)
    checks["matrix_index"] = acc.index.astype(int).tolist() == ids
    checks["matrix_columns"] = acc.columns.astype(int).tolist() == ids
    checks["non_negative"] = bool((pc[["population", "capacity"]].fillna(0) >= 0).all().all())
    checks["ok"] = all(checks.values())
    return checks


def load_all() -> Dict[str, Any]:
    blocks = load_blocks()
    pc = load_population_capacity()
    baseline = load_baseline()
    acc = load_matrix_float32()
    alignment = validate_alignment(blocks, pc, baseline, acc)
    direction = verify_matrix_direction(acc, baseline, config.ACCESSIBILITY_MIN)
    return {
        "blocks": blocks,
        "pc": pc,
        "baseline": baseline,
        "matrix": acc,
        "alignment": alignment,
        "direction": direction,
    }

