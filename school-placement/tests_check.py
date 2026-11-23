"""Проверки из раздела 6 задания (запуск вручную из корня проекта или из school-placement).

Примеры (Windows PowerShell, из корня проекта):
  .\.venv\Scripts\Activate.ps1
  python school-placement\tests_check.py --integrity --baseline
  python school-placement\tests_check.py --autopick-smoke --k 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# добавляем папку приложения в путь
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import config
import data_loader
import provision_engine
import validation
import autopick


def _print(title: str, ok: bool, details=None):
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {title}")
    if details is not None:
        print("  ", details)


def test_integrity(ctx):
    res = validation.check_integrity(ctx["blocks"], ctx["pc"], ctx["baseline"], ctx["matrix"])
    _print("Целостность и выравнивание данных", res.get("ok", False), res)
    return res.get("ok", False)


def test_direction(ctx):
    d = ctx.get("direction", {})
    ok = bool(d.get("direction_confirmed_as_i_to_j", False))
    _print("Направление матрицы acc[i,j]=i(население)->j(объект) подтверждено", ok, d)
    return ok


def test_baseline_reproduction(ctx):
    res = validation.check_no_new_schools_equals_baseline(ctx["pc"], ctx["matrix"], ctx["baseline"], eps=config.EPS)
    ok = res.get("ok", False) and res.get("max_abs_diff_provision_strong", 1) < config.EPS
    _print("Без новых школ результат = baseline (max diff < 1e-6)", ok, res)
    return ok


def test_proxy_all(ctx):
    df = autopick.compute_proxy_all(ctx["matrix"], ctx["baseline"], added_capacity=config.DEFAULT_NEW_SCHOOL_CAPACITY)
    ok = len(df) == config.N_BLOCKS and set(["block_id", "proxy_gain", "mode" if "mode" in df.columns else "proxy_gain"]).issuperset({"block_id", "proxy_gain"})
    # все кварталы присутствуют
    ok = ok and df["block_id"].nunique() == config.N_BLOCKS
    _print("Прокси рассчитан для всех 5643 кварталов", bool(ok), {"rows": len(df), "unique_blocks": int(df["block_id"].nunique())})
    return bool(ok)


def test_autopick_smoke(ctx, k=3):
    res = autopick.run_autopick_topk(
        ctx["pc"], ctx["matrix"], ctx["baseline"],
        added_capacity=config.DEFAULT_NEW_SCHOOL_CAPACITY, k=k, save_intermediate=True,
    )
    table = res["table"]
    ok_all = len(table) == config.N_BLOCKS
    ok_modes = set(table["mode"].unique()).issubset({"точно", "прокси"}) and (table["mode"] == "точно").sum() == min(k, config.N_BLOCKS)
    best = res.get("best", pd.DataFrame())
    # равные лучшие: best содержит все с макс точным критерием
    exact = table[table["mode"] == "точно"]
    vals = pd.to_numeric(exact["exact_deficit_reduction"], errors="coerce")
    if vals.notna().any():
        best_val = vals.max()
        expected_best = exact[np.abs(vals - best_val) <= 1e-9]
        ok_best_all = len(best) == len(expected_best)
    else:
        ok_best_all = False
    _print("Автоподбор: таблица всех 5643 и метки точно/прокси", ok_all and ok_modes, {"rows": len(table), "exact": int((table['mode']=='точно').sum()), "k": k})
    _print("Автоподбор: при равенстве показаны все лучшие", ok_best_all, {"best_count": len(best)})
    # ручное == авто для лучшего
    ok_manual = True
    if len(best) > 0:
        row = best.iloc[0].to_dict()
        cmpm = validation.check_manual_equals_autopick(ctx["pc"], ctx["matrix"], ctx["baseline"], row)
        ok_manual = cmpm.get("ok", False)
        _print("Ручное размещение в найденном квартале = результат автоподбора", ok_manual, cmpm.get("diffs", cmpm))
    return bool(ok_all and ok_modes and ok_best_all and ok_manual)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrity", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--proxy", action="store_true")
    parser.add_argument("--autopick-smoke", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if not any([args.integrity, args.baseline, args.proxy, args.autopick_smoke, args.all]):
        parser.print_help()
        return 0

    print("Загрузка данных...")
    ctx = data_loader.load_all()
    ok = True
    if args.all or args.integrity:
        ok &= test_integrity(ctx)
        ok &= test_direction(ctx)
    if args.all or args.proxy:
        ok &= test_proxy_all(ctx)
    if args.all or args.baseline:
        ok &= test_baseline_reproduction(ctx)
    if args.all or args.autopick_smoke:
        ok &= test_autopick_smoke(ctx, k=args.k)
    print("\nИТОГ:", "OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
