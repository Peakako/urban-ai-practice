"""Streamlit-приложение: оценка вариантов размещения школ (Омск, Практика 3)."""

from __future__ import annotations

import gc
import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import linear
from shapely.geometry import Point
from streamlit_folium import st_folium

import autopick
import config
import data_loader
import provision_engine
import validation


st.set_page_config(page_title="Размещение школ — Омск", layout="wide")


# ---------------------------- Кэширование данных ---------------------------- #

@st.cache_resource(show_spinner=False)
def get_data():
    return data_loader.load_all()


@st.cache_resource(show_spinner=False)
def get_baseline_reproduction(_pc: pd.DataFrame, _acc: pd.DataFrame, _baseline: pd.DataFrame):
    # воспроизведение baseline без новых школ
    scenario = provision_engine.run_scenario(
        _pc, _acc, placements=[], accessibility=config.ACCESSIBILITY_MIN,
        self_supply=config.SELF_SUPPLY, max_depth=config.MAX_DEPTH,
    )
    prov = scenario["prov"]
    cmp_res = validation.compare_to_baseline(prov, _baseline, eps=config.EPS)
    metrics = provision_engine.compute_metrics(prov, baseline_prov=_baseline)
    # сохраняем копию prov (небольшая относительно матрицы)
    prov_copy = prov.copy()
    provision_engine.cleanup(scenario)
    return {"prov": prov_copy, "compare": cmp_res, "metrics": metrics}


@st.cache_resource(show_spinner=False)
def get_blocks_map_gdf(_blocks):
    g = _blocks[["block_id", "geometry"]].copy()

    # 1) чиним невалидные геометрии
    try:
        g["geometry"] = g.geometry.buffer(0)
    except Exception:
        pass

    # 2) упрощаем в МЕТРАХ (UTM 43N для Омска)
    try:
        g_m = g.to_crs(epsg=32643)
        g_m["geometry"] = g_m.geometry.simplify(20, preserve_topology=True)  # 20 м
        g = g_m
    except Exception as e:
        print("simplify failed:", e)

    # 3) переводим в WGS84 для folium
    try:
        g = g.to_crs(epsg=4326)
    except Exception as e:
        print("to_crs failed:", e)

    return g

# ---------------------------- Утилиты ---------------------------- #

def fmt(v, digits=3):
    try:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "—"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.{digits}f}"
    except Exception:
        return str(v)


def ensure_state():
    ss = st.session_state
    if "placements" not in ss:
        ss.placements = []  # list of {block_id, added_capacity, source, label}
    if "scenario_result" not in ss:
        ss.scenario_result = None
    if "autopick_result" not in ss:
        ss.autopick_result = None
    if "selected_block" not in ss:
        ss.selected_block = 0
    if "comparison_results" not in ss:
        ss.comparison_results = None


def placements_signature(placements: List[Dict[str, Any]]) -> str:
    norm = sorted([(int(p["block_id"]), int(p.get("added_capacity", 0))) for p in placements])
    return json.dumps(norm, ensure_ascii=False)


def aggregate_placements(placements: List[Dict[str, Any]]) -> Dict[int, int]:
    agg: Dict[int, int] = {}
    for p in placements or []:
        bid = int(p["block_id"])
        cap = int(p.get("added_capacity", 0))
        agg[bid] = agg.get(bid, 0) + cap
    return agg


def run_and_store_scenario(ctx, placements: List[Dict[str, Any]]):
    acc = ctx["matrix"]
    pc = ctx["pc"]
    baseline = ctx["baseline"]
    sc = provision_engine.run_scenario(pc, acc, placements=placements, accessibility=config.ACCESSIBILITY_MIN)
    prov = sc["prov"].copy()
    metrics = provision_engine.compute_metrics(prov, baseline_prov=baseline)
    cmpb = validation.compare_to_baseline(prov, baseline, eps=config.EPS)
    comp_df = provision_engine.per_block_comparison(pc, baseline, prov, placements=placements)
    provision_engine.cleanup(sc)
    st.session_state.scenario_result = {
        "prov": prov,
        "metrics": metrics,
        "compare_to_baseline": cmpb,
        "comparison_df": comp_df,
        "placements": [dict(p) for p in placements],
        "signature": placements_signature(placements),
    }
    return st.session_state.scenario_result


def build_map(blocks_gdf, df, mode, placements_agg, selected_block):
    # --- нормализация: block_id должен быть колонкой, не индексом ---
    blocks_gdf = blocks_gdf.copy()
    if "block_id" not in blocks_gdf.columns:
        blocks_gdf = blocks_gdf.reset_index()
    if blocks_gdf.index.name == "block_id" or "block_id" in blocks_gdf.index.names:
        blocks_gdf = blocks_gdf.reset_index(drop=True)

    df = df.copy()
    if "block_id" not in df.columns:
        df = df.reset_index()
    if df.index.name == "block_id" or "block_id" in df.index.names:
        df = df.reset_index(drop=True)

    blocks_gdf["block_id"] = blocks_gdf["block_id"].astype(int)
    df["block_id"] = df["block_id"].astype(int)

    # один merge, без дублирования
    g = blocks_gdf.merge(df, on="block_id", how="left")
    added_blocks = set(placements_agg.keys()) if placements_agg else set()

    # выбор колонки для раскраски в зависимости от режима
    if mode == "Исходное":
        col = "provision_strong"
    elif mode == "Сценарий":
        col = "provision_strong_scenario"
    elif mode == "Изменение":
        col = "delta_provision"
    else:
        col = "provision_strong"

    # fallback, если колонки нет
    if col not in g.columns:
        for candidate in ("provision_strong", "provision_strong_scenario", "provision_strong_base"):
            if candidate in g.columns:
                col = candidate
                break
        else:
            col = None

    # диапазон значений для цветовой шкалы
    if col is not None:
        vals = pd.to_numeric(g[col], errors="coerce").dropna()
        if len(vals) > 0:
            vmin = float(vals.quantile(0.02))
            vmax = float(vals.quantile(0.98))
            if vmin == vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = 0.0, 1.0

    cmap = linear.YlGnBu_09.scale(vmin, vmax)
    cmap.caption = f"{col} ({fmt(vmin)} … {fmt(vmax)})" if col else "нет данных"

    def style_fn(feat):
        bid = feat["properties"].get("block_id")
        try:
            bid_int = int(bid)
        except Exception:
            bid_int = None
        val = feat["properties"].get(col) if col else None
        base_style = {
    "color": "#333333",     # тёмно-серая обводка
    "weight": 0.8,           # толщина линии
    "fillOpacity": 0.85,     # почти непрозрачная заливка
    "opacity": 1.0,
}
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            base_style["fillColor"] = "#cccccc"
        else:
            try:
                base_style["fillColor"] = cmap(float(val))
            except Exception:
                base_style["fillColor"] = "#cccccc"
        if bid_int in added_blocks:
            base_style["color"] = "#0057ff"
            base_style["weight"] = 1.5
        if selected_block is not None and bid_int == int(selected_block):
            base_style["color"] = "#ff0000"
            base_style["weight"] = 2.5
        return base_style

    # tooltip: только те поля, что реально есть в g
    desired_fields = [
        "block_id", "population", "school_age_population",
        "capacity_base", "capacity_added", "capacity_scenario",
        "demand", "provision_strong_base", "provision_strong_scenario",
        "delta_provision_strong", "demand_left_base", "demand_left_scenario",
    ]
    tooltip_fields = [f for f in desired_fields if f in g.columns]

    # базовая карта
    if len(g) > 0:
        minx, miny, maxx, maxy = g.total_bounds
        center = [(miny + maxy) / 2, (minx + maxx) / 2]
    else:
        center = [55.0, 73.4]

    m = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap")

    gj = folium.GeoJson(
        g,
        name="Кварталы",
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_fields,
            localize=True,
            sticky=False,
        ) if tooltip_fields else None,
        highlight_function=lambda x: {"weight": 2, "color": "#000000"},
    )
    gj.add_to(m)
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    return m


def render_block_card(comp_df: pd.DataFrame, block_id: int):
    row = comp_df[comp_df["block_id"] == int(block_id)]
    if row.empty:
        st.info("Квартал не найден.")
        return
    r = row.iloc[0]
    st.markdown(f"**Квартал (block_id): {int(block_id)}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Население", fmt(r["population"], 0))
    c1.metric("Школьный возраст (справочно)", fmt(r["school_age_population"], 0))
    c2.metric("Вместимость базовая", fmt(r["capacity_base"], 0))
    c2.metric("Добавлено мест", fmt(r["capacity_added"], 0))
    c2.metric("Вместимость итоговая", fmt(r["capacity_scenario"], 0))
    c3.metric("provision_strong (исх.)", fmt(r["provision_strong_base"], 4))
    c3.metric("provision_strong (сценарий)", fmt(r["provision_strong_scenario"], 4))
    c3.metric("Δ provision_strong", fmt(r["delta_provision_strong"], 4))
    st.caption(f"Спрос (demand, мест): {fmt(r['demand'],0)} | Дефицит исх.: {fmt(r['demand_left_base'],0)} | Дефицит сценарий: {fmt(r['demand_left_scenario'],0)}")


def export_csv_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


# ---------------------------- Основное приложение ---------------------------- #

def main():
    ensure_state()
    st.title("Оценка вариантов размещения школ (Омск)")
    st.caption("Практика 3 — интерактивное приложение. Исходные данные читаются только на чтение.")

    # Загрузка данных
    with st.spinner("Загрузка данных..."):
        ctx = get_data()
    blocks = ctx["blocks"]
    pc = ctx["pc"]
    baseline = ctx["baseline"]
    acc = ctx["matrix"]
    blocks_map = get_blocks_map_gdf(blocks)

    # Проверки/направление
    with st.expander("Целостность данных и направление матрицы", expanded=False):
        integ = validation.check_integrity(blocks, pc, baseline, acc)
        st.write(integ)
        st.json(ctx.get("direction", {}))
        st.markdown(
            "- Интерпретация матрицы: `acc[i,j]` — минуты от квартала **i (население/спрос)** к кварталу **j (объект/школа)**. "
            "Маска достижимости: `acc[i,j] <= ACCESSIBILITY_MIN`."
        )

    # Воспроизведение baseline (обязательно)
    with st.spinner("Воспроизведение baseline (без новых школ)..."):
        repro = get_baseline_reproduction(pc, acc, baseline)
    cmp_res = repro["compare"]
    st.subheader("1) Сходимость с baseline")
    c1, c2, c3 = st.columns(3)
    c1.metric("Макс. |Δ provision_strong|", fmt(cmp_res.get("max_abs_diff_provision_strong"), 8))
    c2.metric("Среднее |Δ provision_strong|", fmt(cmp_res.get("mean_abs_diff_provision_strong"), 8))
    c3.metric("Совпадение (< 1e-6)", "Да" if cmp_res.get("ok") else "Нет")
    st.caption(
        f"Сравнено кварталов с demand>0: {cmp_res.get('compared_blocks')} | "
        f"eps={cmp_res.get('eps')}. Расчёт: competitive_provision, "
        f"DEMAND_PER_1000={config.DEMAND_PER_1000}, ACCESSIBILITY_MIN={config.ACCESSIBILITY_MIN}, "
        f"self_supply={config.SELF_SUPPLY}, max_depth={config.MAX_DEPTH}."
    )

    # Текущий сценарий (результат)
    placements = st.session_state.placements
    if st.session_state.scenario_result is None or st.session_state.scenario_result.get("signature") != placements_signature(placements):
        # для отображения по умолчанию используем baseline-слой как "сценарий без добавлений"
        if placements_signature(placements) == placements_signature([]):
            # можно взять воспроизведённый/или baseline
            base_prov_for_view = repro["prov"]
            comp_df = provision_engine.per_block_comparison(pc, baseline, base_prov_for_view, placements=[])
            st.session_state.scenario_result = {
                "prov": base_prov_for_view,
                "metrics": provision_engine.compute_metrics(base_prov_for_view, baseline_prov=baseline),
                "compare_to_baseline": cmp_res,
                "comparison_df": comp_df,
                "placements": [],
                "signature": placements_signature([]),
            }
        else:
            # есть размещения, но не рассчитано — покажем baseline до расчёта
            base_prov_for_view = repro["prov"]
            comp_df = provision_engine.per_block_comparison(pc, baseline, base_prov_for_view, placements=placements)
            st.session_state.scenario_result = {
                "prov": base_prov_for_view,
                "metrics": provision_engine.compute_metrics(base_prov_for_view, baseline_prov=baseline),
                "compare_to_baseline": cmp_res,
                "comparison_df": comp_df,
                "placements": [dict(p) for p in placements],
                "signature": "pending:" + placements_signature(placements),
                "pending": True,
            }

    scen = st.session_state.scenario_result
    comp_df = scen["comparison_df"].copy()
    placements_agg = aggregate_placements(placements)

    # ---------------- 2) Ручной сценарий ---------------- #
    st.subheader("2) Ручной сценарий")
    colL, colR = st.columns([1.2, 1])
    with colL:
        st.markdown("**Добавление/удаление школы (клик по карте для справки + выбор block_id):**")
        bid_input = st.number_input("block_id квартала", min_value=0, max_value=config.N_BLOCKS-1, value=int(st.session_state.selected_block), step=1)
        cap_input = st.number_input("Места (положительное целое)", min_value=1, max_value=10000, value=int(config.DEFAULT_NEW_SCHOOL_CAPACITY), step=50)
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Добавить/обновить"):
            if cap_input <= 0:
                st.error("Число мест должно быть положительным целым.")
            else:
                # обновляем или добавляем
                updated = False
                new_pl = []
                for p in placements:
                    if int(p["block_id"]) == int(bid_input):
                        new_pl.append({"block_id": int(bid_input), "added_capacity": int(cap_input), "source": "manual", "label": f"manual_{bid_input}"})
                        updated = True
                    else:
                        new_pl.append(p)
                if not updated:
                    new_pl.append({"block_id": int(bid_input), "added_capacity": int(cap_input), "source": "manual", "label": f"manual_{bid_input}"})
                st.session_state.placements = new_pl
                st.session_state.scenario_result = None
                st.rerun()
        if b2.button("Удалить выбранный"):
            st.session_state.placements = [p for p in placements if int(p["block_id"]) != int(bid_input)]
            st.session_state.scenario_result = None
            st.rerun()
        if b3.button("Рассчитать сценарий", type="primary"):
            chk = validation.check_scenario(st.session_state.placements)
            if not chk["ok"]:
                st.error("Ошибки сценария: " + "; ".join(chk["errors"]))
            else:
                with st.spinner("Пересчёт обеспеченности по всей территории..."):
                    run_and_store_scenario(ctx, st.session_state.placements)
                st.success("Сценарий рассчитан.")
                st.rerun()
        if b4.button("Сброс сценария"):
            st.session_state.placements = []
            st.session_state.scenario_result = None
            st.session_state.autopick_result = None
            st.session_state.comparison_results = None
            st.rerun()

        st.write(f"Текущие размещения: {len(st.session_state.placements)}")
        if st.session_state.placements:
            st.dataframe(pd.DataFrame(st.session_state.placements), use_container_width=True)

    with colR:
        st.markdown("**Карточка квартала**")
        # пробуем взять из клика карты позже; пока по вводу
        st.session_state.selected_block = int(bid_input)
        render_block_card(comp_df, int(bid_input))

    # ---------------- 3) Автоподбор ---------------- #
    st.subheader("3) Автоподбор (1 школа)")
    a1, a2, a3, a4 = st.columns(4)
    auto_cap = a1.number_input("Вместимость новой школы, мест", min_value=1, max_value=10000, value=int(config.DEFAULT_NEW_SCHOOL_CAPACITY), step=50, key="auto_cap")
    k_top = a2.number_input("K топ для точной проверки", min_value=1, max_value=config.N_BLOCKS, value=int(config.K_TOP_DEFAULT), step=1)
    run_auto = a3.button("Разместить автоматически")
    run_full = a4.button("Долгий режим: полный перебор")

    if run_auto:
        prog = st.progress(0.0, text="Подготовка...")
        status = st.empty()
        def cb(done, total, phase):
            frac = 0.0 if total <= 0 else min(1.0, done / total)
            prog.progress(frac, text=phase)
            status.write(f"Прогресс: {phase} — проверено {done} из {total}")
        with st.spinner("Автоподбор: прокси + точная проверка топ-K..."):
            res = autopick.run_autopick_topk(
                pc, acc, baseline,
                added_capacity=int(auto_cap), k=int(k_top),
                accessibility=config.ACCESSIBILITY_MIN, progress_cb=cb, save_intermediate=True,
            )
        st.session_state.autopick_result = res
        prog.progress(1.0, text="Готово")
        status.write(res.get("mode_info", "Готово"))
        st.success(f"Готово. {res.get('mode_info')} Результат сохранён: {res.get('result_path')}")

    if run_full:
        st.warning("Полный точный перебор 5643 кварталов может занять длительное время.")
        if st.checkbox("Подтверждаю запуск полного перебора", key="confirm_full"):
            prog = st.progress(0.0, text="Запуск полного перебора...")
            status = st.empty()
            def cb(done, total, phase):
                frac = 0.0 if total <= 0 else min(1.0, done / total)
                prog.progress(frac, text=phase)
                status.write(f"Прогресс: {phase} — проверено {done} из {total}")
            with st.spinner("Полный перебор (точно для всех 5643)..."):
                res = autopick.run_full_exhaustive(
                    pc, acc, baseline,
                    added_capacity=int(auto_cap), accessibility=config.ACCESSIBILITY_MIN, progress_cb=cb,
                )
            st.session_state.autopick_result = res
            prog.progress(1.0, text="Полный перебор завершён")
            st.success(f"Готово. Результат сохранён: {res.get('result_path')}")

    if st.session_state.autopick_result is not None:
        ar = st.session_state.autopick_result
        st.info(
            f"{ar.get('mode_info','')} "
            "Полный точный перебор всех 5643 невозможен по приемлемому времени (в быстром режиме); "
            "использован прокси по всем кварталам с точной проверкой топ-K. "
            "Долгий режим выполняет точный перебор всех 5643 с сохранением в файл."
        )
        best = ar.get("best", pd.DataFrame())
        if best is not None and not best.empty:
            st.markdown("**Лучшие (точно), включая все равные:**")
            st.dataframe(best, use_container_width=True)
            if st.button("Применить лучший автоподбор как сценарий"):
                row = best.iloc[0]
                st.session_state.placements = [{
                    "block_id": int(row["block_id"]),
                    "added_capacity": int(row.get("added_capacity", auto_cap)),
                    "source": "auto",
                    "label": "auto_best",
                }]
                st.session_state.scenario_result = None
                st.rerun()

        table = ar.get("table", pd.DataFrame()).copy()
        if not table.empty:
            # сортировка по убыванию сокращения sum(demand_left) (точные выше), при равенстве — все видны
            st.markdown("**Все варианты (все 5643), сортировка: сокращение дефицита (точно) ↓**")
            show_cols = [c for c in [
                "block_id", "mode", "added_capacity",
                "exact_deficit_reduction", "exact_sum_demand_left",
                "exact_weighted_provision_strong", "exact_mean_provision_strong", "exact_count_lt1",
                "proxy_gain", "proxy_reachable_unmet", "rank_proxy",
            ] if c in table.columns]
            st.dataframe(table[show_cols], use_container_width=True, height=350)
            export_csv_button(table, "Экспорт таблицы автоподбора (CSV)", f"autopick_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # ---------------- 4) Карта ---------------- #
    st.subheader("4) Карта")
    mode = st.radio("Режим карты", ["Исходное", "Сценарий", "Изменение"], horizontal=True)
    # если сценарий не пересчитан после изменения placements — предупредим
    if st.session_state.scenario_result and st.session_state.scenario_result.get("pending"):
        st.warning("Размещения изменены — нажмите «Рассчитать сценарий» для пересчёта по всей территории. На карте/в метриках показан baseline до пересчёта.")

    # актуализируем comp_df с текущими placements для отображения added
    if st.session_state.scenario_result and not st.session_state.scenario_result.get("pending"):
        comp_df = st.session_state.scenario_result["comparison_df"]
    else:
        comp_df = provision_engine.per_block_comparison(pc, baseline, repro["prov"], placements=placements)

    m = build_map(blocks_map, comp_df, mode, placements_agg, st.session_state.get("selected_block"))
    map_out = st_folium(m, width=None, height=600, key=f"map_{mode}_{len(placements)}")
    st.caption("Легенда: цвет — значение показателя (доля 0–1 для provision_strong; изменение доли для Δ). Синий контур — кварталы с добавленной школой, красный — выбранный.")

    # Клик по карте → выбор квартала (добавление/удаление выполняется кнопками выше)
    if map_out and map_out.get("last_clicked"):
        try:
            lat = float(map_out["last_clicked"]["lat"])
            lng = float(map_out["last_clicked"]["lng"])
            pt = Point(lng, lat)
            hit = blocks_map[blocks_map.geometry.contains(pt)]
            if not hit.empty:
                clicked_bid = int(hit.iloc[0]["block_id"])
                if clicked_bid != int(st.session_state.get("selected_block", -1)):
                    st.session_state.selected_block = clicked_bid
                    st.info(f"Выбран квартал по клику: block_id={clicked_bid}. Укажите число мест и нажмите «Добавить/обновить» или «Удалить выбранный».")
                    st.rerun()
        except Exception:
            pass

    # ---------------- 5) и 6) Метрики и таблица ---------------- #
    st.subheader("5-6) Итоговые метрики и сравнение")
    scen_res = st.session_state.scenario_result
    if scen_res:
        met = scen_res.get("metrics", {})
        m1, m2, m3 = st.columns(3)
        m1.metric("Прирост вместимости (мест)", fmt(met.get("capacity_increase", 0), 0))
        m2.metric("Остаточный дефицит (sum demand_left)", fmt(met.get("sum_demand_left", 0), 0))
        m3.metric("Сокращение дефицита", fmt(met.get("deficit_reduction_sum_demand_left", 0), 0))
        m4, m5, m6 = st.columns(3)
        m4.metric("Средний provision_strong (demand>0)", fmt(met.get("mean_provision_strong_demand_gt0"), 4))
        m5.metric("Взвешенный provision_strong", fmt(met.get("weighted_provision_strong"), 4))
        m6.metric("Кварталов с provision_strong < 1 (demand>0)", fmt(met.get("count_provision_lt1_demand_gt0"), 0))

    st.subheader("7) Таблица по кварталам (base vs scenario)")
    table_df = comp_df.copy()
    st.dataframe(table_df, use_container_width=True, height=350)
    export_csv_button(table_df, "Экспорт таблицы по кварталам (CSV)", f"blocks_base_vs_scenario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # ---------------- 9) Сравнение сценариев А, Б, В ---------------- #
    st.subheader("9) Сравнение сценариев А (1×600), Б (2×300), В (авто 600)")
    st.caption(
        "А: авто-лучшее размещение 1 школы × 600 мест. "
        "Б: 2 школы × 300 мест — два лучших квартала по точной оценке автоподбора для 300 мест (одновременное размещение). "
        "В: авто 600 мест (эквивалент А, отдельный расчёт)."
    )
    if st.button("Рассчитать сравнение А/Б/В"):
        with st.spinner("Расчёт сценариев сравнения (несколько прогонов competitive_provision)..."):
            # A/C: авто 1x600
            res600 = autopick.run_autopick_topk(pc, acc, baseline, added_capacity=600, k=min(20, config.N_BLOCKS), save_intermediate=True)
            best600 = res600["best"]
            if not best600.empty:
                pA = [{"block_id": int(best600.iloc[0]["block_id"]), "added_capacity": 600, "source": "auto", "label": "A_1x600"}]
            else:
                pA = []
            scA = provision_engine.run_scenario(pc, acc, placements=pA)
            metA = provision_engine.compute_metrics(scA["prov"], baseline_prov=baseline)
            # B: 2x300
            res300 = autopick.run_autopick_topk(pc, acc, baseline, added_capacity=300, k=min(30, config.N_BLOCKS), save_intermediate=True)
            t300 = res300["table"]
            exact300 = t300[t300["mode"] == "точно"].sort_values("exact_deficit_reduction", ascending=False)
            bids_b = []
            for _, r in exact300.iterrows():
                bids_b.append(int(r["block_id"]))
                if len(bids_b) >= 2:
                    break
            # если меньше 2 точных — доберём по прокси
            if len(bids_b) < 2:
                for _, r in t300.sort_values("proxy_gain", ascending=False).iterrows():
                    b = int(r["block_id"])
                    if b not in bids_b:
                        bids_b.append(b)
                    if len(bids_b) >= 2:
                        break
            pB = [{"block_id": b, "added_capacity": 300, "source": "auto", "label": "B_2x300"} for b in bids_b[:2]]
            scB = provision_engine.run_scenario(pc, acc, placements=pB)
            metB = provision_engine.compute_metrics(scB["prov"], baseline_prov=baseline)
            # C == A
            pC = list(pA)
            scC = provision_engine.run_scenario(pc, acc, placements=pC)
            metC = provision_engine.compute_metrics(scC["prov"], baseline_prov=baseline)

            rows = []
            for name, pls, met in [("A: 1×600 (авто)", pA, metA), ("B: 2×300 (авто)", pB, metB), ("C: авто 600", pC, metC)]:
                rows.append({
                    "Сценарий": name,
                    "Размещения (block_id:мест)": ", ".join([f"{p['block_id']}:{p['added_capacity']}" for p in pls]) or "—",
                    "Прирост вместимости": met.get("capacity_increase", 0),
                    "Остаточный дефицит": met.get("sum_demand_left", 0),
                    "Сокращение дефицита": met.get("deficit_reduction_sum_demand_left", 0),
                    "Взвеш. provision_strong": met.get("weighted_provision_strong", np.nan),
                    "Средн. provision_strong (d>0)": met.get("mean_provision_strong_demand_gt0", np.nan),
                    "Кварталов с prov<1 (d>0)": met.get("count_provision_lt1_demand_gt0", 0),
                })
            st.session_state.comparison_results = {
                "table": pd.DataFrame(rows),
                "A": {"placements": pA, "metrics": metA},
                "B": {"placements": pB, "metrics": metB},
                "C": {"placements": pC, "metrics": metC},
            }
            provision_engine.cleanup(scA, scB, scC, res600, res300)
        st.success("Сравнение рассчитано.")

    if st.session_state.comparison_results:
        st.dataframe(st.session_state.comparison_results["table"], use_container_width=True)
        export_csv_button(st.session_state.comparison_results["table"], "Экспорт сравнения А/Б/В (CSV)", f"comparison_ABC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    # ---------------- 10) Источники и допущения ---------------- #
    st.subheader("10) Источники и допущения")
    st.markdown(
        """
- **Вместимость (baseline):** восстановлена нормативно (550 мест/школа, агрегировано по кварталам) — это оценка, не фактические данные OSM; в интерфейсе и расчётах учитывается как исходная `capacity`. Новые школы по умолчанию — 600 мест.
- **Население/спрос:** `population` — модельная оценка; спрос рассчитан строго как `ceil(population * 120 / 1000)` (DEMAND_PER_1000=120), как в baseline. `school_age_population` приведён справочно и в расчёт спроса не входит.
- **Обеспеченность:** метод `competitive_provision` (BlocksNet 1.0.0a9), `ACCESSIBILITY_MIN=20` мин, `self_supply=True`, `max_depth=1`; матрица доступности (drive, минуты) читается в `float32`.
- **Автоподбор (1 школа):** полный точный перебор 5643 вариантов по времени непрактичен в быстром режиме — используется прокси-покрытие неудовлетворённого спроса в зоне доступности по всем 5643 кварталам + точный `competitive_provision` для топ-K (настраивается). В таблице все варианты помечены `точно/прокси`; равные лучшие показываются все. Доступен «долгий режим: полный перебор» с прогрессом и сохранением в файл.
- **Направление матрицы:** `acc[i,j]` — время от квартала `i` (спрос) к кварталу `j` (объект), проверяется при старте и зафиксировано в README.
- **Ограничения:** после каждого точного расчёта выполняется очистка памяти (`del` + `gc.collect()`), промежуточные батч-результаты сохраняются на диск.
        """
    )

    st.caption(f"Экспорт сценариев/результатов: {config.EXPORTS_DIR}")


if __name__ == "__main__":
    main()
