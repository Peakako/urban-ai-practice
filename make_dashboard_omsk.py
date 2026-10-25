#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_dashboard_omsk.py
Создаёт один HTML-файл dashboard.html с интерактивной картой, графиками, таблицами и выводами
на основе уже готовых результатов расчётов BlocksNet для Омска.

Файл НЕ скачивает данные и НЕ повторяет расчёты.
Использует только готовые файлы:
- output/omsk_blocks_with_indicators.geojson
- output/omsk_accessibility_drive_minutes.csv (опционально)
- data/omsk_blocks.geojson (опционально)

Запуск (Windows PowerShell):
    cd C:\\Users\\xoxom\\Desktop\\ТИИ
    .\.venv\Scripts\Activate.ps1
    python make_dashboard_omsk.py

Результат:
    dashboard.html в корне проекта
"""

import json
from pathlib import Path
import math

import numpy as np
import pandas as pd
import geopandas as gpd

import folium
from folium import plugins
from branca.colormap import linear
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Пути
PROJECT_ROOT = Path(r"C:\Users\xoxom\Desktop\ТИИ")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

BLOCKS_IND_PATH = OUTPUT_DIR / "omsk_blocks_with_indicators.geojson"
ACCESS_PATH = OUTPUT_DIR / "omsk_accessibility_drive_minutes.csv"
BLOCKS_PATH = DATA_DIR / "omsk_blocks.geojson"

DASHBOARD_PATH = PROJECT_ROOT / "dashboard.html"

def main():
    print("Загрузка готовых данных...")
    if not BLOCKS_IND_PATH.exists():
        raise FileNotFoundError(f"Не найден {BLOCKS_IND_PATH}. Сначала нужно выполнить расчёты в ноутбуке.")
    
    gdf = gpd.read_file(BLOCKS_IND_PATH)
    print(f"Загружен {BLOCKS_IND_PATH} | Строк: {len(gdf)} | Колонок: {len(gdf.columns)}")
    
    # Копия для карты в WGS84
    gdf_map = gdf.copy()
    try:
        if gdf_map.crs is not None or gdf_map.crs.to_epsg() != 4326:
            gdf_map = gdf_map.to_crs(epsg=4326)
    except Exception as e:
        print("Не удалось привести к EPSG:4326:", e)
    
    # Числовые колонки для графиков
    num_cols = []
    for c in gdf.columns:
        if c == "geometry":
            continue
        try:
            if pd.api.types.is_numeric_dtype(gdf[c]):
                # Пропускаем ID-подобные
                if gdf[c].nunique() > 1:
                    num_cols.append(c)
        except Exception:
            pass
    
    print(f"Найдено числовых показателей: {len(num_cols)}")
    
    # Создаём карту
    print("Построение карты...")
    m = folium.Map(location=[54.9893, 73.3682], zoom_start=10, tiles="CartoDB positron")
    
    # Выбираем колонку для отображения по умолчанию
    map_col = None
    for cand in ["acc_mean_min", "provision", "school_provision", "far", "FAR", "density"]:
        if cand in gdf_map.columns:
            map_col = cand
            break
    if map_col is None and num_cols:
        map_col = num_cols[0]
    
    if map_col:
        try:
            col_vals = gdf_map[map_col].dropna()
            if len(col_vals) > 0:
                vmin, vmax = col_vals.quantile(0.02), col_vals.quantile(0.98)
                if vmin == vmax:
                    vmin, vmax = col_vals.min(), col_vals.max()
                cmap = linear.YlOrRd_09.scale(vmin, vmax)
                cmap.caption = map_col
                
                def style_fn(feat):
                    val = feat['properties'].get(map_col)
                    if val is None or (isinstance(val, float) and (math.isnan(val))):
                        return {'fillColor': '#cccccc', 'color': '#666666', 'weight': 0.5, 'fillOpacity': 0.4}
                    try:
                        return {'fillColor': cmap(val), 'color': '#333333', 'weight': 0.5, 'fillOpacity': 0.7}
                    except Exception:
                        return {'fillColor': '#cccccc', 'color': '#333333', 'weight': 0.5, 'fillOpacity': 0.5}
                
                # Popup с топ-показателями
                popup_cols = [map_col] + num_cols[:12]
                popup_cols = list(dict.fromkeys(popup_cols))
                
                def popup_fn(feat):
                    props = feat['properties']
                    rows = []
                    for c in popup_cols:
                        if c in props and props[c] is not None:
                            rows.append(f"<b>{c}</b>: {props[c]}")
                    html = "<div style='max-width:250px; font-size:12px;'>" + "<br>".join(rows) + "</div>"
                    return folium.Popup(html, max_width=250)
                
                folium.GeoJson(
                    gdf_map,
                    style_function=style_fn,
                    tooltip=folium.GeoJsonTooltip(fields=popup_cols[:8], aliases=popup_cols[:8], localize=True),
                    popup=folium.GeoJsonTooltip(fields=popup_cols, aliases=popup_cols, localize=True)
                ).add_to(m)
                cmap.add_to(m)
        except Exception as e:
            print("Ошибка при построении слоя карты:", e)
            folium.GeoJson(gdf_map).add_to(m)
    else:
        folium.GeoJson(gdf_map).add_to(m)
    
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl().add_to(m)
    
    map_html = m.get_root().render()
    
    # Графики
    print("Построение графиков...")
    figures_html = []
    
    # 1) Распределение выбранных показателей
    plot_cols = num_cols[:6]
    if plot_cols:
        try:
            fig1 = make_subplots(rows=2, cols=3, subplot_titles=plot_cols)
            for i, c in enumerate(plot_cols):
                row = i // 3 + 1
                col = i % 3 + 1
                s = gdf[c].dropna()
                fig1.add_trace(go.Histogram(x=s, name=c, nbinsx=20), row=row, col=col)
            fig1.update_layout(title_text="Распределение показателей по кварталам", height=600, showlegend=False)
            figures_html.append(fig1.to_html(full_html=False, include_plotlyjs='cdn'))
        except Exception as e:
            print("Ошибка графика 1:", e)
    
    # 2) Корреляция
    if len(num_cols) >= 2:
        try:
            corr_df = gdf[num_cols].corr()
            fig2 = px.imshow(corr_df, text_auto=True, aspect="auto", title="Корреляция показателей")
            fig2.update_layout(height=700)
            figures_html.append(fig2.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            print("Ошибка графика 2:", e)
    
    # 3) Топ по доступности/обеспеченности
    try:
        sort_col = None
        for cand in ["acc_mean_min", "provision", "school_provision"]:
            if cand in gdf.columns:
                sort_col = cand
                break
        if sort_col:
            df_top = gdf[[sort_col]].copy()
            df_top = df_top.dropna().sort_values(sort_col, ascending=(sort_col == "acc_mean_min"))
            df_top = df_top.head(15)
            fig3 = px.bar(df_top, y=df_top.index.astype(str), x=sort_col, orientation='h', title=f"Топ-15 кварталов по {sort_col}")
            figures_html.append(fig3.to_html(full_html=False, include_plotlyjs=False))
    except Exception as e:
        print("Ошибка графика 3:", e)
    
    # Таблица
    print("Построение таблицы...")
    table_df = gdf.drop(columns=['geometry']).copy()
    # Ограничим для отображения
    display_cols = num_cols[:30]
    if display_cols:
        table_df_disp = table_df[display_cols].copy()
    else:
        table_df_disp = table_df.copy()
    
    table_html = table_df_disp.head(100).to_html(classes='table table-striped table-hover table-sm', index=True, border=0)
    
    # Выводы
    print("Формирование выводов...")
    conclusions = []
    conclusions.append("<h2>Выводы</h2>")
    conclusions.append("<ul>")
    conclusions.append(f"<li>Всего проанализировано кварталов: <b>{len(gdf)}</b></li>")
    if map_col and map_col in gdf.columns:
        conclusions.append(f"<li>Основной показатель на карте: <b>{map_col}</b></li>")
    if "acc_mean_min" in gdf.columns:
        conclusions.append(f"<li>Среднее время доступности по кварталам (drive): <b>{gdf['acc_mean_min'].mean():.1f}</b> минут</li>")
    if "school_provision" in gdf.columns:
        below = (gdf['school_provision'] < 1).sum()
        conclusions.append(f"<li>Кварталов с обеспеченностью школами < 1: <b>{below}</b> из {len(gdf)}</li>")
    elif "provision" in gdf.columns:
        below = (gdf['provision'] < 1).sum()
        conclusions.append(f"<li>Кварталов с обеспеченностью < 1: <b>{below}</b> из {len(gdf)}</li>")
    conclusions.append("<li>Все данные взяты из готовых файлов расчёта BlocksNet (повторный расчёт не выполнялся).</li>")
    conclusions.append("</ul>")
    conclusions_html = "\n".join(conclusions)
    
    # Собираем HTML
    print("Сборка dashboard.html...")
    html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>BlocksNet Dashboard | Омск</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: Arial, sans-serif; background: #f8f9fa; }}
        .header {{ background: linear-gradient(90deg, #2c3e50, #3498db); color: white; padding: 20px; }}
        .card {{ margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .map-container {{ height: 600px; }}
        .table-container {{ overflow-x: auto; max-height: 500px; overflow-y: auto; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>BlocksNet - Интерактивный дашборд (Омск)</h1>
        <p>Карта, графики, таблицы и выводы по готовым результатам расчёта</p>
    </div>
    
    <div class="container-fluid mt-4">
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header"><h5>Интерактивная карта кварталов</h5></div>
                    <div class="card-body map-container">
                        {map_html}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header"><h5>Графики и распределения</h5></div>
                    <div class="card-body">
                        {"".join(figures_html)}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header"><h5>Таблица показателей (первые 100 строк)</h5></div>
                    <div class="card-body table-container">
                        {table_html}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-body">
                        {conclusions_html}
                        <p><small>Источник данных: {BLOCKS_IND_PATH}</small></p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    with open(DASHBOARD_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Готово! Создан: {DASHBOARD_PATH}")
    print("Откройте файл dashboard.html в браузере.")

if __name__ == '__main__':
    main()
