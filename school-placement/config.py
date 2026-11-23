"""Конфигурация приложения размещения школ (Омск, Практика 3)."""

from __future__ import annotations

from pathlib import Path

# Корень проекта ТИИ (parent папки school-placement)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
APP_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = APP_DIR / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Исходные данные (только чтение)
BLOCKS_PATH = DATA_DIR / "omsk_blocks.geojson"
POPULATION_PATH = DATA_DIR / "omsk_population.geojson"
SCHOOLS_CAPACITY_PATH = DATA_DIR / "omsk_schools_capacity.geojson"
MATRIX_PATH = OUTPUT_DIR / "omsk_accessibility_drive_minutes.csv"
BASELINE_PATH = OUTPUT_DIR / "omsk_provision_baseline.geojson"
INDICATORS_PATH = OUTPUT_DIR / "omsk_blocks_with_indicators.geojson"
NOTES_PATH = PROJECT_ROOT / "NOTES.md"

# Параметры расчёта (baseline-совместимо, см. NOTES.md)
DEMAND_PER_1000 = 120  # мест на 1000 жителей, строго как в baseline
ACCESSIBILITY_MIN = 20  # минут (drive), порог доступности из NOTES.md
SELF_SUPPLY = True
MAX_DEPTH = 1

# Вместимости
DEFAULT_NEW_SCHOOL_CAPACITY = 600  # мест для новой/добавляемой школы (задание)
BASELINE_EXISTING_CAPACITY_NORM = 550  # норматив для восстановленной вместимости существующих школ

# Автоподбор
K_TOP_DEFAULT = 50
N_BLOCKS = 5643

# Точность сравнения
EPS = 1e-6

# Отображение
MAP_CENTER = [54.9893, 73.3682]
MAP_ZOOM = 11
PROVISION_COL = "provision_strong"
