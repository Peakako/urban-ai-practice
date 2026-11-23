# NOTES — baseline обеспеченности школами (Омск)

## ID
block_id = 0..5642 = порядок features в data/omsk_blocks.geojson (5643 квартала)

## Население
- Файл: data/omsk_population.geojson
- Источник: <WorldPop 2020 constrained | вариант D: равномерно по площади>
- SCHOOL_AGE_SHARE = 0.12 (допущение, 7–17 лет)
- pop_total = <вставьте число>, school_age_total = <вставьте число>

## Школы
- Слой: data/omsk_schools_raw.geojson (источник: <overpass-turbo | Geofabrik>, дата: <...>)
- Агрегат: data/omsk_schools_capacity.geojson
- Вместимость восстановлена нормативно: 550 мест/школа (capacity_is_estimated=true)
- total schools = <...>, total capacity = <...>

## Обеспеченность baseline
- Метод: blocksnet.analysis.provision.competitive_provision (BlocksNet 1.0.0a9)
- demand = 120 мест на 1000 жителей (норматив, спрос рассчитан из population)
- ACCESSIBILITY_MIN = 20 мин (обоснование: доля пар в пределах порога = <...>)
- self_supply = True, max_depth = 1
- Вход: population, capacity (по кварталам)
- Выход: output/omsk_provision_baseline.geojson (+ links.csv)