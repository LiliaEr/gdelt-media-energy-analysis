-- BigQuery / GoogleSQL. 
-- Purpose: document the existing 2022-2024 CSVs; 
-- Its daily grain is inferred: the supplied 18-06-2025-1.sql has geographic grouping.
-- Legacy sanctions/conflict regexes depend on order within V2Themes;
-- they are retained for historical consistency, not endorsed as validated measures.
-- Tone describes the whole document. Topics overlap.
-- Geographic COUNT(*) counts expanded location entries, not unique articles.
-- The original geographic regex is retained, including its extraction limitations.

WITH daily_source AS (
  SELECT
    PARSE_DATE('%Y%m%d', SUBSTR(CAST(date AS STRING), 1, 8)) AS day,
    V2Themes,
    V2Tone
  FROM `gdelt-bq.gdeltv2.gkg`
  WHERE
    SUBSTR(CAST(date AS STRING), 1, 8) BETWEEN '20220101' AND '20241231'
    AND V2Tone IS NOT NULL
    AND (
      REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER|ENV_NATURALGAS|ENV_OIL|ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE|ENV_ENERGY_SECURITY|EPU_POLICY_ENERGY|ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')
      OR REGEXP_CONTAINS(V2Locations, r'(?i)Nord[\s\-]?Stream')
    )
)
SELECT
  FORMAT_DATE('%d/%m/%Y', day) AS date,

  -- Нефть и газ (без отдельного фильтра по России):
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_NATURALGAS|ENV_OIL')) AS fossil_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_NATURALGAS|ENV_OIL')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS fossil_energy_tone,

  -- Ядерная энергетика:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER')) AS nuclear_power_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS nuclear_power_tone,

  -- Зелёная / возобновляемая энергия:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE')) AS green_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS green_energy_tone,

  -- Климат и политика:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')) AS climate_policy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS climate_policy_tone,

  -- Санкции и энергетика:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(?i)(sanctions|embargo).*(Russia|gas|energy|oil)')) AS sanctions_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'(?i)(sanctions|embargo).*(Russia|gas|energy|oil)')
           THEN SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) END) AS sanctions_energy_tone,

  -- Конфликты и энергетика:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(?i)(Ukraine|war|invasion|conflict).*energy')) AS conflict_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'(?i)(Ukraine|war|invasion|conflict).*energy')
           THEN SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) END) AS conflict_energy_tone

FROM daily_source
GROUP BY day
ORDER BY day;

-- Result 2: country.csv / country_article_counts.csv (original SQL).
SELECT
  REGEXP_EXTRACT(location, r'#[0-9]+#([A-Z]{2})#') AS country_code,
  COUNT(*) AS article_count
FROM `gdelt-bq.gdeltv2.gkg`,
UNNEST(SPLIT(V2Locations, ';')) AS location
WHERE
  SUBSTR(CAST(date AS STRING),1,8) BETWEEN '20220101' AND '20241231'
  AND V2Tone IS NOT NULL
  AND (
    REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER|ENV_NATURALGAS|ENV_OIL|ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE|ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY|(sanctions|embargo).*(Russia|gas|energy|oil)|(Ukraine|war|invasion|conflict).*energy')
    OR REGEXP_CONTAINS(V2Locations, r'(?i)Nord[\s\-]?Stream')
  )
  AND REGEXP_CONTAINS(location, r'#[0-9]+#[A-Z]{2}#')
GROUP BY country_code
ORDER BY article_count DESC;

