-- BigQuery / GoogleSQL.
-- Purpose: queries for 2022-2024 exports;
-- Legacy sanctions/conflict regexes depend on order within V2Themes;

-- Validation of these topic definitions requires source documents.
-- Tone describes the whole document. Topics overlap.
-- Geographic COUNT(*) counts expanded location entries, not unique articles.
-- Country codes are extracted by field position in V2Locations.

-- Both results use the same document population. These corrected queries are
CREATE TEMP TABLE energy_source AS
SELECT
  PARSE_DATE('%Y%m%d', SUBSTR(CAST(date AS STRING), 1, 8)) AS day,
  V2Themes,
  V2Tone,
  V2Locations
FROM `gdelt-bq.gdeltv2.gkg`
WHERE
  SUBSTR(CAST(date AS STRING), 1, 8) BETWEEN '20220101' AND '20241231'
  AND V2Tone IS NOT NULL
  AND (
    REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER|ENV_NATURALGAS|ENV_OIL|ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE|ENV_ENERGY_SECURITY|EPU_POLICY_ENERGY|ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')
    OR REGEXP_CONTAINS(V2Themes, r'(?i)(sanctions|embargo).*(Russia|gas|energy|oil)')
    OR REGEXP_CONTAINS(V2Themes, r'(?i)(Ukraine|war|invasion|conflict).*energy')
    OR REGEXP_CONTAINS(V2Locations, r'(?i)Nord[\s\-]?Stream')
  );

SELECT
  FORMAT_DATE('%d/%m/%Y', day) AS date,

  -- Oil and gas (without a separate Russia filter):
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_NATURALGAS|ENV_OIL')) AS fossil_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_NATURALGAS|ENV_OIL')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS fossil_energy_tone,

  -- Nuclear power:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER')) AS nuclear_power_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_NUCLEARPOWER')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS nuclear_power_tone,

  -- Green / renewable energy:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE')) AS green_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_GREEN|ENV_CLEANENERGY|ENV_RENEWABLE')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS green_energy_tone,

  -- Climate and policy:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')) AS climate_policy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'ENV_CLIMATECHANGE|ENV_ENVIRONMENTAL_POLICY')
           THEN SAFE_CAST(SPLIT(V2Tone,',')[OFFSET(0)] AS FLOAT64) END) AS climate_policy_tone,

  -- Sanctions and energy:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(?i)(sanctions|embargo).*(Russia|gas|energy|oil)')) AS sanctions_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'(?i)(sanctions|embargo).*(Russia|gas|energy|oil)')
           THEN SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) END) AS sanctions_energy_tone,

  -- Conflict and energy:
  COUNTIF(REGEXP_CONTAINS(V2Themes, r'(?i)(Ukraine|war|invasion|conflict).*energy')) AS conflict_energy_count,
  AVG(CASE WHEN REGEXP_CONTAINS(V2Themes, r'(?i)(Ukraine|war|invasion|conflict).*energy')
           THEN SAFE_CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) END) AS conflict_energy_tone

FROM energy_source
GROUP BY day
ORDER BY day;

-- Result 2: location mentions by FIPS country code (not unique articles).
WITH location_entries AS (
  SELECT SPLIT(location, '#')[SAFE_OFFSET(2)] AS country_code
  FROM energy_source
  CROSS JOIN UNNEST(SPLIT(V2Locations, ';')) AS location
)
SELECT country_code, COUNT(*) AS article_count
FROM location_entries
WHERE REGEXP_CONTAINS(country_code, r'^[A-Z]{2}$')
GROUP BY country_code
ORDER BY article_count DESC;
