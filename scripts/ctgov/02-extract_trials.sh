#!/usr/bin/env bash

LIMIT=100

# Extract COVID-19 related trials
SIMILAR="SIMILAR"

QUERY="
  SELECT
      t1.nct_id AS \"#nct_id\",
      t1.brief_title AS title,
      CASE WHEN t2.has_us_facility THEN 'true' ELSE 'false' END AS has_us_facility,
      t3.conditions,
      t4.criteria AS eligibility_criteria
  FROM studies t1
  JOIN calculated_values t2
      ON t1.nct_id = t2.nct_id
  JOIN (
      SELECT
          nct_id,
          STRING_AGG(name, '|' ORDER BY name) AS conditions
      FROM conditions
      GROUP BY
          nct_id
  ) t3
      ON t1.nct_id = t3.nct_id
  JOIN eligibilities t4
      ON t1.nct_id = t4.nct_id
  WHERE
      LOWER(conditions) ${SIMILAR} TO '%([^a-z]cov[^a-z]|corona[ v]|covid)%'
      AND t1.study_type = 'Interventional'
      AND t1.overall_status = 'Recruiting'
      AND RANDOM() < 0.2
  ORDER BY
      t1.nct_id DESC
  LIMIT
      ${LIMIT}"


psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "\copy (${QUERY}) to '/data/ctgov/extraction/clinical_trials_similar.csv' with header csv"

# Extract non-COVID-19 related trials
SIMILAR="NOT SIMILAR"


QUERY="
  SELECT
      t1.nct_id AS \"#nct_id\",
      t1.brief_title AS title,
      CASE WHEN t2.has_us_facility THEN 'true' ELSE 'false' END AS has_us_facility,
      t3.conditions,
      t4.criteria AS eligibility_criteria
  FROM studies t1
  JOIN calculated_values t2
      ON t1.nct_id = t2.nct_id
  JOIN (
      SELECT
          nct_id,
          STRING_AGG(name, '|' ORDER BY name) AS conditions
      FROM conditions
      GROUP BY
          nct_id
  ) t3
      ON t1.nct_id = t3.nct_id
  JOIN eligibilities t4
      ON t1.nct_id = t4.nct_id
  WHERE
      LOWER(conditions) ${SIMILAR} TO '%([^a-z]cov[^a-z]|corona[ v]|covid)%'
      AND t1.study_type = 'Interventional'
      AND t1.overall_status = 'Recruiting'
      AND RANDOM() < 0.2
  ORDER BY
      t1.nct_id DESC
  LIMIT
      ${LIMIT}"


psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "\copy (${QUERY}) to '/data/ctgov/extraction/clinical_trials_not_similar.csv' with header csv"
