-- ============================================================
-- WORLD BANK SANITATION PROJECT
-- SQL ANALYSIS QUERIES
-- Indicator: SH.STA.BASS.ZS
-- ============================================================


-- ============================================================
-- QUERY 1: View all sanitation data
-- ============================================================

SELECT *
FROM sanitation_data;


-- ============================================================
-- QUERY 2: Count total records
-- ============================================================

SELECT COUNT(*) AS total_records
FROM sanitation_data;


-- ============================================================
-- QUERY 3: Count unique countries/regions
-- ============================================================

SELECT COUNT(DISTINCT country) AS total_countries
FROM sanitation_data;


-- ============================================================
-- QUERY 4: Find the earliest and latest year
-- ============================================================

SELECT
    MIN(year) AS earliest_year,
    MAX(year) AS latest_year
FROM sanitation_data;


-- ============================================================
-- QUERY 5: Average sanitation percentage
-- ============================================================

SELECT
    ROUND(AVG(sanitation_percent)::numeric, 2)
        AS average_sanitation_percentage
FROM sanitation_data;


-- ============================================================
-- QUERY 6: Highest sanitation percentage
-- ============================================================

SELECT
    country,
    year,
    sanitation_percent
FROM sanitation_data
ORDER BY sanitation_percent DESC
LIMIT 10;


-- ============================================================
-- QUERY 7: Lowest sanitation percentage
-- ============================================================

SELECT
    country,
    year,
    sanitation_percent
FROM sanitation_data
ORDER BY sanitation_percent ASC
LIMIT 10;


-- ============================================================
-- QUERY 8: Average sanitation by year
-- ============================================================

SELECT
    year,
    ROUND(AVG(sanitation_percent)::numeric, 2)
        AS average_sanitation
FROM sanitation_data
GROUP BY year
ORDER BY year;


-- ============================================================
-- QUERY 9: Average sanitation by country
-- ============================================================

SELECT
    country,
    ROUND(AVG(sanitation_percent)::numeric, 2)
        AS average_sanitation
FROM sanitation_data
GROUP BY country
ORDER BY average_sanitation DESC;


-- ============================================================
-- QUERY 10: Sanitation data for Rwanda
-- ============================================================

SELECT
    country,
    country_code,
    year,
    sanitation_percent
FROM sanitation_data
WHERE country = 'Rwanda'
ORDER BY year;


-- ============================================================
-- QUERY 11: Latest sanitation value for each country
-- ============================================================

SELECT DISTINCT ON (country)
    country,
    country_code,
    year,
    sanitation_percent
FROM sanitation_data
ORDER BY country, year DESC;


-- ============================================================
-- QUERY 12: Top 10 countries in the latest available year
-- ============================================================

SELECT
    country,
    country_code,
    year,
    sanitation_percent
FROM sanitation_data
WHERE year = (
    SELECT MAX(year)
    FROM sanitation_data
)
ORDER BY sanitation_percent DESC
LIMIT 10;


-- ============================================================
-- QUERY 13: Bottom 10 countries in the latest available year
-- ============================================================

SELECT
    country,
    country_code,
    year,
    sanitation_percent
FROM sanitation_data
WHERE year = (
    SELECT MAX(year)
    FROM sanitation_data
)
ORDER BY sanitation_percent ASC
LIMIT 10;


-- ============================================================
-- QUERY 14: Sanitation statistics
-- ============================================================

SELECT
    COUNT(*) AS total_records,
    ROUND(AVG(sanitation_percent)::numeric, 2) AS average,
    ROUND(MIN(sanitation_percent)::numeric, 2) AS minimum,
    ROUND(MAX(sanitation_percent)::numeric, 2) AS maximum
FROM sanitation_data;


-- ============================================================
-- QUERY 15: Check for missing values
-- ============================================================

SELECT
    COUNT(*) AS total_rows,
    COUNT(country) AS countries_present,
    COUNT(country_code) AS country_codes_present,
    COUNT(year) AS years_present,
    COUNT(sanitation_percent) AS sanitation_values_present
FROM sanitation_data;