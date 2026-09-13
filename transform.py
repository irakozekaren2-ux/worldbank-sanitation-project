import json
import os

import pandas as pd

from logger import logger


# ==========================================================
# TRANSFORMATION AND VALIDATION
# ==========================================================

def transform_data():
    """
    Reads the raw World Bank JSON data,
    transforms it into a structured DataFrame,
    validates the data,
    removes invalid records,
    and saves the cleaned CSV file.
    """

    logger.info("Starting data transformation and validation.")

    # ======================================================
    # STEP 1 - CHECK RAW FILE
    # ======================================================

    raw_file = "data/raw/sanitation_raw.json"

    if not os.path.exists(raw_file):

        logger.error(
            f"Raw data file not found: {raw_file}"
        )

        raise FileNotFoundError(
            f"Raw data file not found: {raw_file}"
        )

    # ======================================================
    # STEP 2 - READ RAW JSON
    # ======================================================

    try:

        with open(
            raw_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except json.JSONDecodeError as error:

        logger.error(
            f"Raw JSON file is invalid: {error}"
        )

        raise

    # ======================================================
    # STEP 3 - VALIDATE API RESPONSE STRUCTURE
    # ======================================================

    if not isinstance(data, list):

        raise ValueError(
            "Raw World Bank data has an unexpected format."
        )

    if len(data) < 2:

        raise ValueError(
            "Raw World Bank data does not contain records."
        )

    records = data[1]

    if not records:

        raise ValueError(
            "Raw World Bank dataset contains zero records."
        )

    logger.info(
        f"Raw dataset contains {len(records)} records."
    )

    # ======================================================
    # STEP 4 - EXTRACT REQUIRED FIELDS
    # ======================================================

    cleaned_data = []

    skipped_records = 0

    for record in records:

        try:

            country = record.get(
                "country",
                {}
            ).get(
                "value"
            )

            country_code = record.get(
                "countryiso3code"
            )

            year = record.get(
                "date"
            )

            sanitation_percent = record.get(
                "value"
            )

            cleaned_data.append({

                "country": country,

                "country_code": country_code,

                "year": year,

                "sanitation_percent": sanitation_percent

            })

        except (AttributeError, TypeError) as error:

            skipped_records += 1

            logger.warning(
                f"Skipping malformed record: {error}"
            )

    # ======================================================
    # STEP 5 - CREATE DATAFRAME
    # ======================================================

    df = pd.DataFrame(cleaned_data)

    if df.empty:

        logger.error(
            "No usable records were produced."
        )

        raise ValueError(
            "Transformation produced an empty dataset."
        )

    logger.info(
        f"DataFrame created with {len(df)} records."
    )

    # ======================================================
    # STEP 6 - CHECK REQUIRED COLUMNS
    # ======================================================

    required_columns = [
        "country",
        "country_code",
        "year",
        "sanitation_percent"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Required columns are missing: "
            f"{missing_columns}"
        )

    # ======================================================
    # STEP 7 - CONVERT DATA TYPES SAFELY
    # ======================================================

    # Convert year to numeric.
    # Invalid values become NaN instead of crashing
    # the entire transformation immediately.

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce"
    )

    # Convert sanitation percentage to numeric.

    df["sanitation_percent"] = pd.to_numeric(
        df["sanitation_percent"],
        errors="coerce"
    )

    # ======================================================
    # STEP 8 - CHECK MISSING VALUES
    # ======================================================

    missing_before = df.isnull().sum()

    logger.info(
        "Missing values before cleaning:\n"
        f"{missing_before}"
    )

    # Sanitation percentage is essential.
    # A record without a sanitation value cannot
    # be used for analysis.

    missing_sanitation = df[
        "sanitation_percent"
    ].isna().sum()

    if missing_sanitation > 0:

        logger.warning(
            f"Removing {missing_sanitation} records "
            f"with missing sanitation percentages."
        )

        df = df.dropna(
            subset=["sanitation_percent"]
        )

    # Year is also essential.

    missing_year = df["year"].isna().sum()

    if missing_year > 0:

        logger.warning(
            f"Removing {missing_year} records "
            f"with invalid or missing years."
        )

        df = df.dropna(
            subset=["year"]
        )

    # Country is essential for country-level analysis.

    missing_country = df["country"].isna().sum()

    if missing_country > 0:

        logger.warning(
            f"Removing {missing_country} records "
            f"with missing country names."
        )

        df = df.dropna(
            subset=["country"]
        )

    # ======================================================
    # STEP 9 - CONVERT YEAR TO INTEGER
    # ======================================================

    df["year"] = df["year"].astype(int)

    # ======================================================
    # STEP 10 - VALIDATE YEAR RANGE
    # ======================================================

    # The project dataset is based on the period
    # covered by the World Bank data.

    invalid_years = ~df["year"].between(
        1900,
        2100
    )

    invalid_year_count = invalid_years.sum()

    if invalid_year_count > 0:

        logger.warning(
            f"Removing {invalid_year_count} records "
            f"with impossible year values."
        )

        df = df.loc[
            ~invalid_years
        ].copy()

    # ======================================================
    # STEP 11 - VALIDATE SANITATION PERCENTAGE
    # ======================================================

    # A percentage must be between 0 and 100.

    invalid_percentages = ~df[
        "sanitation_percent"
    ].between(
        0,
        100
    )

    invalid_percentage_count = (
        invalid_percentages.sum()
    )

    if invalid_percentage_count > 0:

        logger.warning(
            f"Removing {invalid_percentage_count} "
            f"records with sanitation percentages "
            f"outside the valid range 0-100."
        )

        df = df.loc[
            ~invalid_percentages
        ].copy()

    # ======================================================
    # STEP 12 - REMOVE DUPLICATE RECORDS
    # ======================================================

    duplicate_count = df.duplicated(
        subset=[
            "country",
            "year"
        ]
    ).sum()

    if duplicate_count > 0:

        logger.warning(
            f"Removing {duplicate_count} "
            f"duplicate country-year records."
        )

        df = df.drop_duplicates(
            subset=[
                "country",
                "year"
            ],
            keep="first"
        )

    # ======================================================
    # STEP 13 - SORT DATA
    # ======================================================

    df = df.sort_values(
        by=[
            "country",
            "year"
        ]
    ).reset_index(
        drop=True
    )

    # ======================================================
    # STEP 14 - FINAL DATASET CHECK
    # ======================================================

    if df.empty:

        logger.error(
            "Dataset became empty after validation."
        )

        raise ValueError(
            "No valid records remain after cleaning."
        )

    # ======================================================
    # STEP 15 - FINAL VALIDATION
    # ======================================================

    if not df[
        "sanitation_percent"
    ].between(
        0,
        100
    ).all():

        raise ValueError(
            "Final validation failed: "
            "sanitation percentage contains "
            "values outside 0-100."
        )

    if df[
        "year"
    ].isna().any():

        raise ValueError(
            "Final validation failed: "
            "missing year values remain."
        )

    # ======================================================
    # STEP 16 - SAVE CLEANED DATA
    # ======================================================

    os.makedirs(
        "data/cleaned",
        exist_ok=True
    )

    output_file = (
        "data/cleaned/sanitation_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # ======================================================
    # STEP 17 - LOG FINAL RESULTS
    # ======================================================

    logger.info(
        f"Transformation completed successfully."
    )

    logger.info(
        f"Final cleaned records: {len(df)}"
    )

    logger.info(
        f"Skipped malformed records: "
        f"{skipped_records}"
    )

    logger.info(
        f"Cleaned dataset saved to: {output_file}"
    )

    print(
        "✅ Data transformation and validation "
        "completed successfully."
    )

    print(
        f"✅ Cleaned records: {len(df)}"
    )

    print(
        f"✅ Cleaned data saved to: {output_file}"
    )

    return df