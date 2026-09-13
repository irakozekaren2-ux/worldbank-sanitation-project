import json
import os

import pandas as pd
from sqlalchemy import create_engine, text

from logger import logger
from config import (
    DB_HOST,
    DB_PORT,
    DB_NAME,
    DB_USER,
    DB_PASSWORD
)


# ==========================================================
# DATABASE CONNECTION
# ==========================================================

def get_engine():
    """
    Creates a SQLAlchemy connection to PostgreSQL.
    Credentials are obtained from config.py.
    """

    return create_engine(
        f"postgresql+psycopg2://"
        f"{DB_USER}:{DB_PASSWORD}@"
        f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


# ==========================================================
# RAW DATA
# ==========================================================

def load_raw_data(engine):
    """
    Loads the original World Bank API observations
    into the PostgreSQL raw layer.
    """

    logger.info("Loading raw data into PostgreSQL.")

    raw_file = "data/raw/sanitation_raw.json"

    if not os.path.exists(raw_file):

        raise FileNotFoundError(
            f"Raw file not found: {raw_file}"
        )

    with open(
        raw_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, list) or len(data) < 2:

        raise ValueError(
            "Invalid raw World Bank API structure."
        )

    records = data[1]

    if not records:

        raise ValueError(
            "Raw dataset contains zero records."
        )

    raw_rows = []

    for record in records:

        raw_rows.append({

            "country": record.get(
                "country",
                {}
            ).get(
                "value"
            ),

            "country_code": record.get(
                "countryiso3code"
            ),

            "year": record.get(
                "date"
            ),

            "sanitation_percent": record.get(
                "value"
            )

        })

    raw_df = pd.DataFrame(
        raw_rows
    )

    raw_df.to_sql(
        name="sanitation_raw",
        con=engine,
        if_exists="replace",
        index=False
    )

    logger.info(
        f"Raw table loaded with "
        f"{len(raw_df)} records."
    )


# ==========================================================
# CLEANED DATA
# ==========================================================

def load_cleaned_data(engine):
    """
    Loads cleaned data into PostgreSQL.

    The table uses country + year as the unique
    observation key to prevent duplicate records.
    """

    logger.info(
        "Loading cleaned data into PostgreSQL."
    )

    cleaned_file = (
        "data/cleaned/"
        "sanitation_cleaned.csv"
    )

    if not os.path.exists(cleaned_file):

        raise FileNotFoundError(
            f"Cleaned file not found: {cleaned_file}"
        )

    df = pd.read_csv(
        cleaned_file
    )

    if df.empty:

        raise ValueError(
            "Cleaned dataset is empty."
        )

    required_columns = [
        "country",
        "country_code",
        "year",
        "sanitation_percent"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # ------------------------------------------------------
    # Data validation
    # ------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="raise"
    ).astype(int)

    df["sanitation_percent"] = pd.to_numeric(
        df["sanitation_percent"],
        errors="raise"
    )

    if not df[
        "sanitation_percent"
    ].between(
        0,
        100
    ).all():

        raise ValueError(
            "Sanitation percentage must be "
            "between 0 and 100."
        )

    # ------------------------------------------------------
    # Remove duplicates before loading
    # ------------------------------------------------------

    df = df.drop_duplicates(
        subset=[
            "country",
            "year"
        ],
        keep="last"
    )

    table_name = "sanitation_data"

    # ------------------------------------------------------
    # Create table
    # ------------------------------------------------------

    with engine.begin() as connection:

        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    country TEXT NOT NULL,
                    country_code TEXT,
                    year INTEGER NOT NULL,
                    sanitation_percent DOUBLE PRECISION NOT NULL,
                    CONSTRAINT unique_country_year
                    UNIQUE (country, year)
                )
                """
            )
        )

        # --------------------------------------------------
        # Clear current cleaned observations.
        #
        # The cleaned CSV is the current authoritative
        # dataset, so replacing its contents keeps the
        # pipeline idempotent.
        # --------------------------------------------------

        connection.execute(
            text(
                f"DELETE FROM {table_name}"
            )
        )

    # ------------------------------------------------------
    # Insert current cleaned dataset
    # ------------------------------------------------------

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False
    )

    logger.info(
        f"Cleaned table loaded with "
        f"{len(df)} records."
    )


# ==========================================================
# MAIN LOAD FUNCTION
# ==========================================================

def load_data():
    """
    Loads both raw and cleaned data into PostgreSQL.
    """

    logger.info(
        "Starting PostgreSQL data loading."
    )

    try:

        engine = get_engine()

        # RAW LAYER
        load_raw_data(
            engine
        )

        # CLEANED LAYER
        load_cleaned_data(
            engine
        )

        logger.info(
            "PostgreSQL loading completed successfully."
        )

        print(
            "✅ Raw data loaded into PostgreSQL."
        )

        print(
            "✅ Cleaned data loaded into PostgreSQL."
        )

    except Exception as error:

        logger.exception(
            f"PostgreSQL loading failed: {error}"
        )

        raise