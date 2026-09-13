import os
import pandas as pd


# ==========================================================
# TEST DATASET FILE
# ==========================================================

def test_dataset_exists():

    path = "data/cleaned/sanitation_cleaned.csv"

    assert os.path.exists(path), (
        "sanitation_cleaned.csv does not exist."
    )


# ==========================================================
# TEST DATASET CAN BE LOADED
# ==========================================================

def test_dataset_can_be_loaded():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    assert not df.empty, (
        "Dataset is empty."
    )


# ==========================================================
# TEST REQUIRED COLUMNS
# ==========================================================

def test_required_columns():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    required_columns = {
        "country",
        "country_code",
        "year",
        "sanitation_percent"
    }

    assert required_columns.issubset(
        df.columns
    ), "Required columns are missing."


# ==========================================================
# TEST RECORD COUNT
# ==========================================================

def test_record_count():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    assert len(df) > 0


# ==========================================================
# TEST YEAR RANGE
# ==========================================================

def test_year_range():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    assert df["year"].min() >= 2000

    assert df["year"].max() <= 2024


# ==========================================================
# TEST SANITATION VALUES
# ==========================================================

def test_sanitation_percentage_range():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    assert (
        df["sanitation_percent"] >= 0
    ).all()

    assert (
        df["sanitation_percent"] <= 100
    ).all()


# ==========================================================
# TEST RWANDA 2024
# ==========================================================

def test_rwanda_2024():

    path = "data/cleaned/sanitation_cleaned.csv"

    df = pd.read_csv(path)

    result = df[
        (
            df["country"] == "Rwanda"
        )
        &
        (
            df["year"] == 2024
        )
    ]

    assert len(result) == 1

    value = result.iloc[0][
        "sanitation_percent"
    ]

    assert round(value, 2) == 81.33