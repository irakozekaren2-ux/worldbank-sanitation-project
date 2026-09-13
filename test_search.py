import pandas as pd

DATA_PATH = "data/cleaned/sanitation_cleaned.csv"

df = pd.read_csv(DATA_PATH)

print("\nRwanda records:")
print(
    df[
        df["country"]
        .str.contains("Rwanda", case=False, na=False)
    ][
        ["country", "country_code", "year", "sanitation_percent"]
    ].to_string(index=False)
)

print("\nRwanda 2024:")
print(
    df[
        (
            df["country"]
            .str.contains("Rwanda", case=False, na=False)
        )
        &
        (df["year"] == 2024)
    ][
        ["country", "country_code", "year", "sanitation_percent"]
    ].to_string(index=False)
)