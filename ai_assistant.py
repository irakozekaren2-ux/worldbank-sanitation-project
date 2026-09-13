import os
import re
import pandas as pd

from dotenv import load_dotenv
from google import genai

from logger import logger


# ==========================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in the .env file."
    )


# ==========================================================
# 2. CONNECT TO GEMINI
# ==========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ==========================================================
# 3. LOAD CLEANED DATASET
# ==========================================================

DATA_PATH = "data/cleaned/sanitation_cleaned.csv"

df = pd.read_csv(DATA_PATH)


# ==========================================================
# 4. CLEAN DATA TYPES
# ==========================================================

df["country"] = df["country"].astype(str)

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce"
)

df["sanitation_percent"] = pd.to_numeric(
    df["sanitation_percent"],
    errors="coerce"
)


# ==========================================================
# 5. DATASET SUMMARY
# ==========================================================

def create_data_summary():

    return f"""
WORLD BANK SANITATION DATASET

Indicator:
SH.STA.BASS.ZS

Description:
People using at least basic sanitation services
(% of population).

Number of records:
{len(df)}

Number of countries/regions:
{df["country"].nunique()}

Earliest year:
{int(df["year"].min())}

Latest year:
{int(df["year"].max())}

Average sanitation percentage:
{df["sanitation_percent"].mean():.2f}%

Minimum sanitation percentage:
{df["sanitation_percent"].min():.2f}%

Maximum sanitation percentage:
{df["sanitation_percent"].max():.2f}%
"""


# ==========================================================
# 6. FIND COUNTRY
# ==========================================================

def find_country(question):

    question_lower = question.lower()

    countries = sorted(
        df["country"]
        .dropna()
        .unique(),
        key=len,
        reverse=True
    )

    for country in countries:

        if country.lower() in question_lower:

            return country

    return None


# ==========================================================
# 7. FIND ONE YEAR
# ==========================================================

def find_year(question):

    years = re.findall(
        r"\b(20\d{2})\b",
        question
    )

    if not years:
        return None

    for year in years:

        year = int(year)

        if year in df["year"].values:

            return year

    return None


# ==========================================================
# 8. FIND MULTIPLE YEARS
# ==========================================================

def find_years(question):

    years = re.findall(
        r"\b(20\d{2})\b",
        question
    )

    valid_years = []

    for year in years:

        year = int(year)

        if year in df["year"].values:

            valid_years.append(year)

    return list(dict.fromkeys(valid_years))


# ==========================================================
# 9. ANALYZE DATA
# ==========================================================

def analyze_question(question):

    question_lower = question.lower()

    country = find_country(question)

    year = find_year(question)

    years = find_years(question)


    # ======================================================
    # COUNTRY IMPROVEMENT BETWEEN TWO YEARS
    # ======================================================

    if (
        country
        and len(years) >= 2
        and (
            "improve" in question_lower
            or "improvement" in question_lower
            or "change" in question_lower
            or "increase" in question_lower
            or "difference" in question_lower
        )
    ):

        start_year = min(years)

        end_year = max(years)

        start_data = df[
            (
                df["country"].str.lower()
                == country.lower()
            )
            &
            (df["year"] == start_year)
        ]

        end_data = df[
            (
                df["country"].str.lower()
                == country.lower()
            )
            &
            (df["year"] == end_year)
        ]

        if (
            len(start_data) > 0
            and len(end_data) > 0
        ):

            start_value = (
                start_data.iloc[0]
                ["sanitation_percent"]
            )

            end_value = (
                end_data.iloc[0]
                ["sanitation_percent"]
            )

            change = (
                end_value - start_value
            )

            return f"""
Country improvement analysis from
the actual dataset:

Country/Region:
{country}

Starting year:
{start_year}

Starting sanitation percentage:
{start_value:.2f}%

Ending year:
{end_year}

Ending sanitation percentage:
{end_value:.2f}%

Improvement:
{change:.2f} percentage points
"""


    # ======================================================
    # COUNTRY + YEAR LOOKUP
    # ======================================================

    if country and year:

        result = df[
            (
                df["country"].str.lower()
                == country.lower()
            )
            &
            (df["year"] == year)
        ]

        if len(result) > 0:

            row = result.iloc[0]

            value = row[
                "sanitation_percent"
            ]

            return f"""
Exact dataset result:

Country/Region:
{row["country"]}

Country Code:
{row["country_code"]}

Year:
{int(row["year"])}

Sanitation Percentage:
{value:.2f}%
"""


    # ======================================================
    # COUNTRY ONLY
    # ======================================================

    if country:

        result = df[
            df["country"].str.lower()
            == country.lower()
        ].sort_values("year")

        if len(result) > 0:

            first = result.iloc[0]

            latest = result.iloc[-1]

            average = result[
                "sanitation_percent"
            ].mean()

            change = (
                latest["sanitation_percent"]
                - first["sanitation_percent"]
            )

            return f"""
Country analysis from the actual dataset:

Country/Region:
{country}

Earliest year:
{int(first["year"])}

Earliest sanitation percentage:
{first["sanitation_percent"]:.2f}%

Latest year:
{int(latest["year"])}

Latest sanitation percentage:
{latest["sanitation_percent"]:.2f}%

Average sanitation percentage:
{average:.2f}%

Change between earliest and latest year:
{change:.2f} percentage points
"""


    # ======================================================
    # AVERAGE BY YEAR
    # ======================================================

    if (
        "average" in question_lower
        and "year" in question_lower
    ):

        yearly = (
            df.groupby("year")[
                "sanitation_percent"
            ]
            .mean()
            .reset_index()
        )

        highest = yearly.loc[
            yearly["sanitation_percent"].idxmax()
        ]

        lowest = yearly.loc[
            yearly["sanitation_percent"].idxmin()
        ]

        return f"""
Yearly analysis from the actual dataset:

Highest average sanitation coverage:

Year:
{int(highest["year"])}

Average:
{highest["sanitation_percent"]:.2f}%

Lowest average sanitation coverage:

Year:
{int(lowest["year"])}

Average:
{lowest["sanitation_percent"]:.2f}%
"""


    # ======================================================
    # HIGHEST YEAR
    # ======================================================

    if (
        "highest" in question_lower
        and "year" in question_lower
    ):

        yearly = (
            df.groupby("year")[
                "sanitation_percent"
            ]
            .mean()
        )

        highest_year = yearly.idxmax()

        highest_value = yearly.max()

        return f"""
The year with the highest average sanitation
coverage was {int(highest_year)}, with an average
of {highest_value:.2f}%.
"""


    # ======================================================
    # LOWEST YEAR
    # ======================================================

    if (
        "lowest" in question_lower
        and "year" in question_lower
    ):

        yearly = (
            df.groupby("year")[
                "sanitation_percent"
            ]
            .mean()
        )

        lowest_year = yearly.idxmin()

        lowest_value = yearly.min()

        return f"""
The year with the lowest average sanitation
coverage was {int(lowest_year)}, with an average
of {lowest_value:.2f}%.
"""


    # ======================================================
    # AVERAGE SANITATION
    # ======================================================

    if "average" in question_lower:

        average = df[
            "sanitation_percent"
        ].mean()

        return f"""
The average sanitation coverage across the
dataset is {average:.2f}%.
"""


    # ======================================================
    # MAXIMUM SANITATION
    # ======================================================

    if (
        "maximum" in question_lower
        or "highest percentage" in question_lower
        or "highest sanitation" in question_lower
    ):

        maximum = df[
            "sanitation_percent"
        ].max()

        row = df[
            df["sanitation_percent"]
            == maximum
        ].iloc[0]

        return f"""
The highest sanitation percentage in the dataset
is {maximum:.2f}%.

Country/Region:
{row["country"]}

Year:
{int(row["year"])}
"""


    # ======================================================
    # MINIMUM SANITATION
    # ======================================================

    if (
        "minimum" in question_lower
        or "lowest percentage" in question_lower
        or "lowest sanitation" in question_lower
    ):

        minimum = df[
            "sanitation_percent"
        ].min()

        row = df[
            df["sanitation_percent"]
            == minimum
        ].iloc[0]

        return f"""
The lowest sanitation percentage in the dataset
is {minimum:.2f}%.

Country/Region:
{row["country"]}

Year:
{int(row["year"])}
"""


    # ======================================================
    # NUMBER OF RECORDS
    # ======================================================

    if (
        "how many records" in question_lower
        or "number of records" in question_lower
    ):

        return f"""
The dataset contains {len(df):,} records.
"""


    # ======================================================
    # DATASET YEARS
    # ======================================================

    if (
        "what years" in question_lower
        or "years does the dataset cover"
        in question_lower
    ):

        return f"""
The dataset covers the years
{int(df["year"].min())} to
{int(df["year"].max())}.
"""


    # ======================================================
    # 2024 AVERAGE
    # ======================================================

    if (
        "2024" in question_lower
        and "average" in question_lower
    ):

        result = df[
            df["year"] == 2024
        ]

        if len(result) > 0:

            average = result[
                "sanitation_percent"
            ].mean()

            return f"""
The average sanitation coverage across
the dataset in 2024 was {average:.2f}%.
"""


    # ======================================================
    # NO SPECIAL ANALYSIS
    # ======================================================

    return None


# ==========================================================
# 10. ASK GEMINI
# ==========================================================

def ask_gemini(question):

    try:

        analysis = analyze_question(
            question
        )

        if analysis is None:

            analysis = (
                "No specific calculation was performed "
                "for this question."
            )

        prompt = f"""
You are an AI assistant for a World Bank
sanitation data analysis project.

Indicator:

SH.STA.BASS.ZS

Description:

People using at least basic sanitation services
(% of population).

Dataset summary:

{create_data_summary()}


CALCULATION/RESULT FROM PYTHON:

{analysis}


USER QUESTION:

{question}


Instructions:

1. Use the Python analysis as the factual source.

2. Do not invent statistics.

3. Do not change numerical values provided
   by Python.

4. Explain the result clearly.

5. Mention the relevant country/region and year
   when available.

6. Round percentages to two decimal places.

7. If Python did not calculate the requested
   information, clearly explain that the answer
   cannot be determined from the available analysis.
"""

        response = client.models.generate_content(
            model="models/gemini-flash-latest",
            contents=prompt
        )

        return response.text

    except Exception as e:

        logger.error(
            f"Gemini request failed: {e}"
        )

        return f"AI request failed: {e}"


# ==========================================================
# 11. INTERACTIVE ASSISTANT
# ==========================================================

def run_ai_assistant():

    print("\n")

    print("=" * 60)

    print(
        "WORLD BANK SANITATION AI ASSISTANT"
    )

    print("=" * 60)

    print(
        "Ask questions about the sanitation dataset."
    )

    print(
        "Type 'exit' to stop the assistant."
    )

    while True:

        question = input(
            "\nYou: "
        )

        if question.lower().strip() == "exit":

            print(
                "\nAI Assistant closed."
            )

            break

        if not question.strip():

            print(
                "Please enter a question."
            )

            continue

        print("\nAI:")

        answer = ask_gemini(
            question
        )

        print(answer)


# ==========================================================
# 12. PROGRAM ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    run_ai_assistant()