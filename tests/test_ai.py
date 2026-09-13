import sys
import os

# Allow Python to find project files
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from ai_assistant import (
    find_country,
    find_year,
    find_years,
    analyze_question
)


# ==========================================================
# TEST COUNTRY DETECTION
# ==========================================================

def test_find_country():

    result = find_country(
        "What was Rwanda's sanitation percentage?"
    )

    assert result == "Rwanda"


# ==========================================================
# TEST YEAR DETECTION
# ==========================================================

def test_find_year():

    result = find_year(
        "What was Rwanda's sanitation percentage in 2024?"
    )

    assert result == 2024


# ==========================================================
# TEST MULTIPLE YEARS
# ==========================================================

def test_find_multiple_years():

    result = find_years(
        "How much did Rwanda improve between 2000 and 2024?"
    )

    assert 2000 in result

    assert 2024 in result


# ==========================================================
# TEST RWANDA 2024 ANALYSIS
# ==========================================================

def test_rwanda_2024_analysis():

    result = analyze_question(
        "What was Rwanda's sanitation percentage in 2024?"
    )

    assert "Rwanda" in result

    assert "2024" in result

    assert "81.33" in result


# ==========================================================
# TEST RWANDA IMPROVEMENT
# ==========================================================

def test_rwanda_improvement():

    result = analyze_question(
        "How much did Rwanda improve between 2000 and 2024?"
    )

    assert "43.89" in result

    assert "81.33" in result

    assert "37.44" in result