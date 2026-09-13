import requests
import json
import os
import time

from logger import logger


# ==========================================================
# WORLD BANK INDICATOR
# ==========================================================

INDICATOR = "SH.STA.BASS.ZS"


# ==========================================================
# WORLD BANK API URL
# ==========================================================

URL = (
    f"https://api.worldbank.org/v2/country/all/indicator/"
    f"{INDICATOR}?format=json&per_page=20000"
)


# ==========================================================
# EXTRACTION FUNCTION
# ==========================================================

def extract_data():
    """
    Extracts sanitation data from the World Bank API.

    The function:
    1. Sends a request to the World Bank API.
    2. Uses a timeout to prevent hanging requests.
    3. Retries failed requests up to three times.
    4. Logs every attempt and failure.
    5. Saves the raw API response to JSON.
    6. Handles empty or invalid API responses.
    7. Returns the extracted API data.
    """

    logger.info("Starting World Bank data extraction.")
    logger.info(f"Indicator: {INDICATOR}")
    logger.info(f"API URL: {URL}")

    # ------------------------------------------------------
    # Configuration
    # ------------------------------------------------------

    max_retries = 3
    retry_delay = 5
    timeout = 30

    # ------------------------------------------------------
    # Retry loop
    # ------------------------------------------------------

    for attempt in range(1, max_retries + 1):

        try:

            logger.info(
                f"Extraction attempt {attempt} of {max_retries}."
            )

            # --------------------------------------------------
            # Send request to World Bank API
            # --------------------------------------------------

            response = requests.get(
                URL,
                timeout=timeout
            )

            # --------------------------------------------------
            # Raise an error for HTTP failures
            # --------------------------------------------------

            response.raise_for_status()

            # --------------------------------------------------
            # Check that the API returned content
            # --------------------------------------------------

            if not response.text.strip():

                raise ValueError(
                    "World Bank API returned an empty response."
                )

            # --------------------------------------------------
            # Convert JSON response into Python object
            # --------------------------------------------------

            data = response.json()

            # --------------------------------------------------
            # Validate basic World Bank API structure
            # --------------------------------------------------

            if not isinstance(data, list):

                raise ValueError(
                    "Unexpected API response format."
                )

            if len(data) < 2:

                raise ValueError(
                    "World Bank API response does not contain "
                    "the expected metadata and records."
                )

            # The second element normally contains
            # the actual observations.

            records = data[1]

            if records is None or len(records) == 0:

                raise ValueError(
                    "World Bank API returned zero data records."
                )

            # --------------------------------------------------
            # Create raw data directory
            # --------------------------------------------------

            os.makedirs(
                "data/raw",
                exist_ok=True
            )

            # --------------------------------------------------
            # Save RAW API response
            # --------------------------------------------------

            raw_file = "data/raw/sanitation_raw.json"

            with open(
                raw_file,
                "w",
                encoding="utf-8"
            ) as file:

                # Save the complete API response,
                # including metadata and observations.

                json.dump(
                    data,
                    file,
                    indent=4,
                    ensure_ascii=False
                )

            # --------------------------------------------------
            # Successful extraction
            # --------------------------------------------------

            logger.info(
                f"Successfully extracted {len(records)} "
                f"records from the World Bank API."
            )

            logger.info(
                f"Raw API response saved to {raw_file}."
            )

            print(
                "✅ Extraction completed successfully."
            )

            print(
                f"✅ Records extracted: {len(records)}"
            )

            return data

        # ======================================================
        # REQUEST / NETWORK ERRORS
        # ======================================================

        except requests.exceptions.RequestException as error:

            logger.warning(
                f"Extraction attempt {attempt} failed: {error}"
            )

            if attempt < max_retries:

                logger.info(
                    f"Retrying in {retry_delay} seconds..."
                )

                time.sleep(retry_delay)

            else:

                logger.error(
                    f"Extraction failed after "
                    f"{max_retries} attempts."
                )

                raise

        # ======================================================
        # INVALID / EMPTY API RESPONSE
        # ======================================================

        except (ValueError, json.JSONDecodeError) as error:

            logger.error(
                f"Invalid World Bank API response: {error}"
            )

            # An invalid response is not something that
            # should be silently accepted.

            if attempt < max_retries:

                logger.info(
                    f"Retrying in {retry_delay} seconds..."
                )

                time.sleep(retry_delay)

            else:

                logger.error(
                    f"Extraction failed after "
                    f"{max_retries} attempts."
                )

                raise

    # ----------------------------------------------------------
    # Safety fallback
    # ----------------------------------------------------------

    raise RuntimeError(
        "World Bank data extraction failed."
    )