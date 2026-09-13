from extract import extract_data
from transform import transform_data
from load import load_data
from eda import perform_eda
from train_models import train_models
from logger import logger


def main():
    """
    Main function that runs the complete
    World Bank Sanitation Data Pipeline.
    """

    logger.info("=" * 60)
    logger.info("WORLD BANK SANITATION PROJECT STARTED")
    logger.info("=" * 60)

    try:

        # ==========================================================
        # STEP 1 - EXTRACT
        # ==========================================================

        logger.info("STEP 1: EXTRACT DATA FROM WORLD BANK API")
        logger.info("Starting data extraction...")

        extract_data()

        logger.info("Extraction completed successfully.")

        # ==========================================================
        # STEP 2 - TRANSFORM
        # ==========================================================

        logger.info("STEP 2: TRANSFORM AND VALIDATE DATA")
        logger.info("Starting data transformation and validation...")

        transform_data()

        logger.info("Transformation and validation completed successfully.")

        # ==========================================================
        # STEP 3 - LOAD
        # ==========================================================

        logger.info("STEP 3: LOAD DATA INTO POSTGRESQL")
        logger.info("Starting database loading...")

        load_data()

        logger.info("Data loading completed successfully.")

        # ==========================================================
        # STEP 4 - EXPLORATORY DATA ANALYSIS
        # ==========================================================

        logger.info("STEP 4: EXPLORATORY DATA ANALYSIS")
        logger.info("Starting EDA...")

        perform_eda()

        logger.info("EDA completed successfully.")

        # ==========================================================
        # STEP 5 - MACHINE LEARNING
        # ==========================================================

        logger.info("STEP 5: MACHINE LEARNING")
        logger.info("Starting model training and evaluation...")

        train_models()

        logger.info("Machine Learning completed successfully.")

        # ==========================================================
        # PROJECT COMPLETED
        # ==========================================================

        logger.info("=" * 60)
        logger.info("WORLD BANK SANITATION PROJECT COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

        logger.info("Data extraction: SUCCESS")
        logger.info("Data transformation and validation: SUCCESS")
        logger.info("PostgreSQL loading: SUCCESS")
        logger.info("Exploratory Data Analysis: SUCCESS")
        logger.info("Machine Learning: SUCCESS")
        logger.info("=" * 60)

    except Exception as e:

        logger.exception("=" * 60)
        logger.exception("WORLD BANK SANITATION PROJECT FAILED")
        logger.exception(f"Error: {e}")
        logger.exception("=" * 60)

        # Re-raise the error so that:
        # - automated tests can detect the failure
        # - the pipeline does not silently appear successful
        raise

    finally:

        logger.info("=" * 60)
        logger.info("WORLD BANK SANITATION PROJECT FINISHED")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()