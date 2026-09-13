import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from logger import logger


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "data/cleaned/sanitation_cleaned.csv"
MODEL_DIR = "models"

TARGET = "sanitation_percent"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTION - EVALUATE MODEL
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Calculates evaluation metrics for both training
    and testing data.
    """

    # Predictions on training data
    train_predictions = model.predict(X_train)

    # Predictions on testing data
    test_predictions = model.predict(X_test)

    # -----------------------------
    # Training metrics
    # -----------------------------

    train_mae = mean_absolute_error(
        y_train,
        train_predictions
    )

    train_rmse = np.sqrt(
        mean_squared_error(
            y_train,
            train_predictions
        )
    )

    train_r2 = r2_score(
        y_train,
        train_predictions
    )

    # -----------------------------
    # Testing metrics
    # -----------------------------

    test_mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    test_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    test_r2 = r2_score(
        y_test,
        test_predictions
    )

    return {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }


# ============================================================
# MAIN MACHINE LEARNING FUNCTION
# ============================================================

def train_models():

    logger.info("=" * 60)
    logger.info("MACHINE LEARNING STARTED")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("MACHINE LEARNING")
    print("=" * 60)

    try:

        # ====================================================
        # STEP 1 - LOAD CLEANED DATA
        # ====================================================

        print("\nSTEP 1: Loading cleaned dataset...")

        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(
                f"Cleaned dataset not found: {DATA_FILE}"
            )

        df = pd.read_csv(DATA_FILE)

        print(f"Dataset loaded successfully.")
        print(f"Rows: {len(df)}")
        print(f"Columns: {len(df.columns)}")

        logger.info(
            f"Dataset loaded for ML: {df.shape}"
        )

        # ====================================================
        # STEP 2 - VALIDATE REQUIRED COLUMNS
        # ====================================================

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
                f"Missing required columns: {missing_columns}"
            )

        # ====================================================
        # STEP 3 - CLEAN DATA FOR MODELING
        # ====================================================

        print("\nSTEP 2: Preparing data for machine learning...")

        # Remove rows where target is missing
        df = df.dropna(
            subset=[TARGET]
        ).copy()

        # Make sure year is numeric
        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce"
        )

        # Remove invalid year values
        df = df.dropna(
            subset=["year"]
        )

        # Convert year to integer
        df["year"] = df["year"].astype(int)

        # Fill missing categorical values
        df["country"] = df["country"].fillna(
            "Unknown"
        )

        df["country_code"] = df["country_code"].fillna(
            "Unknown"
        )

        print(
            f"Records available for modeling: {len(df)}"
        )

        # ====================================================
        # STEP 4 - DEFINE FEATURES AND TARGET
        # ====================================================

        X = df[
            [
                "country",
                "country_code",
                "year"
            ]
        ]

        y = df[TARGET]

        print("\nTarget variable:")
        print(f"  {TARGET}")

        print("\nFeatures:")
        print("  country")
        print("  country_code")
        print("  year")

        # ====================================================
        # STEP 5 - TRAIN/TEST SPLIT
        # ====================================================

        print("\nSTEP 3: Creating train/test split...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print(
            f"Training records: {len(X_train)}"
        )

        print(
            f"Testing records: {len(X_test)}"
        )

        print(
            f"Test size: {TEST_SIZE * 100:.0f}%"
        )

        logger.info(
            f"Train/test split: "
            f"{len(X_train)} train, "
            f"{len(X_test)} test"
        )

        # ====================================================
        # STEP 6 - PREPROCESSING
        # ====================================================

        categorical_features = [
            "country",
            "country_code"
        ]

        numerical_features = [
            "year"
        ]

        categorical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        numerical_transformer = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    categorical_transformer,
                    categorical_features
                ),
                (
                    "numerical",
                    numerical_transformer,
                    numerical_features
                )
            ]
        )

        # ====================================================
        # STEP 7 - MODEL 1: LINEAR REGRESSION
        # ====================================================

        print("\n" + "=" * 60)
        print("MODEL 1: LINEAR REGRESSION")
        print("=" * 60)

        linear_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    LinearRegression()
                )
            ]
        )

        logger.info(
            "Training Linear Regression..."
        )

        linear_model.fit(
            X_train,
            y_train
        )

        linear_metrics = evaluate_model(
            linear_model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\nLinear Regression Results:")

        print(
            f"Train MAE : "
            f"{linear_metrics['train_mae']:.4f}"
        )

        print(
            f"Test MAE  : "
            f"{linear_metrics['test_mae']:.4f}"
        )

        print(
            f"Train RMSE: "
            f"{linear_metrics['train_rmse']:.4f}"
        )

        print(
            f"Test RMSE : "
            f"{linear_metrics['test_rmse']:.4f}"
        )

        print(
            f"Train R²  : "
            f"{linear_metrics['train_r2']:.4f}"
        )

        print(
            f"Test R²   : "
            f"{linear_metrics['test_r2']:.4f}"
        )

        # ====================================================
        # STEP 8 - MODEL 2: RANDOM FOREST
        # ====================================================

        print("\n" + "=" * 60)
        print("MODEL 2: RANDOM FOREST")
        print("=" * 60)

        random_forest_model = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )

        logger.info(
            "Training Random Forest..."
        )

        random_forest_model.fit(
            X_train,
            y_train
        )

        rf_metrics = evaluate_model(
            random_forest_model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print("\nRandom Forest Results:")

        print(
            f"Train MAE : "
            f"{rf_metrics['train_mae']:.4f}"
        )

        print(
            f"Test MAE  : "
            f"{rf_metrics['test_mae']:.4f}"
        )

        print(
            f"Train RMSE: "
            f"{rf_metrics['train_rmse']:.4f}"
        )

        print(
            f"Test RMSE : "
            f"{rf_metrics['test_rmse']:.4f}"
        )

        print(
            f"Train R²  : "
            f"{rf_metrics['train_r2']:.4f}"
        )

        print(
            f"Test R²   : "
            f"{rf_metrics['test_r2']:.4f}"
        )

        # ====================================================
        # STEP 9 - OVERFITTING / UNDERFITTING CHECK
        # ====================================================

        print("\n" + "=" * 60)
        print("OVERFITTING / UNDERFITTING CHECK")
        print("=" * 60)

        print(
            "\nThe rubric requires at least two "
            "model configurations."
        )

        # ----------------------------------------------------
        # CONFIGURATION 1
        # ----------------------------------------------------

        print(
            "\nCONFIGURATION 1: Random Forest max_depth=5"
        )

        rf_depth_5 = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )

        rf_depth_5.fit(
            X_train,
            y_train
        )

        depth_5_metrics = evaluate_model(
            rf_depth_5,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            f"Train R²: "
            f"{depth_5_metrics['train_r2']:.4f}"
        )

        print(
            f"Test R² : "
            f"{depth_5_metrics['test_r2']:.4f}"
        )

        print(
            f"Train RMSE: "
            f"{depth_5_metrics['train_rmse']:.4f}"
        )

        print(
            f"Test RMSE : "
            f"{depth_5_metrics['test_rmse']:.4f}"
        )

        # ----------------------------------------------------
        # CONFIGURATION 2
        # ----------------------------------------------------

        print(
            "\nCONFIGURATION 2: Random Forest max_depth=15"
        )

        rf_depth_15 = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=15,
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            ]
        )

        rf_depth_15.fit(
            X_train,
            y_train
        )

        depth_15_metrics = evaluate_model(
            rf_depth_15,
            X_train,
            X_test,
            y_train,
            y_test
        )

        print(
            f"Train R²: "
            f"{depth_15_metrics['train_r2']:.4f}"
        )

        print(
            f"Test R² : "
            f"{depth_15_metrics['test_r2']:.4f}"
        )

        print(
            f"Train RMSE: "
            f"{depth_15_metrics['train_rmse']:.4f}"
        )

        print(
            f"Test RMSE : "
            f"{depth_15_metrics['test_rmse']:.4f}"
        )

        # ====================================================
        # STEP 10 - CALCULATE TRAIN/TEST GAP
        # ====================================================

        gap_depth_5 = (
            depth_5_metrics["train_r2"]
            - depth_5_metrics["test_r2"]
        )

        gap_depth_15 = (
            depth_15_metrics["train_r2"]
            - depth_15_metrics["test_r2"]
        )

        print("\nTrain/Test R² Gaps:")

        print(
            f"max_depth=5 : "
            f"{gap_depth_5:.4f}"
        )

        print(
            f"max_depth=15: "
            f"{gap_depth_15:.4f}"
        )

        # ====================================================
        # STEP 11 - INTERPRET OVERFITTING
        # ====================================================

        print("\nInterpretation:")

        if gap_depth_5 < 0.10:
            print(
                "max_depth=5 shows a relatively small "
                "train/test gap."
            )
        else:
            print(
                "max_depth=5 shows a noticeable "
                "train/test gap."
            )

        if gap_depth_15 < 0.10:
            print(
                "max_depth=15 shows a relatively small "
                "train/test gap."
            )
        else:
            print(
                "max_depth=15 shows a larger "
                "train/test gap, indicating possible "
                "overfitting."
            )

        # ====================================================
        # STEP 12 - COMPARE THE TWO MAIN MODELS
        # ====================================================

        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)

        comparison = pd.DataFrame(
            {
                "Model": [
                    "Linear Regression",
                    "Random Forest"
                ],
                "MAE": [
                    linear_metrics["test_mae"],
                    rf_metrics["test_mae"]
                ],
                "RMSE": [
                    linear_metrics["test_rmse"],
                    rf_metrics["test_rmse"]
                ],
                "R2": [
                    linear_metrics["test_r2"],
                    rf_metrics["test_r2"]
                ]
            }
        )

        print("\nTest-set performance:")

        print(
            comparison.to_string(
                index=False
            )
        )

        # Lower MAE and RMSE are better.
        # Higher R² is better.

        if (
            rf_metrics["test_r2"]
            > linear_metrics["test_r2"]
            and
            rf_metrics["test_rmse"]
            < linear_metrics["test_rmse"]
            and
            rf_metrics["test_mae"]
            < linear_metrics["test_mae"]
        ):

            best_model = random_forest_model
            best_model_name = "Random Forest"

            explanation = (
                "Random Forest performed better because "
                "it achieved a higher test R² and lower "
                "test MAE and RMSE than Linear Regression."
            )

        else:

            best_model = linear_model
            best_model_name = "Linear Regression"

            explanation = (
                "Linear Regression performed better based "
                "on the test-set MAE, RMSE, and R² compared "
                "with Random Forest."
            )

        print(
            f"\nBEST MODEL: {best_model_name}"
        )

        print(
            f"Reason: {explanation}"
        )

        logger.info(
            f"Best model selected: {best_model_name}"
        )

        # ====================================================
        # STEP 13 - SAVE MODELS
        # ====================================================

        print("\n" + "=" * 60)
        print("SAVING MODELS")
        print("=" * 60)

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        joblib.dump(
            linear_model,
            os.path.join(
                MODEL_DIR,
                "linear_regression.pkl"
            )
        )

        joblib.dump(
            random_forest_model,
            os.path.join(
                MODEL_DIR,
                "random_forest.pkl"
            )
        )

        joblib.dump(
            best_model,
            os.path.join(
                MODEL_DIR,
                "best_model.pkl"
            )
        )

        print(
            "✓ linear_regression.pkl saved"
        )

        print(
            "✓ random_forest.pkl saved"
        )

        print(
            "✓ best_model.pkl saved"
        )

        # ====================================================
        # STEP 14 - SAVE RESULTS
        # ====================================================

        results = pd.DataFrame(
            {
                "Model": [
                    "Linear Regression",
                    "Random Forest"
                ],
                "Test_MAE": [
                    linear_metrics["test_mae"],
                    rf_metrics["test_mae"]
                ],
                "Test_RMSE": [
                    linear_metrics["test_rmse"],
                    rf_metrics["test_rmse"]
                ],
                "Test_R2": [
                    linear_metrics["test_r2"],
                    rf_metrics["test_r2"]
                ],
                "Train_MAE": [
                    linear_metrics["train_mae"],
                    rf_metrics["train_mae"]
                ],
                "Train_RMSE": [
                    linear_metrics["train_rmse"],
                    rf_metrics["train_rmse"]
                ],
                "Train_R2": [
                    linear_metrics["train_r2"],
                    rf_metrics["train_r2"]
                ]
            }
        )

        results_file = os.path.join(
            MODEL_DIR,
            "model_results.csv"
        )

        results.to_csv(
            results_file,
            index=False
        )

        print(
            "✓ model_results.csv saved"
        )

        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        print("\n" + "=" * 60)
        print("MACHINE LEARNING COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(
            "\nModels trained:"
        )

        print(
            "1. Linear Regression"
        )

        print(
            "2. Random Forest"
        )

        print(
            f"\nBest model: {best_model_name}"
        )

        print(
            "\nEvaluation metrics:"
        )

        print(
            "✓ MAE"
        )

        print(
            "✓ RMSE"
        )

        print(
            "✓ R²"
        )

        print(
            "\nOverfitting check:"
        )

        print(
            "✓ Random Forest max_depth=5"
        )

        print(
            "✓ Random Forest max_depth=15"
        )

        print(
            "\nModels saved in:"
        )

        print(
            f"  {MODEL_DIR}/"
        )

        logger.info(
            "Machine Learning completed successfully."
        )

        return {
            "linear_regression": linear_metrics,
            "random_forest": rf_metrics,
            "depth_5": depth_5_metrics,
            "depth_15": depth_15_metrics,
            "best_model": best_model_name
        }

    except Exception as error:

        logger.error(
            f"Machine Learning failed: {error}"
        )

        print(
            "\n❌ Machine Learning failed:"
        )

        print(
            f"{error}"
        )

        raise


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    train_models()