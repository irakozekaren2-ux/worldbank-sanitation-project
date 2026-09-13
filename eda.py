import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from logger import logger


def perform_eda():

    logger.info("Starting Exploratory Data Analysis...")

    df = pd.read_csv("data/cleaned/sanitation_cleaned.csv")

    print("\n========== DATASET INFORMATION ==========")
    print(df.info())

    print("\n========== FIRST FIVE ROWS ==========")
    print(df.head())

    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(df.describe())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    # Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(df["sanitation_percent"], bins=20)
    plt.title("Distribution of Basic Sanitation Services")
    plt.savefig("models/histogram.png")
    plt.close()

    # Boxplot
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df["sanitation_percent"])
    plt.title("Boxplot of Sanitation Percentage")
    plt.savefig("models/boxplot.png")
    plt.close()

    logger.info("EDA completed successfully.")

    print("✅ EDA completed successfully.")