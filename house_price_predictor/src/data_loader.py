# Loads and validates raw house price dataset from CSV files.

from pathlib import Path

import pandas as pd

# Path to the dataset, relative to the project root (house_price_predictor/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"

# Known column groups for King County house sales data
TARGET_COLUMN = "price"
NUMERICAL_COLUMNS = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
]
CATEGORICAL_COLUMNS = ["date", "street", "city", "statezip", "country"]


def load_and_analyze() -> pd.DataFrame:
    """
    Load the house price dataset and print a full exploratory summary.

    Returns:
        The loaded DataFrame for use in later pipeline steps.
    """
    # -------------------------------------------------------------------------
    # Step 1: Load the CSV file into a pandas DataFrame
    # A DataFrame is a table-like structure — rows are houses, columns are features.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded: {DATA_PATH}\n")

    # -------------------------------------------------------------------------
    # Step 2: Print the size of the dataset (rows and columns)
    # This tells us how much data we have to work with.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DATASET SIZE")
    print("=" * 60)
    print(f"Total rows:    {df.shape[0]:,}")
    print(f"Total columns: {df.shape[1]}\n")

    # -------------------------------------------------------------------------
    # Step 3: Print column names and their data types
    # Data types matter — numbers need different handling than text categories.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("COLUMN NAMES AND DATA TYPES")
    print("=" * 60)
    print(df.dtypes)
    print()

    # -------------------------------------------------------------------------
    # Step 4: Preview the first 5 rows
    # A quick look helps us understand what the raw data actually looks like.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("FIRST 5 ROWS")
    print("=" * 60)
    print(df.head())
    print()

    # -------------------------------------------------------------------------
    # Step 5: Count missing values per column
    # Missing data can break models, so we need to know where gaps exist.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("MISSING VALUES PER COLUMN")
    print("=" * 60)
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found in any column.")
    else:
        print(missing[missing > 0])
    print()

    # -------------------------------------------------------------------------
    # Step 6: Count duplicate rows
    # Duplicates can skew training, so we check if the same house appears twice.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("DUPLICATE ROWS")
    print("=" * 60)
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate rows: {duplicate_count}\n")

    # -------------------------------------------------------------------------
    # Step 7: Basic statistics for numerical columns
    # describe() gives mean, min, max, etc. — useful for spotting outliers.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("NUMERICAL COLUMN STATISTICS")
    print("=" * 60)
    print(df.describe())
    print()

    # -------------------------------------------------------------------------
    # Step 8: Identify target, numerical, and categorical columns
    # The target is what we predict; features are split by how we encode them.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("COLUMN ROLES")
    print("=" * 60)
    print(f"Target column (what we predict): {TARGET_COLUMN}")
    print(f"Numerical feature columns:       {NUMERICAL_COLUMNS}")
    print(f"Categorical feature columns:     {CATEGORICAL_COLUMNS}")
    print()

    # -------------------------------------------------------------------------
    # Step 9: Count rows where price is zero
    # A price of $0 is invalid — these rows are bad data and should be removed later.
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("INVALID PRICE VALUES")
    print("=" * 60)
    zero_price_count = (df[TARGET_COLUMN] == 0).sum()
    print(f"Rows with price = 0: {zero_price_count}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    load_and_analyze()
