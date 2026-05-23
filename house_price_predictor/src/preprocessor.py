# Cleans, transforms, and prepares features for model training and inference.

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_loader import DATA_PATH, NUMERICAL_COLUMNS, TARGET_COLUMN

# Columns removed because they add no predictive value for MLR
COLUMNS_TO_DROP = ["date", "street", "statezip", "country"]


def preprocess_data(
    filepath: Path | str,
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, list[str]]:
    """
    Clean, encode, split, and scale the house price dataset for modeling.

    Args:
        filepath: Path to the raw CSV file (e.g. data/data.csv).

    Returns:
        X_train_scaled: Scaled training features (numpy array).
        X_test_scaled: Scaled test features (numpy array).
        y_train: Training target values.
        y_test: Test target values.
        scaler: Fitted StandardScaler (reuse for new predictions).
        feature_names: List of feature column names after encoding.
    """
    print("=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)

    # Load the raw dataset using the same path convention as data_loader.py
    df = pd.read_csv(filepath)
    print(f"Loaded: {filepath}  |  Shape: {df.shape}\n")

    # -------------------------------------------------------------------------
    # STEP 1: Remove bad rows
    # Rows with price = 0 have no valid sale price and would corrupt the model.
    # Duplicate rows would overweight the same house during training.
    # -------------------------------------------------------------------------
    print("STEP 1: Removing bad rows")
    rows_before = len(df)
    zero_price_count = (df[TARGET_COLUMN] == 0).sum()
    df = df[df[TARGET_COLUMN] != 0]
    df = df.drop_duplicates()
    print(f"  Dropped {zero_price_count} rows with price = 0")
    print(f"  Rows before: {rows_before:,}  |  Rows after: {len(df):,}\n")

    # -------------------------------------------------------------------------
    # STEP 2: Drop useless columns
    # These columns either have too many unique values or no variation at all,
    # so they would not help (and might hurt) a linear regression model.
    # -------------------------------------------------------------------------
    print("STEP 2: Dropping useless columns")
    df = df.drop(columns=COLUMNS_TO_DROP)
    print(f"  Dropped: {COLUMNS_TO_DROP}")
    print(f"  Remaining columns: {list(df.columns)}\n")

    # -------------------------------------------------------------------------
    # STEP 3: One-Hot Encode the 'city' column
    # ML models need numbers, not text. One-hot encoding creates a 0/1 column
    # per city. drop_first=True removes one city column to avoid multicollinearity
    # (the dropped city is implied when all other city columns are 0).
    # -------------------------------------------------------------------------
    print("STEP 3: One-Hot Encoding 'city' column")
    city_dummies = pd.get_dummies(df["city"], prefix="city", drop_first=True)
    df = pd.concat([df.drop(columns=["city"]), city_dummies], axis=1)
    print(f"  Created {city_dummies.shape[1]} city dummy columns")
    print(f"  New shape: {df.shape}\n")

    # -------------------------------------------------------------------------
    # STEP 4: Handle missing values in numerical columns
    # Median imputation is robust to outliers. We fill before splitting so
    # train and test sets are both complete.
    # -------------------------------------------------------------------------
    print("STEP 4: Handling missing values")
    numerical_cols = [col for col in NUMERICAL_COLUMNS if col in df.columns]
    for col in numerical_cols:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            print(f"  Filled missing values in '{col}' with median: {median_value}")

    remaining_missing = df.isnull().sum().sum()
    if remaining_missing == 0:
        print("  Confirmation: No missing values remain in the dataset.")
    else:
        print(f"  Warning: {remaining_missing} missing values still remain.")
    print()

    # -------------------------------------------------------------------------
    # STEP 5: Split into features (X) and target (y)
    # X contains everything the model learns from; y is what we want to predict.
    # -------------------------------------------------------------------------
    print("STEP 5: Feature and target split")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = X.columns.tolist()
    print(f"  Features (X): {X.shape[1]} columns")
    print(f"  Target (y):   '{TARGET_COLUMN}'\n")

    # -------------------------------------------------------------------------
    # STEP 6: Train/Test split
    # We hold out 20% of data to evaluate the model on unseen houses.
    # random_state=42 makes the split reproducible every time we run the code.
    # -------------------------------------------------------------------------
    print("STEP 6: Train/Test split (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_test shape:  {X_test.shape}")
    print(f"  y_train shape: {y_train.shape}")
    print(f"  y_test shape:  {y_test.shape}\n")

    # -------------------------------------------------------------------------
    # STEP 7: Feature scaling with StandardScaler
    # Scaling puts all features on a similar scale (mean=0, std=1), which helps
    # linear regression converge and makes coefficient sizes comparable.
    # IMPORTANT: Fit only on training data to avoid data leakage from the test set.
    # -------------------------------------------------------------------------
    print("STEP 7: Feature scaling (StandardScaler)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaler fitted on X_train and applied to both train and test sets.")
    print("=" * 60)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


if __name__ == "__main__":
    preprocess_data(DATA_PATH)
