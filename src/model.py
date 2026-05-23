# Defines, trains, evaluates, and persists the house price prediction model.

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> tuple[LinearRegression, np.ndarray, np.ndarray]:
    """
    Train a Multiple Linear Regression model and evaluate its performance.

    Args:
        X_train: Scaled training features.
        X_test: Scaled test features.
        y_train: Training target (house prices).
        y_test: Test target (house prices).
        feature_names: Names of each feature column.

    Returns:
        model: Trained LinearRegression model.
        y_test_pred: Predictions on the test set.
        y_train_pred: Predictions on the training set.
    """
    print("=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 1: Train the Multiple Linear Regression model
    # MLR finds the best-fit line/plane through all features to predict price.
    # It learns a weight (coefficient) for each feature plus an intercept (base price).
    # -------------------------------------------------------------------------
    print("\nSTEP 1: Training Linear Regression model")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("  Model trained successfully on training data.\n")

    # -------------------------------------------------------------------------
    # STEP 2: Make predictions on train and test sets
    # We predict prices for houses the model has and has not seen during training.
    # -------------------------------------------------------------------------
    print("STEP 2: Making predictions")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print("  Predictions generated for training and test sets.\n")

    # -------------------------------------------------------------------------
    # STEP 3: Evaluate model performance
    # We measure how close predictions are to actual prices using standard metrics.
    # -------------------------------------------------------------------------
    print("STEP 3: Model performance metrics")
    print("-" * 60)
    print("TEST SET METRICS:")
    print("-" * 60)

    # MAE — average dollar error in predictions (easy to interpret in USD)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"  MAE  (Mean Absolute Error):     ${test_mae:,.2f}")
    print("       -> Average dollar error in predictions")

    # MSE — penalizes large errors more heavily than small ones
    test_mse = mean_squared_error(y_test, y_test_pred)
    print(f"  MSE  (Mean Squared Error):      ${test_mse:,.2f}")
    print("       -> Penalizes large errors more than small ones")

    # RMSE — same unit as price (USD), easier to understand than MSE
    test_rmse = np.sqrt(test_mse)
    print(f"  RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
    print("       -> Typical prediction error in dollars")

    # R² — how much of price variation the model explains (1.0 = perfect)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"  R2   (R-squared Score):         {test_r2:.4f}")
    print("       -> Fraction of price variation explained (1.0 = perfect)")

    print("-" * 60)
    print("TRAIN SET R2 (overfitting check):")
    print("-" * 60)
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"  Train R2: {train_r2:.4f}")
    print(f"  Test R2:  {test_r2:.4f}")
    if train_r2 - test_r2 > 0.05:
        print("  Note: Train R2 is noticeably higher than test R2 - possible overfitting.")
    else:
        print("  Note: Train and test R2 are similar - model generalizes well.")
    print()

    # -------------------------------------------------------------------------
    # STEP 4: Print model coefficients (feature importance)
    # Positive coefficient = feature increases predicted price.
    # Negative coefficient = feature decreases predicted price.
    # -------------------------------------------------------------------------
    print("STEP 4: Top 15 most important features (by coefficient magnitude)")
    print("-" * 60)
    coefficients_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    coefficients_df = coefficients_df.sort_values(
        "abs_coefficient", ascending=False
    ).drop(columns=["abs_coefficient"])

    print(coefficients_df.head(15).to_string(index=False))
    print(f"\n  Intercept (base price): ${model.intercept_:,.2f}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # STEP 5: Return trained model and predictions for downstream use
    # -------------------------------------------------------------------------
    return model, y_test_pred, y_train_pred


def save_model(model: LinearRegression, scaler: StandardScaler) -> None:
    """
    Save the trained model and scaler to the outputs/ folder using pickle.

    Args:
        model: Trained LinearRegression model.
        scaler: Fitted StandardScaler used during preprocessing.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "house_price_model.pkl"
    scaler_path = OUTPUT_DIR / "scaler.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to:  {model_path}")

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")


def predict_house_price(
    model: LinearRegression,
    scaler: StandardScaler,
    feature_names: list[str],
    input_dict: dict,
) -> float:
    """
    Predict house price for a single property from a feature dictionary.

    Args:
        model: Trained LinearRegression model.
        scaler: Fitted StandardScaler from preprocessing.
        feature_names: Column names the model expects (after one-hot encoding).
        input_dict: House features including 'city' as a string.

    Returns:
        Predicted price in USD.
    """
    try:
        if not isinstance(input_dict, dict):
            raise TypeError("input_dict must be a dictionary of feature names and values.")

        if "city" not in input_dict:
            raise ValueError("Missing required key: 'city'")

        city = str(input_dict["city"]).strip()
        if not city:
            raise ValueError("'city' cannot be empty.")

        # Start with all features set to 0 (including all city dummy columns)
        row = {name: 0 for name in feature_names}

        # Fill numerical / non-city features from the input dictionary
        for key, value in input_dict.items():
            if key == "city":
                continue
            if key not in feature_names:
                raise ValueError(
                    f"Unknown feature '{key}'. "
                    f"Valid features: {[n for n in feature_names if not n.startswith('city_')]}"
                )
            row[key] = value

        # One-hot encode city: set the matching dummy column to 1 if it exists
        city_columns = [name for name in feature_names if name.startswith("city_")]
        city_column = f"city_{city}"

        if city_column in feature_names:
            row[city_column] = 1
        elif city_columns:
            # Unknown city or reference category (dropped by drop_first=True) — all zeros
            pass

        # Build DataFrame with columns in the exact order the scaler expects
        features_df = pd.DataFrame([row], columns=feature_names)
        features_scaled = scaler.transform(features_df)
        predicted_price = float(model.predict(features_scaled)[0])

        if predicted_price < 0:
            print(
                "  Warning: Model predicted a negative price. "
                "This can happen with extreme inputs outside training data."
            )

        return predicted_price

    except (ValueError, TypeError, KeyError) as e:
        raise ValueError(f"Prediction failed: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during prediction: {e}. "
            "Check that the model and scaler were fitted on the same features."
        ) from e
