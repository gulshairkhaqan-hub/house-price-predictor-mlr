# Entry point for the House Price Predictor pipeline.

import sys
import traceback
from pathlib import Path

# Allow imports from src/ when running: python main.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import DATA_PATH, load_and_analyze
from model import predict_house_price, save_model, train_and_evaluate
from preprocessor import preprocess_data
from visualizer import (
    load_clean_data,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    print_summary_stats,
    run_eda,
)

# Example houses for Step 6
EXAMPLE_HOUSES = [
    (
        "Small affordable house",
        {
            "sqft_living": 1200,
            "bedrooms": 3,
            "bathrooms": 1.0,
            "floors": 1.0,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "sqft_above": 1200,
            "sqft_basement": 0,
            "yr_built": 1985,
            "yr_renovated": 0,
            "sqft_lot": 6000,
            "city": "Kent",
        },
    ),
    (
        "Mid-range family house",
        {
            "sqft_living": 2200,
            "bedrooms": 4,
            "bathrooms": 2.5,
            "floors": 2.0,
            "waterfront": 0,
            "view": 0,
            "condition": 4,
            "sqft_above": 2200,
            "sqft_basement": 0,
            "yr_built": 2000,
            "yr_renovated": 0,
            "sqft_lot": 9000,
            "city": "Bellevue",
        },
    ),
    (
        "Premium waterfront house",
        {
            "sqft_living": 3500,
            "bedrooms": 5,
            "bathrooms": 3.5,
            "floors": 2.0,
            "waterfront": 1,
            "view": 4,
            "condition": 5,
            "sqft_above": 3500,
            "sqft_basement": 500,
            "yr_built": 2010,
            "yr_renovated": 2015,
            "sqft_lot": 12000,
            "city": "Seattle",
        },
    ),
]


def print_example_prediction(
    example_num: int,
    title: str,
    input_dict: dict,
    model,
    scaler,
    feature_names: list[str],
) -> None:
    """Print input features and predicted price for one example house."""
    print(f"\nExample {example_num} - {title}")
    print("-" * 40)
    print("Input features:")
    for key, value in input_dict.items():
        print(f"  {key}: {value}")

    predicted_price = predict_house_price(
        model, scaler, feature_names, input_dict
    )
    print(f"\nPredicted price: ${predicted_price:,.0f}")


def main() -> None:
    print("=== House Price Predictor - King County USA ===\n")

    model = None
    scaler = None
    feature_names = None
    y_test = None
    y_test_pred = None

    # -------------------------------------------------------------------------
    # STEP 1: Load and analyze raw data
    # -------------------------------------------------------------------------
    try:
        print("STEP 1: Loading and Analyzing Dataset")
        print("-" * 60)
        load_and_analyze()
        print()
    except FileNotFoundError:
        print("ERROR in Step 1: Dataset not found.")
        print(f"  Place data.csv at: {DATA_PATH}")
        return
    except Exception as e:
        print(f"ERROR in Step 1: Failed to load or analyze data.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # STEP 2: Preprocess data for modeling
    # -------------------------------------------------------------------------
    try:
        print("STEP 2: Cleaning and Preprocessing Data")
        print("-" * 60)
        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(
            DATA_PATH
        )
        print()
    except FileNotFoundError:
        print("ERROR in Step 2: Dataset not found.")
        print(f"  Place data.csv at: {DATA_PATH}")
        return
    except Exception as e:
        print(f"ERROR in Step 2: Preprocessing failed.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # STEP 3: Exploratory data analysis
    # -------------------------------------------------------------------------
    try:
        print("STEP 3: Exploratory Data Analysis")
        print("-" * 60)
        df_eda = load_clean_data(DATA_PATH)
        run_eda(df_eda)
        print()
        print_summary_stats(df_eda)
        print()
    except Exception as e:
        print(f"ERROR in Step 3: EDA failed.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # STEP 4: Train and evaluate the MLR model
    # -------------------------------------------------------------------------
    try:
        print("STEP 4: Training and Evaluating MLR Model")
        print("-" * 60)
        model, y_test_pred, _ = train_and_evaluate(
            X_train, X_test, y_train, y_test, feature_names
        )
        save_model(model, scaler)
        print()
    except Exception as e:
        print(f"ERROR in Step 4: Model training failed.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # STEP 5: Visualize model results
    # -------------------------------------------------------------------------
    try:
        print("STEP 5: Visualizing Results")
        print("-" * 60)
        plot_actual_vs_predicted(y_test, y_test_pred)
        plot_residuals(y_test, y_test_pred)
        plot_feature_importance(model, feature_names)
        print()
    except Exception as e:
        print(f"ERROR in Step 5: Result visualization failed.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    # -------------------------------------------------------------------------
    # STEP 6: Example predictions on new houses
    # -------------------------------------------------------------------------
    try:
        print("STEP 6: Example Predictions")
        print("-" * 60)
        for i, (title, house_features) in enumerate(EXAMPLE_HOUSES, start=1):
            print_example_prediction(
                i, title, house_features, model, scaler, feature_names
            )
        print()
    except (ValueError, RuntimeError) as e:
        print(f"ERROR in Step 6: Prediction failed.")
        print(f"  Details: {e}")
        return
    except Exception as e:
        print(f"ERROR in Step 6: Unexpected error during predictions.")
        print(f"  Details: {e}")
        traceback.print_exc()
        return

    print("Project Complete! Check outputs/ folder for all graphs.")


if __name__ == "__main__":
    main()
