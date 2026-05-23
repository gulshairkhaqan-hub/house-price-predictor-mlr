# Generates and saves exploratory and model-related visualizations.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend — saves plots without opening a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

from data_loader import DATA_PATH, NUMERICAL_COLUMNS, TARGET_COLUMN

# Columns dropped before EDA — same as preprocessor (city is kept for city analysis)
COLUMNS_TO_DROP = ["date", "street", "statezip", "country"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def load_clean_data(filepath: Path | str = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset and apply basic cleaning steps needed for EDA.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        Cleaned DataFrame ready for visualization.
    """
    df = pd.read_csv(filepath)
    df = df[df[TARGET_COLUMN] != 0]
    df = df.drop(columns=COLUMNS_TO_DROP)
    return df


def print_summary_stats(df: pd.DataFrame) -> None:
    """
    Print key summary statistics to understand the dataset at a glance.
    """
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)

    avg_price = df[TARGET_COLUMN].mean()
    min_price = df[TARGET_COLUMN].min()
    max_price = df[TARGET_COLUMN].max()
    print(f"Average price:              ${avg_price:,.2f}")
    print(f"Minimum price:              ${min_price:,.2f}")
    print(f"Maximum price:              ${max_price:,.2f}")

    avg_sqft = df["sqft_living"].mean()
    print(f"Average living area (sqft): {avg_sqft:,.0f}")

    most_common_bedrooms = df["bedrooms"].mode().iloc[0]
    print(f"Most common bedrooms:       {most_common_bedrooms:.0f}")

    waterfront_pct = (df["waterfront"].sum() / len(df)) * 100
    print(f"Waterfront houses:          {waterfront_pct:.1f}%")
    print("=" * 60)


def run_eda(df: pd.DataFrame) -> None:
    """
    Create and save exploratory data analysis plots to the outputs/ folder.

    Args:
        df: Cleaned DataFrame (price=0 rows and useless columns already removed).
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_style("whitegrid")

    # -------------------------------------------------------------------------
    # PLOT 1: Price Distribution
    # Shows how house prices are spread across the dataset.
    # Prices are skewed right — many affordable homes, few very expensive ones.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COLUMN], bins=50, kde=True, color="steelblue")
    plt.title("House Price Distribution")
    plt.xlabel("Price (USD)")
    plt.ylabel("Number of Houses")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "price_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 2: Correlation Heatmap
    # Reveals which numerical features move together with price.
    # Values close to +1 or -1 mean a strong linear relationship.
    # -------------------------------------------------------------------------
    numerical_df = df[[TARGET_COLUMN] + NUMERICAL_COLUMNS]
    plt.figure(figsize=(14, 10))
    correlation = numerical_df.corr()
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        square=True,
    )
    plt.title("Feature Correlation with Price")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "correlation_heatmap.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 3: Price vs Living Area
    # Larger homes generally cost more — this scatter shows that trend visually.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["sqft_living"],
        df[TARGET_COLUMN],
        color="steelblue",
        alpha=0.4,
    )
    plt.title("Price vs Living Area (sqft)")
    plt.xlabel("Living Area (sqft)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "price_vs_sqft.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 4: Price by Bedrooms
    # Boxplots show the median price and spread for each bedroom count.
    # Helps spot whether more bedrooms consistently means higher prices.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="bedrooms", y=TARGET_COLUMN, color="steelblue")
    plt.title("Price Distribution by Number of Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "price_vs_bedrooms.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 5: Price by Condition
    # Condition rates house quality (1=poor, 5=excellent).
    # This plot shows whether better condition homes sell for more.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="condition", y=TARGET_COLUMN, color="steelblue")
    plt.title("Price Distribution by House Condition")
    plt.xlabel("Condition")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "price_vs_condition.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 6: Price by Waterfront
    # Compares prices for waterfront (1) vs non-waterfront (0) properties.
    # Waterfront homes often command a premium — this plot makes that visible.
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="waterfront", y=TARGET_COLUMN, color="steelblue")
    plt.title("Waterfront vs Non-Waterfront Price Comparison")
    plt.xlabel("Waterfront (0 = No, 1 = Yes)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "price_vs_waterfront.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")

    # -------------------------------------------------------------------------
    # PLOT 7: Top 10 Cities by Average Price
    # Location strongly affects price — this bar chart highlights the priciest cities.
    # -------------------------------------------------------------------------
    city_avg_price = (
        df.groupby("city")[TARGET_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=city_avg_price,
        x="city",
        y=TARGET_COLUMN,
        color="steelblue",
    )
    plt.title("Top 10 Cities by Average House Price")
    plt.xlabel("City")
    plt.ylabel("Average Price (USD)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "top_cities_price.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


# =============================================================================
# Model result visualizations
# =============================================================================


def plot_actual_vs_predicted(
    y_test: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """
    Scatter plot comparing actual vs predicted prices on the test set.

    Points close to the red diagonal line indicate accurate predictions.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_test_pred, color="steelblue", alpha=0.4)

    # Perfect predictions would fall on this diagonal line (actual == predicted)
    min_price = min(np.min(y_test), np.min(y_test_pred))
    max_price = max(np.max(y_test), np.max(y_test_pred))
    plt.plot(
        [min_price, max_price],
        [min_price, max_price],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Perfect predictions",
    )

    plt.title("Actual vs Predicted House Prices")
    plt.xlabel("Actual Price (USD)")
    plt.ylabel("Predicted Price (USD)")
    plt.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "actual_vs_predicted.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_residuals(
    y_test: pd.Series | np.ndarray,
    y_test_pred: np.ndarray,
) -> None:
    """
    Plot residual analysis to check model quality.

    Residuals should be randomly scattered around 0 for a good model.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    residuals = np.asarray(y_test) - np.asarray(y_test_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: residuals vs predicted values — look for patterns (bad) vs random scatter (good)
    axes[0].scatter(y_test_pred, residuals, color="steelblue", alpha=0.4)
    axes[0].axhline(y=0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title("Residuals vs Predicted Values")
    axes[0].set_xlabel("Predicted Price (USD)")
    axes[0].set_ylabel("Residual (USD)")

    # Right: distribution of residuals — should be roughly centered at zero
    sns.histplot(residuals, kde=True, color="steelblue", ax=axes[1])
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (USD)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "residuals.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def plot_feature_importance(
    model: LinearRegression,
    feature_names: list[str],
) -> None:
    """
    Horizontal bar chart of the top 15 features by coefficient magnitude.

    Green bars increase predicted price; red bars decrease it.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coefficients_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_}
    )
    coefficients_df["abs_coefficient"] = coefficients_df["coefficient"].abs()
    top_features = coefficients_df.nlargest(15, "abs_coefficient").sort_values(
        "coefficient", ascending=True
    )

    top_features = top_features.copy()
    top_features["direction"] = np.where(
        top_features["coefficient"] >= 0, "positive", "negative"
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=top_features,
        y="feature",
        x="coefficient",
        orient="h",
        hue="direction",
        palette={"positive": "green", "negative": "red"},
        legend=False,
    )
    plt.title("Top 15 Feature Coefficients (Impact on Price)")
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    df = load_clean_data()
    print_summary_stats(df)
    print()
    run_eda(df)
