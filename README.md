# House Price Predictor

[Python] [pandas] [scikit-learn] [matplotlib] [seaborn] [Jupyter]

---

## Project Overview

This project predicts house sale prices in **King County, Washington, USA** using **Multiple Linear Regression (MLR)**. It loads real estate sales data, cleans and explores it, trains a regression model, evaluates performance, and generates visual reports. The dataset contains roughly 4,600 home sales from 2014 with 18 original features such as size, condition, and location.

---

## Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | King County, Washington, USA house sales |
| **Size** | ~4,600 rows, 18 original columns |
| **Time period** | 2014 |
| **Target variable** | `price` (sale price in USD) |

**Columns removed before modeling:** `date`, `street`, `statezip`, `country` (not useful or redundant for MLR).

**Features used in the model:**

| Feature | Description |
|---------|-------------|
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `sqft_living` | Total living area in square feet |
| `sqft_lot` | Lot size in square feet |
| `floors` | Number of floors (including fractional, e.g. 1.5) |
| `waterfront` | Whether the home has waterfront access (0 = no, 1 = yes) |
| `view` | Quality of the view (0 = none, 4 = excellent) |
| `condition` | Overall condition rating (1 = poor, 5 = excellent) |
| `sqft_above` | Square footage above ground |
| `sqft_basement` | Square footage of basement |
| `yr_built` | Year the house was built |
| `yr_renovated` | Year of last renovation (0 if never renovated) |
| `city` | City location (one-hot encoded into dummy columns; one reference city dropped to avoid multicollinearity) |

After encoding, the model uses **55 features** (12 numerical + 43 city dummy columns).

---

## Problem Statement

House prices depend on many factors—size, location, condition, and amenities. Buyers, sellers, and agents need fair estimates to make informed decisions. A data-driven model helps estimate market value from property features, supports comparison across neighborhoods, and provides a repeatable baseline before more complex pricing tools are used.

---

## Tech Stack

- **Python** — core language
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib** — plotting
- **seaborn** — statistical visualizations
- **scikit-learn** — preprocessing, MLR, metrics
- **Jupyter** — interactive analysis notebook

---

## Project Structure

```
house_price_predictor/
├── data/
│   └── data.csv                 # King County house sales dataset
├── notebooks/
│   └── house_price_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Load and explore raw data
│   ├── preprocessor.py          # Clean, encode, split, scale
│   ├── visualizer.py            # EDA and result plots
│   └── model.py                 # Train, evaluate, predict, save
├── outputs/                     # Saved plots and model files (.pkl)
├── main.py                      # Full pipeline entry point
├── requirements.txt
└── README.md
```

---

## Installation & Setup

```bash
git clone <repo-url>
cd house_price_predictor
pip install -r requirements.txt
```

Place the dataset at `data/data.csv` if it is not already included.

Optional: create a virtual environment first.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

---

## How to Run

**Full pipeline (recommended):**

```bash
python main.py
```

This runs data loading, preprocessing, EDA, model training, result plots, and three example predictions. Outputs are saved to `outputs/`.

**Jupyter notebook (step-by-step analysis):**

```bash
jupyter notebook notebooks/house_price_analysis.ipynb
```

Run all cells from the `notebooks/` directory (or ensure paths resolve to the project root).

---

## Model Explanation

Multiple Linear Regression predicts price as a weighted sum of all input features plus a base value (intercept). Each feature gets a coefficient: a positive value means higher values of that feature tend to increase price; a negative value means they tend to decrease it. The model finds the coefficients that best fit the training data in a least-squares sense—minimizing the overall squared error between predicted and actual prices. No hidden layers or complex rules; it is a straightforward linear relationship across many variables at once.

---

## Evaluation Results

Run `python main.py` and record your test-set metrics below.

| Metric | Value |
|--------|-------|
| **MAE** (Mean Absolute Error) | [INSERT MAE HERE] |
| **RMSE** (Root Mean Squared Error) | [INSERT RMSE HERE] |
| **R² Score (test set)** | [INSERT R² SCORE HERE] |
| **R² Score (train set)** | [INSERT TRAIN R² HERE] |

*Example from a typical run: Test R² ≈ 0.68, RMSE ≈ $219,399, MAE ≈ $131,870.*

---

## Key Findings

After running the project and reviewing plots in `outputs/`, summarize your insights:

- [INSERT KEY FINDING 1 — e.g. strongest predictors of price]
- [INSERT KEY FINDING 2 — e.g. effect of location or waterfront]
- [INSERT KEY FINDING 3 — e.g. model limitations or error patterns]

---

## Future Improvements

- Try **Ridge** or **Lasso** regression to reduce overfitting and handle correlated features
- Add **neighborhood-level** features (e.g. zip-level aggregates, school ratings)
- Try **polynomial features** for `sqft_living` to capture non-linear size effects
- Build a simple **web app** using Flask or Streamlit for interactive price estimates

---

## Author

**Gul Shair** | DataCrumbs Assignment 
