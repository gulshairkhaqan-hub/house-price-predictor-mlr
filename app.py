# Streamlit web app for the House Price Predictor (King County, WA).
# Reuses the existing pipeline in src/ — does not modify any src/ file.

import sys
from pathlib import Path

import streamlit as st

# Make src/ importable, mirroring the convention used in main.py
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend, safe for Streamlit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader import DATA_PATH, TARGET_COLUMN
from model import predict_house_price, train_and_evaluate
from preprocessor import preprocess_data

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Custom CSS — premium dark theme
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [class*="css"] {
        font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    }
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    section.main > div { padding-top: 1rem; }

    /* Hide default Streamlit chrome we don't need */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---------- Headings & text ---------- */
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
    p, span, label, div { color: #e2e8f0; }

    /* ---------- Hero banner ---------- */
    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #312e81 60%, #6366f1 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff !important;
    }
    .hero p {
        margin: 0.4rem 0 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    /* ---------- Stat chips ---------- */
    .chip-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin: 0.75rem 0 1.25rem 0;
    }
    .chip {
        background: rgba(99, 102, 241, 0.15);
        color: #c7d2fe;
        border: 1px solid #6366f1;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 500;
    }

    /* ---------- Cards ---------- */
    .card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-left: 4px solid #6366f1;
        border-radius: 12px;
        padding: 1.1rem 1.25rem 0.6rem 1.25rem;
        margin-bottom: 1rem;
    }
    .card-title {
        color: #f8fafc;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0 0 0.6rem 0;
    }

    /* Prediction card with indigo gradient top border */
    .prediction-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.6rem 1.5rem;
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    .prediction-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1 0%, #22c55e 100%);
    }
    .prediction-label {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
        font-weight: 500;
    }
    .prediction-price {
        color: #22c55e;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.2rem 0 0.4rem 0;
    }
    .prediction-note {
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* Mini metrics below the prediction */
    .mini-metrics {
        display: flex;
        gap: 0.75rem;
        margin-top: 0.5rem;
    }
    .mini-metric {
        flex: 1;
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.85rem 0.9rem;
        text-align: center;
    }
    .mini-metric .mm-label {
        color: #94a3b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }
    .mini-metric .mm-value {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* ---------- Inputs ---------- */
    .stSlider [data-baseweb="slider"] > div > div > div { background: #6366f1 !important; }
    .stSlider [role="slider"] { background-color: #6366f1 !important; border-color: #6366f1 !important; }
    .stSlider span { color: #e2e8f0 !important; }

    div[data-baseweb="select"] > div {
        background-color: #1e293b !important;
        color: white !important;
        border-color: #334155 !important;
    }
    [data-baseweb="popover"] {
        background-color: #1e293b !important;
    }
    [data-baseweb="menu"] {
        background-color: #1e293b !important;
    }
    [role="option"] {
        background-color: #1e293b !important;
        color: white !important;
    }
    [role="option"]:hover {
        background-color: #334155 !important;
    }
    .stNumberInput input {
        background-color: #0f172a !important;
        color: #e2e8f0 !important;
        border-color: #334155 !important;
    }
    /* Radio buttons */
    .stRadio label, .stRadio div { color: #e2e8f0 !important; }
    .stRadio [role="radiogroup"] > label {
        background-color: #0f172a;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.35rem 0.75rem;
        margin-right: 0.5rem;
    }
    label, .stSlider label, .stSelectbox label, .stNumberInput label {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        margin-top: 2rem;
        padding: 1.1rem;
        background-color: #0b1220;
        border-top: 1px solid #1e293b;
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        border-radius: 8px;
    }

    /* Section divider */
    .soft-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #334155, transparent);
        margin: 1rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached data + model bootstrap
# Runs once per session; reused across every interaction.
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train():
    """
    Load the dataset, run the full preprocessing pipeline, and train the MLR model.

    Returns:
        model: Trained LinearRegression model.
        scaler: Fitted StandardScaler.
        feature_names: List of feature columns expected by the model.
        df_clean: Cleaned (price>0, deduped) raw DataFrame for charts and city list.
    """
    # Raw frame for charts and city options (kept separate from the modeling pipeline)
    df_clean = pd.read_csv(DATA_PATH)
    df_clean = df_clean[df_clean[TARGET_COLUMN] != 0].drop_duplicates()

    # Full pipeline (same logic as src/preprocessor.py — we just call it)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(DATA_PATH)

    # Train and evaluate the MLR model
    model, _, _ = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names)

    return model, scaler, feature_names, df_clean


# -----------------------------------------------------------------------------
# Hero header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏠 House Price Predictor</h1>
        <p>Premium home valuations powered by Multiple Linear Regression on real King County sales data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Bootstrap (with graceful error handling and spinner)
# -----------------------------------------------------------------------------
try:
    with st.spinner("Training model on King County data..."):
        model, scaler, feature_names, df_clean = load_and_train()
except FileNotFoundError:
    st.error(f"Dataset not found at: {DATA_PATH}. Please place data.csv in the data/ folder.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load data or train the model: {e}")
    st.stop()

# Stat chips below the hero (counts pulled from the actual loaded dataset)
total_houses = len(df_clean)
st.markdown(
    f"""
    <div class="chip-row">
        <span class="chip">📊 {total_houses:,}+ Houses Analyzed</span>
        <span class="chip">📍 King County, WA</span>
        <span class="chip">🤖 MLR Model</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Build the city dropdown from the 15 most frequent cities in the dataset
try:
    top_cities = df_clean["city"].value_counts().head(15).index.tolist()
except Exception as e:
    st.error(f"Could not determine top cities from dataset: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# Two-column layout — inputs on the left, results on the right
# -----------------------------------------------------------------------------
input_col, result_col = st.columns([1.05, 1], gap="large")

# ---------- LEFT: input cards ----------
with input_col:
    try:
        # ---- Card 1: Size & Layout ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📐 Size & Layout</div>', unsafe_allow_html=True)

        sqft_living = st.number_input(
            "Living Area (sqft)",
            min_value=500,
            max_value=8000,
            value=1800,
            step=50,
        )
        bedrooms = st.number_input(
            "Bedrooms", min_value=1, max_value=10, value=3, step=1
        )
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=0.5,
            max_value=8.0,
            value=2.0,
            step=0.25,
        )
        floors = st.number_input(
            "Floors",
            min_value=1.0,
            max_value=3.5,
            value=1.0,
            step=0.5,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Card 2: Quality & Features ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">✨ Quality & Features</div>',
            unsafe_allow_html=True,
        )

        condition = st.number_input(
            "Condition (1=Poor, 5=Excellent)",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
        )
        view = st.number_input(
            "View Quality (0-4)", min_value=0, max_value=4, value=0, step=1
        )
        waterfront_label = st.radio(
            "Waterfront Property?", options=["No", "Yes"], index=0, horizontal=True
        )
        yr_built = st.number_input(
            "Year Built", min_value=1900, max_value=2015, value=1990, step=1
        )
        yr_renovated = st.number_input(
            "Year Renovated (0 if never)",
            min_value=0,
            max_value=2015,
            value=0,
            step=1,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Card 3: Location & Land ----
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-title">📍 Location & Land</div>',
            unsafe_allow_html=True,
        )

        city = st.selectbox("City", options=top_cities, index=0)
        sqft_lot = st.number_input(
            "Lot Size (sqft)",
            min_value=500,
            max_value=100000,
            value=8000,
            step=500,
        )
        sqft_basement = st.number_input(
            "Basement Area (sqft)",
            min_value=0,
            max_value=2000,
            value=0,
            step=50,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error rendering inputs: {e}")
        st.stop()

# -----------------------------------------------------------------------------
# Build the feature dictionary the model expects
# -----------------------------------------------------------------------------
# sqft_above is derived; never asked from the user
sqft_above = max(sqft_living - sqft_basement, 0)
waterfront = 1 if waterfront_label == "Yes" else 0

input_dict = {
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "sqft_living": sqft_living,
    "sqft_lot": sqft_lot,
    "floors": floors,
    "waterfront": waterfront,
    "view": view,
    "condition": condition,
    "sqft_above": sqft_above,
    "sqft_basement": sqft_basement,
    "yr_built": yr_built,
    "yr_renovated": yr_renovated,
    "city": city,
}

# -----------------------------------------------------------------------------
# Prediction (predict_house_price handles the city one-hot encoding internally)
# -----------------------------------------------------------------------------
try:
    predicted_price = predict_house_price(model, scaler, feature_names, input_dict)
    predicted_price = max(predicted_price, 0.0)  # Guard against rare negative outputs
except (ValueError, RuntimeError) as e:
    st.error(f"Prediction failed: {e}")
    st.stop()
except Exception as e:
    st.error(f"Unexpected error during prediction: {e}")
    st.stop()

price_per_sqft = predicted_price / sqft_living if sqft_living > 0 else 0.0

# -----------------------------------------------------------------------------
# Matplotlib styling helper — applied per-figure to keep things isolated
# -----------------------------------------------------------------------------
def _style_dark_axes(fig, ax):
    fig.patch.set_facecolor("#1e293b")
    ax.set_facecolor("#1e293b")
    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#e2e8f0")
    ax.xaxis.label.set_color("#e2e8f0")
    ax.yaxis.label.set_color("#e2e8f0")
    ax.title.set_color("#f8fafc")
    ax.grid(True, color="#334155", alpha=0.6, linewidth=0.6)
    legend = ax.get_legend()
    if legend is not None:
        legend.get_frame().set_facecolor("#0f172a")
        legend.get_frame().set_edgecolor("#334155")
        for text in legend.get_texts():
            text.set_color("#e2e8f0")


# ---------- RIGHT: results ----------
with result_col:
    # Prediction card
    st.markdown(
        f"""
        <div class="prediction-card">
            <div class="prediction-label">🏠 Estimated Price</div>
            <div class="prediction-price">${predicted_price:,.0f}</div>
            <div class="prediction-note">Based on MLR model trained on 2014 data</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mini metrics row
    st.markdown(
        f"""
        <div class="mini-metrics">
            <div class="mini-metric">
                <div class="mm-label">Living Area</div>
                <div class="mm-value">{sqft_living:,} sqft</div>
            </div>
            <div class="mini-metric">
                <div class="mm-label">Bedrooms</div>
                <div class="mm-value">{bedrooms}</div>
            </div>
            <div class="mini-metric">
                <div class="mm-label">Price / sqft</div>
                <div class="mm-value">${price_per_sqft:,.0f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

    # Chart 1: Living area vs price (binned line chart)
    st.markdown("**How does Living Area affect Price?**")
    try:
        with plt.style.context("dark_background"):
            bin_edges = np.arange(500, 8001, 500)
            binned = pd.cut(
                df_clean["sqft_living"], bins=bin_edges, include_lowest=True
            )
            avg_by_bin = (
                df_clean.groupby(binned, observed=True)[TARGET_COLUMN]
                .mean()
                .reset_index()
            )
            avg_by_bin["sqft_mid"] = avg_by_bin["sqft_living"].apply(
                lambda interval: (interval.left + interval.right) / 2
            )
            avg_by_bin = avg_by_bin.dropna(subset=[TARGET_COLUMN])

            fig1, ax1 = plt.subplots(figsize=(6, 3.4))
            ax1.plot(
                avg_by_bin["sqft_mid"],
                avg_by_bin[TARGET_COLUMN],
                color="#6366f1",
                marker="o",
                linewidth=2.2,
            )
            ax1.axvline(
                x=sqft_living,
                color="#ef4444",
                linestyle="--",
                linewidth=2,
                label=f"Your input: {sqft_living:,} sqft",
            )
            ax1.set_xlabel("Living Area (sqft)")
            ax1.set_ylabel("Average Price (USD)")
            ax1.set_title("Average Price by Living Area")
            ax1.legend()
            _style_dark_axes(fig1, ax1)
            fig1.tight_layout()
            st.pyplot(fig1)
            plt.close(fig1)
    except Exception as e:
        st.error(f"Could not render living area chart: {e}")

    # Chart 2: Price distribution histogram
    st.markdown("**Price Distribution in Dataset**")
    try:
        with plt.style.context("dark_background"):
            fig2, ax2 = plt.subplots(figsize=(6, 3.4))
            ax2.hist(
                df_clean[TARGET_COLUMN],
                bins=50,
                color="#6366f1",
                edgecolor="#1e293b",
                alpha=0.9,
            )
            ax2.axvline(
                x=predicted_price,
                color="#ef4444",
                linestyle="--",
                linewidth=2,
                label=f"Predicted: ${predicted_price:,.0f}",
            )
            ax2.set_xlabel("Price (USD)")
            ax2.set_ylabel("Number of Houses")
            ax2.set_title("All House Prices in Training Data")
            ax2.legend()
            _style_dark_axes(fig2, ax2)
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
    except Exception as e:
        st.error(f"Could not render price distribution chart: {e}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        Model: Multiple Linear Regression &nbsp;|&nbsp; Dataset: King County USA (2014)<br/>
        Built for DataCrumbs Internship Assignment<br/>
        © 2024 House Price Predictor | DataCrumbs Internship
    </div>
    """,
    unsafe_allow_html=True,
)
