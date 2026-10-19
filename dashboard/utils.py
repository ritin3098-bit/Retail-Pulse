"""
Shared styling and data-loading helpers for every RetailPulse dashboard page.

All loaders read pre-computed CSVs from data/processed/ (produced by the
src/ pipeline scripts) rather than retraining models at runtime — this is
what keeps the dashboard's startup under the brief's 8-second target.
"""

import os
import pandas as pd
import streamlit as st

DATA_DIR = "data/processed"
FIG_DIR = "reports/figures"

THEME_CSS = """
<style>
    .main { background-color: #f7f9fc; }

    h1, h2, h3 { color: #1b2a4a; font-family: 'Segoe UI', sans-serif; }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    [data-testid="stMetricLabel"] { color: #5b6b8c; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #1b2a4a; }

    .segment-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #1b2a4a;
    }
    section[data-testid="stSidebar"] * { color: #f0f3fa !important; }

    .status-reorder { color: #d62728; font-weight: 700; }
    .status-ok { color: #2ca02c; font-weight: 700; }
</style>
"""

SEGMENT_COLORS = {
    "Champions": "#2ca02c",
    "Loyal Customers": "#1f77b4",
    "Promising": "#9467bd",
    "At Risk": "#ff7f0e",
    "New/Low-Value": "#8c8c8c",
    "Lost/Dormant": "#d62728",
}


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def _safe_read_csv(name, **kwargs):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_clean():
    return _safe_read_csv("clean_all.csv", parse_dates=["InvoiceDate"])


@st.cache_data
def load_customers():
    return _safe_read_csv("clean_customers.csv", parse_dates=["InvoiceDate"])


@st.cache_data
def load_segments():
    return _safe_read_csv("rfm_segmented.csv")


@st.cache_data
def load_forecast_results():
    return _safe_read_csv("forecast_results.csv", parse_dates=["ds"])


@st.cache_data
def load_forecast_metrics():
    return _safe_read_csv("forecast_metrics.csv")


@st.cache_data
def load_churn_results():
    return _safe_read_csv("churn_results.csv")


@st.cache_data
def load_churn_metrics():
    return _safe_read_csv("churn_metrics.csv")


@st.cache_data
def load_inventory():
    return _safe_read_csv("inventory_report.csv")


def figure_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def show_figure_if_exists(name: str, caption: str = None, **kwargs):
    """Display a saved figure, or a friendly placeholder if the pipeline hasn't been run yet."""
    path = figure_path(name)
    if os.path.exists(path):
        st.image(path, caption=caption, **kwargs)
    else:
        st.info(f"Figure not found yet: `{path}`. Run the corresponding src/ script first.")


def missing_data_notice(what: str, command: str):
    st.warning(f"No {what} found yet. Run `{command}` first, then reload this page.")
