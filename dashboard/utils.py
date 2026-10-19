"""
Shared styling and data-loading utilities for the RetailPulse dashboard.

Features:
- Project-relative paths that work locally and on Streamlit Community Cloud.
- Automatic generation of the cleaned sales dataset when it is missing.
- Cached CSV loaders.
- Shared dashboard styling.
- Safe figure loading and missing-data messages.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

# Project root is the parent directory of dashboard/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

RAW_DATA_FILE = RAW_DATA_DIR / "online_retail_II.xlsx"
CLEAN_DATA_FILE = DATA_DIR / "clean_all.csv"


# ============================================================
# DASHBOARD THEME
# ============================================================

THEME_CSS = """
<style>
    .main {
        background-color: #f7f9fc;
    }

    h1, h2, h3 {
        color: #1b2a4a;
        font-family: 'Segoe UI', sans-serif;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e3e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #5b6b8c;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #1b2a4a;
    }

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

    section[data-testid="stSidebar"] * {
        color: #f0f3fa !important;
    }

    .status-reorder {
        color: #d62728;
        font-weight: 700;
    }

    .status-ok {
        color: #2ca02c;
        font-weight: 700;
    }
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


# ============================================================
# THEME
# ============================================================

def inject_theme():
    """Apply the shared RetailPulse dashboard theme."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============================================================
# DATA DIRECTORY INITIALIZATION
# ============================================================

def ensure_output_directories():
    """Create output directories when they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUTOMATIC DATA PREPROCESSING
# ============================================================

def ensure_clean_data():
    """
    Check whether the saved cleaned dataset exists.
    Do not run data preprocessing automatically.
    """

    if CLEAN_DATA_FILE.is_file() and CLEAN_DATA_FILE.stat().st_size > 0:
        return True

    st.error("The cleaned sales dataset is missing or empty.")

    st.info(
        "Expected file: data/processed/clean_all.csv. "
        "Please verify that the file exists in your GitHub repository."
    )

    return False
# ============================================================
# SAFE CSV LOADER
# ============================================================

def _safe_read_csv(name, **kwargs):
    """
    Read a CSV file from data/processed/.

    Returns:
        pandas.DataFrame if the file exists.
        None if the file is missing.
    """

    path = DATA_DIR / name

    if not path.exists():
        return None

    try:
        return pd.read_csv(path, **kwargs)

    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read `{name}`: {exc}")
        return None


# ============================================================
# CACHED DATA LOADERS
# ============================================================

@st.cache_data
def load_clean():
    """Load the cleaned sales dataset."""
    return _safe_read_csv(
        "clean_all.csv",
        parse_dates=["InvoiceDate"],
    )


@st.cache_data
def load_customers():
    """Load cleaned customer-level sales records."""
    return _safe_read_csv(
        "clean_customers.csv",
        parse_dates=["InvoiceDate"],
    )


@st.cache_data
def load_segments():
    """Load customer segmentation results."""
    return _safe_read_csv("rfm_segmented.csv")


@st.cache_data
def load_forecast_results():
    """Load forecasting predictions."""
    return _safe_read_csv(
        "forecast_results.csv",
        parse_dates=["ds"],
    )


@st.cache_data
def load_forecast_metrics():
    """Load forecasting evaluation metrics."""
    return _safe_read_csv("forecast_metrics.csv")


@st.cache_data
def load_churn_results():
    """Load customer churn prediction results."""
    return _safe_read_csv("churn_results.csv")


@st.cache_data
def load_churn_metrics():
    """Load customer churn evaluation metrics."""
    return _safe_read_csv("churn_metrics.csv")


@st.cache_data
def load_inventory():
    """Load inventory and reorder recommendations."""
    return _safe_read_csv("inventory_report.csv")


# ============================================================
# FIGURE HELPERS
# ============================================================

def figure_path(name: str) -> str:
    """Return the absolute path to a generated report figure."""
    return str(FIG_DIR / name)


def show_figure_if_exists(
    name: str,
    caption: str = None,
    **kwargs,
):
    """
    Display a generated figure if available.

    Otherwise, show a friendly message rather than raising an error.
    """

    path = Path(figure_path(name))

    if path.is_file():
        st.image(
            str(path),
            caption=caption,
            **kwargs,
        )
    else:
        st.info(
            f"Figure not found: `{path.relative_to(PROJECT_ROOT)}`. "
            "Run the corresponding pipeline step to generate it."
        )


# ============================================================
# MISSING-DATA MESSAGES
# ============================================================

def missing_data_notice(what: str, command: str):
    """Display instructions when a required generated output is missing."""

    st.warning(
        f"No {what} found yet. "
        f"Run `{command}` first, then reload this page."
    )
    
