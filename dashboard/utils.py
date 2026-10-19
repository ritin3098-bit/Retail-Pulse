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

from src.pipeline import run_pipeline, PipelineError
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
    Generate clean_all.csv if it is missing.

    Requirements:
    - src/data_prep.py must exist.
    - The raw Excel dataset must be available at:
      data/raw/online_retail_II.xlsx

    Returns:
        bool: True when clean_all.csv exists; otherwise False.
    """

    # No preprocessing is necessary if the required file exists.
    if CLEAN_DATA_FILE.exists():
        return True

    # Check whether the source dataset is available.
    if not RAW_DATA_FILE.exists():
        st.error(
            "The cleaned sales dataset is missing, and the original "
            "Excel dataset could not be found."
        )

        st.info(
            "Make the Online Retail II dataset available at "
            "`data/raw/online_retail_II.xlsx`, then restart or rerun "
            "the application."
        )

        return False

    # Ensure that output directories exist.
    ensure_output_directories()

    try:
        with st.spinner(
            "Preparing RetailPulse data for the first time. "
            "This may take several minutes..."
        ):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.data_prep",
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=600,
                check=False,
            )

        if result.returncode != 0:
            st.error("RetailPulse data preprocessing failed.")

            with st.expander("View preprocessing logs"):
                if result.stdout:
                    st.markdown("**Standard output**")
                    st.code(result.stdout[-12000:])

                if result.stderr:
                    st.markdown("**Error output**")
                    st.code(result.stderr[-12000:])

            return False

        # Confirm that the expected output was actually created.
        if not CLEAN_DATA_FILE.exists():
            st.error(
                "The preprocessing script finished, but "
                "`data/processed/clean_all.csv` was not created. "
                "Check the output paths in `src/data_prep.py`."
            )

            with st.expander("View preprocessing output"):
                if result.stdout:
                    st.code(result.stdout[-12000:])

                if result.stderr:
                    st.code(result.stderr[-12000:])

            return False

        # Invalidate cached data so the new CSV is read.
        load_clean.clear()

        return True

    except subprocess.TimeoutExpired:
        st.error(
            "Data preprocessing exceeded the 10-minute time limit. "
            "Consider preparing the data separately or using external "
            "storage for the generated dataset."
        )

        return False

    except OSError as exc:
        st.error(f"Could not start the preprocessing script: {exc}")
        return False

    except Exception as exc:
        st.error(f"Unexpected error during preprocessing: {exc}")
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
    
def ensure_pipeline_outputs():
    """
    Generate missing RetailPulse outputs automatically.

    Run the pipeline only when required outputs are missing.
    """
    required_files = [
        DATA_DIR / "clean_all.csv",
        DATA_DIR / "rfm_segmented.csv",
        DATA_DIR / "forecast_results.csv",
        DATA_DIR / "forecast_metrics.csv",
        DATA_DIR / "churn_results.csv",
        DATA_DIR / "churn_metrics.csv",
        DATA_DIR / "inventory_report.csv",
    ]

    missing_files = [
        path for path in required_files
        if not path.exists()
    ]

    if not missing_files:
        return True

    st.info(
        "Some RetailPulse outputs are missing. "
        "Preparing the required datasets and reports..."
    )

    try:
        with st.spinner(
            "Generating RetailPulse outputs. "
            "This may take several minutes..."
        ):
            run_pipeline()

        # Clear cached loaders so newly generated CSVs are visible.
        load_clean.clear()
        load_customers.clear()
        load_segments.clear()
        load_forecast_results.clear()
        load_forecast_metrics.clear()
        load_churn_results.clear()
        load_churn_metrics.clear()
        load_inventory.clear()

        remaining = [
            path.name for path in required_files
            if not path.exists()
        ]

        if remaining:
            st.error(
                "The pipeline finished, but these files are still missing: "
                + ", ".join(remaining)
            )
            return False

        st.success("RetailPulse datasets and reports are ready.")
        return True

    except PipelineError as exc:
        st.error("RetailPulse could not generate all required outputs.")
        st.code(str(exc))
        return False

    except Exception as exc:
        st.error(f"Unexpected pipeline error: {exc}")
        return False