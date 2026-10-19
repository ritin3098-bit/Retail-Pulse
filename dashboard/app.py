"""
RetailPulse Overview Dashboard.

Uses precomputed overview datasets to reduce memory usage.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.auth import require_login
from dashboard.utils import (
    inject_theme,
    load_segments,
    load_inventory,
    load_churn_metrics,
    show_figure_if_exists,
    missing_data_notice,
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
)

inject_theme()
require_login()

st.title("📊 RetailPulse")
st.caption(
    "AI-Powered Customer Analytics & Demand Forecasting — Overview"
)

# ------------------------------------------------------------
# PATHS AND CACHED LOADERS
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_overview_kpis():
    path = DATA_DIR / "overview_kpis.csv"

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


@st.cache_data
def load_overview_monthly():
    path = DATA_DIR / "overview_monthly.csv"

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


@st.cache_data
def load_overview_countries():
    path = DATA_DIR / "overview_countries.csv"

    if not path.is_file() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


# ------------------------------------------------------------
# LOAD PRECOMPUTED OVERVIEW DATA
# ------------------------------------------------------------

kpis = load_overview_kpis()
monthly = load_overview_monthly()
countries = load_overview_countries()

if kpis is None or monthly is None or countries is None:
    st.error("One or more overview datasets are missing or unreadable.")

    st.write("Expected files:")
    st.code(
        "data/processed/overview_kpis.csv\n"
        "data/processed/overview_monthly.csv\n"
        "data/processed/overview_countries.csv"
    )

    st.info(
        "Generate these files locally, commit them to GitHub, "
        "and redeploy the application."
    )
    st.stop()


# ------------------------------------------------------------
# KPI VALIDATION
# ------------------------------------------------------------

if not {"Metric", "Value"}.issubset(kpis.columns):
    st.error("overview_kpis.csv must contain Metric and Value columns.")
    st.stop()

kpi_values = dict(zip(kpis["Metric"], kpis["Value"]))

required_metrics = {
    "TotalRevenue",
    "TotalOrders",
    "AverageOrderValue",
}

if not required_metrics.issubset(kpi_values):
    st.error(
        "overview_kpis.csv is missing one or more required metrics."
    )
    st.stop()

total_revenue = float(kpi_values["TotalRevenue"])
total_orders = int(float(kpi_values["TotalOrders"]))
avg_order_value = float(kpi_values["AverageOrderValue"])

top_country = "N/A"

if {"Country", "Revenue"}.issubset(countries.columns):
    countries["Revenue"] = pd.to_numeric(
        countries["Revenue"], errors="coerce"
    )
    countries = countries.dropna(subset=["Revenue"])

    if not countries.empty:
        top_country = str(
            countries.loc[countries["Revenue"].idxmax(), "Country"]
        )


# ------------------------------------------------------------
# KPI ROW
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Order Value", f"£{avg_order_value:,.2f}")
col4.metric("Top Market", top_country)

st.divider()


# ------------------------------------------------------------
# MONTHLY REVENUE TREND
# ------------------------------------------------------------

left, right = st.columns([3, 2])

with left:
    st.subheader("Monthly Revenue Trend")

    if {"Month", "Revenue"}.issubset(monthly.columns):
        monthly["Revenue"] = pd.to_numeric(
            monthly["Revenue"], errors="coerce"
        )
        monthly = monthly.dropna(subset=["Revenue"])

        if not monthly.empty:
            monthly_chart = monthly.set_index("Month")["Revenue"]
            st.line_chart(monthly_chart, height=320)
        else:
            st.info("No monthly revenue data is available.")
    else:
        st.error(
            "overview_monthly.csv must contain Month and Revenue columns."
        )

# ------------------------------------------------------------
# CUSTOMER SEGMENTS
# ------------------------------------------------------------

with right:
    st.subheader("Customer Segments")

    rfm = load_segments()

    if rfm is not None and "Segment" in rfm.columns:
        st.bar_chart(rfm["Segment"].value_counts(), height=320)
    else:
        st.info("Saved customer segmentation results are unavailable.")


st.divider()


# ------------------------------------------------------------
# INVENTORY AND CHURN STATUS
# ------------------------------------------------------------

c1, c2 = st.columns(2)

with c1:
    st.subheader("Inventory Health")

    inv = load_inventory()

    if inv is not None and "Status" in inv.columns:
        n_reorder = int((inv["Status"] == "Reorder Now").sum())
        st.metric(
            "Products Needing Reorder",
            f"{n_reorder} / {len(inv)}",
        )
    else:
        st.info("Saved inventory report is unavailable.")


with c2:
    st.subheader("Churn Model Health")

    cm = load_churn_metrics()

    if (
        cm is not None
        and not cm.empty
        and {"AUC_ROC", "ChurnRate"}.issubset(cm.columns)
    ):
        row = cm.iloc[0]

        st.metric("AUC-ROC", f"{float(row['AUC_ROC']):.3f}")
        st.metric(
            "Churn Rate (observed)",
            f"{float(row['ChurnRate']) * 100:.1f}%",
        )
    else:
        st.info("Saved churn metrics are unavailable.")


st.divider()


# ------------------------------------------------------------
# TOP COUNTRIES AND PRODUCTS
# ------------------------------------------------------------

st.subheader("Top Countries & Products")

show_figure_if_exists(
    "02_top_countries_products.png",
    use_container_width=True,
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.caption(
    "Use the sidebar to explore Forecasting, Segments & Churn, "
    "Inventory, and Monitoring in detail."
)