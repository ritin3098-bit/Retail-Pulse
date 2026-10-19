"""
RetailPulse Overview Dashboard.

Displays:
- Business KPIs
- Monthly revenue trend
- Customer segment distribution
- Inventory health
- Churn model health
- Top countries and products figure
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure Python can import the src and dashboard packages.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from dashboard.auth import require_login

from dashboard.utils import (
    inject_theme,
    ensure_clean_data,
    load_clean,
    load_segments,
    load_inventory,
    load_churn_metrics,
    show_figure_if_exists,
    missing_data_notice,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RetailPulse",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# THEME AND AUTHENTICATION
# ============================================================

inject_theme()

# Preserve the existing login/register functionality.
require_login()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 RetailPulse")

st.caption(
    "AI-Powered Customer Analytics & Demand Forecasting — Overview"
)


# ============================================================
# PREPARE AND LOAD CLEANED DATA
# ============================================================

if not ensure_clean_data():
    st.stop()

clean = load_clean()

if clean is None:
    missing_data_notice(
        "cleaned sales data",
        "python -m src.data_prep",
    )
    st.stop()


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Revenue",
    "Invoice",
    "Country",
    "InvoiceDate",
}

missing_columns = required_columns.difference(clean.columns)

if missing_columns:
    st.error(
        "The cleaned sales dataset is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )

    st.info(
        "Check the output of `src/data_prep.py` and confirm that "
        "the generated `clean_all.csv` contains the required columns."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_revenue = clean["Revenue"].sum()

total_orders = clean["Invoice"].nunique()

avg_order_value = (
    total_revenue / total_orders
    if total_orders > 0
    else 0
)

country_revenue = (
    clean.groupby("Country")["Revenue"]
    .sum()
    .dropna()
)

top_country = (
    country_revenue.idxmax()
    if not country_revenue.empty
    else "N/A"
)


# ============================================================
# KPI ROW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Revenue",
    f"£{total_revenue:,.0f}",
)

col2.metric(
    "Total Orders",
    f"{total_orders:,}",
)

col3.metric(
    "Avg Order Value",
    f"£{avg_order_value:,.2f}",
)

col4.metric(
    "Top Market",
    top_country,
)


st.divider()


# ============================================================
# MONTHLY REVENUE TREND
# ============================================================

left, right = st.columns([3, 2])


with left:
    st.subheader("Monthly Revenue Trend")

    monthly = (
        clean.dropna(subset=["InvoiceDate"])
        .set_index("InvoiceDate")["Revenue"]
        .resample("ME")
        .sum()
    )

    if not monthly.empty:
        st.line_chart(
            monthly,
            height=320,
        )
    else:
        st.info("No monthly revenue data is available.")


# ============================================================
# CUSTOMER SEGMENTS
# ============================================================

with right:
    st.subheader("Customer Segments")

    rfm = load_segments()

    if rfm is not None and "Segment" in rfm.columns:
        seg_counts = rfm["Segment"].value_counts()

        st.bar_chart(
            seg_counts,
            height=320,
        )

    else:
        missing_data_notice(
            "segmentation results",
            "python -m src.segmentation",
        )


st.divider()


# ============================================================
# INVENTORY AND CHURN STATUS
# ============================================================

c1, c2 = st.columns(2)


# ------------------------------------------------------------
# Inventory Health
# ------------------------------------------------------------

with c1:
    st.subheader("Inventory Health")

    inv = load_inventory()

    if inv is not None and "Status" in inv.columns:
        n_reorder = (
            inv["Status"] == "Reorder Now"
        ).sum()

        st.metric(
            "Products Needing Reorder",
            f"{n_reorder} / {len(inv)}",
        )

    else:
        missing_data_notice(
            "inventory report",
            "python -m src.inventory",
        )


# ------------------------------------------------------------
# Churn Model Health
# ------------------------------------------------------------

with c2:
    st.subheader("Churn Model Health")

    cm = load_churn_metrics()

    if (
        cm is not None
        and not cm.empty
        and {"AUC_ROC", "ChurnRate"}.issubset(cm.columns)
    ):
        row = cm.iloc[0]

        st.metric(
            "AUC-ROC",
            f"{row['AUC_ROC']:.3f}",
        )

        st.metric(
            "Churn Rate (observed)",
            f"{row['ChurnRate'] * 100:.1f}%",
        )

    else:
        missing_data_notice(
            "churn results",
            "python -m src.churn",
        )


st.divider()


# ============================================================
# TOP COUNTRIES AND PRODUCTS
# ============================================================

st.subheader("Top Countries & Products")

show_figure_if_exists(
    "02_top_countries_products.png",
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Use the sidebar to explore Forecasting, Segments & Churn, "
    "Inventory, and Monitoring in detail."
)