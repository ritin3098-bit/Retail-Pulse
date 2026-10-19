import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
from dashboard.auth import require_login
from dashboard.utils import (
    inject_theme,
    load_clean,
    load_segments,
    load_inventory,
    load_churn_metrics,
    show_figure_if_exists,
    missing_data_notice,
)

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")
inject_theme()
require_login()  # login/register screen; stops the page until the user is signed in

st.title("📊 RetailPulse")
st.caption("AI-Powered Customer Analytics & Demand Forecasting — Overview")

clean = load_clean()

if clean is None:
    missing_data_notice("cleaned sales data", "python -m src.data_prep")
    st.stop()

# ---- KPI row ----
total_revenue = clean["Revenue"].sum()
total_orders = clean["Invoice"].nunique()
avg_order_value = total_revenue / total_orders
top_country = clean.groupby("Country")["Revenue"].sum().idxmax()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"£{total_revenue:,.0f}")
col2.metric("Total Orders", f"{total_orders:,}")
col3.metric("Avg Order Value", f"£{avg_order_value:,.2f}")
col4.metric("Top Market", top_country)

st.divider()

# ---- Revenue trend + segment snapshot side by side ----
left, right = st.columns([3, 2])

with left:
    st.subheader("Monthly Revenue Trend")
    monthly = clean.set_index("InvoiceDate")["Revenue"].resample("ME").sum()
    st.line_chart(monthly, height=320)

with right:
    st.subheader("Customer Segments")
    rfm = load_segments()
    if rfm is not None and "Segment" in rfm.columns:
        seg_counts = rfm["Segment"].value_counts()
        st.bar_chart(seg_counts, height=320)
    else:
        missing_data_notice("segmentation results", "python -m src.segmentation")

st.divider()

# ---- Quick status strip: inventory + churn, so Overview hints at every module ----
c1, c2 = st.columns(2)

with c1:
    st.subheader("Inventory Health")
    inv = load_inventory()
    if inv is not None:
        n_reorder = (inv["Status"] == "Reorder Now").sum()
        st.metric("Products Needing Reorder", f"{n_reorder} / {len(inv)}")
    else:
        missing_data_notice("inventory report", "python -m src.inventory")

with c2:
    st.subheader("Churn Model Health")
    cm = load_churn_metrics()
    if cm is not None:
        row = cm.iloc[0]
        st.metric("AUC-ROC", f"{row['AUC_ROC']:.3f}")
        st.metric("Churn Rate (observed)", f"{row['ChurnRate']*100:.1f}%")
    else:
        missing_data_notice("churn results", "python -m src.churn")

st.divider()
st.subheader("Top Countries & Products")
show_figure_if_exists("02_top_countries_products.png", use_container_width=True)

st.caption("Use the sidebar to explore Forecasting, Segments & Churn, Inventory, and Monitoring in detail.")
