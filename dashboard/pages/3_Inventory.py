import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
from dashboard.utils import inject_theme, load_inventory, show_figure_if_exists, missing_data_notice

st.set_page_config(page_title="Inventory — RetailPulse", page_icon="📦", layout="wide")
inject_theme()

st.title("📦 Inventory Optimization")
st.caption(
    "Demand is computed from real sales history. Stock levels, lead time, and "
    "holding cost are simulated, since the source dataset has no inventory data."
)

inv = load_inventory()

if inv is None:
    missing_data_notice("inventory report", "python -m src.inventory")
    st.stop()

n_reorder = (inv["Status"] == "Reorder Now").sum()
c1, c2, c3 = st.columns(3)
c1.metric("Products Tracked", len(inv))
c2.metric("Need Reordering", n_reorder)
c3.metric("Total Recommended Order Qty", f"{inv['RecommendedOrderQty'].sum():,.0f} units")

st.divider()

status_filter = st.radio("Filter", ["All", "Reorder Now", "OK"], horizontal=True)
filtered = inv if status_filter == "All" else inv[inv["Status"] == status_filter]


def highlight_status(row):
    color = "#fdecea" if row["Status"] == "Reorder Now" else "#eaf7ec"
    return [f"background-color: {color}"] * len(row)


display_cols = [
    "StockCode", "Description", "CurrentStock", "AvgDailyDemand",
    "LeadTimeDays", "ReorderPoint", "Status", "RecommendedOrderQty",
]
display_cols = [c for c in display_cols if c in filtered.columns]

st.dataframe(
    filtered[display_cols].style.apply(highlight_status, axis=1),
    width="stretch",
    height=420,
)

st.download_button(
    "⬇ Download inventory report (CSV)",
    inv.to_csv(index=False),
    file_name="inventory_report.csv",
    mime="text/csv",
)

st.divider()
show_figure_if_exists("13_inventory_status.png", width="stretch")
