import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
from dashboard.utils import (
    inject_theme,
    load_segments,
    load_churn_results,
    load_churn_metrics,
    show_figure_if_exists,
    missing_data_notice,
)

st.set_page_config(page_title="Segments & Churn — RetailPulse", page_icon="🧩", layout="wide")
inject_theme()

st.title("🧩 Customer Segments & Churn Risk")

# ================= Segmentation =================
rfm = load_segments()

if rfm is None:
    missing_data_notice("segmentation results", "python -m src.segmentation")
else:
    st.subheader("Segment Overview")
    seg_summary = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]].mean().round(1)
    )
    seg_summary["Count"] = rfm["Segment"].value_counts()
    st.dataframe(seg_summary.sort_values("Monetary", ascending=False), use_container_width=True)

    segments = sorted(rfm["Segment"].dropna().unique())
    chosen = st.multiselect("Filter by segment", segments, default=segments)
    filtered = rfm[rfm["Segment"].isin(chosen)]

    st.dataframe(
        filtered[["Customer ID", "Segment", "Recency", "Frequency", "Monetary"]].sort_values(
            "Monetary", ascending=False
        ),
        use_container_width=True,
        height=300,
    )
    st.download_button(
        "⬇ Download filtered segments (CSV)",
        filtered.to_csv(index=False),
        file_name="customer_segments.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        show_figure_if_exists("07_cluster_recency_monetary.png", caption="K-Means: Recency vs Monetary", use_container_width=True)
    with col2:
        show_figure_if_exists("07_dbscan_cluster_recency_monetary.png", caption="DBSCAN validation", use_container_width=True)

st.divider()

# ================= Churn =================
st.subheader("Churn Risk")

churn_metrics = load_churn_metrics()
churn = load_churn_results()

if churn_metrics is not None:
    row = churn_metrics.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("AUC-ROC", f"{row['AUC_ROC']:.3f}", help="Brief target: ≥ 0.88")
    c2.metric("Precision@top20%", f"{row['Precision_at_top20']:.3f}", help="Brief target: ≥ 0.75")
    c3.metric("Observed Churn Rate", f"{row['ChurnRate']*100:.1f}%")

if churn is None:
    missing_data_notice("churn results", "python -m src.churn")
else:
    st.markdown("**Highest-risk customers** (ranked by predicted churn probability)")
    top_risk = churn.sort_values("ChurnProbability", ascending=False).head(25)
    display_cols = [c for c in ["Customer ID", "Segment", "Recency", "Frequency", "Monetary", "ChurnProbability"] if c in top_risk.columns]
    st.dataframe(top_risk[display_cols], use_container_width=True, height=400)

    st.download_button(
        "⬇ Download full churn risk table (CSV)",
        churn.to_csv(index=False),
        file_name="churn_risk.csv",
        mime="text/csv",
    )

    col1, col2 = st.columns(2)
    with col1:
        show_figure_if_exists("11_churn_roc.png", use_container_width=True)
    with col2:
        show_figure_if_exists("12_churn_shap_summary.png", caption="SHAP: what drives churn predictions", use_container_width=True)
