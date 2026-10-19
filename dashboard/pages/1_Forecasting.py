import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import streamlit as st
import pandas as pd
from dashboard.utils import (
    inject_theme,
    load_forecast_results,
    load_forecast_metrics,
    show_figure_if_exists,
    missing_data_notice,
)

st.set_page_config(page_title="Forecasting — RetailPulse", page_icon="📈", layout="wide")
inject_theme()

st.title("📈 Demand Forecasting")

results = load_forecast_results()
metrics = load_forecast_metrics()

if results is None:
    missing_data_notice("forecast results", "python -m src.forecasting")
    st.stop()

# ---- Model comparison ----
st.subheader("Model Accuracy (MAPE — lower is better)")
if metrics is not None:
    cols = st.columns(len(metrics))
    best = metrics["MAPE"].min()
    for col, (_, row) in zip(cols, metrics.iterrows()):
        delta = "✅ best" if row["MAPE"] == best else None
        col.metric(row["Model"], f"{row['MAPE']:.2f}%", delta=delta)
    st.caption(
        "Target from the project brief: MAPE ≤ 12%. Forecasting is done on weekly "
        "revenue — daily data was too noisy (a single large order and multiple "
        "closed days distorted the metric)."
    )
else:
    missing_data_notice("forecast metrics", "python -m src.forecasting")

st.divider()

# ---- What-if forecast chart ----
st.subheader("Forecast & What-If Demand Adjustment")

growth_pct = st.slider(
    "Adjust projected demand growth",
    min_value=-20,
    max_value=30,
    value=0,
    step=1,
    format="%d%%",
    help="Scales the forward-looking forecast up or down — simulate a marketing push, a slowdown, or a new market.",
)

adjusted = results.copy()
adjustment_factor = 1 + growth_pct / 100
is_future = adjusted["Type"] == "Forecast"
adjusted.loc[is_future, "Revenue"] = adjusted.loc[is_future, "Revenue"] * adjustment_factor

chart_df = adjusted.pivot_table(index="ds", columns="Type", values="Revenue")
st.line_chart(chart_df, height=400)

future_total = adjusted.loc[is_future, "Revenue"].sum()
baseline_total = results.loc[results["Type"] == "Forecast", "Revenue"].sum()
st.metric(
    f"Projected revenue, next {is_future.sum()} weeks",
    f"£{future_total:,.0f}",
    delta=f"{growth_pct:+d}% vs baseline (£{baseline_total:,.0f})",
)

st.divider()

st.subheader("Prophet Forecast Detail")
col1, col2 = st.columns(2)
with col1:
    show_figure_if_exists("08_prophet_forecast.png", use_container_width=True)
with col2:
    show_figure_if_exists("09_prophet_components.png", use_container_width=True)

st.subheader("LSTM Backtest")
show_figure_if_exists("10_lstm_forecast.png", use_container_width=True)
