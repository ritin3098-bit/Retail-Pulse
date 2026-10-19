"""Tests for src/inventory.py — checks the reorder math and status flagging."""

import pandas as pd
from src.inventory import compute_avg_daily_demand, build_inventory_report


def _sample_clean():
    # Product A: 10 units/day avg over a 10-day window (100 units total)
    # Product B: 1 unit/day avg over the same window
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        rows.append({"StockCode": "A", "Description": "Widget", "Quantity": 10, "InvoiceDate": d, "Revenue": 100})
        rows.append({"StockCode": "B", "Description": "Gadget", "Quantity": 1, "InvoiceDate": d, "Revenue": 50})
    return pd.DataFrame(rows)


def test_avg_daily_demand_matches_known_values():
    clean = _sample_clean()
    demand = compute_avg_daily_demand(clean, ["A", "B"])
    assert round(demand["A"], 1) == 10.0
    assert round(demand["B"], 1) == 1.0


def test_reorder_point_formula_and_status():
    clean = _sample_clean()
    report = build_inventory_report(clean, top_n=2, seed=1, safety_stock_days=3)

    for _, row in report.iterrows():
        expected_safety = round(row["AvgDailyDemand"] * 3, 1)
        expected_rop = round(row["AvgDailyDemand"] * row["LeadTimeDays"] + expected_safety, 1)
        assert row["SafetyStock"] == expected_safety
        assert row["ReorderPoint"] == expected_rop

        if row["CurrentStock"] < row["ReorderPoint"]:
            assert row["Status"] == "Reorder Now"
            assert row["RecommendedOrderQty"] > 0
        else:
            assert row["Status"] == "OK"
            assert row["RecommendedOrderQty"] == 0
