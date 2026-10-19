"""
Inventory optimization for RetailPulse.

The source dataset has no real stock/inventory column, so inventory levels
(current stock, lead time, holding cost) are SIMULATED — state this clearly
in the report. Demand is real, computed from actual sales history.

    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock
    recommended_order_qty = target_stock - current_stock   (only if below reorder point)

Usage:
    from src.inventory import build_inventory_report

    report = build_inventory_report(clean, top_n=20)
"""

import numpy as np
import pandas as pd


def simulate_inventory(clean: pd.DataFrame, top_n: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Simulate lead time and holding cost for the top N products by revenue.
    Random but reproducible (fixed seed).

    CurrentStock is NOT generated here — a fixed range (e.g. 50-500) regardless
    of a product's sales volume is unrealistic: a top seller moving thousands
    of units a month would never realistically be stocked at only 100 units.
    CurrentStock is generated in build_inventory_report() instead, scaled to
    each product's own real demand (see DaysOfStockOnHand below).
    """
    top_products = clean.groupby("StockCode")["Revenue"].sum().nlargest(top_n).index

    rng = np.random.default_rng(seed)
    inventory = pd.DataFrame(
        {
            "StockCode": top_products,
            "LeadTimeDays": rng.integers(3, 14, size=top_n),
            "HoldingCostPerUnit": rng.uniform(0.1, 2.0, size=top_n).round(2),
        }
    )

    # Attach a human-readable product name for the report/dashboard
    names = clean.drop_duplicates("StockCode")[["StockCode", "Description"]]
    inventory = inventory.merge(names, on="StockCode", how="left")
    return inventory


def compute_avg_daily_demand(clean: pd.DataFrame, stock_codes) -> pd.Series:
    """Average units sold per day, per product, over the full date range in the data."""
    date_range_days = (clean["InvoiceDate"].max() - clean["InvoiceDate"].min()).days + 1
    demand = (
        clean[clean["StockCode"].isin(stock_codes)]
        .groupby("StockCode")["Quantity"]
        .sum()
        / date_range_days
    )
    return demand.rename("AvgDailyDemand")


def build_inventory_report(
    clean: pd.DataFrame, top_n: int = 20, seed: int = 42, safety_stock_days: int = 3
) -> pd.DataFrame:
    """
    Full inventory optimization table: simulated stock + real demand ->
    reorder point, status, and recommended order quantity.
    """
    inv = simulate_inventory(clean, top_n=top_n, seed=seed)
    demand = compute_avg_daily_demand(clean, inv["StockCode"])
    inv = inv.merge(demand, on="StockCode")

    # Simulate stock as a random number of DAYS of coverage (5-45), then convert
    # to units using each product's own real demand. This keeps stock levels
    # proportional to how fast a product actually sells, instead of a flat
    # unit range that's meaningless for a product selling 100 vs 10,000/year.
    rng = np.random.default_rng(seed + 1)
    inv["DaysOfStockOnHand"] = rng.integers(5, 45, size=len(inv))
    inv["CurrentStock"] = (inv["AvgDailyDemand"] * inv["DaysOfStockOnHand"]).round(0)

    inv["SafetyStock"] = (inv["AvgDailyDemand"] * safety_stock_days).round(1)
    inv["ReorderPoint"] = (inv["AvgDailyDemand"] * inv["LeadTimeDays"] + inv["SafetyStock"]).round(1)
    inv["Status"] = np.where(inv["CurrentStock"] < inv["ReorderPoint"], "Reorder Now", "OK")

    # Target stock level = 2x reorder point (a simple, explainable policy —
    # covers one full lead-time cycle plus a buffer cycle)
    inv["TargetStock"] = (inv["ReorderPoint"] * 2).round(1)
    inv["RecommendedOrderQty"] = np.where(
        inv["Status"] == "Reorder Now",
        (inv["TargetStock"] - inv["CurrentStock"]).clip(lower=0).round(0),
        0,
    )

    return inv.sort_values("Status", ascending=False)  # "Reorder Now" rows first


if __name__ == "__main__":
    from src import visualize as viz

    clean = pd.read_csv("data/processed/clean_all.csv", parse_dates=["InvoiceDate"], dtype={"Invoice": str})

    report = build_inventory_report(clean, top_n=20)
    print(report[["StockCode", "Description", "CurrentStock", "ReorderPoint", "Status", "RecommendedOrderQty"]])

    n_reorder = (report["Status"] == "Reorder Now").sum()
    print(f"\n{n_reorder} of {len(report)} products need reordering.")

    report.to_csv("data/processed/inventory_report.csv", index=False)

    print("Saving figures:")
    print(" -", viz.plot_inventory_status(report))
