"""
Data loading, cleaning, and RFM feature engineering for RetailPulse.

Usage:
    from src.data_prep import load_raw_data, clean_transactions, add_engineered_features, compute_rfm

    df = load_raw_data("data/raw/online_retail_II.xlsx")
    clean, cust = clean_transactions(df)
    clean = add_engineered_features(clean)   # adds Month, IsWeekend, IsPeakSeason
    rfm = compute_rfm(cust)
"""

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load every sheet of the Online Retail II Excel file into one DataFrame."""
    sheets = pd.read_excel(path, sheet_name=None)
    df = pd.concat(sheets.values(), ignore_index=True)
    return df


def audit_data_quality(df: pd.DataFrame) -> dict:
    """Return counts of common data quality issues, for reporting."""
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    return {
        "missing_customer_id": int(df["Customer ID"].isna().sum()),
        "cancelled_invoices": int(is_cancel.sum()),
        "quantity_le_zero": int((df["Quantity"] <= 0).sum()),
        "price_le_zero": int((df["Price"] <= 0).sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean raw transactions.

    Returns:
        clean: all valid sales rows (Customer ID may be missing) — use for
               demand forecasting.
        cust:  valid sales rows with a known Customer ID — use for
               segmentation and churn.
    """
    df = df.drop_duplicates()
    is_cancel = df["Invoice"].astype(str).str.startswith("C")

    clean = df[~is_cancel & (df["Quantity"] > 0) & (df["Price"] > 0)].copy()
    clean["Revenue"] = clean["Quantity"] * clean["Price"]

    cust = clean.dropna(subset=["Customer ID"]).copy()
    cust["Customer ID"] = cust["Customer ID"].astype(int)

    return clean, cust


def get_cancellations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return cancelled/returned transactions (invoices starting with 'C').
    These were dropped from `clean`/`cust` for revenue analysis, but are kept
    here separately — used as a churn feature (ReturnRate) in src/churn.py.
    """
    is_cancel = df["Invoice"].astype(str).str.startswith("C")
    return df[is_cancel].drop_duplicates().copy()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add new columns to the dataset itself, beyond the raw source columns:
      - Month           : calendar month of the transaction
      - IsWeekend       : 1 if Saturday/Sunday, else 0
      - IsPeakSeason    : 1 if Oct-Dec (holiday shopping season), else 0

    IsPeakSeason is later used as a Prophet regressor (src/forecasting.py)
    to measurably improve demand forecast accuracy — not just an unused
    column, but one that feeds directly into a model result.
    """
    df = df.copy()
    df["Month"] = df["InvoiceDate"].dt.month
    df["IsWeekend"] = df["InvoiceDate"].dt.dayofweek.isin([5, 6]).astype(int)
    df["IsPeakSeason"] = df["Month"].isin([10, 11, 12]).astype(int)
    return df


def build_daily_series(clean: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into a complete daily revenue series (Prophet format)."""
    daily = clean.set_index("InvoiceDate").resample("D")["Revenue"].sum().reset_index()
    daily.columns = ["ds", "y"]

    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = (
        daily.set_index("ds")
        .reindex(full_range, fill_value=0)
        .rename_axis("ds")
        .reset_index()
    )
    daily.columns = ["ds", "y"]
    daily["IsPeakSeason"] = daily["ds"].dt.month.isin([10, 11, 12]).astype(int)
    return daily


def compute_rfm(cust: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary per customer."""
    snapshot_date = cust["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        cust.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )
    return rfm


if __name__ == "__main__":
    from src import visualize as viz

    RAW_PATH = "data/raw/online_retail_II.xlsx"

    df = load_raw_data(RAW_PATH)
    print("Raw shape:", df.shape)
    print("Audit:", audit_data_quality(df))

    clean, cust = clean_transactions(df)
    clean = add_engineered_features(clean)
    cust = add_engineered_features(cust)
    clean.to_csv("data/processed/clean_all.csv", index=False)
    cust.to_csv("data/processed/clean_customers.csv", index=False)
    print("Clean shape:", clean.shape, "| Customers shape:", cust.shape)
    print("Engineered columns added: Month, IsWeekend, IsPeakSeason")

    cancels = get_cancellations(df)
    cancels.to_csv("data/processed/cancellations.csv", index=False)
    print("Cancellations shape:", cancels.shape, "(saved for churn ReturnRate feature)")

    daily = build_daily_series(clean)
    daily.to_csv("data/processed/daily_sales.csv", index=False)
    print("Daily series shape:", daily.shape)

    rfm = compute_rfm(cust)
    rfm.to_csv("data/processed/rfm.csv", index=False)
    print("RFM shape:", rfm.shape)

    # EDA figures -> reports/figures/
    print("Saving figures:")
    print(" -", viz.plot_monthly_revenue(clean))
    print(" -", viz.plot_top_countries_products(clean))
    print(" -", viz.plot_distributions(clean, cust))
    print(" -", viz.plot_correlation(clean))
    print(" -", viz.plot_rfm_distributions(rfm, log=False))
