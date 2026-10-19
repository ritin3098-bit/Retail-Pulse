"""Basic sanity tests for src/data_prep.py. Run with: pytest tests/"""

import pandas as pd
from src.data_prep import clean_transactions, compute_rfm


def _sample_df():
    return pd.DataFrame(
        {
            "Invoice": ["1", "1", "C2", "3", "3"],
            "StockCode": ["A", "B", "A", "A", "C"],
            "Description": ["Widget", "Gadget", "Widget", "Widget", "Gizmo"],
            "Quantity": [2, 1, -1, 3, -5],  # row 5 is invalid (negative qty, not cancel)
            "InvoiceDate": pd.to_datetime(
                ["2023-01-01", "2023-01-01", "2023-01-02", "2023-01-05", "2023-01-05"]
            ),
            "Price": [10.0, 5.0, 10.0, 10.0, 2.0],
            "Customer ID": [100, 100, 100, None, 200],
            "Country": ["UK", "UK", "UK", "UK", "UK"],
        }
    )


def test_clean_transactions_removes_cancellations_and_bad_rows():
    df = _sample_df()
    clean, cust = clean_transactions(df)

    # Cancelled invoice (C2) and negative-quantity row should be gone
    assert "C2" not in clean["Invoice"].values
    assert (clean["Quantity"] > 0).all()
    assert (clean["Price"] > 0).all()

    # Revenue column should be computed correctly
    row = clean[(clean["Invoice"] == "1") & (clean["StockCode"] == "A")].iloc[0]
    assert row["Revenue"] == 20.0  # 2 * 10.0

    # cust should exclude the row with missing Customer ID
    assert cust["Customer ID"].notna().all()
    assert 100 in cust["Customer ID"].values


def test_compute_rfm_shape_and_columns():
    df = _sample_df()
    _, cust = clean_transactions(df)
    rfm = compute_rfm(cust)

    assert set(["Customer ID", "Recency", "Frequency", "Monetary"]).issubset(rfm.columns)
    assert len(rfm) == cust["Customer ID"].nunique()
    assert (rfm["Monetary"] > 0).all()
