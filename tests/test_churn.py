"""Tests for src/churn.py — mainly checks the cutoff logic doesn't leak future data."""

import pandas as pd
from src.churn import build_churn_dataset, attach_return_rate


def _sample_cust():
    return pd.DataFrame(
        {
            "Invoice": ["1", "2", "3", "4", "5"],
            "StockCode": ["A", "A", "B", "A", "A"],
            "InvoiceDate": pd.to_datetime(
                ["2023-01-01", "2023-01-10", "2023-02-01", "2023-02-15", "2023-04-01"]
            ),
            "Revenue": [10.0, 10.0, 20.0, 15.0, 5.0],
            "Customer ID": [1, 1, 2, 2, 1],  # customer 1 buys again on 2023-04-01
        }
    )


def test_churned_customer_has_no_future_purchase():
    cust = _sample_cust()
    # cutoff right after customer 2's last visible purchase, horizon covers up to 2023-03-17
    feats = build_churn_dataset(cust, cutoff_date="2023-02-16", horizon_days=30)

    cust2 = feats[feats["Customer ID"] == 2].iloc[0]
    assert cust2["Churned"] == 1  # no purchase for customer 2 after cutoff within horizon

    cust1 = feats[feats["Customer ID"] == 1].iloc[0]
    # customer 1's only pre-cutoff purchases are Jan 1 and Jan 10; next real purchase (Apr 1)
    # is outside the 30-day horizon, so they should ALSO be labeled churned here
    assert cust1["Churned"] == 1


def test_features_use_only_data_before_cutoff():
    cust = _sample_cust()
    feats = build_churn_dataset(cust, cutoff_date="2023-02-16", horizon_days=30)

    cust1 = feats[feats["Customer ID"] == 1].iloc[0]
    # Only the Jan 1 and Jan 10 purchases should count (2 invoices), NOT the Apr 1 one
    assert cust1["Frequency"] == 2


def test_return_rate_defaults_to_zero_without_cancellations():
    cust = _sample_cust()
    feats = build_churn_dataset(cust, cutoff_date="2023-02-16", horizon_days=30)  # cancels=None
    assert (feats["ReturnRate"] == 0.0).all()


def test_return_rate_computed_correctly():
    cust = _sample_cust()
    feats = build_churn_dataset(cust, cutoff_date="2023-02-16", horizon_days=30)

    # Customer 1 made 2 purchases before cutoff and cancelled 1 order before cutoff
    cancels = pd.DataFrame(
        {
            "Invoice": ["C9"],
            "InvoiceDate": pd.to_datetime(["2023-01-15"]),
            "Customer ID": [1],
        }
    )
    feats_with_returns = attach_return_rate(feats, cancels, pd.Timestamp("2023-02-16"))

    cust1 = feats_with_returns[feats_with_returns["Customer ID"] == 1].iloc[0]
    assert cust1["ReturnRate"] == 1 / 3  # 1 cancelled / (2 purchased + 1 cancelled)

    cust2 = feats_with_returns[feats_with_returns["Customer ID"] == 2].iloc[0]
    assert cust2["ReturnRate"] == 0.0  # no cancellations for customer 2


def test_behavior_features_use_only_pre_cutoff_data():
    cust = _sample_cust()
    feats = build_churn_dataset(cust, cutoff_date="2023-02-16", horizon_days=30)

    c1 = feats[feats["Customer ID"] == 1].iloc[0]
    assert c1["AvgGapDays"] == 9.0          # Jan 1 -> Jan 10 only; the Apr 1 order is ignored
    assert c1["Orders_90d"] == 2            # both pre-cutoff orders fall inside the last 90 days

    c2 = feats[feats["Customer ID"] == 2].iloc[0]
    assert c2["AvgGapDays"] == 14.0         # Feb 1 -> Feb 15
    assert c2["RecencyRatio"] == c2["Recency"] / 14.0


def test_single_order_customer_gets_nan_gap_not_error():
    import numpy as np

    cust = _sample_cust()
    feats = build_churn_dataset(cust, cutoff_date="2023-01-05", horizon_days=30)  # customer 1 has one order
    c1 = feats[feats["Customer ID"] == 1].iloc[0]
    assert np.isnan(c1["AvgGapDays"]) and np.isnan(c1["RecencyRatio"])


def test_all_feature_columns_are_built():
    from src.churn import FEATURE_COLS

    feats = build_churn_dataset(_sample_cust(), cutoff_date="2023-02-16", horizon_days=30)
    assert set(FEATURE_COLS).issubset(feats.columns)
