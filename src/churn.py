"""
Churn prediction for RetailPulse.

Approach: pick a cutoff date. Build features ONLY from transactions before
the cutoff (so the model only knows what it would know in production).
Label a customer as churned (1) if they make no purchase in the
`horizon_days` after the cutoff, else 0 (0 = still active).

Targets (brief, F-04): AUC-ROC >= 0.88 and precision@top20% >= 0.75.

What drives the score up compared with plain RFM:
  * Purchase-rhythm features: the average/std gap between a customer's orders,
    and Recency relative to that gap (RecencyRatio). A customer who normally
    buys every 10 days and has been silent for 60 is far more at risk than one
    who normally buys every 120 days.
  * Recent-activity windows: orders / revenue in the last 30/60/90 days and the
    trend versus the previous 90 days (a customer slowing down is a churn signal).
  * ReturnRate (cancelled orders), as before.
  * A cross-validated hyper-parameter search optimised directly for AUC.

Usage:
    from src.churn import build_churn_dataset, train_churn_model, precision_at_k

    feats = build_churn_dataset(cust, cutoff_date="2011-09-09", horizon_days=90, cancels=cancels)
    model, X_test, y_test, probs, auc = train_churn_model(feats)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # keeps the module importable (and testable) without xgboost
    from sklearn.ensemble import HistGradientBoostingClassifier

    HAS_XGB = False

TARGET_AUC = 0.88
TARGET_PRECISION_AT_20 = 0.75

BASE_FEATURES = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "Tenure", "UniqueProducts", "ReturnRate"]
BEHAVIOR_FEATURES = [
    "AvgGapDays",
    "StdGapDays",
    "MaxGapDays",
    "RecencyRatio",
    "Orders_30d",
    "Orders_60d",
    "Orders_90d",
    "Revenue_90d",
    "OrderTrend",
    "RevenueTrend",
    "AvgLinesPerOrder",
    "OrdersPerMonth",
]
FEATURE_COLS = BASE_FEATURES + BEHAVIOR_FEATURES


def attach_return_rate(feats: pd.DataFrame, cancels: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Engineered feature: ReturnRate = cancelled orders / (purchased + cancelled orders),
    computed only from activity before the cutoff. A customer who frequently
    returns items is a meaningfully different churn risk than one with the
    same RFM numbers who never returns anything.
    """
    cancels_before = cancels[cancels["InvoiceDate"] <= cutoff_date]
    returns_per_customer = cancels_before.groupby("Customer ID")["Invoice"].nunique()

    feats = feats.merge(returns_per_customer.rename("ReturnedOrders"), on="Customer ID", how="left")
    feats["ReturnedOrders"] = feats["ReturnedOrders"].fillna(0)
    feats["ReturnRate"] = feats["ReturnedOrders"] / (feats["Frequency"] + feats["ReturnedOrders"])
    return feats.drop(columns=["ReturnedOrders"])


def attach_behavior_features(feats: pd.DataFrame, past: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """
    Purchase-rhythm and recent-activity features, built from pre-cutoff data only.

    `past` must already be filtered to InvoiceDate <= cutoff, so nothing here can
    see the label window.
    """
    # One row per (customer, invoice) so a multi-line basket counts as one order.
    orders = (
        past.groupby(["Customer ID", "Invoice"])
        .agg(Date=("InvoiceDate", "max"), Rev=("Revenue", "sum"), Lines=("StockCode", "size"))
        .reset_index()
        .sort_values(["Customer ID", "Date"])
    )
    orders["Gap"] = orders.groupby("Customer ID")["Date"].diff().dt.total_seconds() / 86400.0
    orders["DaysAgo"] = (snapshot - orders["Date"]).dt.total_seconds() / 86400.0

    gaps = orders.groupby("Customer ID")["Gap"].agg(AvgGapDays="mean", StdGapDays="std", MaxGapDays="max")

    in30, in60, in90 = orders["DaysAgo"] <= 30, orders["DaysAgo"] <= 60, orders["DaysAgo"] <= 90
    prev90 = (orders["DaysAgo"] > 90) & (orders["DaysAgo"] <= 180)
    windows = pd.DataFrame(
        {
            "Orders_30d": in30.groupby(orders["Customer ID"]).sum(),
            "Orders_60d": in60.groupby(orders["Customer ID"]).sum(),
            "Orders_90d": in90.groupby(orders["Customer ID"]).sum(),
            "Orders_prev90d": prev90.groupby(orders["Customer ID"]).sum(),
            "Revenue_90d": orders["Rev"].where(in90, 0.0).groupby(orders["Customer ID"]).sum(),
            "Revenue_prev90d": orders["Rev"].where(prev90, 0.0).groupby(orders["Customer ID"]).sum(),
            "AvgLinesPerOrder": orders.groupby("Customer ID")["Lines"].mean(),
        }
    )

    out = feats.merge(gaps, on="Customer ID", how="left").merge(windows, on="Customer ID", how="left")

    # Positive trend = customer is buying more recently than before; negative = slowing down.
    out["OrderTrend"] = out["Orders_90d"] - out["Orders_prev90d"]
    out["RevenueTrend"] = out["Revenue_90d"] - out["Revenue_prev90d"]

    # How overdue is the customer relative to their own rhythm? NaN for one-time buyers.
    out["RecencyRatio"] = np.where(out["AvgGapDays"] > 0, out["Recency"] / out["AvgGapDays"], np.nan)

    months_active = np.maximum(out["Tenure"] / 30.0, 1.0)
    out["OrdersPerMonth"] = out["Frequency"] / months_active

    return out.drop(columns=["Orders_prev90d", "Revenue_prev90d"])


def build_churn_dataset(
    cust: pd.DataFrame, cutoff_date: str, horizon_days: int = 90, cancels: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build features (before cutoff) and churn labels (based on activity after cutoff).

    Only customers with at least one purchase before the cutoff are included —
    a customer can't be scored on history they don't have yet.

    cancels: output of src.data_prep.get_cancellations(df). If provided,
    adds the ReturnRate engineered feature; if None, ReturnRate is set to 0
    for every customer (so the function still runs without it).
    """
    cutoff_date = pd.Timestamp(cutoff_date)
    horizon_end = cutoff_date + pd.Timedelta(days=horizon_days)

    past = cust[cust["InvoiceDate"] <= cutoff_date]
    future = cust[(cust["InvoiceDate"] > cutoff_date) & (cust["InvoiceDate"] <= horizon_end)]

    snapshot = cutoff_date + pd.Timedelta(days=1)
    feats = (
        past.groupby("Customer ID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot - x.max()).days),
            Frequency=("Invoice", "nunique"),
            Monetary=("Revenue", "sum"),
            Tenure=("InvoiceDate", lambda x: (x.max() - x.min()).days),
            UniqueProducts=("StockCode", "nunique"),
        )
        .reset_index()
    )
    feats["AvgOrderValue"] = feats["Monetary"] / feats["Frequency"]

    if cancels is not None:
        feats = attach_return_rate(feats, cancels, cutoff_date)
    else:
        feats["ReturnRate"] = 0.0

    feats = attach_behavior_features(feats, past, snapshot)

    active_customers = set(future["Customer ID"].unique())
    feats["Churned"] = (~feats["Customer ID"].isin(active_customers)).astype(int)

    return feats


def precision_at_k(y_true: np.ndarray, probs: np.ndarray, k_frac: float = 0.2) -> float:
    """Of the top k% customers ranked by predicted churn risk, what fraction actually churned."""
    n = len(y_true)
    k = max(1, int(n * k_frac))
    top_k_idx = np.argsort(-probs)[:k]
    return float(np.asarray(y_true)[top_k_idx].mean())


def _make_model(**params):
    if HAS_XGB:
        defaults = dict(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            eval_metric="auc",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(params)
        return XGBClassifier(**defaults)

    defaults = dict(max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=1.0, random_state=42)
    defaults.update({k: v for k, v in params.items() if k in defaults})
    return HistGradientBoostingClassifier(**defaults)


def _search_space():
    if HAS_XGB:
        return {
            "n_estimators": [200, 300, 500, 700],
            "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
            "max_depth": [2, 3, 4, 5, 6],
            "min_child_weight": [1, 3, 5, 10],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
            "gamma": [0, 0.1, 0.5, 1.0],
        }
    return {
        "max_iter": [150, 300, 500],
        "learning_rate": [0.02, 0.05, 0.1],
        "max_depth": [2, 3, 4, 6],
        "l2_regularization": [0.0, 1.0, 5.0],
    }


def train_churn_model(feats: pd.DataFrame, test_size: float = 0.2, tune: bool = True, n_iter: int = 30):
    """
    Train the churn classifier.

    With tune=True (default) a 5-fold stratified randomized search picks the
    hyper-parameters that maximise cross-validated AUC on the TRAINING split only;
    the held-out test split is never used for tuning, so the reported AUC is honest.

    Returns: model, X_test, y_test, probs (predicted churn probability), auc
    """
    X = feats[FEATURE_COLS]
    y = feats["Churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Tuning needs enough rows in each class for 5 folds; skip it on tiny datasets.
    can_tune = tune and y_train.value_counts().min() >= 25

    if can_tune:
        search = RandomizedSearchCV(
            _make_model(),
            param_distributions=_search_space(),
            n_iter=n_iter,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            random_state=42,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f"Best CV AUC: {search.best_score_:.3f} | params: {search.best_params_}")
    else:
        model = _make_model()
        model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    return model, X_test, y_test, probs, auc


if __name__ == "__main__":
    from src import visualize as viz

    cust = pd.read_csv("data/processed/clean_customers.csv", parse_dates=["InvoiceDate"])

    try:
        cancels = pd.read_csv("data/processed/cancellations.csv", parse_dates=["InvoiceDate"])
    except FileNotFoundError:
        cancels = None
        print("Note: cancellations.csv not found — run src/data_prep.py first for the ReturnRate feature.")

    # Cutoff ~90 days before the end of the dataset, leaving room to observe churn after it
    max_date = cust["InvoiceDate"].max()
    cutoff = max_date - pd.Timedelta(days=90)
    print(f"Data range: {cust['InvoiceDate'].min()} to {max_date}")
    print(f"Cutoff date: {cutoff}")

    feats = build_churn_dataset(cust, cutoff_date=cutoff, horizon_days=90, cancels=cancels)
    print("Churn dataset shape:", feats.shape)
    print("Churn rate:", feats["Churned"].mean().round(3))
    print("Mean ReturnRate (engineered feature):", feats["ReturnRate"].mean().round(3))

    model, X_test, y_test, probs, auc = train_churn_model(feats)
    p20 = precision_at_k(y_test.values, probs, k_frac=0.2)
    print(f"AUC-ROC: {auc:.3f}  (target >= {TARGET_AUC})")
    print(f"Precision@top20%: {p20:.3f}  (target >= {TARGET_PRECISION_AT_20})")
    if auc < TARGET_AUC or p20 < TARGET_PRECISION_AT_20:
        print("Warning: below the brief's acceptance targets on this data/cutoff.")

    feats.to_csv("data/processed/churn_features.csv", index=False)

    # --- Dashboard-ready outputs ---
    # Score every customer (not just the held-out test split) for the risk table
    feats["ChurnProbability"] = model.predict_proba(feats[FEATURE_COLS])[:, 1].round(3)

    try:
        seg = pd.read_csv("data/processed/rfm_segmented.csv")[["Customer ID", "Segment"]]
        feats = feats.merge(seg, on="Customer ID", how="left")
    except FileNotFoundError:
        print("Note: run src/segmentation.py first to attach Segment names to churn results.")

    feats.to_csv("data/processed/churn_results.csv", index=False)

    pd.DataFrame(
        [{"AUC_ROC": round(auc, 3), "Precision_at_top20": round(p20, 3), "ChurnRate": round(feats["Churned"].mean(), 3)}]
    ).to_csv("data/processed/churn_metrics.csv", index=False)

    print("Saving figures:")
    print(" -", viz.plot_roc_curve(y_test, probs))
    print(" -", viz.plot_shap_summary(model, X_test))
