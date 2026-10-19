import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from src.churn import (
    build_churn_dataset,
    FEATURE_COLS,
    _make_model,
)
import pandas as pd


# Load data
cust = pd.read_csv(
    "data/processed/clean_customers.csv",
    parse_dates=["InvoiceDate"],
)

try:
    cancels = pd.read_csv(
        "data/processed/cancellations.csv",
        parse_dates=["InvoiceDate"],
    )
except FileNotFoundError:
    cancels = None


# Use the same cutoff as the current experiment
max_date = cust["InvoiceDate"].max()
cutoff = max_date - pd.Timedelta(days=90)

feats = build_churn_dataset(
    cust,
    cutoff_date=cutoff,
    horizon_days=90,
    cancels=cancels,
)

X = feats[FEATURE_COLS].replace([np.inf, -np.inf], np.nan)
y = feats["Churned"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

models = {
    "XGBoost": _make_model(),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Mean CV AUC": cv_scores.mean(),
        "CV Std": cv_scores.std(),
        "Test AUC": test_auc,
    })

    print(f"Mean CV AUC: {cv_scores.mean():.4f}")
    print(f"Test AUC: {test_auc:.4f}")

print("\nFINAL COMPARISON")
print(pd.DataFrame(results).sort_values("Mean CV AUC", ascending=False))