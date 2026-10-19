"""
Customer segmentation for RetailPulse: K-Means (primary) and DBSCAN (validation).

Usage:
    from src.segmentation import prepare_features, fit_kmeans, fit_dbscan, SEGMENT_NAMES

    X, scaler = prepare_features(rfm)
    rfm["Cluster"] = fit_kmeans(X, k=6)
    rfm["Segment"] = rfm["Cluster"].map(SEGMENT_NAMES)
    rfm["DBSCAN_Cluster"] = fit_dbscan(X, eps=0.3, min_samples=5)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

FEATURES = ["Recency_log", "Frequency_log", "Monetary_log"]

# Fill in after reviewing your own cluster summary table (see README / Step 5).
# Keys are K-Means cluster ids, values are business-friendly names.
SEGMENT_NAMES = {
    0: "New/Low-Value",
    1: "At Risk",
    2: "Lost/Dormant",
    3: "Champions",
    4: "Promising",
    5: "Loyal Customers",
}


def prepare_features(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform skewed RFM columns and scale them for clustering."""
    rfm["Recency_log"] = np.log1p(rfm["Recency"])
    rfm["Frequency_log"] = np.log1p(rfm["Frequency"])
    rfm["Monetary_log"] = np.log1p(rfm["Monetary"])

    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[FEATURES])
    return X, scaler


def evaluate_k_range(X: np.ndarray, k_range=range(2, 11)) -> pd.DataFrame:
    """Compute inertia and silhouette score for a range of k, to choose k."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        rows.append(
            {"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, labels)}
        )
    return pd.DataFrame(rows)


def fit_kmeans(X: np.ndarray, k: int) -> np.ndarray:
    """Fit final K-Means model and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    return km.fit_predict(X)


def fit_dbscan(X: np.ndarray, eps: float, min_samples: int = 5) -> np.ndarray:
    """Fit DBSCAN and return cluster labels (-1 = noise/outlier)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    return db.fit_predict(X)


def cluster_summary(rfm: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    """Mean RFM values and size per cluster — use this to name segments."""
    summary = rfm.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean().round(1)
    summary["Count"] = rfm[cluster_col].value_counts()
    return summary.sort_values("Monetary", ascending=False)


if __name__ == "__main__":
    from src import visualize as viz

    rfm = pd.read_csv("data/processed/rfm.csv")

    X, scaler = prepare_features(rfm)

    k_eval = evaluate_k_range(X)
    print(k_eval)

    rfm["Cluster"] = fit_kmeans(X, k=6)
    rfm["Segment"] = rfm["Cluster"].map(SEGMENT_NAMES)
    rfm["DBSCAN_Cluster"] = fit_dbscan(X, eps=0.3, min_samples=5)

    print(cluster_summary(rfm, "Cluster"))
    print(cluster_summary(rfm, "DBSCAN_Cluster"))

    rfm.to_csv("data/processed/rfm_segmented.csv", index=False)

    # Segmentation figures -> reports/figures/
    print("Saving figures:")
    print(" -", viz.plot_rfm_distributions(rfm, log=True))
    print(" -", viz.plot_elbow(k_eval))
    print(" -", viz.plot_clusters(rfm, "Cluster", x="Recency", y="Monetary"))
    print(" -", viz.plot_clusters(rfm, "Cluster", x="Recency", y="Frequency"))
    print(" -", viz.plot_clusters(rfm, "DBSCAN_Cluster", x="Recency", y="Monetary"))
