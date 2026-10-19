"""
All plotting for RetailPulse. Every function saves a PNG to reports/figures/
and returns the path, so the report-writing step can just list the folder.

Usage:
    from src import visualize as viz
    viz.plot_monthly_revenue(clean)
    viz.plot_top_countries_products(clean)
    viz.plot_distributions(clean, cust)
    viz.plot_correlation(clean)
    viz.plot_rfm_distributions(rfm)
    viz.plot_rfm_distributions(rfm, log=True)
    viz.plot_elbow(k_eval_df)
    viz.plot_clusters(rfm, "Cluster")
    viz.plot_prophet_forecast(model, forecast)
    viz.plot_lstm_forecast(actual, preds)
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = "reports/figures"
sns.set_theme(style="whitegrid")


def _save(fig_or_plt, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig_or_plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_monthly_revenue(clean):
    monthly = clean.set_index("InvoiceDate")["Revenue"].resample("ME").sum()
    plt.figure(figsize=(10, 4))
    monthly.plot(marker="o")
    plt.title("Monthly Revenue")
    plt.ylabel("Revenue")
    plt.tight_layout()
    return _save(plt, "01_monthly_revenue.png")


def plot_top_countries_products(clean, n=10):
    top_countries = clean.groupby("Country")["Revenue"].sum().nlargest(n)
    top_products = clean.groupby("Description")["Revenue"].sum().nlargest(n)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    top_countries.sort_values().plot.barh(ax=ax[0], title=f"Top {n} Countries by Revenue")
    top_products.sort_values().plot.barh(ax=ax[1], title=f"Top {n} Products by Revenue")
    plt.tight_layout()
    return _save(fig, "02_top_countries_products.png")


def plot_distributions(clean, cust):
    order_value = clean.groupby("Invoice")["Revenue"].sum()
    orders_per_customer = cust.groupby("Customer ID")["Invoice"].nunique()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    sns.histplot(order_value, log_scale=True, bins=50, ax=ax[0])
    ax[0].set_title("Order Value Distribution (log scale)")
    sns.histplot(orders_per_customer, log_scale=True, bins=40, ax=ax[1])
    ax[1].set_title("Orders per Customer (log scale)")
    plt.tight_layout()
    return _save(fig, "03_distributions.png")


def plot_correlation(clean):
    corr = clean[["Quantity", "Price", "Revenue"]].corr()
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return _save(plt, "04_correlation.png")


def plot_rfm_distributions(rfm, log=False):
    cols = ["Recency_log", "Frequency_log", "Monetary_log"] if log else ["Recency", "Frequency", "Monetary"]
    suffix = "log" if log else "raw"
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, col in enumerate(cols):
        sns.histplot(rfm[col], ax=ax[i], bins=40).set_title(col)
    plt.tight_layout()
    return _save(fig, f"05_rfm_distributions_{suffix}.png")


def plot_elbow(k_eval_df):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(k_eval_df["k"], k_eval_df["inertia"], marker="o")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia"); ax[0].set_title("Elbow Method")
    ax[1].plot(k_eval_df["k"], k_eval_df["silhouette"], marker="o", color="orange")
    ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette Score"); ax[1].set_title("Silhouette Analysis")
    plt.tight_layout()
    return _save(fig, "06_k_selection.png")


def plot_clusters(rfm, cluster_col, x="Recency", y="Monetary"):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=rfm, x=x, y=y, hue=cluster_col, palette="tab10", alpha=0.6)
    plt.title(f"Customer Segments ({cluster_col}): {x} vs {y}")
    plt.tight_layout()
    return _save(plt, f"07_{cluster_col.lower()}_{x.lower()}_{y.lower()}.png")


def plot_prophet_forecast(model, forecast):
    fig1 = model.plot(forecast)
    plt.title("Prophet Forecast")
    p1 = _save(fig1, "08_prophet_forecast.png")

    fig2 = model.plot_components(forecast)
    p2 = _save(fig2, "09_prophet_components.png")
    return p1, p2


def plot_roc_curve(y_test, probs):
    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Churn Model ROC Curve")
    plt.legend()
    plt.tight_layout()
    return _save(plt, "11_churn_roc.png")


def plot_shap_summary(model, X_test):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    return _save(plt, "12_churn_shap_summary.png")


def plot_inventory_status(report):
    import matplotlib.patches as mpatches

    df = report.sort_values("CurrentStock")
    colors = df["Status"].map({"Reorder Now": "#d62728", "OK": "#2ca02c"})

    plt.figure(figsize=(10, max(4, 0.35 * len(df))))
    plt.barh(df["StockCode"].astype(str), df["CurrentStock"], color=colors, label="Current Stock")
    plt.plot(df["ReorderPoint"], df["StockCode"].astype(str), "ko", markersize=4, label="Reorder Point")
    plt.xlabel("Units")
    plt.title("Inventory Status: Current Stock vs Reorder Point")
    handles = [
        mpatches.Patch(color="#d62728", label="Reorder Now"),
        mpatches.Patch(color="#2ca02c", label="OK"),
    ]
    plt.legend(handles=handles)
    plt.tight_layout()
    return _save(plt, "13_inventory_status.png")


def plot_drift_summary(drift_df, title="Drift Check"):
    colors = drift_df["drift_detected"].map({True: "#d62728", False: "#2ca02c"})
    plt.figure(figsize=(7, 4))
    plt.bar(drift_df["column"], drift_df["ks_statistic"], color=colors)
    plt.ylabel("KS Statistic")
    plt.title(title)
    plt.tight_layout()
    fname = "14_drift_" + title.lower().replace(" ", "_") + ".png"
    return _save(plt, fname)


def plot_lstm_forecast(actual, preds):
    plt.figure(figsize=(10, 4))
    plt.plot(actual, label="Actual", marker="o")
    plt.plot(preds, label="LSTM Predicted", marker="o")
    plt.legend()
    plt.title("LSTM Forecast vs Actual")
    plt.tight_layout()
    return _save(plt, "10_lstm_forecast.png")
