"""
Drift detection for RetailPulse.

Two ways to check drift:
  1. generate_drift_report() — full Evidently AI report (HTML + summary dict).
  2. quick_drift_check()     — lightweight KS-test per column, no extra
                                dependency beyond scipy, used by the dashboard
                                for instant feedback.

simulate_drift() artificially shifts a column's distribution — use this to
demonstrate drift detection working in your demo video, since the real
dataset's two halves may not show much natural drift.

Usage:
    from src.monitoring import split_reference_current, generate_drift_report, simulate_drift

    reference, current = split_reference_current(clean)
    summary, html_path = generate_drift_report(reference, current)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

MONITORED_COLUMNS = ["Quantity", "Price", "Revenue"]


def split_reference_current(clean: pd.DataFrame, split_date=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split transactions in two by date: first half = reference, second half = current."""
    if split_date is None:
        span = clean["InvoiceDate"].max() - clean["InvoiceDate"].min()
        split_date = clean["InvoiceDate"].min() + span / 2

    reference = clean[clean["InvoiceDate"] < split_date][MONITORED_COLUMNS].reset_index(drop=True)
    current = clean[clean["InvoiceDate"] >= split_date][MONITORED_COLUMNS].reset_index(drop=True)
    return reference, current


def simulate_drift(current: pd.DataFrame, column: str = "Price", shift_factor: float = 1.6) -> pd.DataFrame:
    """Artificially shift a column's distribution, to demonstrate drift detection working."""
    drifted = current.copy()
    drifted[column] = drifted[column] * shift_factor
    return drifted


def quick_drift_check(reference: pd.DataFrame, current: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Fast per-column drift check using the Kolmogorov-Smirnov test.
    No Evidently dependency — used by the dashboard for instant results.
    """
    rows = []
    for col in reference.columns:
        stat, p_value = ks_2samp(reference[col], current[col])
        rows.append(
            {
                "column": col,
                "ks_statistic": round(stat, 4),
                "p_value": round(p_value, 5),
                "drift_detected": p_value < alpha,
            }
        )
    return pd.DataFrame(rows)


def generate_drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, out_path: str = "reports/drift_report.html"
) -> tuple[dict, str]:
    """
    Full Evidently AI drift report. Saves an interactive HTML report and
    returns a summary dict for quick display (e.g. in the dashboard).
    """
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    # Force the KS test explicitly for numeric columns (threshold 0.05) so this
    # report agrees with quick_drift_check() above — Evidently otherwise picks
    # a different default test (e.g. Wasserstein distance) for larger datasets,
    # which can disagree with a straightforward KS-based check.
    report = Report(metrics=[DataDriftPreset(num_stattest="ks", num_stattest_threshold=0.05)])
    report.run(reference_data=reference, current_data=current)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    report.save_html(out_path)

    result = report.as_dict()["metrics"][0]["result"]
    summary = {
        "dataset_drift": bool(result["dataset_drift"]),
        "number_of_drifted_columns": int(result["number_of_drifted_columns"]),
        "share_of_drifted_columns": round(float(result["share_of_drifted_columns"]), 3),
    }
    return summary, out_path


if __name__ == "__main__":
    from src import visualize as viz

    clean = pd.read_csv("data/processed/clean_all.csv", parse_dates=["InvoiceDate"], dtype={"Invoice": str})
    reference, current = split_reference_current(clean)

    print("=== Natural drift check (first half vs second half of real data) ===")
    quick = quick_drift_check(reference, current)
    print(quick)

    summary, html_path = generate_drift_report(reference, current, "reports/drift_report_natural.html")
    print("Evidently summary:", summary)

    print("\n=== Simulated drift check (Price artificially shifted x1.6) ===")
    drifted = simulate_drift(current, column="Price", shift_factor=1.6)
    quick_sim = quick_drift_check(reference, drifted)
    print(quick_sim)

    summary_sim, html_path_sim = generate_drift_report(reference, drifted, "reports/drift_report_simulated.html")
    print("Evidently summary (simulated):", summary_sim)

    print("\nSaving figures:")
    print(" -", viz.plot_drift_summary(quick, title="Natural Drift Check"))
    print(" -", viz.plot_drift_summary(quick_sim, title="Simulated Drift Check"))
