"""
RetailPulse automatic data and model pipeline.

Run from the project root:
    python -m src.pipeline

Pipeline order:
    1. Data preparation
    2. Customer segmentation
    3. Demand forecasting
    4. Churn prediction
    5. Inventory optimization
    6. Drift monitoring
"""

import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ============================================================
# PIPELINE STAGES
# ============================================================

STAGES = [
    {
        "name": "Data Preparation",
        "module": "src.data_prep",
        "required": [
            DATA_DIR / "clean_all.csv",
            DATA_DIR / "clean_customers.csv",
            DATA_DIR / "daily_sales.csv",
            DATA_DIR / "rfm.csv",
        ],
    },
    {
        "name": "Customer Segmentation",
        "module": "src.segmentation",
        "required": [
            DATA_DIR / "rfm_segmented.csv",
        ],
    },
    {
        "name": "Demand Forecasting",
        "module": "src.forecasting",
        "required": [
            DATA_DIR / "forecast_results.csv",
            DATA_DIR / "forecast_metrics.csv",
        ],
    },
    {
        "name": "Churn Prediction",
        "module": "src.churn",
        "required": [
            DATA_DIR / "churn_results.csv",
            DATA_DIR / "churn_metrics.csv",
        ],
    },
    {
        "name": "Inventory Optimization",
        "module": "src.inventory",
        "required": [
            DATA_DIR / "inventory_report.csv",
        ],
    },
    {
        "name": "Drift Monitoring",
        "module": "src.monitoring",
        "required": [
            REPORTS_DIR / "drift_report_natural.html",
            REPORTS_DIR / "drift_report_simulated.html",
        ],
    },
]


class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


def outputs_exist(paths):
    """Return True when all expected output files exist and are non-empty."""
    return all(
        path.is_file() and path.stat().st_size > 0
        for path in paths
    )


def run_stage(stage, force=False, log_callback=None):
    """Run one pipeline stage if its required outputs are missing."""

    name = stage["name"]
    module = stage["module"]
    required = stage["required"]

    def log(message):
        print(message, flush=True)

        if log_callback is not None:
            log_callback(message)

    if not force and outputs_exist(required):
        log(f"SKIPPED: {name} — output files already exist.")
        return

    log(f"\n{'=' * 60}")
    log(f"STARTING: {name}")
    log(f"Module: {module}")
    log(f"{'=' * 60}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                module,
            ],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
        )

    except subprocess.TimeoutExpired as exc:
        raise PipelineError(
            f"{name} exceeded the 30-minute time limit. "
            "Check the model's training time and available resources."
        ) from exc

    output = result.stdout or ""

    if output:
        log(output[-10000:])

    if result.returncode != 0:
        raise PipelineError(
            f"{name} failed with exit code {result.returncode}.\n"
            f"Last output:\n{output[-8000:]}"
        )

    missing = [
        str(path.relative_to(PROJECT_ROOT))
        for path in required
        if not path.is_file() or path.stat().st_size == 0
    ]

    if missing:
        raise PipelineError(
            f"{name} completed, but these output files are missing:\n"
            + "\n".join(missing)
        )

    log(f"COMPLETED: {name}")


def run_pipeline(force=False, log_callback=None):
    """
    Generate missing RetailPulse outputs.

    Args:
        force: Run every stage, even when outputs exist.
        log_callback: Optional function for displaying progress.
    """

    raw_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "online_retail_II.xlsx"
    )

    if not raw_file.is_file():
        raise PipelineError(
            "The raw dataset was not found at:\n"
            "data/raw/online_retail_II.xlsx\n\n"
            "Make the original Excel dataset available to the "
            "application before running the pipeline."
        )

    # Create output directories.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def log(message):
        print(message, flush=True)

        if log_callback is not None:
            log_callback(message)

    log("RetailPulse automatic pipeline started.")

    for stage in STAGES:
        run_stage(
            stage,
            force=force,
            log_callback=log_callback,
        )

    log("\nRetailPulse pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except PipelineError as exc:
        print(f"\nPIPELINE ERROR:\n{exc}", file=sys.stderr)
        sys.exit(1)