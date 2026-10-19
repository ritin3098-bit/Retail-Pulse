"""
RetailPulse automatic data and model pipeline.

Run from the project root:
    python -m src.pipeline

Pipeline stages:
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
import threading
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"

# Prevent concurrent runs within this Python process.
_PIPELINE_LOCK = threading.Lock()


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


# ============================================================
# CUSTOM EXCEPTION
# ============================================================

class PipelineError(RuntimeError):
    """Raised when a pipeline stage fails."""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def outputs_exist(paths):
    """Return True if every expected output exists and is non-empty."""

    return all(
        path.is_file() and path.stat().st_size > 0
        for path in paths
    )


def _log(message, log_callback=None):
    """Print a message and optionally send it to the dashboard."""

    print(message, flush=True)

    if log_callback is not None:
        log_callback(message)


# ============================================================
# RUN ONE PIPELINE STAGE
# ============================================================

def run_stage(stage, force=False, log_callback=None):
    """Execute one stage, skipping it when outputs already exist."""

    name = stage["name"]
    module = stage["module"]
    required = stage["required"]

    if not force and outputs_exist(required):
        _log(
            f"SKIPPED: {name} — required output files already exist.",
            log_callback,
        )
        return

    _log(f"\n{'=' * 60}", log_callback)
    _log(f"STARTING: {name}", log_callback)
    _log(f"Module: {module}", log_callback)
    _log(f"{'=' * 60}", log_callback)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
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
        partial_output = exc.stdout or ""

        if isinstance(partial_output, bytes):
            partial_output = partial_output.decode(
                "utf-8",
                errors="replace",
            )

        raise PipelineError(
            f"{name} exceeded the 30-minute time limit.\n"
            f"Last output:\n{partial_output[-4000:]}"
        ) from exc

    except OSError as exc:
        raise PipelineError(
            f"Could not start {name} ({module}): {exc}"
        ) from exc

    output = result.stdout or ""

    # Show the latest output in the logs.
    if output:
        _log(output[-10000:], log_callback)

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
            f"{name} completed, but these output files "
            f"are missing or empty:\n"
            + "\n".join(missing)
        )

    _log(f"COMPLETED: {name}", log_callback)


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

def run_pipeline(force=False, log_callback=None):
    """
    Generate missing RetailPulse datasets and reports.

    Args:
        force:
            If True, run every stage even if its outputs exist.
        log_callback:
            Optional callback for reporting progress.

    Raises:
        PipelineError:
            If another run is active, a stage fails, or the
            required raw dataset is unavailable.
    """

    # Avoid starting another pipeline in the same process.
    if not _PIPELINE_LOCK.acquire(blocking=False):
        raise PipelineError(
            "The RetailPulse pipeline is already running in this "
            "Python process. Wait for it to finish before retrying."
        )

    try:
        # Validate the source dataset.
        if not RAW_FILE.is_file():
            raise PipelineError(
                "The raw dataset was not found:\n"
                "data/raw/online_retail_II.xlsx\n\n"
                "Ensure the original Excel dataset is available "
                "to the application."
            )

        # Prepare directories.
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        _log(
            "RetailPulse automatic pipeline started.",
            log_callback,
        )

        # Execute stages sequentially.
        for stage in STAGES:
            run_stage(
                stage=stage,
                force=force,
                log_callback=log_callback,
            )

        _log(
            "\nRetailPulse pipeline completed successfully.",
            log_callback,
        )

    finally:
        _PIPELINE_LOCK.release()


# ============================================================
# COMMAND-LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":
    try:
        run_pipeline()

    except PipelineError as exc:
        print(
            f"\nPIPELINE ERROR:\n{exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    except Exception as exc:
        print(
            f"\nUNEXPECTED PIPELINE ERROR: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)