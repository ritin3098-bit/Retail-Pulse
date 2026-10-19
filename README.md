# RetailPulse — AI-Powered Customer Analytics & Demand Forecasting

Zidio Development — Data Science & Analytics project (March 2026 edition).

## Project structure

```

retailpulse/

├── data/

│   ├── raw/                  # original Online Retail II file (not committed)

│   └── processed/            # cleaned CSVs: clean_all, clean_customers, daily_sales, rfm(_segmented)

├── src/                     # reusable, tested logic — imported by the dashboard and tests

│   ├── data_prep.py          # loading, cleaning, RFM           [done]

│   ├── segmentation.py       # K-Means + DBSCAN                 [done]

│   ├── forecasting.py        # naive, Prophet, LSTM, ensemble   [done]

│   ├── visualize.py          # all plots -> reports/figures/    [done]

│   ├── churn.py              # cutoff labels, XGBoost, SHAP     [done]

│   ├── inventory.py          # simulated stock + reorder logic  [done]

│   └── monitoring.py         # Evidently drift detection        [TODO]

├── dashboard/                # Streamlit app                    [done]

│   ├── app.py                 # Overview page (KPIs, revenue trend, segment snapshot)

│   ├── utils.py                # shared CSS theme + cached data loaders

│   └── pages/

│       ├── 1_Forecasting.py        # MAPE comparison, what-if growth slider

│       ├── 2_Segments_Churn.py     # segment table, churn risk table, SHAP

│       ├── 3_Inventory.py          # reorder status table, CSV export

│       └── 4_Monitoring.py         # live drift check + simulated drift demo

├── models/                   # saved model files (not committed)

├── reports/figures/          # PNGs for the PDF report

├── tests/                    # pytest unit tests

├── requirements.txt

└── README.md

```

## Setup

```bash

python -m venv .venv

source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

```

Place the Online Retail II Excel file at `data/raw/online_retail_II.xlsx`.

## Run the pipeline so far

```bash

python -m src.data_prep        # cleans data, builds daily series + RFM, saves EDA figures

python -m src.segmentation     # K-Means + DBSCAN clustering, saves segmentation figures

python -m src.forecasting      # Prophet + LSTM + ensemble, prints MAPE, saves forecast figures

python -m src.churn            # builds churn labels, trains XGBoost, prints AUC + precision@20%

python -m src.inventory        # simulates stock, computes reorder points, flags products to reorder

python -m src.monitoring       # drift check (natural + simulated), saves Evidently HTML reports

pytest tests/                  # sanity checks (run from the project root — see pytest.ini)

```

Run the first five in order once — each saves CSVs/figures the next step or the

dashboard depends on. `src.monitoring` and `pytest` can run any time after.

## Run the dashboard

```bash

streamlit run dashboard/app.py

```

The dashboard reads only the CSVs and PNGs the pipeline above already saved —

it does **not** retrain any model at runtime, which is what keeps page loads

fast. If a page shows "run this command first", it means that pipeline step

hasn't been run yet (or `data/processed/` was cleared).

Each command prints the path of every figure it saves. All figures land in

`reports/figures/`, numbered in the order they'd appear in the report

(01_monthly_revenue.png ... 10_lstm_forecast.png) — ready to drop straight

into the PDF.

## Progress & key decisions

**Data cleaning** — Of ~1.07M raw rows: 243,007 had no Customer ID, 19,494 were

cancellations, 22,950 had non-positive quantity, 6,207 had non-positive price,

34,335 were exact duplicates (the two source sheets overlap in early Dec 2010).

Two cleaned tables are kept: `clean` (all valid sales, used for forecasting)

and `cust` (sales with a known customer, used for segmentation/churn).

**Segmentation** — RFM features were log-transformed before clustering

(Frequency and Monetary are heavily right-skewed). Silhouette analysis

peaked at k=4 (0.365); k=6 was chosen instead (0.335) to meet the project's

requirement of 6-8 business-interpretable segments — documented as a

deliberate trade-off, not an oversight. DBSCAN was run as a validation

method: it isolated high-value customers as "noise" (mean Monetary ≈ 29,988),

independently confirming the K-Means "Champions" segment, but produced

highly imbalanced cluster sizes (4 to 3,118 customers) so K-Means remains

the primary method feeding the dashboard.

Segment names (edit `SEGMENT_NAMES` in `src/segmentation.py` after reviewing

your own `cluster_summary()` output — cluster IDs can shift between runs):

Champions, Loyal Customers, Promising, At Risk, New/Low-Value, Lost/Dormant.

**Forecasting** — Daily revenue forecasting gave MAPE of 23-26% for both a

seasonal-naive baseline and Prophet — too noisy, driven by day-to-day

variance and a single large outlier near the end of the dataset

(2011-12-09, ~200,919 revenue in one day). Switching to **weekly

aggregation** reduced Prophet's MAPE to 12.99%. Adding the engineered

`IsPeakSeason` column as a Prophet regressor reduced it further to

**11.73%**, crossing the project's <=12% target.

Final results (weekly, 6-week holdout):

| Model                              | MAPE   |

|-------------------------------------|--------|

| Naive Baseline                      | 16.36% |

| Prophet (no engineered feature)     | 12.99% |

| **Prophet (+ IsPeakSeason)**         | **11.73%** |

| LSTM                                 | 18.31% |

| Prophet+LSTM Ensemble                | 13.63% |

**Model selection decision:** Prophet + IsPeakSeason was chosen as the

final/production forecasting model, NOT the Prophet+LSTM ensemble the brief

originally sketches. With only ~100 weekly data points, the LSTM

underperforms even the naive baseline, and blending it into an ensemble

drags down an already-strong Prophet model rather than improving it. This

is reported as a deliberate, data-driven choice — the LSTM and ensemble

code remain in `src/forecasting.py` as a documented experiment, since

showing an approach was tried and correctly rejected is itself evidence of

technical rigor, not a gap.

**Churn prediction** — Cutoff date set to 90 days before the end of the

dataset. Features (Recency, Frequency, Monetary, average order value,

tenure, unique products bought) are computed only from transactions

*before* the cutoff; the label (churned = 1) is based only on whether the

customer purchased again in the 90 days *after* the cutoff — this avoids

leaking future information into the features. XGBoost is trained on this

split; `src/churn.py` prints AUC-ROC and precision@top-20% for your dataset

(brief targets: AUC-ROC >= 0.88, precision@top20% >= 0.75). Actual results:

**AUC-ROC 0.785** (below target — report this honestly; next steps to try

would be Optuna hyperparameter tuning and a longer feature-engineering

pass) and **Precision@top20% 0.834** (exceeds target). SHAP summary plot

explains which features drive predictions.

**Inventory optimization** — The source dataset has no stock/inventory

column, so current stock, lead time, and holding cost are **simulated**

(fixed random seed, so results are reproducible) for the top 20 products by

revenue — this is stated explicitly here and should be stated in the report

too. Demand, however, is real: average daily units sold is computed from

actual sales history. `reorder_point = avg_daily_demand * lead_time_days +

safety_stock`; a product is flagged "Reorder Now" when current (simulated)

stock falls below its reorder point, with a recommended order quantity to

bring it back to a target level (2x reorder point — a simple, explainable

policy, not a claim of optimality).

Simulated current stock is expressed as a random **number of days of

coverage** (5-45 days) converted to units via each product's own real

demand — not a flat unit range. A flat range (e.g. 50-500 units for every

product) would be unrealistic: a top seller moving thousands of units a

month would never realistically be stocked at only 100 units, yet share

the same reorder math as a slow-moving product. Scaling by demand keeps

the simulation internally consistent.

**Drift detection** — `src/monitoring.py` splits the cleaned transaction

data in two by date (first half = reference, second half = current) and

checks `Quantity`, `Price`, and `Revenue` for distribution drift two ways:

a fast Kolmogorov-Smirnov test (used live in the dashboard) and a full

Evidently AI report (saved as HTML, with per-feature plots). Since the

natural data may not show much drift, `simulate_drift()` artificially

shifts a column (default: `Price` × 1.6) to demonstrate the detector

actually catching drift — this is what to show in the demo video, alongside

the natural check for contrast. The Evidently report's `DataDriftPreset` is

explicitly configured to use the KS test (`num_stattest="ks"`) so its

results agree with the dashboard's live KS-based check — Evidently's

default test choice for larger datasets can otherwise disagree with a

plain KS test.

**Dashboard** — Four pages, all reading pre-computed CSVs/PNGs (no retraining

at runtime): **Overview** (KPIs, revenue trend, segment snapshot, inventory

and churn health strip), **Forecasting** (MAPE comparison across all four

models, a what-if slider that scales the forward forecast ±20-30% to

simulate demand scenarios), **Segments & Churn** (filterable segment table,

churn risk leaderboard, SHAP explanation, CSV exports), and **Monitoring**

(live drift check with a toggle between natural and simulated drift).

**Still to do:** deployment (Docker, Streamlit Community Cloud / Hugging

Face Spaces), CI/CD, and the final report/README polish.

## Known install note

`evidently`'s API changed significantly between versions. This project

pins `evidently>=0.4,<0.5`. If that range is no longer installable by the

time you set this up, install the latest available 0.4.x release

explicitly (e.g. `pip install evidently==0.4.33`) rather than letting pip

pull a newer major version, or `src/monitoring.py`'s `generate_drift_report`

import will need updating to match the new API (the `quick_drift_check`

KS-test function has no such dependency and will work regardless).

## Engineered features added to the dataset

Per mentor feedback, the raw/cleaned dataset was not used as-is — new

columns were engineered and shown to measurably change results, not just

added for appearance:

- **Month, IsWeekend, IsPeakSeason** (`src/data_prep.add_engineered_features`)

  — added directly to `clean_all.csv` and `clean_customers.csv`.

  `IsPeakSeason` (Oct-Dec) is fed into Prophet as an external regressor

  (`src/forecasting.fit_prophet(use_regressor=True)`). Running

  `python -m src.forecasting` prints the MAPE **with and without** this

  regressor side by side, so the improvement (or lack of one, if that's

  what your data shows) is directly visible and reportable.

- **ReturnRate** (`src/churn.attach_return_rate`) — cancelled orders /

  (purchased + cancelled orders) per customer, computed only from

  cancellation history before the churn cutoff date (saved separately as

  `cancellations.csv` by `src/data_prep.get_cancellations`). This captures

  a return/complaint-driven risk signal that plain RFM cannot — two

  customers with identical Recency/Frequency/Monetary can have very

  different churn risk if one returns items constantly. It's included in

  `FEATURE_COLS` and will appear in the SHAP summary plot

  (`12_churn_shap_summary.png`) if it has meaningful predictive power.

## Notes for the report

- Inventory data does not exist in the source dataset and will be

  simulated — state this explicitly in the report.

- The MAPE metric excludes zero-revenue (non-operating) days; including

  them distorts the percentage error toward infinity when actual revenue

  is 0. See `mape()` in `src/forecasting.py`.