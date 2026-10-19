Keep one notebook per phase, importing logic from `src/` rather than
redefining it inline - this avoids the kernel/variable mixups from early
development.

- 01_eda.ipynb            - load data, audit quality, EDA plots (uses src.data_prep)
- 02_segmentation.ipynb   - K-Means + DBSCAN, cluster naming (uses src.segmentation)
- 03_forecasting.ipynb    - Prophet + LSTM + ensemble, plots (uses src.forecasting)
- 04_churn.ipynb          - TODO
- 05_inventory.ipynb      - TODO
