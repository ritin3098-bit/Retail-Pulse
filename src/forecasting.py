"""
Demand forecasting for RetailPulse: naive baseline, Prophet, and LSTM.

Daily revenue was too noisy (one large outlier + zero-revenue closed days
distorted MAPE), so forecasting is done on a WEEKLY aggregation. See README
for the full reasoning.

Usage:
    from src.forecasting import build_weekly_series, mape, fit_prophet, fit_lstm

    weekly = build_weekly_series(daily)
    train_w, test_w = weekly.iloc[:-6], weekly.iloc[-6:]
    preds_prophet, model, forecast = fit_prophet(train_w, periods=6)
    print(mape(test_w["y"].values, preds_prophet))
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, excluding zero-revenue days (closed days)."""
    actual, pred = np.array(actual), np.array(pred)
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def build_weekly_series(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a daily (ds, y, IsPeakSeason) series into weekly totals.
    IsPeakSeason is carried through (max over the week) as an engineered
    regressor for Prophet — see fit_prophet(use_regressor=True).
    """
    if "IsPeakSeason" not in daily.columns:
        daily = daily.copy()
        daily["IsPeakSeason"] = daily["ds"].dt.month.isin([10, 11, 12]).astype(int)

    weekly = daily.set_index("ds").resample("W").agg({"y": "sum", "IsPeakSeason": "max"}).reset_index()
    return weekly


def fit_prophet(train: pd.DataFrame, periods: int = 6, use_regressor: bool = True):
    """
    Fit Prophet on weekly data.

    use_regressor: if True AND the data has an 'IsPeakSeason' column, add it
    as an external regressor — an engineered feature (Oct-Dec holiday flag)
    that Prophet would not otherwise know about. Compare MAPE with this
    on/off to see the measurable effect of the engineered column.

    Returns:
        preds: predicted values for the next `periods` weeks
        model: the fitted Prophet model (needed for plotting)
        forecast: the full forecast DataFrame (needed for plotting)
    """
    has_regressor = use_regressor and "IsPeakSeason" in train.columns

    model = Prophet(weekly_seasonality=False, yearly_seasonality=True)
    if has_regressor:
        model.add_regressor("IsPeakSeason")
    model.fit(train)

    future = model.make_future_dataframe(periods=periods, freq="W")
    if has_regressor:
        future["IsPeakSeason"] = future["ds"].dt.month.isin([10, 11, 12]).astype(int)

    forecast = model.predict(future)
    preds = forecast.iloc[-periods:]["yhat"].values
    return preds, model, forecast


class SimpleLSTM(nn.Module):
    """Single-layer LSTM for univariate time series forecasting."""

    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def make_sequences(data: np.ndarray, window: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Turn a scaled 1D series into sliding-window (X, y) sequences."""
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i : i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


def fit_lstm(
    weekly: pd.DataFrame, window: int = 8, test_size: int = 6, epochs: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the LSTM on all but the last `test_size` weeks, predict the held-out weeks.

    Returns:
        preds: predicted values, in original revenue scale
        actual: true values for the held-out weeks, in original revenue scale
    """
    values = weekly["y"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X_seq, y_seq = make_sequences(scaled, window)
    X_train, y_train = X_seq[:-test_size], y_seq[:-test_size]
    X_test, y_test = X_seq[-test_size:], y_seq[-test_size:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = SimpleLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds_scaled = model(X_test_t).numpy()

    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test).flatten()
    return preds, actual


def ensemble(preds_prophet: np.ndarray, preds_lstm: np.ndarray, weight_prophet: float = 0.7) -> np.ndarray:
    """Weighted average of Prophet and LSTM predictions. Tune weight_prophet on validation data."""
    return weight_prophet * preds_prophet + (1 - weight_prophet) * preds_lstm


def forecast_future(weekly: pd.DataFrame, periods: int = 8):
    """
    Fit Prophet on ALL available weekly data and forecast `periods` weeks
    beyond it. Used for the dashboard's forward-looking forecast (not for
    backtesting/MAPE — that uses the train/test split above).

    Prophet is used here rather than the LSTM: iteratively forecasting many
    steps ahead with an LSTM trained on ~100 points compounds error quickly,
    while Prophet's trend+seasonality model extrapolates more reliably with
    this little data. The LSTM remains valuable for the ensemble/backtest
    comparison above.
    """
    has_regressor = "IsPeakSeason" in weekly.columns

    model = Prophet(weekly_seasonality=False, yearly_seasonality=True)
    if has_regressor:
        model.add_regressor("IsPeakSeason")
    model.fit(weekly)

    future = model.make_future_dataframe(periods=periods, freq="W")
    if has_regressor:
        future["IsPeakSeason"] = future["ds"].dt.month.isin([10, 11, 12]).astype(int)

    forecast = model.predict(future)
    future_only = forecast[forecast["ds"] > weekly["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return future_only, model, forecast


if __name__ == "__main__":
    from src import visualize as viz

    daily = pd.read_csv("data/processed/daily_sales.csv", parse_dates=["ds"])
    weekly = build_weekly_series(daily)

    train_w, test_w = weekly.iloc[:-6], weekly.iloc[-6:]
    actual_w = test_w["y"].values

    naive_preds_w = weekly["y"].shift(1).iloc[-6:].values
    mape_naive = mape(actual_w, naive_preds_w)
    print(f"Naive baseline MAPE: {mape_naive:.2f}%")

    # Prove the engineered IsPeakSeason column actually changes the result,
    # not just sits in the dataset unused.
    preds_prophet_base, _, _ = fit_prophet(train_w[["ds", "y"]], periods=6, use_regressor=False)
    mape_prophet_base = mape(actual_w, preds_prophet_base)
    print(f"Prophet MAPE (no engineered feature): {mape_prophet_base:.2f}%")

    preds_prophet, model, forecast = fit_prophet(train_w, periods=6, use_regressor=True)
    mape_prophet = mape(actual_w, preds_prophet)
    print(f"Prophet MAPE (+ IsPeakSeason regressor): {mape_prophet:.2f}%")

    preds_lstm, actual_lstm = fit_lstm(weekly)
    mape_lstm = mape(actual_lstm, preds_lstm)
    print(f"LSTM MAPE: {mape_lstm:.2f}%")

    preds_ens = ensemble(preds_prophet, preds_lstm)
    mape_ens = mape(actual_w, preds_ens)
    print(f"Ensemble MAPE: {mape_ens:.2f}%")

    # --- Dashboard-ready outputs ---
    future_only, future_model, future_forecast = forecast_future(weekly, periods=8)

    hist = weekly.rename(columns={"y": "Revenue"}).copy()
    hist["Type"] = "Actual"
    future_df = future_only.rename(columns={"yhat": "Revenue"})[["ds", "Revenue"]]
    future_df["Type"] = "Forecast"
    results = pd.concat([hist[["ds", "Revenue", "Type"]], future_df], ignore_index=True)
    results.to_csv("data/processed/forecast_results.csv", index=False)

    metrics = pd.DataFrame(
        [
            {"Model": "Naive Baseline", "MAPE": round(mape_naive, 2)},
            {"Model": "Prophet (no engineered feature)", "MAPE": round(mape_prophet_base, 2)},
            {"Model": "Prophet (+ IsPeakSeason)", "MAPE": round(mape_prophet, 2)},
            {"Model": "LSTM", "MAPE": round(mape_lstm, 2)},
            {"Model": "Ensemble", "MAPE": round(mape_ens, 2)},
        ]
    )
    metrics.to_csv("data/processed/forecast_metrics.csv", index=False)
    print("\n", metrics)

    # Forecasting figures -> reports/figures/
    print("Saving figures:")
    for p in viz.plot_prophet_forecast(model, forecast):
        print(" -", p)
    print(" -", viz.plot_lstm_forecast(actual_lstm, preds_lstm))
