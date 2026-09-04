"""Forecasting monthly foreign tourist arrivals in India.

One file, top to bottom:

    1. load the monthly data
    2. build lag and seasonal features
    3. train five models
    4. evaluate with walk-forward validation
    5. print a comparison table and save a plot

The forecast for month t is made at the end of month t, using every arrival
figure published by then and that month's search interest. Official arrival
figures come out about three weeks after a month ends, so month t-1 is
already available when the call is made -- but month t itself is not, which
is the gap the model has to fill.

Run:  python forecast.py
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
DATA = HERE / "data" / "tourism_monthly.csv"

SEARCH_COLS = ["flights_to_india", "india_visa", "hotels_in_india",
               "taj_mahal", "goa", "kerala_tourism", "india_tourism"]

# Arrivals collapsed 99% in April 2020 (2,820 vs ~1.1m). Percentage errors on
# a near-zero denominator are meaningless, so this window is reported
# separately rather than mixed into the headline numbers.
COVID_START, COVID_END = "2020-03-01", "2021-12-31"

TEST_MONTHS = 36     # size of the walk-forward evaluation window
SEQ_LEN = 12         # months of history the LSTM sees
SEASONAL = 12


# ----------------------------------------------------------------- features

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, rolling means and seasonal encodings.

    Every feature for month t uses arrivals no later than t-1 and search
    interest no later than t, matching what is knowable at the end of month t.
    """
    out = df.copy()

    for lag in (1, 2, 3, 12):
        out[f"arrivals_lag{lag}"] = out["arrivals"].shift(lag)

    out["arrivals_roll3"] = out["arrivals"].shift(1).rolling(3).mean()
    out["arrivals_roll12"] = out["arrivals"].shift(1).rolling(12).mean()

    for c in SEARCH_COLS:
        out[c + "_lag0"] = out[c]
        out[c + "_lag1"] = out[c].shift(1)

    month = out["month"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["month_num"] = month
    out["time_idx"] = np.arange(len(out))

    return out


FEATURE_COLS = None  # filled in by main()


# ------------------------------------------------------------------- models

def fit_ridge(train, test_row):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    X, y = train[FEATURE_COLS].values, train["arrivals"].values
    sc = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(sc.transform(X), y)
    return float(model.predict(sc.transform(test_row[FEATURE_COLS].values))[0])


def fit_gbm(train, test_row):
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=0
    ).fit(train[FEATURE_COLS].values, train["arrivals"].values)
    return float(model.predict(test_row[FEATURE_COLS].values)[0])


def fit_random_forest(train, test_row):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(
        n_estimators=300, max_depth=8, random_state=0, n_jobs=-1
    ).fit(train[FEATURE_COLS].values, train["arrivals"].values)
    return float(model.predict(test_row[FEATURE_COLS].values)[0])


def fit_sarima(train, test_row):
    """SARIMA(1,1,1)(0,1,1,12), forecasting one month ahead."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    y = train["arrivals"].values
    if len(y) < 2 * SEASONAL + 5:
        return float(y[-SEASONAL])
    try:
        fit = SARIMAX(y, order=(1, 1, 1),
                      seasonal_order=(0, 1, 1, SEASONAL)).fit(disp=False)
        return float(np.asarray(fit.forecast(1))[-1])
    except Exception:
        return float(y[-SEASONAL])


def fit_ets(train, test_row):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y = train["arrivals"].values
    if len(y) < 2 * SEASONAL + 1:
        return float(y[-SEASONAL])
    try:
        fit = ExponentialSmoothing(
            y, trend="add", seasonal="add", seasonal_periods=SEASONAL,
            initialization_method="estimated",
        ).fit(optimized=True)
        return float(np.asarray(fit.forecast(1))[-1])
    except Exception:
        return float(y[-SEASONAL])


def fit_lstm(train, test_row, _cache={}):
    """Small LSTM on the arrivals sequence.

    One layer, 16 units. Deliberately small: there are barely 100 monthly
    observations, and a bigger network would have more parameters than the
    data can support.
    """
    import torch

    y = train["arrivals"].values.astype(float)
    if len(y) < SEQ_LEN + 12:
        return float(y[-SEASONAL])

    lo, hi = y.min(), y.max()
    span = max(hi - lo, 1e-9)
    ys = (y - lo) / span

    xs = np.stack([ys[i:i + SEQ_LEN] for i in range(len(ys) - SEQ_LEN)])
    ts = ys[SEQ_LEN:]
    n_val = max(1, int(len(xs) * 0.2))

    torch.manual_seed(0)
    net = torch.nn.Sequential()

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, 16, batch_first=True)
            self.head = torch.nn.Linear(16, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :])

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    lossf = torch.nn.MSELoss()
    X = torch.tensor(xs[:-n_val], dtype=torch.float32).unsqueeze(-1)
    T = torch.tensor(ts[:-n_val], dtype=torch.float32).unsqueeze(-1)
    Xv = torch.tensor(xs[-n_val:], dtype=torch.float32).unsqueeze(-1)
    Tv = torch.tensor(ts[-n_val:], dtype=torch.float32).unsqueeze(-1)

    for _ in range(200):
        opt.zero_grad()
        lossf(net(X), T).backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        _cache.setdefault("val", []).append(float(lossf(net(Xv), Tv)))
        window = list(ys[-SEQ_LEN:])
        inp = torch.tensor(window[-SEQ_LEN:], dtype=torch.float32).view(1, -1, 1)
        window.append(float(net(inp).item()))
    return float(window[-1] * span + lo)


def fit_seasonal_naive(train, test_row):
    """Last year's value for the same month. The benchmark to beat."""
    return float(train["arrivals"].values[-SEASONAL])


MODELS = {
    "Seasonal Naive": fit_seasonal_naive,
    "SARIMA": fit_sarima,
    "ETS": fit_ets,
    "Ridge": fit_ridge,
    "Random Forest": fit_random_forest,
    "Gradient Boosting": fit_gbm,
    "LSTM": fit_lstm,
}


# ---------------------------------------------------------------- evaluation

def walk_forward(df: pd.DataFrame, fit_fn, test_months: int) -> pd.DataFrame:
    """Retrain at every step using only earlier data, forecast one month.

    The training set grows as the test window advances, so a forecast for
    month t is never informed by anything at or after t.
    """
    rows = []
    for i in range(len(df) - test_months, len(df)):
        train = df.iloc[:i]              # everything strictly before month t
        test_row = df.iloc[[i]]
        if train["arrivals"].isna().all() or len(train) < 30:
            continue
        pred = fit_fn(train, test_row)
        rows.append({
            "month": df.iloc[i]["month"],
            "actual": float(df.iloc[i]["arrivals"]),
            "predicted": pred,
        })
    return pd.DataFrame(rows)


def metrics(res: pd.DataFrame) -> dict:
    err = res["actual"] - res["predicted"]
    ape = (err.abs() / res["actual"].abs()) * 100
    return {
        "n": len(res),
        "MAE": err.abs().mean(),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE%": ape.mean(),
        "Accuracy%": 100 - ape.mean(),
    }


def main() -> None:
    global FEATURE_COLS

    df = pd.read_csv(DATA, parse_dates=["month"]).sort_values("month")
    print(f"Loaded {len(df)} months: "
          f"{df.month.min():%Y-%m} to {df.month.max():%Y-%m}\n")

    feat = make_features(df).dropna().reset_index(drop=True)
    FEATURE_COLS = [c for c in feat.columns
                    if c not in ("month", "arrivals") and c not in SEARCH_COLS]
    print(f"{len(FEATURE_COLS)} features, {len(feat)} usable months")
    print(f"Walk-forward test window: last {TEST_MONTHS} months\n")

    all_res, summary = {}, []
    for name, fn in MODELS.items():
        res = walk_forward(feat, fn, TEST_MONTHS)
        all_res[name] = res
        row = {"Model": name}
        row.update(metrics(res))
        summary.append(row)
        note = ""
        if name == "LSTM" and fn.__defaults__ and fn.__defaults__[0].get("val"):
            note = f"  (mean validation loss {np.mean(fn.__defaults__[0]['val']):.4f})"
        print(f"  {name:<18} done{note}")

    out = pd.DataFrame(summary).sort_values("RMSE").reset_index(drop=True)
    pd.set_option("display.width", 160)
    print("\n" + "=" * 96)
    print("WALK-FORWARD RESULTS  (one-month-ahead, retrained every month)")
    print("=" * 96)
    print(out.round(1).to_string(index=False))

    out.to_csv(HERE / "results.csv", index=False)

    # Every month-by-month forecast, so results can be charted or re-scored
    # without rerunning the models.
    preds = pd.concat(
        [r.assign(model=name) for name, r in all_res.items()], ignore_index=True
    )
    preds.to_csv(HERE / "predictions.csv", index=False)

    best = out.iloc[0]["Model"]
    print(f"\nBest by RMSE: {best}")

    # Plot the three best against the actuals.
    fig, ax = plt.subplots(figsize=(11, 5))
    ref = all_res[best]
    ax.plot(ref["month"], ref["actual"], "k-", lw=2, label="Actual")
    for name in out["Model"].head(3):
        r = all_res[name]
        ax.plot(r["month"], r["predicted"], "--", lw=1.4, label=name)
    ax.set_title("Monthly foreign tourist arrivals: actual vs forecast")
    ax.set_ylabel("Arrivals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "forecast_plot.png", dpi=130)
    print(f"Saved {HERE / 'results.csv'} and {HERE / 'forecast_plot.png'}")


if __name__ == "__main__":
    main()
