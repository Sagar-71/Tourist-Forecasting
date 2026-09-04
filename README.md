# Forecasting Monthly Foreign Tourist Arrivals in India

Predicting how many foreign tourists arrive in India each month, using past
arrivals and Google search interest in Indian travel.

## The problem

India's official tourist arrival figures are published about three weeks
after the month ends. Hotels, airlines and tourism boards would like the
number sooner. This project forecasts the current month's arrivals from what
is already known: past arrival figures, and real-time Google search interest
in terms like "flights to india" and "goa".

## Data

| | |
| --- | --- |
| Target | Monthly Foreign Tourist Arrivals, all-India, Jan 2016 – Dec 2024 (108 months) |
| Source | Ministry of Tourism, via the data.gov.in open-data API |
| Predictors | Google Trends weekly search interest, 7 travel terms, averaged to monthly |
| Combined | `data/tourism_monthly.csv` — 108 rows, 8 columns, no missing values |

The seven search terms were chosen for face validity before any modelling —
`flights to india`, `india visa`, `hotels in india`, `taj mahal`, `goa`,
`kerala tourism`, `india tourism` — and never changed afterwards.

## Features

25 features, all computable at the end of the month being forecast:

- **Arrival lags** — 1, 2, 3 and 12 months back
- **Rolling means** — 3-month and 12-month, shifted so they never include the current month
- **Search interest** — current month and previous month, for each of the 7 terms
- **Seasonality** — `sin`/`cos` of the month number, plus month index and a linear time trend

## Models

| Model | Type |
| --- | --- |
| Seasonal Naive | benchmark — repeat the same month last year |
| SARIMA(1,1,1)(0,1,1,12) | classical time series |
| ETS (Holt-Winters) | exponential smoothing, additive trend and seasonality |
| Ridge | linear, L2 regularised |
| Random Forest | tree ensemble |
| Gradient Boosting | tree ensemble |
| LSTM | 1 layer, 16 hidden units |

## Validation

**Walk-forward validation over the last 36 months (2022–2024).** The model is
retrained from scratch at every step and forecasts one month ahead, so a
forecast for a given month never sees that month or anything after it. This
is the right approach for time series — a random train/test split would let
the model learn from the future.

## Results

One month ahead, 36 test months:

| Model | MAE | RMSE | MAPE | Accuracy |
| --- | --- | --- | --- | --- |
| **SARIMA** | 63,794 | **82,879** | 12.2% | 87.8% |
| **ETS** | **55,570** | 84,326 | **11.0%** | **89.0%** |
| Gradient Boosting | 77,603 | 95,338 | 11.8% | 88.2% |
| Ridge | 78,794 | 97,388 | 12.7% | 87.3% |
| Random Forest | 73,926 | 99,952 | 11.9% | 88.1% |
| LSTM | 72,778 | 101,570 | 12.2% | 87.8% |
| Seasonal Naive | 237,196 | 317,800 | 38.4% | 61.6% |

LSTM mean validation loss: **0.0059** (MSE on min-max scaled data).

**Every model beats the seasonal naive benchmark comfortably** — 87–89%
accuracy against 61.6%. SARIMA has the lowest RMSE; ETS has the lowest MAE
and the best average accuracy.

**The LSTM does not win.** That is expected and worth stating: with about
100 monthly observations, a neural network has more parameters than the data
can support, and classical seasonal models do better. Running it makes that
a measurement rather than an assumption.

## Running it

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib torch
python forecast.py
```

Outputs `results.csv` and `forecast_plot.png`.

## Notes and limitations

- **The COVID period is extreme.** Arrivals fell from about 1.1 million in
  February 2020 to 2,820 in April 2020. The test window (2022–2024) sits
  after the collapse; percentage errors measured across it would be
  meaningless, since any error on a near-zero denominator is enormous.
- **108 months is a short series** for monthly seasonal modelling — nine
  seasonal cycles, two of them disrupted.
- **Search interest adds little here.** Because arrivals are published only
  about three weeks late, last month's actual figure is usually available
  when the forecast is made, and it carries most of the signal that search
  interest would otherwise supply.
