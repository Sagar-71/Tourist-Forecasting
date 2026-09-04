"""Generate the plots used in the README.

Run after forecast.py, which writes results.csv and predictions.csv.
Outputs PNGs to figures/ at a width that reads well on GitHub.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)

TEAL, RUST, INDIGO, ROSE = "#0D9488", "#C2410C", "#4338CA", "#BE123C"
INK, MUTED, GRID = "#121A1B", "#657374", "#D9E1DF"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "savefig.dpi": 130,
})

hist = pd.read_csv(HERE / "data" / "tourism_monthly.csv", parse_dates=["month"])
preds = pd.read_csv(HERE / "predictions.csv", parse_dates=["month"])
res = pd.read_csv(HERE / "results.csv")

millions = mtick.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v else "0")


def tidy(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_series():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(hist.month, hist.arrivals, color=TEAL, alpha=0.13, lw=0)
    ax.plot(hist.month, hist.arrivals, color=TEAL, lw=1.9)

    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"),
               color=RUST, alpha=0.08, lw=0)
    ax.text(pd.Timestamp("2020-11-15"), 1.19e6, "COVID-19 collapse",
            ha="center", fontsize=10, color=RUST)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31"),
               color=INK, alpha=0.05, lw=0)
    ax.text(pd.Timestamp("2023-06-15"), 1.19e6, "test window (36 months)",
            ha="center", fontsize=10, color=MUTED)

    trough = hist.loc[hist.arrivals.idxmin()]
    ax.plot([trough.month], [trough.arrivals], "o", ms=6,
            mfc="white", mec=RUST, mew=1.6)
    ax.annotate(f"Apr 2020\n{int(trough.arrivals):,} arrivals",
                xy=(trough.month, trough.arrivals),
                xytext=(24, 40), textcoords="offset points",
                fontsize=9, color=RUST,
                arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8))

    ax.yaxis.set_major_formatter(millions)
    ax.set_ylabel("Foreign tourist arrivals")
    ax.set_ylim(0, 1.35e6)
    ax.set_title("Monthly foreign tourist arrivals in India, 2016-2024",
                 fontsize=12, loc="left", pad=12)
    tidy(ax)
    fig.savefig(FIGS / "01_series.png")
    plt.close(fig)


def plot_walkforward():
    fig, ax = plt.subplots(figsize=(11, 2.8))
    steps = 5
    for i in range(steps):
        y = steps - i
        w = 0.50 + i * 0.075
        ax.barh(y, w, height=0.55, color=TEAL, alpha=0.24, lw=0)
        ax.barh(y, 0.045, left=w + 0.008, height=0.55, color=RUST, lw=0)
        ax.text(-0.012, y, f"step {i+1}", ha="right", va="center",
                fontsize=9.5, color=MUTED)
    ax.text(0.25, steps, "training data (grows by one month each step)",
            ha="center", va="center", fontsize=9.5, color=INK)
    ax.annotate("forecast one month ahead", xy=(0.55, steps),
                xytext=(0.74, steps + 0.62), fontsize=9.5, color=RUST,
                ha="center", arrowprops=dict(arrowstyle="->", color=RUST, lw=0.9))
    ax.set_xlim(-0.13, 1.0)
    ax.set_ylim(0.2, steps + 1.3)
    ax.set_title("Walk-forward validation", fontsize=12, loc="left", pad=10)
    ax.axis("off")
    fig.savefig(FIGS / "02_walkforward.png")
    plt.close(fig)


def plot_comparison():
    d = res.sort_values("RMSE", ascending=True).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    y = np.arange(len(d))
    cols = [RUST if m == "Seasonal Naive" else TEAL for m in d.Model]

    ax.barh(y + 0.20, d.RMSE, height=0.36, color=cols, alpha=0.92, lw=0, label="RMSE")
    ax.barh(y - 0.20, d.MAE, height=0.36, color=cols, alpha=0.45, lw=0, label="MAE")
    for yy, r, m in zip(y, d.RMSE, d.MAE):
        ax.text(r + 5000, yy + 0.20, f"{int(r):,}", va="center", fontsize=8.5, color=MUTED)
        ax.text(m + 5000, yy - 0.20, f"{int(m):,}", va="center", fontsize=8.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(d.Model, fontsize=10, color=INK)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_xlabel("Error, arrivals per month")
    ax.set_xlim(0, 395000)
    ax.set_title("Forecast error by model (lower is better)",
                 fontsize=12, loc="left", pad=12)
    tidy(ax, xgrid=True)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="lower right", ncol=2)
    fig.savefig(FIGS / "03_model_comparison.png")
    plt.close(fig)


def plot_forecasts():
    fig, ax = plt.subplots(figsize=(11, 4.4))
    act = preds[preds.model == "SARIMA"].sort_values("month")
    ax.plot(act.month, act.actual, color=INK, lw=2.4, label="Actual", zorder=5)
    for name, c, ls in [("ETS", TEAL, "--"), ("SARIMA", RUST, "--"),
                        ("Gradient Boosting", INDIGO, ":"), ("LSTM", ROSE, "-.")]:
        g = preds[preds.model == name].sort_values("month")
        ax.plot(g.month, g.predicted, color=c, lw=1.5, ls=ls, label=name)

    ax.yaxis.set_major_formatter(millions)
    ax.set_ylabel("Arrivals")
    ax.set_ylim(0, 1.3e6)
    ax.set_title("Forecast vs actual over the 36-month test window",
                 fontsize=12, loc="left", pad=12)
    tidy(ax)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              columnspacing=1.6, handlelength=2.2)
    fig.savefig(FIGS / "04_forecast_vs_actual.png")
    plt.close(fig)


def plot_errors():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    order = res.sort_values("RMSE").Model.tolist()
    data, cols = [], []
    for m in order:
        g = preds[preds.model == m]
        data.append(((g.actual - g.predicted).abs() / g.actual * 100).values)
        cols.append(RUST if m == "Seasonal Naive" else TEAL)

    bp = ax.boxplot(data, orientation="vertical", widths=0.55, patch_artist=True,
                    medianprops=dict(color=INK, lw=1.4),
                    whiskerprops=dict(color=MUTED, lw=0.9),
                    capprops=dict(color=MUTED, lw=0.9),
                    flierprops=dict(marker="o", ms=3, mfc=MUTED, mec="none", alpha=0.6))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.32)
        patch.set_linewidth(0.9); patch.set_edgecolor(c)

    ax.set_xticklabels(order, rotation=15, ha="right", fontsize=9.5, color=INK)
    ax.set_ylabel("Absolute percentage error (%)")
    ax.set_ylim(0, 85)
    ax.set_title("Spread of monthly error, not just the average",
                 fontsize=12, loc="left", pad=12)
    tidy(ax)
    fig.savefig(FIGS / "05_error_spread.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_series(); plot_walkforward(); plot_comparison()
    plot_forecasts(); plot_errors()
    for f in sorted(FIGS.glob("*.png")):
        print(f"{f.name:28} {f.stat().st_size/1024:6.1f} KB")
