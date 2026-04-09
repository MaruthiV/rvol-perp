"""Data quality and index properties analysis — paper figures for Section 4.

Produces:
  1. Return distribution: histogram + QQ-plot vs Normal (heavy tails → motivates cap)
  2. Autocorrelation of r² (volatility clustering → motivates rolling window)
  3. Data completeness heatmap (by month)
  4. Index time series: RV7/14/30 over full history

Output PNGs saved to data/processed/

Usage:
    python scripts/analyze_data_quality.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RETURNS_PATH = PROCESSED_DIR / "returns" / "btcusdt_1m_returns.parquet"
INDEX_PATH = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"


def load_returns() -> pd.DataFrame:
    tbl = pq.read_table(RETURNS_PATH)
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df.set_index("ts")


def load_index() -> pd.DataFrame:
    tbl = pq.read_table(INDEX_PATH)
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df.set_index("ts")


def main() -> None:
    print("Loading data...")
    ret_df = load_returns()
    idx_df = load_index()

    rets = ret_df["log_return"].dropna()
    print(f"  Returns: {len(rets):,} observations ({rets.index[0].date()} → {rets.index[-1].date()})")
    print(f"  Capped:  {ret_df['is_capped'].sum():,} ({ret_df['is_capped'].mean()*100:.3f}%)")
    print(f"  Gaps:    {ret_df['is_gap'].sum():,} ({ret_df['is_gap'].mean()*100:.3f}%)")

    _plot_return_distribution(rets)
    _plot_autocorrelation(rets)
    _plot_completeness(ret_df)
    _plot_index_timeseries(idx_df)
    _print_summary_stats(rets, idx_df)


def _plot_return_distribution(rets: pd.Series) -> None:
    """Histogram + QQ-plot showing heavy tails vs Normal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("BTC 1-Minute Log Return Distribution", fontsize=12)

    # Histogram with normal overlay
    ax = axes[0]
    sigma = rets.std()
    x = np.linspace(rets.quantile(0.001), rets.quantile(0.999), 500)
    ax.hist(rets, bins=300, density=True, color="#2196F3", alpha=0.7,
            label=f"Empirical (σ={sigma*100:.3f}%)")
    ax.plot(x, stats.norm.pdf(x, 0, sigma), "r-", lw=2, label="Normal fit")
    ax.set_xlim(rets.quantile(0.001), rets.quantile(0.999))
    ax.set_xlabel("1-minute log return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Excess kurtosis annotation
    kurt = rets.kurtosis()
    ax.text(0.97, 0.95, f"Excess kurtosis: {kurt:.1f}\n(Normal = 0)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # QQ-plot
    ax2 = axes[1]
    (osm, osr), (slope, intercept, r) = stats.probplot(rets, dist="norm")
    ax2.scatter(osm, osr, s=0.3, alpha=0.3, color="#2196F3")
    ax2.plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5, label="Normal reference")
    ax2.set_xlabel("Theoretical quantiles (Normal)")
    ax2.set_ylabel("Sample quantiles")
    ax2.set_title("QQ-Plot vs Normal")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    path = PROCESSED_DIR / "return_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_autocorrelation(rets: pd.Series, max_lags: int = 100) -> None:
    """Autocorrelation of r² — shows volatility clustering."""
    r2 = rets ** 2
    acf_vals = [r2.autocorr(lag=k) for k in range(1, max_lags + 1)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Volatility Clustering: Autocorrelation of Squared Returns", fontsize=12)

    # ACF of r²
    ax = axes[0]
    lags = np.arange(1, max_lags + 1)
    bars = ax.bar(lags, acf_vals, color=["#2196F3" if v > 0 else "#F44336" for v in acf_vals],
                  alpha=0.7, width=0.8)
    conf = 1.96 / np.sqrt(len(rets))
    ax.axhline(conf, color="red", ls="--", lw=1, label=f"95% CI (±{conf:.4f})")
    ax.axhline(-conf, color="red", ls="--", lw=1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("Lag (minutes)")
    ax.set_ylabel("Autocorrelation of r²")
    ax.set_title("ACF of r² (up to 100 lags)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ACF of |r| (alternative view, longer lags)
    ax2 = axes[1]
    abs_r = rets.abs()
    acf_abs = [abs_r.autocorr(lag=k) for k in range(1, 1441, 14)]  # up to 1 day, every 14 min
    lags2 = np.arange(1, 1441, 14)
    ax2.plot(lags2, acf_abs, color="#FF5722", lw=1.5)
    ax2.axhline(conf, color="red", ls="--", lw=1, label="95% CI")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Lag (minutes)")
    ax2.set_ylabel("Autocorrelation of |r|")
    ax2.set_title("ACF of |r| (up to 1 day)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PROCESSED_DIR / "autocorrelation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_completeness(ret_df: pd.DataFrame) -> None:
    """Heatmap of data completeness by month (% of expected 1-min observations)."""
    ret_df2 = ret_df.copy()
    ret_df2["year"] = ret_df2.index.year
    ret_df2["month"] = ret_df2.index.month

    # Count actual non-gap observations per month
    actual = ret_df2[~ret_df2["is_gap"]].groupby(["year", "month"]).size()
    # Expected: days_in_month × 1440
    expected = ret_df2.groupby(["year", "month"]).size()
    completeness = (actual / expected * 100).clip(upper=100)

    # Pivot for heatmap
    pivot = completeness.unstack(level=1)
    pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=90, vmax=100)
    ax.set_xticks(range(12))
    ax.set_xticklabels(pivot.columns, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title("Data Completeness (% of expected 1-minute observations)", fontsize=11)
    plt.colorbar(im, ax=ax, label="%")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=7, color="black" if val > 95 else "white")

    plt.tight_layout()
    path = PROCESSED_DIR / "data_completeness.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_index_timeseries(idx_df: pd.DataFrame) -> None:
    """RV7/14/30 annualized vol equivalent over full history."""
    valid = idx_df[idx_df["is_valid"]].copy()
    # Daily resample
    daily = valid[["rv7", "rv14", "rv30"]].resample("1D").last().dropna(how="all")

    # Convert to annualized vol %
    ann7  = np.sqrt(daily["rv7"]  * 365 / 7)  * 100
    ann14 = np.sqrt(daily["rv14"] * 365 / 14) * 100
    ann30 = np.sqrt(daily["rv30"] * 365 / 30) * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ann30.index, ann30, label="RV30 (30-day)", color="#2196F3", lw=1.5)
    ax.plot(ann14.index, ann14, label="RV14 (14-day)", color="#FF9800", lw=1, alpha=0.8)
    ax.plot(ann7.index,  ann7,  label="RV7 (7-day)",   color="#F44336", lw=0.8, alpha=0.7)

    # Annotate key events
    events = {
        "COVID\nMar 2020": "2020-03-12",
        "FTX\nNov 2022":   "2022-11-09",
        "ETF\nJan 2024":   "2024-01-11",
    }
    for label, date in events.items():
        try:
            y = ann30.loc[date]
            ax.annotate(label, xy=(pd.Timestamp(date, tz="UTC"), y),
                        xytext=(0, 15), textcoords="offset points",
                        ha="center", fontsize=7.5,
                        arrowprops=dict(arrowstyle="->", lw=0.8))
        except KeyError:
            pass

    ax.set_ylabel("Annualized Volatility (%)")
    ax.set_title("BTC Realized Variance Index — All Windows", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout()
    path = PROCESSED_DIR / "rv_index_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _print_summary_stats(rets: pd.Series, idx_df: pd.DataFrame) -> None:
    print("\n--- Return distribution stats ---")
    print(f"  Mean:             {rets.mean()*100:.5f}%")
    print(f"  Std (1-min):      {rets.std()*100:.4f}%  (~{rets.std()*np.sqrt(365*1440)*100:.0f}% annualized)")
    print(f"  Skewness:         {rets.skew():.3f}  (Normal=0)")
    print(f"  Excess kurtosis:  {rets.kurtosis():.1f}  (Normal=0, heavy tails > 0)")
    print(f"  Max 1-min return: {rets.max()*100:+.3f}%")
    print(f"  Min 1-min return: {rets.min()*100:+.3f}%")

    valid = idx_df[idx_df["is_valid"]]
    rv30_daily = valid["rv30"].resample("1D").last().dropna()
    print("\n--- RV30 index stats ---")
    print(f"  Median ann vol:   {np.sqrt(rv30_daily.median() * 365/30)*100:.1f}%")
    print(f"  Daily change std: {rv30_daily.pct_change().std()*100:.1f}%  (index moves ~this % per day)")
    print(f"  Max 1-day change: {rv30_daily.pct_change().max()*100:+.1f}%")
    print(f"  Min 1-day change: {rv30_daily.pct_change().min()*100:+.1f}%")


if __name__ == "__main__":
    main()
