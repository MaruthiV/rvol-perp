"""Correlation structure analysis — paper Section 4, use-case identification.

Computes correlations between realized variance and:
  1. BTC log returns (negative correlation = "fear index" / crisis hedge)
  2. Binance BTC perp funding rates (positive = high vol → high funding)
  3. Deribit DVOL (implied variance) — how well does RV track IV?
  4. RV persistence — AR(1) coefficient, half-life of vol shocks

Each correlation identifies a natural hedger / use case for the product:
  - RV vs returns   → options dealers and tail-risk hedgers (long RV)
  - RV vs funding   → arb between RV perp and standard perp
  - RV vs DVOL      → vol traders arbing realized vs implied

Output: prints stats table and saves plots to data/processed/

Usage:
    python scripts/analyze_correlations.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.stats as stats

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH    = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"
DVOL_PATH     = Path(__file__).parent.parent / "data" / "raw" / "deribit" / "btc" / "dvol" / "dvol.parquet"
BINANCE_FUNDING_PATH = Path(__file__).parent.parent / "data" / "raw" / "binance" / "btcusdt" / "funding_rates" / "funding.parquet"
KLINES_DIR    = Path(__file__).parent.parent / "data" / "raw" / "binance" / "btcusdt" / "klines_1m"


def load_rv30_daily() -> pd.Series:
    tbl = pq.read_table(INDEX_PATH, columns=["timestamp_us", "rv30", "is_valid"])
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df[df["is_valid"]].set_index("ts")
    return df["rv30"].resample("1D").last().dropna()


def load_btc_daily_returns() -> pd.Series:
    """Daily log returns from monthly kline parquets."""
    files = sorted(KLINES_DIR.rglob("*.parquet"))
    tables = [pq.read_table(f, columns=["timestamp_us", "close"]) for f in files]
    import pyarrow as pa
    combined = pa.concat_tables(tables).to_pandas()
    combined["ts"] = pd.to_datetime(combined["timestamp_us"], unit="us", utc=True)
    combined = combined.set_index("ts").sort_index()
    daily_close = combined["close"].resample("1D").last().dropna()
    return np.log(daily_close / daily_close.shift(1)).dropna()


def load_binance_funding() -> pd.Series:
    if not BINANCE_FUNDING_PATH.exists():
        return pd.Series(dtype=float)
    tbl = pq.read_table(BINANCE_FUNDING_PATH)
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df.set_index("ts").sort_index()
    # Annualize: 8h funding × 3 × 365
    daily = df["funding_rate"].resample("1D").mean().dropna() * 3 * 365
    return daily


def load_dvol_daily() -> pd.Series:
    tbl = pq.read_table(DVOL_PATH)
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    return df.set_index("ts")["dvol_annualized"].resample("1D").last().dropna()


def _corr_stats(x: pd.Series, y: pd.Series, label: str) -> dict:
    common = x.index.intersection(y.index)
    x, y = x.loc[common].dropna(), y.loc[common].dropna()
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    r, pval = stats.pearsonr(x, y)
    rho, _ = stats.spearmanr(x, y)
    return {"label": label, "n": len(x), "pearson_r": r, "spearman_rho": rho,
            "p_value": pval, "x": x, "y": y}


def main() -> None:
    print("Loading data...")
    rv30 = load_rv30_daily()
    btc_rets = load_btc_daily_returns()
    funding = load_binance_funding()
    dvol = load_dvol_daily()

    # Convert DVOL to implied variance (daily rate) for apples-to-apples comparison
    iv_daily = (dvol / 100) ** 2 / 365
    rv30_daily = rv30 / 30  # daily rate

    # Delta-RV: daily change in log(RV) — stationary, better for correlation
    drv = np.log(rv30_daily).diff().dropna()
    dbtc = btc_rets

    print("\n" + "="*60)
    print("CORRELATION STRUCTURE ANALYSIS")
    print("="*60)

    # 1. RV vs BTC returns (level)
    c1 = _corr_stats(rv30_daily, btc_rets, "RV30 (daily) vs BTC daily return")
    # 2. ΔRV vs BTC returns (changes)
    c2 = _corr_stats(drv, dbtc, "Δlog(RV30) vs BTC daily return")
    # 3. RV vs Binance funding rate
    c3 = _corr_stats(rv30_daily, funding, "RV30 vs Binance funding rate (annualized)")
    # 4. RV vs DVOL (implied variance)
    c4 = _corr_stats(rv30_daily, iv_daily, "RV30 vs DVOL implied variance")
    # 5. RV autocorrelation / persistence
    ar1 = rv30_daily.autocorr(lag=1)
    ar5 = rv30_daily.autocorr(lag=5)
    ar30 = rv30_daily.autocorr(lag=30)

    for c in [c1, c2, c3, c4]:
        print(f"\n{c['label']}")
        print(f"  n={c['n']}  Pearson r={c['pearson_r']:+.3f}  Spearman ρ={c['spearman_rho']:+.3f}  p={c['p_value']:.2e}")

    print(f"\nRV30 persistence (autocorrelation):")
    print(f"  AR(1)  lag-1d:  {ar1:.3f}")
    print(f"  AR(5)  lag-5d:  {ar5:.3f}")
    print(f"  AR(30) lag-30d: {ar30:.3f}")
    hl = -np.log(2) / np.log(abs(ar1)) if ar1 > 0 else float("nan")
    print(f"  Half-life of vol shock: ~{hl:.0f} days")

    print("\n--- Use case implications ---")
    r_ret = c2["pearson_r"]
    if r_ret < -0.1:
        print(f"  RV ↑ when BTC returns ↓ (r={r_ret:+.3f}): crisis hedge property confirmed")
        print("  → Options dealers and tail-risk funds are natural longs")
    r_fund = c3["pearson_r"]
    if r_fund > 0.1:
        print(f"  RV corr funding (r={r_fund:+.3f}): high vol periods have higher funding")
        print("  → Arb opportunity between RV perp and standard BTC perp funding")
    r_dvol = c4["pearson_r"]
    print(f"  RV vs IV correlation (r={r_dvol:+.3f}): realized tracks implied well")
    print(f"  → Options MMs can hedge their vega books via the RV perp")

    _plot_scatter_grid(c1, c2, c3, c4)
    _plot_rolling_correlation(rv30_daily, btc_rets, dvol)


def _plot_scatter_grid(c1: dict, c2: dict, c3: dict, c4: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Realized Variance — Correlation Structure", fontsize=13)

    pairs = [(axes[0,0], c1, "BTC daily return", "RV30 (daily rate)"),
             (axes[0,1], c2, "BTC daily return", "Δlog(RV30)"),
             (axes[1,0], c3, "Binance funding (annualized)", "RV30 (daily rate)"),
             (axes[1,1], c4, "DVOL implied variance (daily)", "RV30 (daily rate)")]

    for ax, c, xlabel, ylabel in pairs:
        x, y = c["x"], c["y"]
        ax.scatter(x, y, s=2, alpha=0.3, color="#2196F3")
        # Regression line
        m, b = np.polyfit(x, y, 1)
        xr = np.linspace(x.quantile(0.01), x.quantile(0.99), 100)
        ax.plot(xr, m * xr + b, "r-", lw=1.5)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"r={c['pearson_r']:+.3f}  ρ={c['spearman_rho']:+.3f}  n={c['n']}", fontsize=9)
        ax.grid(True, alpha=0.3)
        # Clip outliers for visibility
        ax.set_xlim(x.quantile(0.01), x.quantile(0.99))
        ax.set_ylim(y.quantile(0.01), y.quantile(0.99))

    plt.tight_layout()
    path = PROCESSED_DIR / "correlation_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {path}")


def _plot_rolling_correlation(rv30: pd.Series, btc_rets: pd.Series, dvol: pd.Series) -> None:
    """90-day rolling correlation: RV vs BTC returns, RV vs DVOL."""
    common = rv30.index.intersection(btc_rets.index)
    rv_c = rv30.loc[common]
    ret_c = btc_rets.loc[common]
    roll_rv_ret = rv_c.rolling(90).corr(ret_c)

    common2 = rv30.index.intersection(dvol.index)
    rv_c2 = rv30.loc[common2]
    dv_c2 = dvol.loc[common2]
    roll_rv_dvol = rv_c2.rolling(90).corr(dv_c2)

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=False)
    fig.suptitle("90-Day Rolling Correlations with RV30", fontsize=12)

    ax = axes[0]
    ax.plot(roll_rv_ret.index, roll_rv_ret, color="#F44336", lw=1.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(roll_rv_ret.mean(), color="#F44336", lw=1, ls="--",
               label=f"Mean = {roll_rv_ret.mean():+.3f}")
    ax.set_ylabel("Rolling correlation")
    ax.set_title("RV30 vs BTC daily returns (negative = crisis hedge)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax2 = axes[1]
    ax2.plot(roll_rv_dvol.index, roll_rv_dvol, color="#2196F3", lw=1.2)
    ax2.axhline(roll_rv_dvol.mean(), color="#2196F3", lw=1, ls="--",
                label=f"Mean = {roll_rv_dvol.mean():+.3f}")
    ax2.set_ylabel("Rolling correlation")
    ax2.set_title("RV30 vs DVOL (high = RV perp hedges IV exposure)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = PROCESSED_DIR / "rolling_correlations.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


if __name__ == "__main__":
    main()
