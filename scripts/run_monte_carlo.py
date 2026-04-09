"""Run the Monte Carlo lifecycle simulator — Phase 5.

Generates SV price paths, computes RV30 index, simulates funding + mark,
runs leveraged long/short positions, and reports PnL distributions and
liquidation rates.

Usage:
    python scripts/run_monte_carlo.py
    python scripts/run_monte_carlo.py --n-paths 2048 --horizon-days 120 --leverage 5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simulations.monte_carlo.lifecycle import LifecycleParams, run_lifecycle
from simulations.monte_carlo.paths import SVParams, simulate_paths

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-paths", type=int, default=512)
    p.add_argument("--horizon-days", type=int, default=90)
    p.add_argument("--leverage", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    n_steps = args.horizon_days * 1440

    print(f"Simulating {args.n_paths} paths × {args.horizon_days} days "
          f"({n_steps:,} minutes each)...")

    sv = SVParams()
    t0 = time.time()
    prices_j, log_var_j = simulate_paths(sv, args.n_paths, n_steps, seed=args.seed)
    prices = np.asarray(prices_j)
    log_var = np.asarray(log_var_j)
    t_sim = time.time() - t0
    print(f"  SV paths generated in {t_sim:.1f}s")

    # Sanity: realized 1-month vol from simulated paths
    returns = np.diff(np.log(prices), axis=1)
    # Annualized realized vol over the last 30 days of each path
    last_30d = returns[:, -30 * 1440 :]
    realized_var_annual = (last_30d ** 2).sum(axis=1) * (365.0 / 30.0)
    realized_vol_pct = np.sqrt(realized_var_annual) * 100
    target_vol = np.sqrt(np.exp(sv.theta) * 365) * 100
    print(f"  Realized vol (last 30d) — median: {np.median(realized_vol_pct):.1f}% "
          f"mean: {realized_vol_pct.mean():.1f}%  (target: {target_vol:.1f}%)")

    lc = LifecycleParams(leverage=args.leverage)
    t0 = time.time()
    result = run_lifecycle(prices, lc, seed=args.seed)
    t_lc = time.time() - t0
    print(f"  Lifecycle (index + funding + margin) in {t_lc:.1f}s")

    # --- Summary stats ---
    long_pnl = result.long_final_pnl
    short_pnl = result.short_final_pnl
    entry_hour = lc.entry_minute // lc.funding_interval_minutes
    # Convert PnL (in variance units) to % of notional — notional = idx_entry
    idx_entry = result.rv30_index_hourly[:, entry_hour]
    long_pnl_pct = long_pnl / idx_entry * 100
    short_pnl_pct = short_pnl / idx_entry * 100

    print("\n" + "=" * 60)
    print("LIFECYCLE RESULTS")
    print("=" * 60)
    print(f"  Paths:             {args.n_paths}")
    print(f"  Horizon:           {args.horizon_days} days")
    print(f"  Entry:             day {lc.entry_minute // 1440} (after index warmup)")
    print(f"  Leverage:          {lc.leverage}×")
    print(f"  Maintenance margin {lc.maintenance_margin_frac * 100:.1f}%")

    print(f"\n  Long PnL  (% of notional):")
    print(f"    median       {np.median(long_pnl_pct):+7.2f}%")
    print(f"    mean         {long_pnl_pct.mean():+7.2f}%")
    print(f"    std          {long_pnl_pct.std():7.2f}%")
    print(f"    5th  %ile    {np.percentile(long_pnl_pct, 5):+7.2f}%")
    print(f"    95th %ile    {np.percentile(long_pnl_pct, 95):+7.2f}%")

    print(f"\n  Short PnL (% of notional):")
    print(f"    median       {np.median(short_pnl_pct):+7.2f}%")
    print(f"    mean         {short_pnl_pct.mean():+7.2f}%")
    print(f"    std          {short_pnl_pct.std():7.2f}%")
    print(f"    5th  %ile    {np.percentile(short_pnl_pct, 5):+7.2f}%")
    print(f"    95th %ile    {np.percentile(short_pnl_pct, 95):+7.2f}%")

    print(f"\n  Liquidations at {lc.leverage}× leverage:")
    print(f"    Long  liquidated: {result.long_liquidated.mean() * 100:.2f}%")
    print(f"    Short liquidated: {result.short_liquidated.mean() * 100:.2f}%")

    # --- Gate checks ---
    print("\n" + "=" * 60)
    print("GATE CHECKS (from design doc)")
    print("=" * 60)

    # Gate 1: Median short-side PnL > 0 (VRP carry)
    if np.median(short_pnl_pct) > 0:
        print(f"  ✓ Median short PnL > 0  ({np.median(short_pnl_pct):+.2f}%)")
    else:
        print(f"  ✗ Median short PnL ≤ 0  ({np.median(short_pnl_pct):+.2f}%) — "
              "VRP carry not present in SV model")

    # Gate 2: < 1% liquidation rate at 3× for both sides (if leverage == 3)
    if args.leverage == 3.0:
        liq_long = result.long_liquidated.mean()
        liq_short = result.short_liquidated.mean()
        ok = liq_long < 0.01 and liq_short < 0.01
        mark = "✓" if ok else "✗"
        print(f"  {mark} Liquidation rate < 1% at 3×  "
              f"(long {liq_long*100:.2f}%, short {liq_short*100:.2f}%)")

    # Gate 3: realized vol matches target within 5%
    err = abs(np.median(realized_vol_pct) - target_vol) / target_vol
    mark = "✓" if err < 0.05 else "✗"
    print(f"  {mark} Realized vol matches target within 5%  ({err*100:.1f}% error)")

    # Gate 4: runtime budget
    mark = "✓" if (t_sim + t_lc) < 60 else "✗"
    print(f"  {mark} Total runtime < 60s  ({t_sim + t_lc:.1f}s)")

    # --- Save summary ---
    summary = pd.DataFrame({
        "long_pnl_pct": long_pnl_pct,
        "short_pnl_pct": short_pnl_pct,
        "long_liquidated": result.long_liquidated,
        "short_liquidated": result.short_liquidated,
        "realized_vol_last30d_pct": realized_vol_pct,
    })
    out_csv = PROCESSED_DIR / "mc_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nSummary → {out_csv}")

    _plot_pnl_distribution(long_pnl_pct, short_pnl_pct)
    _plot_sample_paths(prices, log_var, result)
    _plot_margin_usage(result, lc)


def _plot_pnl_distribution(long_pct: np.ndarray, short_pct: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(min(long_pct.min(), short_pct.min()),
                       max(long_pct.max(), short_pct.max()), 80)
    ax.hist(long_pct, bins=bins, color="#2196F3", alpha=0.6, label=f"Long (median {np.median(long_pct):+.1f}%)")
    ax.hist(short_pct, bins=bins, color="#F44336", alpha=0.6, label=f"Short (median {np.median(short_pct):+.1f}%)")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Final PnL (% of notional)")
    ax.set_ylabel("Paths")
    ax.set_title("Monte Carlo — Variance Perp PnL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PROCESSED_DIR / "mc_pnl_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_sample_paths(prices: np.ndarray, log_var: np.ndarray, result) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle("Sample Monte Carlo Paths", fontsize=12)

    n_show = 10
    days = np.arange(prices.shape[1]) / 1440.0

    ax = axes[0]
    for i in range(n_show):
        ax.plot(days, prices[i], lw=0.6, alpha=0.7)
    ax.set_ylabel("Price")
    ax.set_title("SV price paths")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    for i in range(n_show):
        ann_vol = np.sqrt(np.exp(log_var[i]) * 365) * 100
        ax2.plot(days, ann_vol, lw=0.6, alpha=0.7)
    ax2.set_ylabel("Instantaneous vol (% ann)")
    ax2.set_title("Latent vol process")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    n_hours = result.rv30_index_hourly.shape[1]
    hours = np.arange(n_hours) / 24.0
    for i in range(n_show):
        rv_ann = np.sqrt(result.rv30_index_hourly[i] * 365) * 100
        ax3.plot(hours, rv_ann, lw=0.6, alpha=0.7)
    ax3.set_xlabel("Days")
    ax3.set_ylabel("RV30 (% ann vol)")
    ax3.set_title("Realized variance index")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PROCESSED_DIR / "mc_sample_paths.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


def _plot_margin_usage(result, lc: LifecycleParams) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle(f"Margin Equity Over Time ({lc.leverage}× leverage)", fontsize=12)

    entry_hour = lc.entry_minute // lc.funding_interval_minutes
    n_hours = result.long_equity.shape[1]
    hours = (np.arange(n_hours) - entry_hour) / 24.0
    im = 1.0 / lc.leverage

    n_show = 50
    for i in range(n_show):
        idx_entry = result.rv30_index_hourly[i, entry_hour]
        axes[0].plot(hours, result.long_equity[i] / idx_entry, lw=0.4, alpha=0.5,
                     color="#F44336" if result.long_liquidated[i] else "#2196F3")
        axes[1].plot(hours, result.short_equity[i] / idx_entry, lw=0.4, alpha=0.5,
                     color="#F44336" if result.short_liquidated[i] else "#4CAF50")

    for ax, title in [(axes[0], f"Long (liq rate {result.long_liquidated.mean()*100:.1f}%)"),
                      (axes[1], f"Short (liq rate {result.short_liquidated.mean()*100:.1f}%)")]:
        ax.axhline(im, color="black", lw=0.8, ls="--", label="Initial margin")
        ax.axhline(lc.maintenance_margin_frac, color="red", lw=0.8, ls="--", label="Maint. margin")
        ax.set_xlabel("Days since entry")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Equity / entry notional")

    plt.tight_layout()
    path = PROCESSED_DIR / "mc_margin_usage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


if __name__ == "__main__":
    main()
