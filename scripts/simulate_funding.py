"""Historical funding-mechanism replay — Phase 4.

Replays the RV30 index as "truth" and simulates a traded mark driven by a
reduced-form AR(1) basis process with funding feedback. Sweeps (dampening, cap)
parameter grid and reports:
  - mean |basis|, max |basis|, cap-hit fraction
  - long/short cumulative PnL (should be zero-sum, long = variance exposure)
  - time-to-converge from a +20% initial basis shock
  - funding vs VRP: is funding noise dominating the variance risk premium?

Outputs:
  - data/processed/funding_sim_summary.csv — grid search results
  - data/processed/funding_sim_timeseries.png — best-parameter replay plot
  - data/processed/funding_sim_convergence.png — shock convergence curves

Usage:
    python scripts/simulate_funding.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rvol.funding import BasisProcess, FundingParams, simulate_mark_convergence

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
INDEX_PATH = PROCESSED_DIR / "index" / "btcusdt_rv_index.parquet"


def load_rv30_hourly() -> pd.Series:
    """Load RV30 resampled to hourly close — the index the perp tracks."""
    tbl = pq.read_table(INDEX_PATH, columns=["timestamp_us", "rv30", "is_valid"])
    df = tbl.to_pandas()
    df["ts"] = pd.to_datetime(df["timestamp_us"], unit="us", utc=True)
    df = df[df["is_valid"]].set_index("ts")
    # RV30 in daily-rate units (already normalized by 30 in analyze_vrp, but
    # build_index leaves it as the 30-day sum). Convert to daily rate.
    rv30_daily_rate = df["rv30"] / 30.0
    return rv30_daily_rate.resample("1h").last().dropna()


def grid_search(
    index: pd.Series,
    dampenings: list[float],
    caps: list[float],
    basis_sigma: float,
    gamma: float,
    initial_basis: float,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    bp = BasisProcess(rho=0.90, sigma=basis_sigma, gamma=gamma, initial_basis=initial_basis)
    for k in dampenings:
        for c in caps:
            fp = FundingParams(dampening=k, cap=c)
            r = simulate_mark_convergence(index, fp, bp, seed=seed)
            rows.append({
                "dampening": k,
                "cap": c,
                "mean_abs_basis": r.mean_abs_basis,
                "max_abs_basis": r.max_abs_basis,
                "cap_hit_frac": r.cap_hit_fraction,
                "final_long_pnl": r.long_pnl.iloc[-1],
                "funding_std_ann_pp": r.funding.std() * 24 * 365 * 100,
            })
    return pd.DataFrame(rows)


def convergence_from_shock(
    index: pd.Series,
    funding_params: FundingParams,
    shocks: list[float],
    gamma: float,
) -> dict[float, pd.Series]:
    """Return basis path from each initial shock (no new noise)."""
    # Short horizon — only need to see convergence
    short_index = index.iloc[:240]  # 10 days
    out = {}
    for s in shocks:
        bp = BasisProcess(rho=0.95, sigma=0.0, gamma=gamma, initial_basis=s)
        r = simulate_mark_convergence(short_index, funding_params, bp, seed=0)
        out[s] = r.basis
    return out


def main() -> None:
    print("Loading RV30 index (hourly)...")
    index = load_rv30_hourly()
    print(f"  {len(index):,} hourly obs  |  {index.index[0].date()} → {index.index[-1].date()}")
    print(f"  Median daily variance: {index.median():.6f}  (~{np.sqrt(index.median()*365)*100:.0f}% ann vol)")

    # --- Grid search ---
    print("\n" + "="*60)
    print("PARAMETER GRID SEARCH")
    print("="*60)
    dampenings = [100.0, 200.0, 333.0, 500.0, 1000.0]
    caps       = [0.001, 0.002, 0.005, 0.01]
    # Basis innovation sigma: 0.5%/hour — realistic market noise scale
    grid = grid_search(
        index, dampenings, caps,
        basis_sigma=0.005, gamma=10.0, initial_basis=0.0,
    )
    print(grid.to_string(index=False, float_format=lambda v: f"{v:8.4f}"))

    summary_path = PROCESSED_DIR / "funding_sim_summary.csv"
    grid.to_csv(summary_path, index=False)
    print(f"\nGrid → {summary_path}")

    # Pick the "best" params: lowest mean |basis| subject to cap_hit_frac < 5%
    feasible = grid[grid["cap_hit_frac"] < 0.05]
    if feasible.empty:
        best = grid.sort_values("mean_abs_basis").iloc[0]
    else:
        best = feasible.sort_values("mean_abs_basis").iloc[0]
    print(f"\nSelected: k={best['dampening']:.0f}  cap={best['cap']:.4f}")
    print(f"  mean|b|={best['mean_abs_basis']:.4f}  max|b|={best['max_abs_basis']:.4f}  "
          f"cap_hit={best['cap_hit_frac']*100:.2f}%")

    # --- Full replay with selected params ---
    fp_best = FundingParams(dampening=float(best["dampening"]), cap=float(best["cap"]))
    bp_noise = BasisProcess(rho=0.90, sigma=0.005, gamma=10.0, initial_basis=0.0)
    full = simulate_mark_convergence(index, fp_best, bp_noise, seed=1)
    print("\nFull historical replay (noise-driven basis):")
    print(f"  mean|basis|         {full.mean_abs_basis:.4f}")
    print(f"  max|basis|          {full.max_abs_basis:.4f}")
    print(f"  cap hit fraction    {full.cap_hit_fraction*100:.2f}%")
    print(f"  funding std (ann)   {full.funding.std()*24*365*100:+.2f} pp")

    # VRP comparison — funding noise must not swamp the ~14pp/yr VRP
    vrp_ann_pp = 14.0  # from Phase 2 gate
    funding_noise_ann_pp = full.funding.std() * 24 * 365 * 100
    print(f"  VRP (Phase 2)       +{vrp_ann_pp:.1f} pp/yr")
    print(f"  Funding noise / VRP {funding_noise_ann_pp/vrp_ann_pp:.2f}x")
    if funding_noise_ann_pp < vrp_ann_pp:
        print("  ✓ Funding noise < VRP — carry signal preserved")
    else:
        print("  ⚠ Funding noise ≥ VRP — tighten dampening or cap")

    _plot_timeseries(full, fp_best)

    # --- Convergence from shocks ---
    print("\n" + "="*60)
    print("CONVERGENCE FROM INITIAL SHOCK")
    print("="*60)
    paths = convergence_from_shock(index, fp_best, shocks=[0.05, 0.10, 0.20, 0.50], gamma=10.0)
    for s, p in paths.items():
        # Hours until |basis| < 1%
        below = np.where(np.abs(p.to_numpy()) < 0.01)[0]
        hrs = int(below[0]) if len(below) else -1
        print(f"  Initial basis {s*100:+.0f}%  →  converges to <1% in "
              f"{hrs if hrs >= 0 else '>240'} hours")

    _plot_convergence(paths)


def _plot_timeseries(r, fp: FundingParams) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(f"Funding Mechanism Replay  (k={fp.dampening:.0f}, cap={fp.cap:.4f})",
                 fontsize=12)

    ax = axes[0]
    ax.plot(r.index.index, r.index.values, label="Index (RV30 daily rate)",
            color="#2196F3", lw=1.0)
    ax.plot(r.mark.index, r.mark.values, label="Simulated mark",
            color="#FF9800", lw=0.6, alpha=0.7)
    ax.set_ylabel("Daily variance")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(r.basis.index, r.basis.values * 100, color="#9C27B0", lw=0.6)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_ylabel("Basis (%)")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(r.funding.index, r.funding.values * 100, color="#F44336", lw=0.5)
    ax3.axhline(fp.cap * 100, color="black", ls="--", lw=0.8, label=f"cap ±{fp.cap*100:.2f}%")
    ax3.axhline(-fp.cap * 100, color="black", ls="--", lw=0.8)
    ax3.set_ylabel("Funding rate (% / hr)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    path = PROCESSED_DIR / "funding_sim_timeseries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot → {path}")


def _plot_convergence(paths: dict[float, pd.Series]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Basis Convergence From Initial Shock (no new noise)", fontsize=12)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    for (s, p), c in zip(paths.items(), colors):
        hours = np.arange(len(p))
        ax.plot(hours, p.to_numpy() * 100, label=f"b₀ = {s*100:+.0f}%", color=c, lw=1.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.axhline(1, color="gray", lw=0.5, ls="--")
    ax.axhline(-1, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("Hours since shock")
    ax.set_ylabel("Basis (%)")
    ax.set_xlim(0, 240)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = PROCESSED_DIR / "funding_sim_convergence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


if __name__ == "__main__":
    main()
