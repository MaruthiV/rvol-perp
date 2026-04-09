"""Manipulation cost analysis — Phase 7.

Injects adversarial returns into the real BTC 1-minute return series and
reports the economic cost of moving the RV30 index by a given amount,
with and without the 4σ MAD cap. See docs/design/05-manipulation-cost.md
for the threat model.

Outputs:
  - data/processed/manipulation_cost_table.csv
  - data/processed/manipulation_attack_curves.png

Usage:
    python scripts/analyze_manipulation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from rvol.attacks.inject import (
    AttackParams,
    inject_single_spike,
    inject_sustained,
)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
RETURNS_PATH = PROCESSED_DIR / "returns" / "btcusdt_1m_returns.parquet"


def load_returns() -> np.ndarray:
    tbl = pq.read_table(RETURNS_PATH, columns=["log_return"])
    return tbl["log_return"].to_numpy()


def main() -> None:
    print("Loading returns...")
    r = load_returns()
    # Pick an attack offset near the end of the series so the RV30 window
    # is fully populated with real data.
    offset = len(r) - 1 - 1000  # 1000 minutes from the end
    print(f"  {len(r):,} minutes  |  attack offset: {offset:,}")

    # --- Attack A: single-minute spike sweep over k ---
    print("\n" + "=" * 70)
    print("ATTACK A — SINGLE-MINUTE SPIKE")
    print("=" * 70)

    spike_rows = []
    for k in [2, 4, 6, 8, 10, 15, 20, 30, 50]:
        p = AttackParams(k_sigma=float(k), n_minutes=1, attack_offset=offset)
        res = inject_single_spike(r, p)
        spike_rows.append({
            "attack": "spike",
            "k_sigma": k,
            "n_minutes": 1,
            "d_vol_pt_uncapped": res.delta_ann_vol_pt_uncapped,
            "d_vol_pt_capped": res.delta_ann_vol_pt_capped,
            "cap_reduction": res.cap_reduction,
            "cost_usd": res.cost_usd,
            "usd_per_vol_pt": res.usd_per_vol_pt_capped,
        })

    df_spike = pd.DataFrame(spike_rows)
    print(df_spike.to_string(
        index=False,
        float_format=lambda v: f"{v:>10.3f}" if abs(v) < 1e9 else f"{v:.2e}",
    ))

    # --- Attack B: sustained attack sweep over (k, N) ---
    print("\n" + "=" * 70)
    print("ATTACK B — SUSTAINED ATTACK")
    print("=" * 70)

    sustain_rows = []
    for k in [4, 6, 10, 15]:
        for n_minutes in [10, 60, 240, 720]:
            p = AttackParams(k_sigma=float(k), n_minutes=n_minutes, attack_offset=offset)
            res = inject_sustained(r, p)
            sustain_rows.append({
                "attack": "sustained",
                "k_sigma": k,
                "n_minutes": n_minutes,
                "d_vol_pt_uncapped": res.delta_ann_vol_pt_uncapped,
                "d_vol_pt_capped": res.delta_ann_vol_pt_capped,
                "cap_reduction": res.cap_reduction,
                "cost_usd": res.cost_usd,
                "usd_per_vol_pt": res.usd_per_vol_pt_capped,
            })

    df_sustain = pd.DataFrame(sustain_rows)
    print(df_sustain.to_string(
        index=False,
        float_format=lambda v: f"{v:>10.3f}" if abs(v) < 1e9 else f"{v:.2e}",
    ))

    # --- Combine and save ---
    df = pd.concat([df_spike, df_sustain], ignore_index=True)
    out_csv = PROCESSED_DIR / "manipulation_cost_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nTable → {out_csv}")

    # --- Gate checks ---
    print("\n" + "=" * 70)
    print("GATE CHECKS")
    print("=" * 70)

    # Gate 1: Cap reduces single-minute attack impact by ≥90% for k≥10σ
    large_spikes = df_spike[df_spike["k_sigma"] >= 10]
    min_reduction = large_spikes["cap_reduction"].min()
    ok1 = min_reduction >= 0.90
    print(f"  {'✓' if ok1 else '✗'} Cap reduces large spike impact ≥ 90%  "
          f"(min {min_reduction*100:.1f}%)")

    # Gate 2: $ per vol point ≥ $1M for sustained attacks (capped)
    # Filter rows where cap actually moved the index a meaningful amount
    meaningful = df_sustain[df_sustain["d_vol_pt_capped"].abs() > 0.1]
    if len(meaningful) > 0:
        min_cost_per_pt = meaningful["usd_per_vol_pt"].min()
        ok2 = min_cost_per_pt >= 1_000_000
        print(f"  {'✓' if ok2 else '✗'} Cost per vol point ≥ $1M (sustained)  "
              f"(min ${min_cost_per_pt/1e6:.2f}M/pt)")
    else:
        print(f"  ✓ No sustained attack moves index ≥ 0.1 vol pt — cap is effective")

    # Gate 3: Sustained attack at 15σ × 720 min should cost > $10M AND move < 5 vol pts
    row = df_sustain[(df_sustain["k_sigma"] == 15) & (df_sustain["n_minutes"] == 720)]
    if len(row):
        cost = row["cost_usd"].iloc[0]
        move = row["d_vol_pt_capped"].iloc[0]
        print(f"  Attack at k=15σ, N=720m (12h):")
        print(f"    Cost: ${cost/1e6:.1f}M  |  ΔRV30 (capped): {move:+.3f} vol pt")
        if cost > 10_000_000 and abs(move) < 5:
            print(f"  ✓ Large attack costs >$10M for <5 vol pt move")
        else:
            print(f"  ⚠ Large attack economics: review design")

    _plot_attack_curves(df_spike, df_sustain)


def _plot_attack_curves(df_spike: pd.DataFrame, df_sustain: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Manipulation Cost Analysis", fontsize=12)

    # Left: single-spike impact (capped vs uncapped) as function of k
    ax = axes[0]
    ax.plot(df_spike["k_sigma"], df_spike["d_vol_pt_uncapped"],
            marker="o", color="#F44336", label="Uncapped")
    ax.plot(df_spike["k_sigma"], df_spike["d_vol_pt_capped"],
            marker="s", color="#2196F3", label="Capped (4σ MAD)")
    ax.axvline(4, color="gray", ls="--", lw=1, label="Cap threshold (4σ)")
    ax.set_xlabel("Attack size k (units of σ_MAD)")
    ax.set_ylabel("ΔRV30 (annualized vol points)")
    ax.set_title("Single-minute spike")
    ax.set_yscale("symlog", linthresh=0.001)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: sustained attack cost per vol point
    ax2 = axes[1]
    for k, group in df_sustain.groupby("k_sigma"):
        ax2.plot(group["n_minutes"], group["cost_usd"] / 1e6,
                 marker="o", label=f"k={k}σ")
    ax2.set_xlabel("Attack duration (minutes)")
    ax2.set_ylabel("Total cost ($M)")
    ax2.set_title("Sustained attack cost")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = PROCESSED_DIR / "manipulation_attack_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot → {path}")


if __name__ == "__main__":
    main()
