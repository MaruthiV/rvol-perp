from dataclasses import dataclass, field


@dataclass(frozen=True)
class IndexSpec:
    """All parameters that define a realized variance index.

    Frozen so that specs can be used as dict keys and passed safely
    across simulator boundaries without mutation risk.
    """

    window_days: int = 30
    """Rolling window length in calendar days."""

    interval_minutes: int = 1
    """Return sampling interval in minutes."""

    sigma_cap: float = 4.0
    """Return contribution cap in units of the rolling MAD-based sigma."""

    min_obs_fraction: float = 0.9
    """Minimum fraction of expected observations required for is_valid=True."""

    mad_lookback_minutes: int = 1440
    """Lookback window for the MAD-based sigma estimator (default: 1 day)."""

    annualize: bool = False
    """If True, rolling_realized_variance returns annualized variance."""

    annualization_days: int = 365
    """Days per year for annualization. Crypto convention: 365 (not 252)."""

    gap_threshold_factor: float = 1.5
    """Gap detection: intervals longer than this × interval_minutes are NaN'd."""

    expected_obs_per_window: int = field(init=False)
    min_obs_count: int = field(init=False)

    def __post_init__(self) -> None:
        if not (1 <= self.window_days <= 365):
            raise ValueError(f"window_days must be in [1, 365], got {self.window_days}")
        if self.sigma_cap <= 0:
            raise ValueError(f"sigma_cap must be > 0, got {self.sigma_cap}")
        if not (0.01 <= self.min_obs_fraction <= 1.0):
            raise ValueError(
                f"min_obs_fraction must be in [0.01, 1.0], got {self.min_obs_fraction}"
            )
        if self.interval_minutes < 1:
            raise ValueError(f"interval_minutes must be >= 1, got {self.interval_minutes}")
        if self.mad_lookback_minutes < 60:
            raise ValueError(
                f"mad_lookback_minutes must be >= 60, got {self.mad_lookback_minutes}"
            )
        # Derived fields — use object.__setattr__ because the dataclass is frozen
        expected = self.window_days * (24 * 60 // self.interval_minutes)
        object.__setattr__(self, "expected_obs_per_window", expected)
        object.__setattr__(self, "min_obs_count", int(expected * self.min_obs_fraction))

    @property
    def window_us(self) -> int:
        """Window length in microseconds (for timestamp arithmetic)."""
        return self.window_days * 24 * 60 * 60 * 1_000_000

    @property
    def interval_us(self) -> int:
        """Interval length in microseconds."""
        return self.interval_minutes * 60 * 1_000_000

    @property
    def gap_threshold_us(self) -> int:
        """Gap detection threshold in microseconds."""
        return int(self.gap_threshold_factor * self.interval_us)


# Pre-built standard specs
SPEC_30D = IndexSpec(window_days=30)
SPEC_14D = IndexSpec(window_days=14)
SPEC_7D = IndexSpec(window_days=7)
