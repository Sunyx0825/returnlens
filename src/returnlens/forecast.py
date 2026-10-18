"""Expanding-window OLS with a training-only slope direction and forecast floor."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastConfig:
    initial_months: int = 240
    start: str = "1927-12-01"
    end: str = "2005-12-01"


def fit_ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Fit an intercept and slope to available training pairs only."""
    available = np.isfinite(x) & np.isfinite(y)
    if available.sum() < 3:
        raise ValueError("OLS requires at least three finite training pairs.")
    design = np.column_stack([np.ones(available.sum()), x[available]])
    if np.ptp(x[available]) == 0:
        raise ValueError("The predictor is constant in the training window.")
    intercept, slope = np.linalg.pinv(design, rcond=1e-15) @ y[available]
    return float(intercept), float(slope), int(available.sum())


def constrain_forecast(intercept: float, slope: float, value: float, sign: int) -> float:
    """Zero a conflicting slope, keep the fitted intercept, then floor at zero.

    This is a forecast restriction, not a re-estimated constrained least-squares
    fit. Keeping this distinction explicit makes the experiment reproducible.
    """
    if sign not in (-1, 1):
        raise ValueError("The initial slope sign must be -1 or 1.")
    restricted_slope = 0.0 if slope * sign < 0 else slope
    return max(intercept + restricted_slope * value, 0.0)


def expanding_forecasts(
    data: pd.DataFrame, predictor: str, config: ForecastConfig = ForecastConfig()
) -> pd.DataFrame:
    """At each month-end, fit on known returns and forecast the following month.

    Inflation has one extra month of publication delay. The slope direction is
    estimated once at the first forecast origin and never uses test outcomes.
    """
    if config.initial_months < 3:
        raise ValueError("initial_months must be at least three.")
    if not data.index.is_unique or not data.index.is_monotonic_increasing:
        raise ValueError("Forecast dates must be unique and increasing.")
    if not data.index.equals(pd.date_range(data.index.min(), data.index.max(), freq="MS")):
        raise ValueError("Forecast dates must be consecutive months.")
    start = max(config.start, "1937-05-01") if predictor == "csp" else config.start
    end = min(config.end, "2002-12-01") if predictor == "csp" else config.end
    sample = data.loc[start:end]
    if len(sample) < config.initial_months + 2:
        raise ValueError("Not enough observations for training and a test forecast.")
    lag = 2 if predictor == "infl" else 1
    training_x = sample[predictor].shift(lag).to_numpy(dtype=float)
    targets = sample.equity_premium.to_numpy(dtype=float)
    if not np.isfinite(targets).all():
        raise ValueError("Target returns must be finite.")
    first_origin = config.initial_months
    _, initial_slope, _ = fit_ols(training_x[: first_origin + 1], targets[: first_origin + 1])
    initial_sign = 1 if initial_slope > 0 else -1
    historical_means = np.cumsum(targets) / np.arange(1, len(targets) + 1)
    rows = []
    for origin in range(first_origin, len(sample) - 1):
        intercept, slope, nobs = fit_ols(training_x[: origin + 1], targets[: origin + 1])
        predictor_position = origin - (lag - 1)
        value = float(sample[predictor].iloc[predictor_position])
        if not np.isfinite(value):
            raise ValueError(f"Missing {predictor} at forecast origin {sample.index[origin]}.")
        rows.append(
            {
                "predictor": predictor,
                "origin": sample.index[origin],
                "target": sample.index[origin + 1],
                "predictor_date": sample.index[predictor_position],
                "sign_training_end": sample.index[first_origin],
                "n_training": nobs,
                "initial_sign": initial_sign,
                "intercept": intercept,
                "slope": slope,
                "actual": targets[origin + 1],
                "historical_mean": historical_means[origin],
                "unconstrained": intercept + slope * value,
                "constrained": constrain_forecast(intercept, slope, value, initial_sign),
            }
        )
    return pd.DataFrame(rows)
