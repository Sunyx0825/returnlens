import numpy as np
import pandas as pd
import pytest

from returnlens.data import load_monthly
from returnlens.evaluate import comparison, evaluate
from returnlens.forecast import ForecastConfig, constrain_forecast, expanding_forecasts, fit_ols


def example_data():
    rng = np.random.default_rng(12)
    x = rng.normal(size=48)
    return pd.DataFrame(
        {"x": x, "infl": x, "equity_premium": 0.01 + 0.03 * np.roll(x, 1)},
        index=pd.date_range("2000-01-01", periods=len(x), freq="MS"),
    )


def test_ols_known_line():
    x = np.arange(10, dtype=float)
    intercept, slope, n = fit_ols(x, 2 + 3 * x)
    assert (intercept, slope) == pytest.approx((2, 3))
    assert n == 10


def test_constraints_have_explicit_intercept_behavior():
    assert constrain_forecast(0.02, -0.4, 10, 1) == 0.02
    assert constrain_forecast(-0.02, -0.4, 10, 1) == 0
    assert constrain_forecast(0.02, 0.4, 10, 1) == pytest.approx(4.02)


def test_future_changes_cannot_change_earlier_predictions():
    data = example_data()
    config = ForecastConfig(initial_months=12, start="2000-01-01", end="2003-12-01")
    original = expanding_forecasts(data, "x", config)
    altered = data.copy()
    cutoff = pd.Timestamp("2002-01-01")
    altered.loc[cutoff:, ["x", "equity_premium"]] *= -1000
    changed = expanding_forecasts(altered, "x", config)
    columns = ["unconstrained", "constrained", "historical_mean", "initial_sign"]
    pd.testing.assert_frame_equal(
        original.loc[original.origin < cutoff, columns],
        changed.loc[changed.origin < cutoff, columns],
    )


def test_inflation_waits_an_additional_month():
    config = ForecastConfig(initial_months=12, start="2000-01-01", end="2003-12-01")
    frame = expanding_forecasts(example_data(), "infl", config)
    assert ((frame.origin.dt.to_period("M") - 1) == frame.predictor_date.dt.to_period("M")).all()
    assert (frame.origin < frame.target).all()
    assert (frame.sign_training_end <= frame.origin).all()


def test_missing_calendar_month_is_rejected():
    with pytest.raises(ValueError, match="consecutive"):
        expanding_forecasts(example_data().drop(pd.Timestamp("2001-01-01")), "x")


def test_term_spread_is_government_bond_yield_minus_tbill():
    from pathlib import Path

    data = load_monthly(Path(__file__).parents[1] / "data/GW05_original_monthly.csv")
    np.testing.assert_allclose(data.tms, data.lty - data.tbl)


def test_oos_r2_uses_shared_benchmark_error():
    frame = pd.DataFrame(
        {
            "predictor": ["x"] * 2,
            "actual": [1, 3],
            "historical_mean": [0, 0],
            "unconstrained": [0, 0],
            "constrained": [1, 1],
            "target": ["a", "b"],
        }
    )
    table = comparison(evaluate(frame)).iloc[0]
    assert table.unconstrained == 0
    assert table.constrained == pytest.approx(60)
