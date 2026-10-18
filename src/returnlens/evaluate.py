"""Evaluate both models on exactly the same test observations."""

import numpy as np
import pandas as pd


def evaluate(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for predictor, frame in forecasts.groupby("predictor", sort=False):
        actual = frame.actual.to_numpy()
        benchmark_sse = float(np.square(actual - frame.historical_mean).sum())
        if benchmark_sse <= 0:
            raise ValueError("Out-of-sample R² needs nonzero benchmark error.")
        for variant in ("unconstrained", "constrained"):
            prediction = frame[variant].to_numpy()
            if not np.isfinite(prediction).all():
                raise ValueError("Forecasts must be finite.")
            sse = float(np.square(actual - prediction).sum())
            rows.append(
                {
                    "predictor": predictor,
                    "variant": variant,
                    "n_test": len(frame),
                    "first_target": frame.target.iloc[0],
                    "last_target": frame.target.iloc[-1],
                    "oos_r2_pct": 100 * (1 - sse / benchmark_sse),
                    "mse": sse / len(frame),
                    "benchmark_mse": benchmark_sse / len(frame),
                }
            )
    return pd.DataFrame(rows)


def comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    table = metrics.pivot(index="predictor", columns="variant", values="oos_r2_pct")
    table["improvement_pp"] = table.constrained - table.unconstrained
    return table.reset_index()
