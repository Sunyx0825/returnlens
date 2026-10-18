"""Check a fresh forecast run and independently recompute its error metric."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
expected = pd.read_csv(ROOT / "results/comparison.csv")
actual = pd.read_csv(ROOT / "output/comparison.csv")
pd.testing.assert_frame_equal(actual, expected, atol=1e-7, rtol=1e-7)
summary = json.loads((ROOT / "output/summary.json").read_text(encoding="utf-8"))
assert (summary["improved"], summary["baseline_beats_mean"], summary["constrained_beats_mean"]) == (
    12,
    4,
    8,
)
forecasts = pd.read_csv(ROOT / "output/forecasts.csv", parse_dates=["origin", "target"])
assert (forecasts.origin < forecasts.target).all()
for predictor, group in forecasts.groupby("predictor"):
    benchmark_sse = ((group.actual - group.historical_mean) ** 2).sum()
    row = actual.set_index("predictor").loc[predictor]
    for variant in ("constrained", "unconstrained"):
        r_squared = 100 * (1 - ((group.actual - group[variant]) ** 2).sum() / benchmark_sse)
        np.testing.assert_allclose(r_squared, row[variant], atol=1e-7)
print("15 models reproduce the reference; 12 improve and 8 beat the mean forecast.")
