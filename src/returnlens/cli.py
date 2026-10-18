"""Command-line entry point; all writes stay inside the requested output folder."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PREDICTORS, load_monthly
from .evaluate import comparison, evaluate
from .forecast import expanding_forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReturnLens monthly forecasting study.")
    parser.add_argument("--data", type=Path, default=Path("data/GW05_original_monthly.csv"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    args = parser.parse_args()
    data = load_monthly(args.data)
    forecasts = pd.concat(
        [expanding_forecasts(data, name) for name in PREDICTORS], ignore_index=True
    )
    metrics = evaluate(forecasts)
    table = comparison(metrics)
    summary = {
        "models": len(table),
        "improved": int((table.improvement_pp > 1e-10).sum()),
        "unchanged": int((table.improvement_pp.abs() <= 1e-10).sum()),
        "baseline_beats_mean": int((table.unconstrained > 0).sum()),
        "constrained_beats_mean": int((table.constrained > 0).sum()),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "term_spread": "lty - tbl",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("forecasts", forecasts), ("metrics", metrics), ("comparison", table)]:
        frame.to_csv(args.output / f"{name}.csv", index=False, float_format="%.17g")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
