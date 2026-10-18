"""Load the original monthly data vintage and construct predictors."""

from pathlib import Path

import numpy as np
import pandas as pd

PREDICTORS = (
    "de",
    "svar",
    "lty",
    "ltr",
    "infl",
    "tms",
    "tbl",
    "dfy",
    "dp",
    "dy",
    "ep",
    "b/m",
    "e10p",
    "csp",
    "ntis",
)


def load_monthly(path: str | Path) -> pd.DataFrame:
    """Read semicolon-separated data; return a unique, regular monthly index.

    The archived experiment starts feature construction in November 1927 and
    evaluation in December. Missing early e10p values are warm-up observations,
    not values to fill using later data.
    """
    raw = pd.read_csv(path, sep=";", decimal=",")
    required = {
        "yyyymm",
        "Index",
        "D12",
        "E12",
        "AAA",
        "BAA",
        "lty",
        "tbl",
        "Rfree",
        "CRSP_SPvw",
        *set(PREDICTORS) - {"de", "tms", "dfy", "dp", "dy", "ep", "e10p"},
    }
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw.index = pd.to_datetime(raw.pop("yyyymm").astype(str), format="%Y%m")
    raw.index.name = "date"
    if not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing.")
    expected = pd.date_range(raw.index.min(), raw.index.max(), freq="MS")
    if not raw.index.equals(expected):
        raise ValueError("The input must contain consecutive calendar months.")
    data = raw.loc["1927-11-01":].copy()
    for name in ("Index", "D12", "E12"):
        if (data[name].dropna() <= 0).any():
            raise ValueError(f"{name} must be positive before taking logarithms.")
    data["dp"] = np.log(data.D12) - np.log(data.Index)
    data["dy"] = np.log(data.D12) - np.log(data.Index.shift(1))
    data["ep"] = np.log(data.E12) - np.log(data.Index)
    data["de"] = np.log(data.D12) - np.log(data.E12)
    data["e10p"] = np.log(data.E12.rolling(120).mean()) - np.log(data.Index)
    data["tms"] = data.lty - data.tbl
    data["dfy"] = data.BAA - data.AAA
    data["equity_premium"] = data.CRSP_SPvw - data.Rfree
    if not np.isfinite(data.equity_premium).all():
        raise ValueError("Target returns must be finite.")
    return data.iloc[1:]
