"""
build_historical_features.py

Builds leakage-safe rolling features from historical_gameweeks.csv, for
training V2's regression model.

Critical rule: every feature for a given (player, season, gameweek) row
uses ONLY prior gameweeks within that same season — never the current
row's own result, and never future rows.

Usage:
    python -m src.features.build_historical_features
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROLLING_WINDOW = 5


def load_historical() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "historical_gameweeks.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/cleaning/clean_historical_data.py first.")
    return pd.read_csv(path)


def _add_rolling_features(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Adds leakage-safe rolling features using groupby + shift + transform.

    Deliberately avoids groupby(...).apply() here — in some pandas
    versions it silently drops the grouping columns from the output,
    which would corrupt the dataset (verified against this exact issue
    while building this file). transform() on a shifted column has no
    such problem and is also faster.
    """
    df = df.sort_values(["name", "season", "gameweek_id"]).copy()
    grouped = df.groupby(["name", "season"])

    shifted_minutes = grouped["minutes"].shift(1)
    shifted_xg = grouped["expected_goals"].shift(1)
    shifted_xa = grouped["expected_assists"].shift(1)
    shifted_points = grouped["total_points"].shift(1)
    shifted_bonus = grouped["bonus"].shift(1)
    shifted_bps = grouped["bps"].shift(1)

    def _rolling(series: pd.Series, agg: str) -> pd.Series:
        return series.groupby([df["name"], df["season"]]).transform(
            lambda s: s.rolling(window, min_periods=1).agg(agg)
        )

    roll_minutes_sum = _rolling(shifted_minutes, "sum")
    roll_xg_sum = _rolling(shifted_xg, "sum")
    roll_xa_sum = _rolling(shifted_xa, "sum")

    df["games_in_window"] = _rolling(shifted_minutes, "count")
    df["avg_minutes_last_n"] = _rolling(shifted_minutes, "mean")
    df["avg_points_last_n"] = _rolling(shifted_points, "mean")
    df["avg_bonus_last_n"] = _rolling(shifted_bonus, "mean")
    df["avg_bps_last_n"] = _rolling(shifted_bps, "mean")

    df["xG_per90_last_n"] = (roll_xg_sum / roll_minutes_sum * 90).where(roll_minutes_sum > 0, 0.0)
    df["xA_per90_last_n"] = (roll_xa_sum / roll_minutes_sum * 90).where(roll_minutes_sum > 0, 0.0)

    return df


def build_historical_features() -> pd.DataFrame:
    history = load_historical()
    featured = _add_rolling_features(history)

    fill_cols = [
        "games_in_window", "avg_minutes_last_n", "avg_points_last_n",
        "avg_bonus_last_n", "avg_bps_last_n", "xG_per90_last_n", "xA_per90_last_n",
    ]
    featured[fill_cols] = featured[fill_cols].fillna(0)

    return featured


def run() -> None:
    featured = build_historical_features()
    out_path = DATA_PROCESSED_DIR / "historical_features.csv"
    featured.to_csv(out_path, index=False)
    logger.info("Saved %d rows with historical features to %s", len(featured), out_path)


if __name__ == "__main__":
    run()