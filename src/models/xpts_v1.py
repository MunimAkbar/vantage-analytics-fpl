"""
xpts_v1.py

V1 baseline expected-points model: a rule-based formula built directly
from the FPL scoring table, driven by P(start), rolling per-90 xG/xA,
and a fixture-difficulty-based clean sheet proxy.

This is intentionally crude — it exists to (a) produce a usable first
prediction and (b) become the feature set for V2 (regression) later.
Every intermediate term is kept as its own column, not just baked into
the final score.

Usage:
    python -m src.models.xpts_v1
    python -m src.models.xpts_v1 --position MID --top 15
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from configs.fpl_scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    GOAL_POINTS,
    MINUTES_POINTS_60_PLUS,
    MINUTES_POINTS_UNDER_60,
)
from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_features() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "features.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/features/build_features.py first."
        )
    return pd.read_csv(path)


def minutes_points(row: pd.Series) -> float:
    """Expected points from appearance alone.

    P(start) is split into P(60+ min) vs P(1-59 min) using recent
    average minutes as a proxy — a player who starts but is usually
    subbed off early won't reliably clear the 60-minute bonus threshold.
    """
    p_start = row["p_start"]
    minutes_share = min(row["avg_minutes_last_n"] / 90.0, 1.0) if pd.notna(row["avg_minutes_last_n"]) else 0.0

    p_60_plus = p_start * minutes_share
    p_under_60 = max(p_start - p_60_plus, 0.0)

    return p_60_plus * MINUTES_POINTS_60_PLUS + p_under_60 * MINUTES_POINTS_UNDER_60


def expected_match_involvement(row: pd.Series) -> tuple[float, float]:
    """xG and xA scaled to expected minutes this match (not per-90)."""
    minutes_fraction = min(row["avg_minutes_last_n"] / 90.0, 1.0) if pd.notna(row["avg_minutes_last_n"]) else 0.0
    xg_match = (row["xG_per90_last_n"] or 0.0) * minutes_fraction
    xa_match = (row["xA_per90_last_n"] or 0.0) * minutes_fraction
    return xg_match, xa_match


def clean_sheet_probability(fdr: float) -> float:
    """Crude V1 proxy: easier fixtures (lower FDR) -> higher clean sheet odds.

    FDR runs 1 (easiest) to 5 (hardest). This is deliberately simple —
    a real team-strength/Poisson model replaces this in V2.
    """
    if pd.isna(fdr):
        return 0.25  # unknown fixture, use a league-average-ish default
    return max(0.05, min(0.5, (6 - fdr) / 10))


def bonus_points_proxy(xg_match: float, xa_match: float) -> float:
    """Crude V1 proxy: more involvement in goals -> more likely to be
    in the top 3 BPS and earn bonus. Capped at 3 (the max possible)."""
    return min((xg_match + xa_match) * 1.2, 3.0)


def compute_xpts(row: pd.Series) -> pd.Series:
    position = row["position"]
    p_start = row["p_start"]

    min_pts = minutes_points(row)
    xg_match, xa_match = expected_match_involvement(row)

    goal_pts = p_start * xg_match * GOAL_POINTS.get(position, 0)
    assist_pts = p_start * xa_match * ASSIST_POINTS

    cs_prob = clean_sheet_probability(row.get("next_fixture_difficulty"))
    cs_pts = p_start * cs_prob * CLEAN_SHEET_POINTS.get(position, 0)

    bonus_pts = p_start * bonus_points_proxy(xg_match, xa_match)

    xpts = min_pts + goal_pts + assist_pts + cs_pts + bonus_pts

    return pd.Series({
        "minutes_pts": round(min_pts, 2),
        "goal_pts": round(goal_pts, 2),
        "assist_pts": round(assist_pts, 2),
        "clean_sheet_pts": round(cs_pts, 2),
        "bonus_pts": round(bonus_pts, 2),
        "xPts": round(xpts, 2),
    })


def run(position: str | None = None, top: int = 20) -> pd.DataFrame:
    features = load_features()

    breakdown = features.apply(compute_xpts, axis=1)
    predictions = pd.concat([features, breakdown], axis=1)

    predictions = predictions.sort_values("xPts", ascending=False)

    out_path = DATA_PROCESSED_DIR / "predictions_v1.csv"
    predictions.to_csv(out_path, index=False)
    logger.info("Saved %d predictions to %s", len(predictions), out_path)

    display_cols = [
        "web_name", "team_name", "position", "now_cost",
        "p_start", "xPts", "minutes_pts", "goal_pts",
        "assist_pts", "clean_sheet_pts", "bonus_pts",
    ]

    view = predictions
    if position:
        view = view[view["position"] == position.upper()]

    print(view[display_cols].head(top).to_string(index=False))
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V1 xPts baseline model.")
    parser.add_argument("--position", choices=["GK", "DEF", "MID", "FWD"], default=None)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    run(position=args.position, top=args.top)


if __name__ == "__main__":
    main()