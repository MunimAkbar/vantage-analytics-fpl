"""
gameweek_review.py

Compares a locked model-version decision (from decision_log.py) against
what actually happened once a Gameweek's matches are finished.
Version-aware: run separately for v1 and v2, then use compare_models.py
for the head-to-head.

Usage:
    python -m src.evaluation.gameweek_review --gameweek 4 --version v1
    python -m src.evaluation.gameweek_review --gameweek 4 --version v2
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def load_decision(gameweek: int, version: str) -> pd.DataFrame:
    path = DECISIONS_DIR / f"gw{gameweek}_decision_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.evaluation.decision_log "
            f"--gameweek {gameweek} --version {version}' before the deadline, next time."
        )
    return pd.read_csv(path)


def load_actual_results(gameweek: int) -> pd.DataFrame:
    history = pd.read_csv(DATA_PROCESSED_DIR / "player_history.csv")
    actual = history[history["gameweek_id"] == gameweek]

    if actual.empty:
        raise ValueError(
            f"No results found for gameweek {gameweek} in player_history.csv. "
            "Either the gameweek hasn't started yet, or you need to re-run: "
            "fetch_fpl_data.py --player-history and clean_player_history.py."
        )

    return actual[["player_id", "minutes", "total_points", "goals_scored", "assists",
                   "clean_sheets", "bonus", "yellow_cards", "red_cards"]]


def team_fixture_status(gameweek: int) -> dict[str, bool]:
    """A fixture counts as 'done enough to review' once finished_provisional
    is True — real scores are already in at that point. The strict
    'finished' flag only flips after FPL's official bonus-point
    confirmation pass, which can lag the final whistle by hours, so
    relying on it alone would leave reviews stuck on 'pending' long
    after a gameweek is actually over.
    """
    teams = pd.read_csv(DATA_PROCESSED_DIR / "teams.csv")
    fixtures = pd.read_csv(DATA_PROCESSED_DIR / "fixtures.csv")

    gw_fixtures = fixtures[fixtures["gameweek_id"] == gameweek]
    team_lookup = teams.set_index("team_id")["name"].to_dict()

    status: dict[str, bool] = {}
    for _, row in gw_fixtures.iterrows():
        home_name = team_lookup.get(row["team_h"])
        away_name = team_lookup.get(row["team_a"])
        is_done = bool(row.get("finished_provisional", row["finished"]))
        if home_name:
            status[home_name] = is_done
        if away_name:
            status[away_name] = is_done

    return status


def build_review(decision: pd.DataFrame, actual: pd.DataFrame, fixture_status: dict[str, bool]) -> pd.DataFrame:
    review = decision.merge(actual, on="player_id", how="left")

    fill_cols = ["minutes", "total_points", "goals_scored", "assists",
                 "clean_sheets", "bonus", "yellow_cards", "red_cards"]
    review[fill_cols] = review[fill_cols].fillna(0)

    review["actual_points_scored"] = review["total_points"]
    review["fixture_finished"] = review["team_name"].map(fixture_status).fillna(False)

    review["prediction_error"] = review.apply(
        lambda r: (r["actual_points_scored"] - r["predicted_xPts"]) if r["fixture_finished"] else float("nan"),
        axis=1,
    )

    review["points_toward_team_total"] = review.apply(
        lambda r: (r["actual_points_scored"] * 2 if r["is_captain"] else r["actual_points_scored"])
        if r["squad_role"] == "starting" and r["fixture_finished"] else 0,
        axis=1,
    )

    return review


def summarize(review: pd.DataFrame, gameweek: int, version: str) -> dict:
    starting = review[review["squad_role"] == "starting"]
    finished = starting[starting["fixture_finished"]]
    pending = starting[~starting["fixture_finished"]]

    total_predicted = finished["predicted_xPts"].sum()
    captain_row = finished[finished["is_captain"]]
    if not captain_row.empty:
        total_predicted += captain_row["predicted_xPts"].values[0]

    total_actual = finished["points_toward_team_total"].sum()

    print(f"\n{'='*72}")
    print(f"GAMEWEEK {gameweek} REVIEW — {version.upper()}")
    print(f"{'='*72}")

    if not pending.empty:
        pending_names = ", ".join(pending["web_name"])
        print(f"⚠  {len(pending)} starter(s) haven't played yet — excluded from totals: {pending_names}\n")

    print(f"Predicted total so far ({len(finished)}/{len(starting)} starters played): {total_predicted:.2f}")
    print(f"Actual total so far:                                          {total_actual:.2f}")
    print(f"Difference: {total_actual - total_predicted:+.2f}")

    print(f"\n{'-'*72}")
    print(f"STARTING XI ({version.upper()}) — predicted vs actual (finished fixtures only)")
    print(f"{'-'*72}")
    display = finished.sort_values("actual_points_scored", ascending=False)
    for _, row in display.iterrows():
        tag = " (C)" if row["is_captain"] else (" (VC)" if row["is_vice_captain"] else "")
        print(
            f"  {row['web_name']:<18}{tag:<5} predicted={row['predicted_xPts']:>5.2f}  "
            f"actual={row['actual_points_scored']:>5.0f}  diff={row['prediction_error']:+.2f}"
        )

    print(f"{'='*72}\n")

    return {
        "version": version,
        "gameweek": gameweek,
        "total_predicted": total_predicted,
        "total_actual": total_actual,
        "starters_finished": len(finished),
        "starters_total": len(starting),
        "fully_complete": len(pending) == 0,
    }


def run(gameweek: int, version: str) -> pd.DataFrame:
    decision = load_decision(gameweek, version)
    actual = load_actual_results(gameweek)
    fixture_status = team_fixture_status(gameweek)
    review = build_review(decision, actual, fixture_status)

    out_path = DECISIONS_DIR / f"gw{gameweek}_review_{version}.csv"
    review.to_csv(out_path, index=False)
    logger.info("Saved detailed review to %s", out_path)

    summary = summarize(review, gameweek, version)

    summary_path = DECISIONS_DIR / f"gw{gameweek}_summary_{version}.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return review


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a completed Gameweek's decision vs actual results.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    run(args.gameweek, args.version)


if __name__ == "__main__":
    main()