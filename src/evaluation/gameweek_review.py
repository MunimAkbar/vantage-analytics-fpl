"""
gameweek_review.py

Compares a locked AI decision (from decision_log.py) against what
actually happened once a Gameweek's matches are finished. This is the
core "was the AI right" checkpoint for the human-vs-AI experiment.

Correctly handles partial gameweeks: a player whose team hasn't played
yet is excluded from scoring/error calculations rather than being
treated as if they'd already returned 0 points.

Prerequisites before running this for a given gameweek:
  1. Re-run ingestion + cleaning so player_history.csv and fixtures.csv
     reflect the latest state:
       python -m src.ingestion.fetch_fpl_data
       python -m src.ingestion.fetch_fpl_data --player-history
       python -m src.cleaning.run_cleaning
       python -m src.cleaning.clean_player_history

Usage:
    python -m src.evaluation.gameweek_review --gameweek 3
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def load_decision(gameweek: int) -> pd.DataFrame:
    path = DECISIONS_DIR / f"gw{gameweek}_decision.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python -m src.evaluation.decision_log "
            f"--gameweek {gameweek}' before the deadline, next time."
        )
    return pd.read_csv(path)


def load_actual_results(gameweek: int) -> pd.DataFrame:
    history = pd.read_csv(DATA_PROCESSED_DIR / "player_history.csv")
    actual = history[history["gameweek_id"] == gameweek]

    if actual.empty:
        raise ValueError(
            f"No results found for gameweek {gameweek} in player_history.csv. "
            "Either the gameweek hasn't started yet, or you need to re-run: "
            "fetch_fpl_data.py --player-history and clean_player_history.py "
            "to pull the latest results."
        )

    return actual[["player_id", "minutes", "total_points", "goals_scored", "assists",
                   "clean_sheets", "bonus", "yellow_cards", "red_cards"]]


def team_fixture_status(gameweek: int) -> dict[str, bool]:
    """Maps each team_name -> whether their fixture in this gameweek has
    finished. A player whose team hasn't played yet must not be scored
    as if they'd already returned 0 — that's a pending result, not a fact.
    """
    teams = pd.read_csv(DATA_PROCESSED_DIR / "teams.csv")
    fixtures = pd.read_csv(DATA_PROCESSED_DIR / "fixtures.csv")

    gw_fixtures = fixtures[fixtures["gameweek_id"] == gameweek]
    team_lookup = teams.set_index("team_id")["name"].to_dict()

    status: dict[str, bool] = {}
    for _, row in gw_fixtures.iterrows():
        home_name = team_lookup.get(row["team_h"])
        away_name = team_lookup.get(row["team_a"])
        if home_name:
            status[home_name] = bool(row["finished"])
        if away_name:
            status[away_name] = bool(row["finished"])

    return status


def build_review(decision: pd.DataFrame, actual: pd.DataFrame, fixture_status: dict[str, bool]) -> pd.DataFrame:
    review = decision.merge(actual, on="player_id", how="left")

    fill_cols = ["minutes", "total_points", "goals_scored", "assists",
                 "clean_sheets", "bonus", "yellow_cards", "red_cards"]
    review[fill_cols] = review[fill_cols].fillna(0)

    review["actual_points_scored"] = review["total_points"]
    review["fixture_finished"] = review["team_name"].map(fixture_status).fillna(False)

    # Prediction error only makes sense once the result is real — a player
    # whose match hasn't happened yet isn't "wrong", they're just pending.
    review["prediction_error"] = review.apply(
        lambda r: (r["actual_points_scored"] - r["predicted_xPts"]) if r["fixture_finished"] else float("nan"),
        axis=1,
    )

    # Captain's actual points count double when scoring the team total.
    # Unplayed fixtures contribute 0 for now — they'll count once finished.
    review["points_toward_team_total"] = review.apply(
        lambda r: (r["actual_points_scored"] * 2 if r["is_captain"] else r["actual_points_scored"])
        if r["squad_role"] == "starting" and r["fixture_finished"] else 0,
        axis=1,
    )

    return review


def summarize(review: pd.DataFrame, gameweek: int) -> None:
    starting = review[review["squad_role"] == "starting"]

    finished = starting[starting["fixture_finished"]]
    pending = starting[~starting["fixture_finished"]]

    total_predicted = finished["predicted_xPts"].sum()
    captain_row = finished[finished["is_captain"]]
    if not captain_row.empty:
        total_predicted += captain_row["predicted_xPts"].values[0]

    total_actual = finished["points_toward_team_total"].sum()

    print(f"\n{'='*72}")
    print(f"GAMEWEEK {gameweek} REVIEW")
    print(f"{'='*72}")

    if not pending.empty:
        pending_names = ", ".join(pending["web_name"])
        print(f"⚠  {len(pending)} starter(s) haven't played yet — excluded from totals below: {pending_names}")
        print(f"   Re-run this review once all GW{gameweek} fixtures are finished for a complete picture.\n")

    print(f"Predicted total so far ({len(finished)}/{len(starting)} starters played): {total_predicted:.2f}")
    print(f"Actual total so far:                                          {total_actual:.2f}")
    print(f"Difference: {total_actual - total_predicted:+.2f}")

    print(f"\n{'-'*72}")
    print("STARTING XI — predicted vs actual (finished fixtures only)")
    print(f"{'-'*72}")
    display = finished.sort_values("actual_points_scored", ascending=False)
    for _, row in display.iterrows():
        tag = " (C)" if row["is_captain"] else (" (VC)" if row["is_vice_captain"] else "")
        print(
            f"  {row['web_name']:<18}{tag:<5} predicted={row['predicted_xPts']:>5.2f}  "
            f"actual={row['actual_points_scored']:>5.0f}  diff={row['prediction_error']:+.2f}"
        )

    finished_all = review[review["fixture_finished"]]

    print(f"\n{'-'*72}")
    print("BIGGEST OVERPERFORMANCE (AI underrated them)")
    print(f"{'-'*72}")
    top_over = finished_all.sort_values("prediction_error", ascending=False).head(3)
    for _, row in top_over.iterrows():
        print(f"  {row['web_name']:<18} predicted={row['predicted_xPts']:.2f}  actual={row['actual_points_scored']:.0f}")

    print(f"\n{'-'*72}")
    print("BIGGEST UNDERPERFORMANCE (AI overrated them)")
    print(f"{'-'*72}")
    top_under = finished_all.sort_values("prediction_error", ascending=True).head(3)
    for _, row in top_under.iterrows():
        print(f"  {row['web_name']:<18} predicted={row['predicted_xPts']:.2f}  actual={row['actual_points_scored']:.0f}")

    bench_finished = finished_all[finished_all["squad_role"] == "bench"]
    starting_finished = finished_all[finished_all["squad_role"] == "starting"]
    if not bench_finished.empty and not starting_finished.empty:
        best_bench = bench_finished.loc[bench_finished["actual_points_scored"].idxmax()]
        worst_starter = starting_finished.loc[starting_finished["actual_points_scored"].idxmin()]
        if best_bench["actual_points_scored"] > worst_starter["actual_points_scored"]:
            print(f"\n{'-'*72}")
            print("SELECTION MISS")
            print(f"{'-'*72}")
            print(
                f"  Bench player {best_bench['web_name']} scored {best_bench['actual_points_scored']:.0f} pts "
                f"— more than starter {worst_starter['web_name']} ({worst_starter['actual_points_scored']:.0f} pts)."
            )

    print(f"{'='*72}\n")


def run(gameweek: int) -> pd.DataFrame:
    decision = load_decision(gameweek)
    actual = load_actual_results(gameweek)
    fixture_status = team_fixture_status(gameweek)
    review = build_review(decision, actual, fixture_status)

    out_path = DECISIONS_DIR / f"gw{gameweek}_review.csv"
    review.to_csv(out_path, index=False)
    logger.info("Saved detailed review to %s", out_path)

    summarize(review, gameweek)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description="Review a completed Gameweek's AI decision vs actual results.")
    parser.add_argument("--gameweek", type=int, required=True)
    args = parser.parse_args()
    run(args.gameweek)


if __name__ == "__main__":
    main()