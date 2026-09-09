"""
weekly_report.py

One consolidated report for the "gameweek just ended" moment.

Usage:
    python -m src.evaluation.weekly_report --completed-gameweek 4 --version v1 --free-transfers 1
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR
from src.evaluation import gameweek_review
from src.evaluation.suggest_transfers import find_latest_locked_squad, load_live_predictions
from src.optimization.transfer_selector import select_squad_with_transfers

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def fixture_context(player_row: pd.Series, teams: pd.DataFrame) -> str:
    parts = []
    if "next_opponent_team" in player_row and pd.notna(player_row["next_opponent_team"]):
        team_lookup = teams.set_index("team_id")["name"].to_dict()
        opp_name = team_lookup.get(int(player_row["next_opponent_team"]), "Unknown")
        parts.append(f"next opponent: {opp_name}")
    if "next_fixture_difficulty" in player_row and pd.notna(player_row["next_fixture_difficulty"]):
        fdr = player_row["next_fixture_difficulty"]
        label = "easy" if fdr <= 2 else ("tough" if fdr >= 4 else "moderate")
        parts.append(f"difficulty {fdr:.0f}/5 ({label})")
    if "fixture_run_difficulty" in player_row and pd.notna(player_row["fixture_run_difficulty"]):
        parts.append(f"next-5 avg difficulty {player_row['fixture_run_difficulty']:.1f}/5")
    return " | ".join(parts) if parts else "(no fixture data available)"


def run(completed_gameweek: int, version: str, free_transfers: int) -> None:
    print(f"\n{'#'*70}")
    print(f"# WEEKLY REPORT — after GW{completed_gameweek} ({version.upper()})")
    print(f"{'#'*70}")

    review_path = DECISIONS_DIR / f"gw{completed_gameweek}_review_{version}.csv"
    summary_path = DECISIONS_DIR / f"gw{completed_gameweek}_summary_{version}.csv"

    if not summary_path.exists():
        print(f"\nRunning gameweek review for GW{completed_gameweek} ({version})...")
        gameweek_review.run(completed_gameweek, version)
    else:
        summary = pd.read_csv(summary_path).iloc[0]
        print(f"\nGW{completed_gameweek} RESULT ({version.upper()})")
        print(f"  Predicted: {summary['total_predicted']:.2f}   Actual: {summary['total_actual']:.2f}"
              f"   Diff: {summary['total_actual'] - summary['total_predicted']:+.2f}")
        if not summary["fully_complete"]:
            print("  ⚠ Not all fixtures had finished when this was last reviewed — re-run gameweek_review for a final number.")

    print(f"\n{'-'*70}")
    print(f"NEXT WEEK — suggested squad, captain, and reasoning ({version.upper()})")
    print(f"{'-'*70}")

    try:
        predictions = load_live_predictions(version)
    except FileNotFoundError as e:
        print(f"\n⚠  {e}")
        print("Run the predict step for next week's gameweek first, then re-run this report.")
        return

    _, old_squad_names = find_latest_locked_squad(version)
    new_squad, transfers_made, hits_taken = select_squad_with_transfers(
        predictions, old_squad_names, free_transfers, name_col="name",
    )

    ranked = new_squad.sort_values("xPts", ascending=False)
    captain = ranked.iloc[0]
    vice = ranked.iloc[1]

    print(f"\nSuggested transfers: {transfers_made}  |  Hits: {hits_taken}"
          f"  ({'-' + str(hits_taken * 4) + ' points' if hits_taken else 'no cost'})")

    teams_path = DATA_PROCESSED_DIR / "teams.csv"
    teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame(columns=["team_id", "name"])

    print(f"\nRecommended CAPTAIN: {captain['name']} ({captain['team_name']})")
    print(f"  xPts: {captain['xPts']:.2f}  |  {fixture_context(captain, teams)}")
    print(f"\nRecommended VICE-CAPTAIN: {vice['name']} ({vice['team_name']})")
    print(f"  xPts: {vice['xPts']:.2f}  |  {fixture_context(vice, teams)}")

    new_names = set(new_squad["name"])
    players_in = new_names - old_squad_names
    if players_in:
        print(f"\nSuggested incoming transfer(s):")
        for name in sorted(players_in):
            row = new_squad[new_squad["name"] == name].iloc[0]
            print(f"  IN: {name} ({row['team_name']}, £{row['now_cost']:.1f}m)")
            print(f"      {fixture_context(row, teams)}")
    else:
        print(f"\nNo beneficial transfer found this week — squad unchanged is optimal.")

    print(f"\n{'#'*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidated weekly report: review + transfer + captain suggestion.")
    parser.add_argument("--completed-gameweek", type=int, required=True)
    parser.add_argument("--version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--free-transfers", type=int, required=True)
    args = parser.parse_args()
    run(args.completed_gameweek, args.version, args.free_transfers)


if __name__ == "__main__":
    main()