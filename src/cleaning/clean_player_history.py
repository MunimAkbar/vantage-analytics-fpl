"""
clean_player_history.py

Parses per-player element-summary JSON (from fetch_fpl_data.py --player-history)
into a single long-format DataFrame: one row per (player, gameweek).

Requires you to have run:
    python -m src.ingestion.fetch_fpl_data --player-history
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import DATA_RAW_DIR, load_json, save_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HISTORY_DIR = DATA_RAW_DIR / "player_history"

HISTORY_COLUMNS = [
    "element",          # player id
    "round",            # gameweek number
    "kickoff_time",
    "was_home",
    "opponent_team",
    "team_h_score",
    "team_a_score",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
    "yellow_cards",
    "red_cards",
    "saves",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "total_points",
    "value",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
]


def run() -> None:
    if not HISTORY_DIR.exists():
        raise FileNotFoundError(
            f"{HISTORY_DIR} doesn't exist. Run "
            "'python -m src.ingestion.fetch_fpl_data --player-history' first."
        )

    files = sorted(HISTORY_DIR.glob("player_*.json"))
    if not files:
        raise FileNotFoundError(f"No player history files found in {HISTORY_DIR}.")

    logger.info("Parsing %d player history files...", len(files))

    all_rows = []
    for path in files:
        data = load_json(path)
        # element-summary returns {"history": [...], "history_past": [...], "fixtures": [...]}
        # "history" = this season's per-gameweek rows, which is what we want.
        this_season = data.get("history", [])
        all_rows.extend(this_season)

    if not all_rows:
        logger.warning("No per-gameweek rows found across all player files.")
        return

    history = pd.DataFrame(all_rows)
    available = [c for c in HISTORY_COLUMNS if c in history.columns]
    history = history[available].rename(columns={"element": "player_id", "round": "gameweek_id"})

    save_processed(history, "player_history.csv")
    logger.info("Saved %d player-gameweek rows to data/processed/player_history.csv", len(history))


if __name__ == "__main__":
    run()