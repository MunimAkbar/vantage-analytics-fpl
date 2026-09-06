"""
clean_historical_data.py

Combines the raw multi-season merged_gw.csv files into one unified,
training-ready historical dataset: one row per (player, season,
gameweek), tagged by season so V2's walk-forward validation can split
cleanly on season boundaries (never randomly — see AGENTS.md leakage rules).

Note on the 'xP' column: this is FPL's own official pre-match expected
points estimate. The source dataset's own documentation warns this can
cause lookahead bias if used unshifted as a training feature — so this
cleaning step drops it entirely rather than risk it leaking in later.

Usage:
    python -m src.cleaning.clean_historical_data
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import DATA_RAW_DIR, save_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HISTORICAL_RAW_DIR = DATA_RAW_DIR / "historical"

KEEP_COLUMNS = [
    "name", "position", "team", "element", "GW", "opponent_team", "was_home",
    "minutes", "starts", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_missed", "penalties_saved",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "influence", "creativity", "threat", "ict_index",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "total_points", "value",
]


def discover_season_files() -> list[tuple[str, "Path"]]:
    files = sorted(HISTORICAL_RAW_DIR.glob("*_merged_gw.csv"))
    if not files:
        raise FileNotFoundError(
            f"No historical files found in {HISTORICAL_RAW_DIR}. "
            "Run src/ingestion/fetch_historical_data.py first."
        )
    return [(f.stem.replace("_merged_gw", ""), f) for f in files]


def clean_season(path, season: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Position label fixes, found by inspecting real data across seasons:
    # - 2021-22 used "GKP" for part of the season instead of "GK" — normalize.
    # - 2024-25 introduced "AM" (Assistant Manager) — a real FPL feature
    #   letting managers pick an actual PL manager as a squad slot, scored
    #   on entirely different rules (team results, not player stats). These
    #   are not players and must be excluded, not just relabeled, or they'd
    #   corrupt a player-points regression model.
    df["position"] = df["position"].replace({"GKP": "GK"})
    df = df[df["position"] != "AM"].copy()

    available = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[available].copy()
    df["season"] = season
    df = df.rename(columns={"GW": "gameweek_id", "element": "season_player_id"})
    return df


def run() -> pd.DataFrame:
    season_files = discover_season_files()
    logger.info("Found %d season file(s): %s", len(season_files), [s for s, _ in season_files])

    frames = [clean_season(path, season) for season, path in season_files]
    combined = pd.concat(frames, ignore_index=True)

    combined = combined.sort_values(["season", "gameweek_id"]).reset_index(drop=True)

    save_processed(combined, "historical_gameweeks.csv")
    logger.info(
        "Saved %d historical player-gameweek rows across %d seasons to "
        "data/processed/historical_gameweeks.csv",
        len(combined), combined["season"].nunique(),
    )
    return combined


if __name__ == "__main__":
    run()