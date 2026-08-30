"""
clean_bootstrap.py

Parses the raw bootstrap-static JSON dump into three clean, joined
DataFrames: players, teams, gameweeks.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import latest_raw_file, load_json, save_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

PLAYER_COLUMNS = [
    "id", "web_name", "first_name", "second_name", "team", "team_name",
    "position", "now_cost", "status", "chance_of_playing_this_round",
    "chance_of_playing_next_round", "news", "total_points", "points_per_game",
    "form", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "selected_by_percent",
]

TEAM_COLUMNS = [
    "id", "name", "short_name", "strength", "strength_overall_home",
    "strength_overall_away", "strength_attack_home", "strength_attack_away",
    "strength_defence_home", "strength_defence_away",
]

GAMEWEEK_COLUMNS = [
    "id", "name", "deadline_time", "finished", "is_current", "is_next",
    "average_entry_score", "highest_score",
]


def load_bootstrap_raw() -> dict:
    path = latest_raw_file("bootstrap", "bootstrap_static_")
    logger.info("Loading bootstrap-static from %s", path)
    return load_json(path)


def clean_teams(bootstrap: dict) -> pd.DataFrame:
    teams = pd.DataFrame(bootstrap["teams"])
    available = [c for c in TEAM_COLUMNS if c in teams.columns]
    return teams[available].rename(columns={"id": "team_id"})


def clean_players(bootstrap: dict, teams: pd.DataFrame) -> pd.DataFrame:
    players = pd.DataFrame(bootstrap["elements"])
    players["position"] = players["element_type"].map(POSITION_MAP)
    team_lookup = teams.set_index("team_id")["name"].to_dict()
    players["team_name"] = players["team"].map(team_lookup)
    players["now_cost"] = players["now_cost"] / 10.0
    available = [c for c in PLAYER_COLUMNS if c in players.columns]
    return players[available]


def clean_gameweeks(bootstrap: dict) -> pd.DataFrame:
    gameweeks = pd.DataFrame(bootstrap["events"])
    available = [c for c in GAMEWEEK_COLUMNS if c in gameweeks.columns]
    return gameweeks[available].rename(columns={"id": "gameweek_id"})


def run() -> None:
    bootstrap = load_bootstrap_raw()
    teams = clean_teams(bootstrap)
    players = clean_players(bootstrap, teams)
    gameweeks = clean_gameweeks(bootstrap)

    save_processed(teams, "teams.csv")
    save_processed(players, "players.csv")
    save_processed(gameweeks, "gameweeks.csv")

    logger.info(
        "Saved %d teams, %d players, %d gameweeks to data/processed/",
        len(teams), len(players), len(gameweeks),
    )


if __name__ == "__main__":
    run()
    