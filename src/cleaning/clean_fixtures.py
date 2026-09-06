"""
clean_fixtures.py

Parses the raw fixtures JSON into a clean DataFrame with FPL's
Fixture Difficulty Rating (FDR) for both sides of each match.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import latest_raw_file, load_json, save_processed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FIXTURE_COLUMNS = [
    "id", "event", "kickoff_time", "finished", "finished_provisional", "team_h", "team_a",
    "team_h_score", "team_a_score", "team_h_difficulty", "team_a_difficulty",
]


def load_fixtures_raw() -> list[dict]:
    path = latest_raw_file("fixtures", "fixtures_")
    logger.info("Loading fixtures from %s", path)
    return load_json(path)


def clean_fixtures(raw_fixtures: list[dict], teams: pd.DataFrame | None = None) -> pd.DataFrame:
    fixtures = pd.DataFrame(raw_fixtures)
    available = [c for c in FIXTURE_COLUMNS if c in fixtures.columns]
    fixtures = fixtures[available].rename(columns={"event": "gameweek_id"})

    if teams is not None:
        team_lookup = teams.set_index("team_id")["name"].to_dict()
        fixtures["team_h_name"] = fixtures["team_h"].map(team_lookup)
        fixtures["team_a_name"] = fixtures["team_a"].map(team_lookup)

    return fixtures


def run() -> None:
    raw_fixtures = load_fixtures_raw()

    try:
        from src.cleaning.utils import DATA_PROCESSED_DIR
        teams = pd.read_csv(DATA_PROCESSED_DIR / "teams.csv")
    except FileNotFoundError:
        logger.warning("teams.csv not found yet — run clean_bootstrap.py first for team names.")
        teams = None

    fixtures = clean_fixtures(raw_fixtures, teams)
    save_processed(fixtures, "fixtures.csv")

    logger.info("Saved %d fixtures to data/processed/fixtures.csv", len(fixtures))


if __name__ == "__main__":
    run()