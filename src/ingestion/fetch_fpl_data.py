"""
fetch_fpl_data.py

Pulls raw data from the official Fantasy Premier League API and saves it
to data/raw/ as timestamped JSON files.

The official FPL API requires no authentication. Key endpoints:

- bootstrap-static  : all players, teams, gameweeks, and current prices
                      (the main "everything" endpoint)
- fixtures          : full season fixture list with difficulty ratings
- element-summary/{id} : per-player detailed history (per gameweek, per season)
- event/{gw}/live  : live/final stats for a specific gameweek

Usage:
    python -m src.ingestion.fetch_fpl_data
    python -m src.ingestion.fetch_fpl_data --player-history
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

REQUEST_TIMEOUT_SECONDS = 15
REQUEST_DELAY_SECONDS = 0.5  # be polite to the API between calls


def _timestamp() -> str:
    """Returns a UTC timestamp string safe for filenames.

    Every fetch is timestamped so the pipeline can respect the
    'no lookahead' rule later: predictions for a Gameweek must only use
    data that existed before that Gameweek's deadline.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _get_json(url: str) -> dict:
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _save_json(data: dict, subfolder: str, filename: str) -> Path:
    out_dir = DATA_RAW_DIR / subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Saved %s", out_path)
    return out_path


def fetch_bootstrap_static() -> dict:
    """Fetches the main FPL 'everything' endpoint.

    Contains: all players (elements), all teams, all gameweeks (events),
    scoring rules (game_settings), and current prices/ownership.
    This is the single most useful endpoint to start with.
    """
    logger.info("Fetching bootstrap-static...")
    data = _get_json(f"{BASE_URL}/bootstrap-static/")
    _save_json(data, "bootstrap", f"bootstrap_static_{_timestamp()}.json")
    return data


def fetch_fixtures() -> dict:
    """Fetches the full season fixture list, including FPL's own
    difficulty ratings (FDR) for each side of each fixture."""
    logger.info("Fetching fixtures...")
    data = _get_json(f"{BASE_URL}/fixtures/")
    _save_json(data, "fixtures", f"fixtures_{_timestamp()}.json")
    return data


def fetch_player_history(player_id: int) -> dict:
    """Fetches a single player's detailed history: past-season summaries
    and this-season per-gameweek breakdown (minutes, xG, xA, bonus, etc.)."""
    data = _get_json(f"{BASE_URL}/element-summary/{player_id}/")
    return data


def fetch_all_player_histories(bootstrap: dict, delay: float = REQUEST_DELAY_SECONDS) -> None:
    """Loops through every player in bootstrap-static and fetches their
    detailed history. This is a lot of requests (600+), so it's opt-in
    via --player-history and rate-limited to be polite to the API.
    """
    players = bootstrap["elements"]
    logger.info("Fetching per-player history for %d players...", len(players))

    for i, player in enumerate(players, start=1):
        player_id = player["id"]
        web_name = player["web_name"]
        try:
            history = fetch_player_history(player_id)
            _save_json(history, "player_history", f"player_{player_id}_{web_name}.json")
        except requests.RequestException as e:
            logger.warning("Failed to fetch player %s (%s): %s", player_id, web_name, e)
        time.sleep(delay)

        if i % 50 == 0:
            logger.info("Progress: %d/%d players fetched", i, len(players))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch raw data from the FPL API.")
    parser.add_argument(
        "--player-history",
        action="store_true",
        help="Also fetch detailed per-gameweek history for every player (slow, ~600+ requests).",
    )
    args = parser.parse_args()

    bootstrap = fetch_bootstrap_static()
    fetch_fixtures()

    if args.player_history:
        fetch_all_player_histories(bootstrap)

    logger.info("Done.")


if __name__ == "__main__":
    main()
