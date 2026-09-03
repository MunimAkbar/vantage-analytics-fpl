"""
decision_log.py

Records a permanent, timestamped snapshot of the AI's squad/lineup
decision for a given Gameweek — never overwritten by later pipeline runs.

This is the backbone of the human-vs-AI experiment: without a locked
record of what the AI decided *before* a Gameweek's deadline, there's no
way to later prove the AI wasn't quietly adjusted after seeing results.

Usage:
    python -m src.evaluation.decision_log --gameweek 3
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def load_current_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    squad = pd.read_csv(DATA_PROCESSED_DIR / "squad_v1.csv")
    starting_xi = pd.read_csv(DATA_PROCESSED_DIR / "starting_xi_v1.csv")
    bench = pd.read_csv(DATA_PROCESSED_DIR / "bench_v1.csv")
    return squad, starting_xi, bench


def determine_captaincy(starting_xi: pd.DataFrame) -> tuple[str, str]:
    ranked = starting_xi.sort_values("xPts", ascending=False)
    return ranked.iloc[0]["web_name"], ranked.iloc[1]["web_name"]


def log_decision(gameweek: int) -> Path:
    """Writes a locked snapshot for this gameweek. Raises if one already
    exists — a locked decision should never be silently overwritten.
    """
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DECISIONS_DIR / f"gw{gameweek}_decision.csv"

    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists — a decision for GW{gameweek} is already locked. "
            "This is intentional: the AI's decision must not be changed after the fact. "
            "If this snapshot was a mistake, delete it manually and re-run."
        )

    squad, starting_xi, bench = load_current_outputs()
    captain, vice_captain = determine_captaincy(starting_xi)

    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for _, row in squad.iterrows():
        is_starting = row["id"] in starting_xi["id"].values
        rows.append({
            "gameweek": gameweek,
            "decision_timestamp_utc": timestamp,
            "player_id": row["id"],
            "web_name": row["web_name"],
            "team_name": row["team_name"],
            "position": row["position"],
            "now_cost": row["now_cost"],
            "predicted_xPts": row["xPts"],
            "squad_role": "starting" if is_starting else "bench",
            "is_captain": row["web_name"] == captain,
            "is_vice_captain": row["web_name"] == vice_captain,
        })

    decision = pd.DataFrame(rows)
    decision.to_csv(out_path, index=False)

    logger.info("Locked GW%d decision at %s (%s)", gameweek, timestamp, out_path)
    logger.info("Captain: %s | Vice-captain: %s", captain, vice_captain)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock in the AI's decision for a Gameweek.")
    parser.add_argument("--gameweek", type=int, required=True, help="Gameweek number, e.g. 3")
    args = parser.parse_args()

    log_decision(args.gameweek)


if __name__ == "__main__":
    main()