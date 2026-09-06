"""
decision_log.py

Locks a permanent, timestamped snapshot of a model version's decision
for a given Gameweek — never overwritten.

Usage:
    python -m src.evaluation.decision_log --gameweek 4 --version v1
    python -m src.evaluation.decision_log --gameweek 4 --version v2
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def load_current_outputs(version):
    squad = pd.read_csv(DATA_PROCESSED_DIR / f"squad_{version}.csv")
    starting_xi = pd.read_csv(DATA_PROCESSED_DIR / f"starting_xi_{version}.csv")
    bench = pd.read_csv(DATA_PROCESSED_DIR / f"bench_{version}.csv")
    return squad, starting_xi, bench


def determine_captaincy(starting_xi):
    ranked = starting_xi.sort_values("xPts", ascending=False)
    return ranked.iloc[0]["web_name"], ranked.iloc[1]["web_name"]


def log_decision(gameweek, version):
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DECISIONS_DIR / f"gw{gameweek}_decision_{version}.csv"

    if out_path.exists():
        raise FileExistsError(f"{out_path} already exists — decision already locked. Not overwriting.")

    squad, starting_xi, bench = load_current_outputs(version)
    captain, vice_captain = determine_captaincy(starting_xi)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []
    for _, row in squad.iterrows():
        is_starting = row["id"] in starting_xi["id"].values
        rows.append({
            "gameweek": gameweek, "model_version": version, "decision_timestamp_utc": timestamp,
            "player_id": row["id"], "web_name": row["web_name"], "team_name": row["team_name"],
            "position": row["position"], "now_cost": row["now_cost"], "predicted_xPts": row["xPts"],
            "squad_role": "starting" if is_starting else "bench",
            "is_captain": row["web_name"] == captain, "is_vice_captain": row["web_name"] == vice_captain,
        })

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("Locked GW%d (%s) decision at %s (%s)", gameweek, version, timestamp, out_path)
    logger.info("Captain: %s | Vice-captain: %s", captain, vice_captain)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    log_decision(args.gameweek, args.version)


if __name__ == "__main__":
    main()