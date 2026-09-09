"""
suggest_transfers.py

Live weekly transfer advisor. Compares your CURRENT squad (from the
most recent locked decision) against fresh predictions, suggests the
best transfer(s) under real FPL rules. Advisory only — never touches
locked files or auto-applies anything.

Usage:
    python -m src.evaluation.suggest_transfers --version v1 --free-transfers 1
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR
from src.optimization.transfer_selector import select_squad_with_transfers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"
LIVE_NAME_COLUMN = "web_name"
SCORE_COLUMN_BY_VERSION = {"v1": "xPts", "v2": "xPts_v2"}


def find_latest_locked_squad(version: str) -> tuple[int, set[str]]:
    files = sorted(DECISIONS_DIR.glob(f"gw*_decision_{version}.csv"))
    if not files:
        raise FileNotFoundError(f"No locked decisions found for {version} in {DECISIONS_DIR}.")

    def gw_num(path):
        return int(path.stem.split("_")[0].replace("gw", ""))

    latest = max(files, key=gw_num)
    gw = gw_num(latest)
    decision = pd.read_csv(latest)
    return gw, set(decision[LIVE_NAME_COLUMN])


def load_live_predictions(version: str) -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / f"predictions_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run this week's prediction step first.")
    df = pd.read_csv(path)
    score_col = SCORE_COLUMN_BY_VERSION[version]
    return df.rename(columns={score_col: "xPts", LIVE_NAME_COLUMN: "name"})


def suggest(version: str, free_transfers: int, current_gameweek: int | None) -> None:
    if current_gameweek is None:
        current_gameweek, old_squad_names = find_latest_locked_squad(version)
    else:
        path = DECISIONS_DIR / f"gw{current_gameweek}_decision_{version}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        decision = pd.read_csv(path)
        old_squad_names = set(decision[LIVE_NAME_COLUMN])

    logger.info("Current squad taken from GW%d locked decision (%s).", current_gameweek, version)
    predictions = load_live_predictions(version)

    new_squad, transfers_made, hits_taken = select_squad_with_transfers(
        predictions, old_squad_names, free_transfers, name_col="name",
    )

    new_names = set(new_squad["name"])
    players_out = old_squad_names - new_names
    players_in = new_names - old_squad_names

    print(f"\n{'='*70}")
    print(f"TRANSFER SUGGESTION ({version.upper()}) — based on GW{current_gameweek} squad")
    print(f"{'='*70}")
    print(f"Free transfers you told me you have: {free_transfers}")
    print(f"Suggested transfers: {transfers_made}  |  Hits: {hits_taken}"
          f"  ({'-' + str(hits_taken * 4) + ' points' if hits_taken else 'no cost'})")

    if not players_out:
        print("\nNo beneficial transfer found — current squad is already optimal.")
    else:
        print(f"\n{'OUT':<20}{'IN':<20}")
        print(f"{'-'*40}")
        for position in ["GK", "DEF", "MID", "FWD"]:
            outs_this_pos = sorted(
                n for n in players_out
                if not predictions[predictions["name"] == n].empty
                and predictions[predictions["name"] == n]["position"].values[0] == position
            )
            ins_this_pos = sorted(
                n for n in players_in
                if not new_squad[new_squad["name"] == n].empty
                and new_squad[new_squad["name"] == n]["position"].values[0] == position
            )
            for o, i in zip(outs_this_pos, ins_this_pos):
                out_xpts = predictions[predictions["name"] == o]["xPts"].values[0]
                in_xpts = new_squad[new_squad["name"] == i]["xPts"].values[0]
                print(f"{o:<20}{i:<20}  ({out_xpts:.2f} -> {in_xpts:.2f})")

    print(f"{'='*70}\n")

    out_path = DECISIONS_DIR / f"gw{current_gameweek + 1}_transfer_suggestion_{version}.csv"
    new_squad.to_csv(out_path, index=False)
    logger.info("Saved suggested squad to %s (review only — nothing locked).", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest transfers based on your current locked squad.")
    parser.add_argument("--version", default="v1", choices=list(SCORE_COLUMN_BY_VERSION.keys()))
    parser.add_argument("--free-transfers", type=int, required=True)
    parser.add_argument("--current-gameweek", type=int, default=None)
    args = parser.parse_args()
    suggest(args.version, args.free_transfers, args.current_gameweek)


if __name__ == "__main__":
    main()