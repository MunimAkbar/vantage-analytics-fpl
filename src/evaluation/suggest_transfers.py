"""
suggest_transfers.py

Live weekly transfer advisor. Compares the squad you currently OWN against
this week's fresh predictions and suggests the best transfer(s) under real
FPL rules (free transfers, -4 per extra, capped at 2 paid hits).

"The squad you currently own" is whichever is NEWER of:
  - the latest locked decision (gw{N}_decision_{version}.csv), or
  - your synced real-account squad (current_squad.csv, made by sync_squad.py)

Advisory only: it never touches locked decision files.

Usage:
    python -m src.evaluation.suggest_transfers --version v1 --free-transfers 3
    python -m src.evaluation.suggest_transfers --version v1 --free-transfers 3 --budget 100.4
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from configs.fpl_constraints import SQUAD_BUDGET
from src.cleaning.utils import DATA_PROCESSED_DIR
from src.optimization.transfer_selector import select_squad_with_transfers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"
SNAPSHOT_PATH = DECISIONS_DIR / "current_squad.csv"
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


def load_current_squad(version: str) -> tuple[int, set[str], float, str]:
    """Returns (latest_locked_gameweek, squad_names, total_budget, source_description).

    Uses the synced account squad only if it is newer than the latest lock,
    so a stale sync can never silently override a more recent locked decision.
    """
    gw, locked_names = find_latest_locked_squad(version)
    locked_path = DECISIONS_DIR / f"gw{gw}_decision_{version}.csv"
    locked_ts = pd.to_datetime(pd.read_csv(locked_path)["decision_timestamp_utc"].iloc[0], utc=True)

    if SNAPSHOT_PATH.exists():
        snap = pd.read_csv(SNAPSHOT_PATH)
        snap_ts = pd.to_datetime(snap["synced_at_utc"].iloc[0], utc=True)
        if snap_ts > locked_ts:
            budget = SQUAD_BUDGET
            if "budget_total" in snap.columns and pd.notna(snap["budget_total"].iloc[0]):
                budget = float(snap["budget_total"].iloc[0])
            source = f"your synced account squad (synced {snap_ts:%Y-%m-%d %H:%M} UTC)"
            return gw, set(snap[LIVE_NAME_COLUMN]), budget, source

    return gw, locked_names, SQUAD_BUDGET, f"the locked GW{gw} decision"


def load_live_predictions(version: str) -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / f"predictions_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run this week's prediction step first.")
    df = pd.read_csv(path)
    score_col = SCORE_COLUMN_BY_VERSION[version]
    return df.rename(columns={score_col: "xPts", LIVE_NAME_COLUMN: "name"})


def suggest(version: str, free_transfers: int, current_gameweek: int | None, budget_override: float | None) -> None:
    if current_gameweek is None:
        current_gameweek, old_squad_names, budget, source = load_current_squad(version)
    else:
        path = DECISIONS_DIR / f"gw{current_gameweek}_decision_{version}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found.")
        old_squad_names = set(pd.read_csv(path)[LIVE_NAME_COLUMN])
        budget = SQUAD_BUDGET
        source = f"the locked GW{current_gameweek} decision"

    if budget_override is not None:
        budget = budget_override

    predictions = load_live_predictions(version)

    new_squad, transfers_made, hits_taken = select_squad_with_transfers(
        predictions, old_squad_names, free_transfers, name_col="name", budget=budget,
    )

    new_names = set(new_squad["name"])
    players_out = old_squad_names - new_names
    players_in = new_names - old_squad_names

    print(f"\n{'='*70}")
    print(f"TRANSFER SUGGESTION ({version.upper()})")
    print(f"{'='*70}")
    print(f"Starting from: {source}")
    print(f"Total budget assumed: £{budget:.1f}m  |  Free transfers: {free_transfers}")
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
    parser = argparse.ArgumentParser(description="Suggest transfers based on the squad you currently own.")
    parser.add_argument("--version", default="v1", choices=list(SCORE_COLUMN_BY_VERSION.keys()))
    parser.add_argument("--free-transfers", type=int, required=True)
    parser.add_argument("--current-gameweek", type=int, default=None)
    parser.add_argument("--budget", type=float, default=None,
                         help="Total budget (squad value + in the bank). Overrides the synced value.")
    args = parser.parse_args()
    suggest(args.version, args.free_transfers, args.current_gameweek, args.budget)


if __name__ == "__main__":
    main()