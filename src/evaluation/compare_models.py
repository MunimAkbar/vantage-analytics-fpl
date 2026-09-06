"""
compare_models.py

Head-to-head comparison of V1 vs V2 for a given Gameweek, plus a running
season-long log.

Usage:
    python -m src.evaluation.compare_models --gameweek 4
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"
SEASON_LOG_PATH = DECISIONS_DIR / "season_comparison.csv"


def load_summary(gameweek, version):
    path = DECISIONS_DIR / f"gw{gameweek}_summary_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run gameweek_review.py --gameweek {gameweek} --version {version} first.")
    return pd.read_csv(path).iloc[0].to_dict()


def compare(gameweek):
    v1 = load_summary(gameweek, "v1")
    v2 = load_summary(gameweek, "v2")

    print(f"\n{'='*72}\nGAMEWEEK {gameweek} — MODEL COMPARISON (V1 vs V2)\n{'='*72}")
    if not v1["fully_complete"] or not v2["fully_complete"]:
        print("⚠  At least one version has starters who haven't played yet.\n")

    print(f"{'Metric':<28}{'V1':>15}{'V2':>15}")
    print(f"{'-'*58}")
    print(f"{'Predicted total':<28}{v1['total_predicted']:>15.2f}{v2['total_predicted']:>15.2f}")
    print(f"{'Actual total':<28}{v1['total_actual']:>15.2f}{v2['total_actual']:>15.2f}")

    winner = None
    if v1["fully_complete"] and v2["fully_complete"]:
        winner = "v1" if v1["total_actual"] > v2["total_actual"] else ("v2" if v2["total_actual"] > v1["total_actual"] else "tie")
        print(f"\nGameweek {gameweek} winner: {winner.upper()}")
    else:
        print("\nWinner: TBD")
    print(f"{'='*72}\n")

    result = {
        "gameweek": gameweek, "v1_actual": v1["total_actual"], "v2_actual": v2["total_actual"],
        "v1_predicted": v1["total_predicted"], "v2_predicted": v2["total_predicted"],
        "winner": winner, "complete": v1["fully_complete"] and v2["fully_complete"],
    }
    _append_to_season_log(result)
    return result


def _append_to_season_log(result):
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    log = pd.read_csv(SEASON_LOG_PATH) if SEASON_LOG_PATH.exists() else pd.DataFrame()
    if not log.empty:
        log = log[log["gameweek"] != result["gameweek"]]
    log = pd.concat([log, pd.DataFrame([result])], ignore_index=True).sort_values("gameweek").reset_index(drop=True)
    log.to_csv(SEASON_LOG_PATH, index=False)

    completed = log[log["complete"] == True]  # noqa: E712
    if not completed.empty:
        v1_wins = (completed["winner"] == "v1").sum()
        v2_wins = (completed["winner"] == "v2").sum()
        ties = (completed["winner"] == "tie").sum()
        print(f"SEASON SO FAR ({len(completed)} GW complete): V1 wins={v1_wins}  V2 wins={v2_wins}  Ties={ties}")
        print(f"Full log: {SEASON_LOG_PATH}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameweek", type=int, required=True)
    args = parser.parse_args()
    compare(args.gameweek)


if __name__ == "__main__":
    main()