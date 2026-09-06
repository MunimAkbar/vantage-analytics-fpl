"""
squad_selector.py

Selects the optimal 15-man FPL squad from a given model version's
predictions. Version-aware: works for predictions_{version}.csv.

Usage:
    python -m src.optimization.squad_selector --version v1
    python -m src.optimization.squad_selector --version v2
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
# pyrefly: ignore [missing-import]
import pulp

from configs.fpl_constraints import MAX_PLAYERS_PER_CLUB, SQUAD_BUDGET, SQUAD_POSITION_COUNTS
from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCORE_COLUMN_BY_VERSION = {"v1": "xPts", "v2": "xPts_v2"}


def load_predictions(version: str) -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / f"predictions_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the {version} model first.")
    df = pd.read_csv(path)
    score_col = SCORE_COLUMN_BY_VERSION[version]
    if score_col not in df.columns:
        raise ValueError(f"Expected score column '{score_col}' not found in {path}.")
    return df.rename(columns={score_col: "xPts"})


def select_squad(predictions, budget=SQUAD_BUDGET, position_counts=SQUAD_POSITION_COUNTS, max_per_club=MAX_PLAYERS_PER_CLUB):
    players = predictions.reset_index(drop=True)
    n = len(players)
    prob = pulp.LpProblem("fpl_squad_selection", pulp.LpMaximize)
    select = [pulp.LpVariable(f"select_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(select[i] * players.loc[i, "xPts"] for i in range(n))
    prob += pulp.lpSum(select[i] * players.loc[i, "now_cost"] for i in range(n)) <= budget

    for position, count in position_counts.items():
        idxs = players.index[players["position"] == position]
        prob += pulp.lpSum(select[i] for i in idxs) == count

    for club in players["team_name"].unique():
        idxs = players.index[players["team_name"] == club]
        prob += pulp.lpSum(select[i] for i in idxs) <= max_per_club

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Squad selection failed (status: {pulp.LpStatus[status]}).")

    selected_idx = [i for i in range(n) if select[i].value() == 1]
    return players.loc[selected_idx].sort_values(["position", "xPts"], ascending=[True, False]).reset_index(drop=True)


def summarize(squad, version):
    total_cost = squad["now_cost"].sum()
    total_xpts = squad["xPts"].sum()
    print(f"\n{'='*70}\nOPTIMAL 15-MAN SQUAD ({version.upper()})  —  £{total_cost:.1f}m  —  xPts: {total_xpts:.2f}\n{'='*70}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        group = squad[squad["position"] == position]
        print(f"\n{position}:")
        for _, row in group.iterrows():
            print(f"  {row['web_name']:<18} {row['team_name']:<16} £{row['now_cost']:.1f}m   xPts={row['xPts']:.2f}")
    club_counts = squad["team_name"].value_counts()
    over_limit = club_counts[club_counts > MAX_PLAYERS_PER_CLUB]
    if len(over_limit) > 0:
        logger.warning("Club limit violated: %s", over_limit.to_dict())
    else:
        logger.info("Club constraint satisfied.")


def run(version="v1"):
    predictions = load_predictions(version)
    squad = select_squad(predictions)
    out_path = DATA_PROCESSED_DIR / f"squad_{version}.csv"
    squad.to_csv(out_path, index=False)
    logger.info("Saved squad to %s", out_path)
    summarize(squad, version)
    return squad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1", choices=list(SCORE_COLUMN_BY_VERSION.keys()))
    args = parser.parse_args()
    run(args.version)


if __name__ == "__main__":
    main()