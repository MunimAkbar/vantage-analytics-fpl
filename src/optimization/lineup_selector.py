"""
lineup_selector.py

Selects Starting XI, Captain, Vice-captain, Bench from squad_{version}.csv.

Usage:
    python -m src.optimization.lineup_selector --version v1
    python -m src.optimization.lineup_selector --version v2
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import pulp

from configs.fpl_constraints import VALID_FORMATIONS
from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_squad(version):
    path = DATA_PROCESSED_DIR / f"squad_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run squad_selector.py --version {version} first.")
    return pd.read_csv(path)


def select_starting_xi(squad):
    players = squad.reset_index(drop=True)
    n = len(players)
    prob = pulp.LpProblem("fpl_starting_xi", pulp.LpMaximize)
    start = [pulp.LpVariable(f"start_{i}", cat="Binary") for i in range(n)]

    prob += pulp.lpSum(start[i] * players.loc[i, "xPts"] for i in range(n))
    prob += pulp.lpSum(start) == 11

    formation_vars = [pulp.LpVariable(f"formation_{f}", cat="Binary") for f in range(len(VALID_FORMATIONS))]
    prob += pulp.lpSum(formation_vars) == 1

    for pos_idx, position in enumerate(["GK", "DEF", "MID", "FWD"]):
        idxs = players.index[players["position"] == position]
        prob += pulp.lpSum(start[i] for i in idxs) == pulp.lpSum(
            formation_vars[f] * VALID_FORMATIONS[f][pos_idx] for f in range(len(VALID_FORMATIONS))
        )

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Starting XI selection failed (status: {pulp.LpStatus[status]}).")

    starting_idx = [i for i in range(n) if start[i].value() == 1]
    bench_idx = [i for i in range(n) if i not in starting_idx]
    starting_xi = players.loc[starting_idx].sort_values("xPts", ascending=False).reset_index(drop=True)
    bench = players.loc[bench_idx].sort_values("xPts", ascending=False).reset_index(drop=True)
    return starting_xi, bench


def assign_captaincy(starting_xi):
    ranked = starting_xi.sort_values("xPts", ascending=False)
    return ranked.iloc[0]["web_name"], ranked.iloc[1]["web_name"]


def summarize(starting_xi, bench, captain, vice_captain, version):
    total_xi_xpts = starting_xi["xPts"].sum()
    captain_bonus = starting_xi[starting_xi["web_name"] == captain]["xPts"].values[0]
    total_with_captaincy = total_xi_xpts + captain_bonus

    print(f"\n{'='*70}\nSTARTING XI ({version.upper()})\n{'='*70}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        group = starting_xi[starting_xi["position"] == position]
        if len(group) == 0:
            continue
        print(f"\n{position}:")
        for _, row in group.iterrows():
            tag = "  (C)" if row["web_name"] == captain else ("  (VC)" if row["web_name"] == vice_captain else "")
            print(f"  {row['web_name']:<18} {row['team_name']:<16} xPts={row['xPts']:.2f}{tag}")

    print(f"\n{'='*70}\nBENCH (in play order)\n{'='*70}")
    for i, (_, row) in enumerate(bench.iterrows(), start=1):
        print(f"  {i}. {row['web_name']:<18} {row['team_name']:<16} {row['position']:<4} xPts={row['xPts']:.2f}")

    print(f"\n{'='*70}")
    print(f"Starting XI total xPts (before captaincy): {total_xi_xpts:.2f}")
    print(f"Captain: {captain} (+{captain_bonus:.2f})  |  Vice-captain: {vice_captain}")
    print(f"Projected Gameweek total (with captaincy): {total_with_captaincy:.2f}")
    print(f"{'='*70}")


def run(version="v1"):
    squad = load_squad(version)
    starting_xi, bench = select_starting_xi(squad)
    captain, vice_captain = assign_captaincy(starting_xi)

    starting_xi.to_csv(DATA_PROCESSED_DIR / f"starting_xi_{version}.csv", index=False)
    bench.to_csv(DATA_PROCESSED_DIR / f"bench_{version}.csv", index=False)
    logger.info("Saved starting_xi_%s.csv and bench_%s.csv", version, version)

    summarize(starting_xi, bench, captain, vice_captain, version)
    return {"starting_xi": starting_xi, "bench": bench, "captain": captain, "vice_captain": vice_captain}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()
    run(args.version)


if __name__ == "__main__":
    main()