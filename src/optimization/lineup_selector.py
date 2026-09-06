"""
lineup_selector.py

Given the optimal 15-man squad (from squad_selector.py), selects:
  - Starting XI (best valid formation)
  - Captain (highest xPts starter — points doubled)
  - Vice-captain (second highest xPts starter — backup if captain doesn't play)
  - Bench order (4 non-starters, ranked by xPts)

Usage:
    python -m src.optimization.lineup_selector
"""

from __future__ import annotations

import logging

import pandas as pd
# pyrefly: ignore [missing-import]
import pulp

from configs.fpl_constraints import VALID_FORMATIONS
from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_squad() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "squad_v1.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/optimization/squad_selector.py first.")
    return pd.read_csv(path)


def select_starting_xi(squad: pd.DataFrame) -> pd.DataFrame:
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


def assign_captaincy(starting_xi: pd.DataFrame) -> tuple[str, str]:
    ranked = starting_xi.sort_values("xPts", ascending=False)
    captain = ranked.iloc[0]["web_name"]
    vice_captain = ranked.iloc[1]["web_name"]
    return captain, vice_captain


def summarize(starting_xi: pd.DataFrame, bench: pd.DataFrame, captain: str, vice_captain: str) -> None:
    total_xi_xpts = starting_xi["xPts"].sum()
    captain_bonus = starting_xi[starting_xi["web_name"] == captain]["xPts"].values[0]
    total_with_captaincy = total_xi_xpts + captain_bonus

    print(f"\n{'='*70}")
    print("STARTING XI")
    print(f"{'='*70}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        group = starting_xi[starting_xi["position"] == position]
        if len(group) == 0:
            continue
        print(f"\n{position}:")
        for _, row in group.iterrows():
            tag = ""
            if row["web_name"] == captain:
                tag = "  (C)"
            elif row["web_name"] == vice_captain:
                tag = "  (VC)"
            print(f"  {row['web_name']:<18} {row['team_name']:<16} xPts={row['xPts']:.2f}{tag}")

    print(f"\n{'='*70}")
    print("BENCH (in play order)")
    print(f"{'='*70}")
    for i, (_, row) in enumerate(bench.iterrows(), start=1):
        print(f"  {i}. {row['web_name']:<18} {row['team_name']:<16} {row['position']:<4} xPts={row['xPts']:.2f}")

    print(f"\n{'='*70}")
    print(f"Starting XI total xPts (before captaincy): {total_xi_xpts:.2f}")
    print(f"Captain: {captain} (+{captain_bonus:.2f} bonus for doubled points)")
    print(f"Vice-captain: {vice_captain}")
    print(f"Projected Gameweek total (with captaincy): {total_with_captaincy:.2f}")
    print(f"{'='*70}")


def run() -> dict:
    squad = load_squad()
    starting_xi, bench = select_starting_xi(squad)
    captain, vice_captain = assign_captaincy(starting_xi)

    starting_xi.to_csv(DATA_PROCESSED_DIR / "starting_xi_v1.csv", index=False)
    bench.to_csv(DATA_PROCESSED_DIR / "bench_v1.csv", index=False)
    logger.info("Saved starting_xi_v1.csv and bench_v1.csv")

    summarize(starting_xi, bench, captain, vice_captain)

    return {"starting_xi": starting_xi, "bench": bench, "captain": captain, "vice_captain": vice_captain}


if __name__ == "__main__":
    run()