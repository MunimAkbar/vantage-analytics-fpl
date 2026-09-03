"""
squad_selector.py

Selects the optimal 15-man FPL squad from V1 (or any future version's)
predictions, respecting real FPL constraints:

- exactly 2 GK, 5 DEF, 5 MID, 3 FWD
- total cost <= budget
- max 3 players from any one club

Usage:
    python -m src.optimization.squad_selector
"""

from __future__ import annotations

import logging

import pandas as pd
import pulp

from configs.fpl_constraints import MAX_PLAYERS_PER_CLUB, SQUAD_BUDGET, SQUAD_POSITION_COUNTS
from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_predictions() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "predictions_v1.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/models/xpts_v1.py first.")
    return pd.read_csv(path)


def select_squad(
    predictions: pd.DataFrame,
    budget: float = SQUAD_BUDGET,
    position_counts: dict[str, int] = SQUAD_POSITION_COUNTS,
    max_per_club: int = MAX_PLAYERS_PER_CLUB,
) -> pd.DataFrame:
    """Solves the 15-man squad selection ILP: maximize total xPts subject
    to budget, position, and club constraints."""

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
        raise RuntimeError(
            f"Squad selection did not find an optimal solution (status: {pulp.LpStatus[status]}). "
            "This usually means the constraints are infeasible — e.g. budget too low for the "
            "required position counts. Check for players with missing/NaN prices or xPts."
        )

    selected_idx = [i for i in range(n) if select[i].value() == 1]
    squad = players.loc[selected_idx].sort_values(
        ["position", "xPts"], ascending=[True, False]
    ).reset_index(drop=True)

    return squad


def summarize(squad: pd.DataFrame) -> None:
    total_cost = squad["now_cost"].sum()
    total_xpts = squad["xPts"].sum()

    print(f"\n{'='*70}")
    print(f"OPTIMAL 15-MAN SQUAD  —  Total cost: £{total_cost:.1f}m  —  Total xPts: {total_xpts:.2f}")
    print(f"{'='*70}")

    for position in ["GK", "DEF", "MID", "FWD"]:
        group = squad[squad["position"] == position]
        print(f"\n{position}:")
        for _, row in group.iterrows():
            print(f"  {row['web_name']:<18} {row['team_name']:<16} £{row['now_cost']:.1f}m   xPts={row['xPts']:.2f}")

    print(f"\n{'='*70}")
    club_counts = squad["team_name"].value_counts()
    over_limit = club_counts[club_counts > MAX_PLAYERS_PER_CLUB]
    if len(over_limit) > 0:
        logger.warning("Club limit violated (should never happen): %s", over_limit.to_dict())
    else:
        logger.info("Club constraint satisfied (max %d per club).", MAX_PLAYERS_PER_CLUB)


def run() -> pd.DataFrame:
    predictions = load_predictions()
    squad = select_squad(predictions)

    out_path = DATA_PROCESSED_DIR / "squad_v1.csv"
    squad.to_csv(out_path, index=False)
    logger.info("Saved squad to %s", out_path)

    summarize(squad)
    return squad


if __name__ == "__main__":
    run()