"""
transfer_selector.py

Selects the best new squad given an EXISTING squad and real FPL transfer
constraints (1 free transfer/week, banking up to 5, -4 points per extra
transfer). Shared by both backtest_v2_transfers.py and suggest_transfers.py.
"""

from __future__ import annotations

import pandas as pd
import pulp

from configs.fpl_constraints import (
    MAX_PLAYERS_PER_CLUB,
    SQUAD_BUDGET,
    SQUAD_POSITION_COUNTS,
    TRANSFER_HIT_COST,
)


def select_squad_with_transfers(
    predictions: pd.DataFrame,
    old_squad_names: set[str],
    free_transfers: int,
    budget: float = SQUAD_BUDGET,
    position_counts: dict[str, int] = SQUAD_POSITION_COUNTS,
    max_per_club: int = MAX_PLAYERS_PER_CLUB,
    hit_cost: float = TRANSFER_HIT_COST,
    name_col: str = "name",
    max_hits: int | None = 2,
) -> tuple[pd.DataFrame, int, int]:
    """Solves the transfer-constrained squad selection ILP.

    max_hits caps the number of paid transfers (hits) the solver is
    allowed to recommend in a single week. Without this, the ILP can
    legally decide that tearing up 8 players for a +2 xPts gain is
    worthwhile — technically correct per the objective but completely
    unrealistic for a real FPL manager. Default of 2 reflects the
    practical maximum any rational manager would consider in one week.
    Set to None to remove the cap (used in backtesting only).
    """
    players = predictions.reset_index(drop=True)
    n = len(players)

    prob = pulp.LpProblem("fpl_squad_with_transfers", pulp.LpMaximize)
    select = [pulp.LpVariable(f"select_{i}", cat="Binary") for i in range(n)]

    old_indices = players.index[players[name_col].isin(old_squad_names)].tolist()
    num_old_in_pool = len(old_indices)
    missing_from_pool = len(old_squad_names) - num_old_in_pool
    transfers_out_expr = (num_old_in_pool - pulp.lpSum(select[i] for i in old_indices)) + missing_from_pool

    hits = pulp.LpVariable("hits", lowBound=0, cat="Continuous")
    prob += hits >= transfers_out_expr - free_transfers
    if max_hits is not None:
        prob += hits <= max_hits

    prob += pulp.lpSum(select[i] * players.loc[i, "xPts"] for i in range(n)) - hit_cost * hits
    prob += pulp.lpSum(select[i] * players.loc[i, "now_cost"] for i in range(n)) <= budget

    for position, count in position_counts.items():
        idxs = players.index[players["position"] == position]
        prob += pulp.lpSum(select[i] for i in idxs) == count

    for club in players["team_name"].unique():
        idxs = players.index[players["team_name"] == club]
        prob += pulp.lpSum(select[i] for i in idxs) <= max_per_club

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Transfer-constrained squad selection failed (status: {pulp.LpStatus[status]}).")

    selected_idx = [i for i in range(n) if select[i].value() == 1]
    new_squad = players.loc[selected_idx].sort_values(
        ["position", "xPts"], ascending=[True, False]
    ).reset_index(drop=True)

    kept_old = sum(1 for i in old_indices if select[i].value() == 1)
    transfers_made = len(old_squad_names) - kept_old
    hits_taken = max(0, transfers_made - free_transfers)

    return new_squad, transfers_made, hits_taken