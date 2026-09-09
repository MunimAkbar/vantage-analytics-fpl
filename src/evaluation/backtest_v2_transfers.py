"""
backtest_v2_transfers.py

Realistic version of backtest_v2.py: instead of rebuilding an optimal
squad from scratch every gameweek, this carries last week's squad
forward and only allows transfers under real FPL rules:

- 1 free transfer earned per gameweek
- up to 5 free transfers can be banked (2024/25 rule)
- each transfer beyond the free allowance costs -4 points

Simplification, stated plainly: squad budget is treated as a constant
£100m each gameweek (assumes sale price == current listed value). Real
FPL sell-price rules (you only keep half of a price rise) aren't
modeled — this makes the realistic version slightly MORE generous than
true reality, not less.

Usage:
    python -m src.evaluation.backtest_v2_transfers --season 2024-25
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
from src.optimization.transfer_selector import select_squad_with_transfers

from configs.fpl_constraints import (
    FREE_TRANSFERS_PER_GAMEWEEK,
    MAX_BANKED_FREE_TRANSFERS,
    MAX_PLAYERS_PER_CLUB,
    SQUAD_BUDGET,
    SQUAD_POSITION_COUNTS,
    TRANSFER_HIT_COST,
)
from src.cleaning.utils import DATA_PROCESSED_DIR
from src.evaluation.backtest_v2 import (
    aggregate_double_gameweeks,
    load_model,
    load_season_features,
    predict_gameweek,
)
from src.optimization.lineup_selector import select_starting_xi
from src.optimization.squad_selector import select_squad

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

"""
def select_squad_with_transfers(
    predictions: pd.DataFrame,
    old_squad_names: set[str],
    free_transfers: int,
    budget: float = SQUAD_BUDGET,
    position_counts: dict[str, int] = SQUAD_POSITION_COUNTS,
    max_per_club: int = MAX_PLAYERS_PER_CLUB,
    hit_cost: float = TRANSFER_HIT_COST,
) -> tuple[pd.DataFrame, int, int]:
    players = predictions.reset_index(drop=True)
    n = len(players)

    prob = pulp.LpProblem("fpl_squad_with_transfers", pulp.LpMaximize)
    select = [pulp.LpVariable(f"select_{i}", cat="Binary") for i in range(n)]

    old_indices = players.index[players["name"].isin(old_squad_names)].tolist()
    num_old_in_pool = len(old_indices)
    missing_from_pool = len(old_squad_names) - num_old_in_pool
    transfers_out_expr = (num_old_in_pool - pulp.lpSum(select[i] for i in old_indices)) + missing_from_pool

    hits = pulp.LpVariable("hits", lowBound=0, cat="Continuous")
    prob += hits >= transfers_out_expr - free_transfers

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

"""
def simulate_season(season: str) -> pd.DataFrame:
    model, feature_columns = load_model()
    season_df = load_season_features(season)
    gameweeks = sorted(season_df["gameweek_id"].unique())

    results = []
    current_squad_names: set[str] = set()
    free_transfers = FREE_TRANSFERS_PER_GAMEWEEK

    for i, gw in enumerate(gameweeks):
        gw_df = season_df[season_df["gameweek_id"] == gw]
        gw_df = aggregate_double_gameweeks(gw_df)
        predicted = predict_gameweek(gw_df, model, feature_columns)

        if i == 0:
            squad = select_squad(predicted, budget=SQUAD_BUDGET,
                                  position_counts=SQUAD_POSITION_COUNTS,
                                  max_per_club=MAX_PLAYERS_PER_CLUB)
            transfers_made, hits_taken = 0, 0
        else:
            try:
                squad, transfers_made, hits_taken = select_squad_with_transfers(
                    predicted, current_squad_names, free_transfers,
                )
            except RuntimeError as e:
                logger.warning("GW%d: transfer-constrained selection failed (%s) — skipping.", gw, e)
                continue

        current_squad_names = set(squad["name"])

        starting_xi, bench = select_starting_xi(squad)
        ranked = starting_xi.sort_values("xPts", ascending=False)
        captain_name = ranked.iloc[0]["name"]

        actual_total = 0
        for _, row in starting_xi.iterrows():
            pts = row["total_points"]
            if row["name"] == captain_name:
                pts *= 2
            actual_total += pts
        actual_total_after_hits = actual_total - (hits_taken * TRANSFER_HIT_COST)

        results.append({
            "gameweek": gw,
            "actual_total_before_hits": actual_total,
            "hits_taken": hits_taken,
            "actual_total_after_hits": actual_total_after_hits,
            "transfers_made": transfers_made,
            "free_transfers_available": free_transfers,
            "captain": captain_name,
            "squad_cost": squad["now_cost"].sum(),
        })

        used_free = min(transfers_made, free_transfers)
        leftover = max(0, free_transfers - used_free)
        free_transfers = min(MAX_BANKED_FREE_TRANSFERS, leftover + FREE_TRANSFERS_PER_GAMEWEEK)

        if gw % 10 == 0 or i == len(gameweeks) - 1:
            running_total = sum(r["actual_total_after_hits"] for r in results)
            logger.info("GW%d done. Running total (after hits): %d", gw, running_total)

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame, season: str) -> None:
    total_before = results["actual_total_before_hits"].sum()
    total_after = results["actual_total_after_hits"].sum()
    total_hits = results["hits_taken"].sum()
    total_transfers = results["transfers_made"].sum()

    print(f"\n{'='*72}")
    print(f"REALISTIC BACKTEST SEASON {season} — V2 (1 free transfer/week, -4 per extra)")
    print(f"{'='*72}")
    print(f"Gameweeks simulated: {len(results)}")
    print(f"Total transfers made across season: {int(total_transfers)}")
    print(f"Total hits taken (paid transfers): {int(total_hits)}  (-{int(total_hits)*4} points)")
    print(f"Total points BEFORE hit deductions: {total_before:.0f}")
    print(f"Total points AFTER hit deductions:  {total_after:.0f}")
    print(f"{'='*72}\n")


def run(season: str) -> pd.DataFrame:
    results = simulate_season(season)
    out_path = DATA_PROCESSED_DIR / f"backtest_v2_transfers_{season}.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved gameweek-by-gameweek results to %s", out_path)
    summarize(results, season)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Realistic transfer-constrained backtest of V2.")
    parser.add_argument("--season", default="2024-25")
    args = parser.parse_args()
    run(args.season)


if __name__ == "__main__":
    main()