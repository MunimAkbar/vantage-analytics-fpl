"""
backtest_v2.py

Simulates the full weekly pipeline (predict -> select squad -> select
lineup -> score against actual results) gameweek-by-gameweek across a
real historical season, using V2's trained model.

Methodologically this is genuinely safe: the season used (2024-25) was
V2's held-out TEST season during training (see xpts_v2.py's
walk_forward_split) — V2 never saw this season's outcomes during
training, so this is a true out-of-season simulation, not hindsight.

IMPORTANT CAVEAT (by design, not a bug): this backtest rebuilds an
entirely new optimal squad from scratch every gameweek, with no
transfer limits. That is NOT how real FPL works (1 free transfer/week,
-4 hit for extras) — this measures "how good are V2's weekly picks in
isolation", not "how would a realistically-constrained AI manager have
actually performed all season". The transfer-constrained version is
next.

Usage:
    python -m src.evaluation.backtest_v2 --season 2024-25
"""

from __future__ import annotations

import argparse
import logging

# pyrefly: ignore [missing-import]
import joblib
import pandas as pd

from configs.fpl_constraints import MAX_PLAYERS_PER_CLUB, SQUAD_BUDGET, SQUAD_POSITION_COUNTS
from src.cleaning.utils import DATA_PROCESSED_DIR
from src.optimization.lineup_selector import select_starting_xi
from src.optimization.squad_selector import select_squad

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = DATA_PROCESSED_DIR.parents[1] / "models"

FEATURE_COLUMNS = [
    "avg_minutes_last_n", "avg_points_last_n", "avg_bonus_last_n",
    "avg_bps_last_n", "xG_per90_last_n", "xA_per90_last_n",
    "games_in_window", "value", "was_home",
]


def load_model():
    path = MODELS_DIR / "xpts_v2_ridge.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python -m src.models.xpts_v2 --train' first.")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_columns"]


def load_season_features(season: str) -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "historical_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/features/build_historical_features.py first.")
    df = pd.read_csv(path)
    season_df = df[df["season"] == season]
    if season_df.empty:
        raise ValueError(f"No data found for season '{season}'. Available: {sorted(df['season'].unique())}")
    return season_df


def aggregate_double_gameweeks(gw_df: pd.DataFrame) -> pd.DataFrame:
    """Collapses duplicate (name, gameweek) rows from double gameweeks:
    sums actual points (matches real FPL scoring across both fixtures)
    but keeps one row's features/price/position for squad-building.
    """
    agg = gw_df.groupby("name", as_index=False).agg({
        **{c: "first" for c in gw_df.columns if c not in ("name", "total_points")},
        "total_points": "sum",
    })
    return agg


def predict_gameweek(gw_df: pd.DataFrame, model, feature_columns: list[str]) -> pd.DataFrame:
    df = gw_df.copy()
    df["was_home"] = df["was_home"].astype(int)

    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    X = pd.concat([df[[c for c in FEATURE_COLUMNS if c in df.columns]], position_dummies], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)

    df["xPts"] = model.predict(X)
    df["now_cost"] = df["value"] / 10.0
    df["team_name"] = df["team"]
    return df


def simulate_season(season: str) -> pd.DataFrame:
    model, feature_columns = load_model()
    season_df = load_season_features(season)

    gameweeks = sorted(season_df["gameweek_id"].unique())
    results = []

    for gw in gameweeks:
        gw_df = season_df[season_df["gameweek_id"] == gw]
        gw_df = aggregate_double_gameweeks(gw_df)

        predicted = predict_gameweek(gw_df, model, feature_columns)

        try:
            squad = select_squad(predicted, budget=SQUAD_BUDGET,
                                  position_counts=SQUAD_POSITION_COUNTS,
                                  max_per_club=MAX_PLAYERS_PER_CLUB)
        except RuntimeError as e:
            logger.warning("GW%d: squad selection failed (%s) — skipping this gameweek.", gw, e)
            continue

        starting_xi, bench = select_starting_xi(squad)

        ranked = starting_xi.sort_values("xPts", ascending=False)
        captain_name = ranked.iloc[0]["name"]

        actual_total = 0
        for _, row in starting_xi.iterrows():
            pts = row["total_points"]
            if row["name"] == captain_name:
                pts *= 2
            actual_total += pts

        predicted_total = starting_xi["xPts"].sum() + ranked.iloc[0]["xPts"]

        results.append({
            "gameweek": gw,
            "predicted_total": predicted_total,
            "actual_total": actual_total,
            "captain": captain_name,
            "squad_cost": squad["now_cost"].sum(),
        })

        if gw % 10 == 0 or gw == gameweeks[-1]:
            logger.info("GW%d done. Running actual total so far: %d", gw, sum(r["actual_total"] for r in results))

    return pd.DataFrame(results)


def summarize(results: pd.DataFrame, season: str) -> None:
    total_actual = results["actual_total"].sum()
    total_predicted = results["predicted_total"].sum()
    avg_per_gw = results["actual_total"].mean()

    print(f"\n{'='*72}")
    print(f"BACKTEST SEASON {season} — V2 (fresh-squad-every-week, no transfer limits)")
    print(f"{'='*72}")
    print(f"Gameweeks simulated: {len(results)}")
    print(f"Total predicted points (season): {total_predicted:.1f}")
    print(f"Total ACTUAL points (season):    {total_actual:.1f}")
    print(f"Average actual points per gameweek: {avg_per_gw:.1f}")
    print(f"\nFor rough context: a real FPL manager scoring ~2000-2200 points across a full season")
    print(f"is a solid, above-average result. This backtest is NOT directly comparable to that")
    print(f"figure though — it assumes unlimited free squad changes every week, which no real")
    print(f"manager gets. Treat this as V2's weekly-picking skill in isolation, not a season score.")
    print(f"{'='*72}\n")


def run(season: str) -> pd.DataFrame:
    results = simulate_season(season)
    out_path = DATA_PROCESSED_DIR / f"backtest_v2_{season}.csv"
    results.to_csv(out_path, index=False)
    logger.info("Saved gameweek-by-gameweek backtest results to %s", out_path)
    summarize(results, season)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest V2 across a historical season.")
    parser.add_argument("--season", default="2024-25", help="Season string, e.g. 2024-25")
    args = parser.parse_args()
    run(args.season)


if __name__ == "__main__":
    main()