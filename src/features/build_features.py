"""
build_features.py

Builds the feature table that src/models/ will consume for V1 (and later
V2+) expected-points prediction. One row per player, features computed
as of the next upcoming gameweek — never using that gameweek's own results.

Usage:
    python -m src.features.build_features
"""

from __future__ import annotations

import logging

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROLLING_WINDOW = 5  # games


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = pd.read_csv(DATA_PROCESSED_DIR / "players.csv")
    teams = pd.read_csv(DATA_PROCESSED_DIR / "teams.csv")
    fixtures = pd.read_csv(DATA_PROCESSED_DIR / "fixtures.csv")
    history = pd.read_csv(DATA_PROCESSED_DIR / "player_history.csv")
    return players, teams, fixtures, history


def rolling_form_features(history: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Per-player rolling averages over their last `window` played gameweeks.

    Only games with minutes > 0 count toward the window, so an unused
    substitute gameweek doesn't dilute a player's recent form.
    """
    played = history[history["minutes"] > 0].copy()
    played = played.sort_values(["player_id", "gameweek_id"])

    def _last_n(group: pd.DataFrame) -> pd.Series:
        recent = group.tail(window)
        minutes_total = recent["minutes"].sum()

        # Per-90 rates guard against divide-by-zero if minutes_total is 0.
        per90 = lambda col: (recent[col].sum() / minutes_total * 90) if minutes_total > 0 else 0.0

        return pd.Series({
            "avg_minutes_last_n": recent["minutes"].mean(),
            "avg_points_last_n": recent["total_points"].mean(),
            "xG_per90_last_n": per90("expected_goals"),
            "xA_per90_last_n": per90("expected_assists"),
            "games_played_last_n": len(recent),
        })

    form = played.groupby("player_id").apply(_last_n).reset_index()
    return form


def start_probability(players: pd.DataFrame, form: pd.DataFrame) -> pd.Series:
    """Simple V1 P(start) estimate.

    Combines FPL's own 'chance of playing' signal (injury/rotation news)
    with recent minutes share. This is the crude version flagged in the
    baseline model spec — a dedicated classifier can replace it later.
    """
    merged = players[["id", "chance_of_playing_this_round", "status"]].merge(
        form[["player_id", "avg_minutes_last_n"]],
        left_on="id",
        right_on="player_id",
        how="left",
    )

    # chance_of_playing_this_round is 0-100 or NaN (NaN usually means "fully fit").
    fitness_prob = merged["chance_of_playing_this_round"].fillna(100) / 100.0

    # Share of a full 90 minutes played recently, as a rotation-risk proxy.
    minutes_prob = (merged["avg_minutes_last_n"].fillna(0) / 90.0).clip(upper=1.0)

    # Players flagged unavailable ('i' injured, 's' suspended) get zeroed out
    # regardless of recent minutes.
    unavailable = merged["status"].isin(["i", "s", "u"])

    p_start = (fitness_prob * 0.5 + minutes_prob * 0.5).where(~unavailable, 0.0)
    return p_start


def next_fixture_difficulty(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Attaches each player's team's next unplayed fixture and its FDR."""
    upcoming = fixtures[fixtures["finished"] == False].copy()  # noqa: E712
    upcoming = upcoming.sort_values("gameweek_id")

    next_for_team = {}
    for _, row in upcoming.iterrows():
        for team_col, opp_col, fdr_col in [
            ("team_h", "team_a", "team_h_difficulty"),
            ("team_a", "team_h", "team_a_difficulty"),
        ]:
            team_id = row[team_col]
            if team_id not in next_for_team:
                next_for_team[team_id] = {
                    "next_gameweek_id": row["gameweek_id"],
                    "next_opponent_team": row[opp_col],
                    "next_fixture_difficulty": row[fdr_col],
                }

    fixture_df = pd.DataFrame.from_dict(next_for_team, orient="index").reset_index()
    fixture_df = fixture_df.rename(columns={"index": "team"})

    return players[["id", "team"]].merge(fixture_df, on="team", how="left")


def build_features() -> pd.DataFrame:
    players, teams, fixtures, history = load_processed()

    form = rolling_form_features(history)
    fixture_info = next_fixture_difficulty(players, teams, fixtures)
    p_start = start_probability(players, form)

    features = players[["id", "web_name", "team_name", "position", "now_cost"]].merge(
        form, left_on="id", right_on="player_id", how="left"
    ).drop(columns=["player_id"])

    features["p_start"] = p_start.values

    features = features.merge(
        fixture_info.drop(columns=["team"]), on="id", how="left"
    )

    # Fill players with no history yet (e.g. new signings) with conservative defaults.
    features[["avg_minutes_last_n", "avg_points_last_n", "xG_per90_last_n",
              "xA_per90_last_n", "games_played_last_n"]] = features[
        ["avg_minutes_last_n", "avg_points_last_n", "xG_per90_last_n",
         "xA_per90_last_n", "games_played_last_n"]
    ].fillna(0)

    return features


def run() -> None:
    features = build_features()
    out_path = DATA_PROCESSED_DIR / "features.csv"
    features.to_csv(out_path, index=False)
    logger.info("Saved %d player feature rows to %s", len(features), out_path)


if __name__ == "__main__":
    run()