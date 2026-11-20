"""
lock_squad.py

Locks your squad for the next gameweek, using the squad you actually hold in
the FPL app (synced with sync_squad.py). It picks the best starting XI,
captain, vice-captain and bench from this week's V1 predictions, shows you
the plan, asks you to confirm, then locks it with decision_log (timestamped,
never overwritten).

Safety checks before anything is written:
  - the gameweek must be the NEXT gameweek, and its deadline must not have passed
  - your synced squad must exist
  - predictions must exist (a warning is shown if they look stale)
  - a lock for that gameweek must not already exist

V1 only: your real account follows the V1 squad, so locking it as V2 would
mislabel the record.

Usage:
    python -m src.evaluation.lock_squad --gameweek 10
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR
from src.evaluation import decision_log
from src.evaluation.suggest_transfers import DECISIONS_DIR, SNAPSHOT_PATH
from src.optimization.lineup_selector import select_starting_xi

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

VERSION = "v1"

def check_gameweek_and_deadline(gameweek: int, skip_checks: bool) -> None:
    path = DATA_PROCESSED_DIR / "gameweeks.csv"
    if not path.exists():
        raise SystemExit("gameweeks.csv not found — run the fetch and run_cleaning steps first.")
    gws = pd.read_csv(path)
    row = gws[gws["gameweek_id"] == gameweek]
    if row.empty:
        raise SystemExit(f"GW{gameweek} not found in gameweeks.csv.")

    deadline = pd.to_datetime(row["deadline_time"].iloc[0], utc=True)
    now = datetime.now(timezone.utc)

    problems = []
    nxt = gws[gws["is_next"] == True]  # noqa: E712
    if not nxt.empty and int(nxt["gameweek_id"].iloc[0]) != gameweek:
        problems.append(
            f"GW{int(nxt['gameweek_id'].iloc[0])} is the next gameweek, not GW{gameweek}. "
            "Predictions are built for the next gameweek."
        )
    if now > deadline:
        problems.append(
            f"The GW{gameweek} deadline passed at {deadline:%Y-%m-%d %H:%M} UTC — "
            "a lock now would not be a pre-deadline decision."
        )

    if problems:
        print("\nProblems found:")
        for p in problems:
            print(f"  - {p}")
        if not skip_checks:
            print("\nNothing was locked. (Use --skip-checks only if you are sure.)")
            raise SystemExit(1)
        print("  ⚠ --skip-checks given: continuing anyway.")

    hours = (deadline - now).total_seconds() / 3600
    local = deadline.tz_convert(datetime.now().astimezone().tzinfo)  # this PC's own time zone
    print(f"GW{gameweek} deadline: {deadline:%a %d %b %H:%M} UTC / {local:%a %d %b %H:%M} your local time (UTC{local:%z})"
          f"  ({hours:.1f}h from now)")


def build_squad() -> tuple[pd.DataFrame, pd.Timestamp]:
    if not SNAPSHOT_PATH.exists():
        raise SystemExit("No synced squad found. Run sync_squad first so the system matches your app.")
    snap = pd.read_csv(SNAPSHOT_PATH)
    synced_at = pd.to_datetime(snap["synced_at_utc"].iloc[0], utc=True)

    pred_path = DATA_PROCESSED_DIR / f"predictions_{VERSION}.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found — run build_features and xpts_v1 first.")
    age_hours = (datetime.now(timezone.utc).timestamp() - pred_path.stat().st_mtime) / 3600
    if age_hours > 24:
        print(f"⚠ Predictions are {age_hours:.0f}h old. Refresh the data and rerun xpts_v1 before locking.")
    else:
        print(f"Predictions generated {age_hours:.1f}h ago.")

    preds = pd.read_csv(pred_path)
    squad = snap[["id", "web_name", "team_name", "position"]].merge(
        preds[["id", "now_cost", "xPts"]], on="id", how="left"
    )
    missing = squad[squad["xPts"].isna()]
    if not missing.empty:
        raise SystemExit(f"These players have no prediction: {', '.join(missing['web_name'])}. Refresh and rebuild.")
    return squad, synced_at


def print_plan(starting_xi, bench, captain, vice, squad) -> None:
    formation = "-".join(str((starting_xi['position'] == p).sum()) for p in ["DEF", "MID", "FWD"])
    print(f"\n{'='*66}\n  PLAN — formation {formation}\n{'='*66}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        for r in starting_xi[starting_xi["position"] == position].itertuples():
            tag = "  (C)" if r.web_name == captain else ("  (VC)" if r.web_name == vice else "")
            print(f"  {position:<4}{r.web_name:<18}{r.team_name:<16}xPts {r.xPts:>5.2f}{tag}")
    print("\n  BENCH (substitute order)")
    for i, r in enumerate(bench.itertuples(), start=1):
        print(f"    {i}. {r.web_name:<18}{r.team_name:<16}{r.position}")
    print(f"\n  Squad value: £{squad['now_cost'].sum():.1f}m")

    players_path = DATA_PROCESSED_DIR / "players.csv"
    if players_path.exists():
        pl = pd.read_csv(players_path)
        flags = squad[["id", "web_name"]].merge(pl[["id", "status", "news"]], on="id", how="left")
        flags = flags[(flags["status"].fillna("a") != "a") | flags["news"].fillna("").ne("")]
        if not flags.empty:
            print("\n  ⚠ Availability flags in your squad:")
            for r in flags.itertuples():
                print(f"    {r.web_name}: status {r.status} — {r.news}")

    suggestions = sorted(DECISIONS_DIR.glob(f"gw*_transfer_suggestion_{VERSION}.csv"), key=lambda p: p.stat().st_mtime)
    if suggestions:
        latest = suggestions[-1]
        age = (datetime.now(timezone.utc).timestamp() - latest.stat().st_mtime) / 3600
        suggested = set(pd.read_csv(latest)["name"])
        mine = set(squad["web_name"])
        print(f"\n  Versus the AI's latest transfer suggestion ({age:.0f}h old):")
        if suggested == mine:
            print("    Your squad matches it exactly.")
        else:
            print(f"    AI suggested but you don't have: {', '.join(sorted(suggested - mine)) or '—'}")
            print(f"    You have but the AI didn't pick:  {', '.join(sorted(mine - suggested)) or '—'}")
    print(f"{'='*66}")


def run(gameweek: int, skip_checks: bool, assume_yes: bool) -> None:
    lock_path = DECISIONS_DIR / f"gw{gameweek}_decision_{VERSION}.csv"
    if lock_path.exists():
        raise SystemExit(f"GW{gameweek} is already locked ({lock_path.name}). Not overwriting.")

    check_gameweek_and_deadline(gameweek, skip_checks)
    squad, synced_at = build_squad()
    print(f"Using the squad synced at {synced_at:%Y-%m-%d %H:%M} UTC. "
          "If your app squad has changed since, update account_squad.txt and re-run sync_squad first.")

    starting_xi, bench = select_starting_xi(squad)
    ranked = starting_xi.sort_values("xPts", ascending=False)
    captain, vice = ranked.iloc[0]["web_name"], ranked.iloc[1]["web_name"]
    print_plan(starting_xi, bench, captain, vice, squad)

    if not assume_yes:
        answer = input(f"\nLock this as GW{gameweek}'s decision? This cannot be undone. Type LOCK to confirm: ")
        if answer.strip() != "LOCK":
            print("Cancelled — nothing was locked.")
            raise SystemExit(0)

    squad.sort_values(["position", "xPts"], ascending=[True, False]).to_csv(DATA_PROCESSED_DIR / f"squad_{VERSION}.csv", index=False)
    starting_xi.to_csv(DATA_PROCESSED_DIR / f"starting_xi_{VERSION}.csv", index=False)
    bench.to_csv(DATA_PROCESSED_DIR / f"bench_{VERSION}.csv", index=False)

    try:
        path = decision_log.log_decision(gameweek, VERSION)
    except FileExistsError as e:
        raise SystemExit(str(e))
    print(f"\nLocked GW{gameweek}: {path}")
    print("View it any time with: python -m src.evaluation.show_squad --version v1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock the squad you hold in the app for the next gameweek.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--skip-checks", action="store_true", help="Bypass the next-gameweek/deadline checks.")
    parser.add_argument("--yes", action="store_true", help="Skip the LOCK confirmation prompt.")
    args = parser.parse_args()
    run(args.gameweek, args.skip_checks, args.yes)


if __name__ == "__main__":
    main()