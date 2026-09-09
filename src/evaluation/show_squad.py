"""
show_squad.py

Displays an already-locked squad decision in a clean, readable format —
exactly what you'd copy into a real FPL account by hand. Reads existing
files only, never touches the pipeline or re-runs anything.

Usage:
    python -m src.evaluation.show_squad --version v1
    python -m src.evaluation.show_squad --version v1 --gameweek 4
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.cleaning.utils import DATA_PROCESSED_DIR

DECISIONS_DIR = DATA_PROCESSED_DIR / "decisions"


def find_latest_gameweek(version: str) -> int:
    files = sorted(DECISIONS_DIR.glob(f"gw*_decision_{version}.csv"))
    if not files:
        raise FileNotFoundError(f"No locked decisions found for {version} in {DECISIONS_DIR}.")

    def gw_num(path):
        return int(path.stem.split("_")[0].replace("gw", ""))

    return max(gw_num(f) for f in files)


def show(version: str, gameweek: int | None) -> None:
    if gameweek is None:
        gameweek = find_latest_gameweek(version)

    path = DECISIONS_DIR / f"gw{gameweek}_decision_{version}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")

    squad = pd.read_csv(path)

    starting = squad[squad["squad_role"] == "starting"]
    bench = squad[squad["squad_role"] == "bench"].sort_values("predicted_xPts", ascending=False)
    captain = squad[squad["is_captain"]]["web_name"].values[0]
    vice = squad[squad["is_vice_captain"]]["web_name"].values[0]
    total_cost = squad["now_cost"].sum()

    print(f"\n{'='*66}")
    print(f"  GAMEWEEK {gameweek} SQUAD ({version.upper()}) — locked, ready to copy into FPL")
    print(f"{'='*66}")
    print(f"  Total squad value: £{total_cost:.1f}m\n")

    print("  STARTING XI")
    print(f"  {'-'*62}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        group = starting[starting["position"] == position]
        if group.empty:
            continue
        print(f"  {position}:")
        for _, row in group.iterrows():
            tag = "  (C)" if row["web_name"] == captain else ("  (VC)" if row["web_name"] == vice else "")
            print(f"    {row['web_name']:<20} {row['team_name']:<15} £{row['now_cost']:.1f}m{tag}")

    print(f"\n  BENCH (in substitute order)")
    print(f"  {'-'*62}")
    for i, (_, row) in enumerate(bench.iterrows(), start=1):
        print(f"    {i}. {row['web_name']:<18} {row['team_name']:<15} {row['position']:<4} £{row['now_cost']:.1f}m")

    print(f"\n  Captain: {captain}")
    print(f"  Vice-captain: {vice}")
    print(f"{'='*66}\n")
    print(f"  File location: {path}")
    print(f"  (open in Excel/Sheets if you'd rather browse the raw table)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Display a locked squad decision for manual replication.")
    parser.add_argument("--version", default="v1", choices=["v1", "v2"])
    parser.add_argument("--gameweek", type=int, default=None, help="Defaults to the latest locked gameweek.")
    args = parser.parse_args()
    show(args.version, args.gameweek)


if __name__ == "__main__":
    main()