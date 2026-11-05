"""
sync_squad.py

Records the squad you actually hold in your real FPL account, so the transfer
tools start from what you own instead of the last locked AI decision.

This is a SNAPSHOT, not a lock. It never touches or replaces a locked decision
file, and it is not a "before the deadline" record. suggest_transfers uses it
only while it is newer than your latest locked decision.

Input: a plain text file (default: data/processed/account_squad.txt),
one player per line:   name[, price[, club]]
Use plain ASCII spellings. Names match accent-insensitively. Price and club
are optional and only used to tell apart players who share a surname.

Usage:
    python -m src.evaluation.sync_squad --budget 100.4
"""

from __future__ import annotations

import argparse
import difflib
import logging
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from configs.fpl_constraints import MAX_PLAYERS_PER_CLUB, SQUAD_POSITION_COUNTS
from src.cleaning.utils import DATA_PROCESSED_DIR
from src.evaluation.suggest_transfers import DECISIONS_DIR, SNAPSHOT_PATH, find_latest_locked_squad

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_INPUT = DATA_PROCESSED_DIR / "account_squad.txt"
PRICE_TOLERANCE = 0.3  # players.csv prices can lag the app by a day or two


def normalize(text) -> str:
    # Escapes used (not literal characters) so this file works whatever
    # encoding your editor saves it in.
    text = str(text).replace("\u00df", "ss").replace("\u00f8", "o").replace("\u00d8", "O").replace("\u00e6", "ae")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.casefold().strip()


def parse_squad_file(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Create it with one player per line.")
    entries = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        line = raw.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        price, club = None, None
        if len(parts) > 1 and parts[1]:
            try:
                price = float(parts[1])
            except ValueError:
                raise ValueError(f"Line {line_no}: '{parts[1]}' is not a valid price.")
        if len(parts) > 2 and parts[2]:
            club = parts[2]
        entries.append({"name": parts[0], "price": price, "club": club})
    return entries


def match_player(entry: dict, players: pd.DataFrame) -> pd.Series:
    wanted = normalize(entry["name"])
    cands = players[players["_norm"] == wanted]

    if entry["club"]:
        club = normalize(entry["club"])
        cands = cands[cands["team_name"].map(normalize).str.contains(club, regex=False)]

    if len(cands) > 1 and entry["price"] is not None:
        near = cands[(cands["now_cost"] - entry["price"]).abs() <= PRICE_TOLERANCE]
        if len(near) >= 1:
            cands = near

    if len(cands) == 1:
        return cands.iloc[0]

    if cands.empty:
        close = difflib.get_close_matches(wanted, players["_norm"].unique().tolist(), n=3, cutoff=0.6)
        shown = sorted(set(players[players["_norm"].isin(close)]["web_name"]))
        hint = f" Did you mean: {', '.join(shown)}?" if shown else ""
        raise ValueError(f"No player found matching '{entry['name']}'.{hint}")

    options = "; ".join(f"{r.web_name} ({r.team_name}, £{r.now_cost:.1f}m)" for r in cands.itertuples())
    raise ValueError(f"'{entry['name']}' is ambiguous — add a price and/or club on that line. Candidates: {options}")


def validate_squad(squad: pd.DataFrame) -> list[str]:
    problems = []
    if len(squad) != 15:
        problems.append(f"Expected 15 players, found {len(squad)}.")
    if squad["id"].duplicated().any():
        problems.append("The same player appears more than once.")
    counts = squad["position"].value_counts().to_dict()
    for pos, need in SQUAD_POSITION_COUNTS.items():
        if counts.get(pos, 0) != need:
            problems.append(f"Expected {need} {pos}, found {counts.get(pos, 0)}.")
    for club, n in squad["team_name"].value_counts().items():
        if n > MAX_PLAYERS_PER_CLUB:
            problems.append(f"{n} players from {club} (max {MAX_PLAYERS_PER_CLUB}).")
    return problems


def sync(input_path: Path, budget: float | None) -> None:
    players = pd.read_csv(DATA_PROCESSED_DIR / "players.csv")
    players["_norm"] = players["web_name"].map(normalize)

    entries = parse_squad_file(input_path)
    matched, errors = [], []
    for entry in entries:
        try:
            matched.append(match_player(entry, players))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        print(f"\nCould not sync — fix these in {input_path}:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    squad = pd.DataFrame(matched)
    problems = validate_squad(squad)
    if problems:
        print("\nThat doesn't look like a legal FPL squad:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    out = squad[["id", "web_name", "team_name", "position", "now_cost"]].copy()
    out["synced_at_utc"] = datetime.now(timezone.utc).isoformat()
    out["budget_total"] = budget if budget is not None else float("nan")
    DECISIONS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(SNAPSHOT_PATH, index=False)

    print(f"\n{'='*66}")
    print("  SYNCED ACCOUNT SQUAD")
    print(f"{'='*66}")
    for position in ["GK", "DEF", "MID", "FWD"]:
        group = squad[squad["position"] == position]
        names = ", ".join(f"{r.web_name} (£{r.now_cost:.1f}m)" for r in group.itertuples())
        print(f"  {position}: {names}")
    print(f"\n  Squad value at current prices: £{squad['now_cost'].sum():.1f}m")
    if budget is None:
        print("  ⚠ No --budget given — suggestions will assume £100.0m total, which may be wrong.")
        print("    Budget = squad value + 'in the bank', both from the app's Transfers page.")
    else:
        print(f"  Total budget recorded: £{budget:.1f}m")

    flagged = squad[(squad.get("status", "a") != "a") | squad.get("news", pd.Series("", index=squad.index)).fillna("").ne("")]
    if not flagged.empty:
        print("\n  Availability flags:")
        for r in flagged.itertuples():
            print(f"    {r.web_name}: status {getattr(r, 'status', '?')} — {getattr(r, 'news', '')}")

    try:
        gw, locked_names = find_latest_locked_squad("v1")
        synced_names = set(out["web_name"])
        dropped, added = sorted(locked_names - synced_names), sorted(synced_names - locked_names)
        print(f"\n  Compared with your locked GW{gw} (v1) decision:")
        print(f"    In the system's lock but not your account: {', '.join(dropped) or '—'}")
        print(f"    In your account but not the lock:          {', '.join(added) or '—'}")
    except FileNotFoundError:
        pass

    print(f"\n  Saved to {SNAPSHOT_PATH}")
    print("  (a snapshot only — your locked decision files were not touched)")
    print(f"{'='*66}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync your real FPL account squad into the system.")
    parser.add_argument("--file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--budget", type=float, default=None,
                         help="Total budget = squad value + in the bank (from the app's Transfers page).")
    args = parser.parse_args()
    sync(args.file, args.budget)


if __name__ == "__main__":
    main()