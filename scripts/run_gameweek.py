"""
run_gameweek.py

Runs the full weekly pipeline in one command instead of ~12 separate
steps: refresh data, build features, predict with both V1 and V2,
optimize squad + lineup for both, and lock the decision for a given
gameweek.

Usage:
    python -m scripts.run_gameweek --gameweek 5
    python -m scripts.run_gameweek --gameweek 5 --skip-fetch   (data already fresh)
    python -m scripts.run_gameweek --gameweek 5 --no-lock      (predict/optimize only, don't lock yet)
    python -m scripts.run_gameweek --gameweek 5 --v1-only      (skip V2, e.g. before it's trained)
"""

from __future__ import annotations

import argparse
import logging

from src.cleaning import clean_player_history, run_cleaning
from src.evaluation import decision_log
from src.features import build_features
from src.ingestion import fetch_fpl_data
from src.models import xpts_v1, xpts_v2
from src.optimization import lineup_selector, squad_selector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def step(name: str, fn, *args, **kwargs):
    print(f"\n{'#'*70}\n# STEP: {name}\n{'#'*70}")
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error("Step '%s' failed: %s", name, e)
        raise SystemExit(1)


def run(gameweek: int, skip_fetch: bool = False, lock: bool = True, versions=("v1", "v2")) -> None:
    if not skip_fetch:
        bootstrap = step("Fetch bootstrap + fixtures", fetch_fpl_data.fetch_bootstrap_static)
        step("Fetch fixtures", fetch_fpl_data.fetch_fixtures)
        step("Fetch per-player history (slow)", fetch_fpl_data.fetch_all_player_histories, bootstrap)
    else:
        logger.info("Skipping fetch (--skip-fetch) — using whatever is already in data/raw/.")

    step("Clean bootstrap + fixtures", run_cleaning.main)
    step("Clean player history", clean_player_history.run)
    step("Build features", build_features.run)

    if "v1" in versions:
        step("Predict V1", xpts_v1.run)
    if "v2" in versions:
        step("Predict V2", xpts_v2.predict_current_season)

    for version in versions:
        step(f"Select squad ({version})", squad_selector.run, version)
        step(f"Select lineup ({version})", lineup_selector.run, version)

    if lock:
        for version in versions:
            try:
                step(f"Lock decision GW{gameweek} ({version})", decision_log.log_decision, gameweek, version)
            except SystemExit:
                logger.warning(
                    "GW%d (%s) appears to already be locked — skipping (not overwriting).",
                    gameweek, version,
                )
    else:
        logger.info("Skipping decision locking (--no-lock). Run decision_log.py manually when ready.")

    print(f"\n{'='*70}")
    print(f"GAMEWEEK {gameweek} PIPELINE COMPLETE — versions run: {list(versions)}")
    print(f"{'='*70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full weekly FPL pipeline in one command.")
    parser.add_argument("--gameweek", type=int, required=True, help="Gameweek number, e.g. 5")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip re-fetching raw data from the FPL API.")
    parser.add_argument("--no-lock", action="store_true", help="Predict/optimize only, don't lock the decision.")
    parser.add_argument("--v1-only", action="store_true", help="Only run V1 (skip V2).")
    args = parser.parse_args()

    versions = ("v1",) if args.v1_only else ("v1", "v2")

    run(
        gameweek=args.gameweek,
        skip_fetch=args.skip_fetch,
        lock=not args.no_lock,
        versions=versions,
    )


if __name__ == "__main__":
    main()