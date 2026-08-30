"""
run_cleaning.py

Single entrypoint for the cleaning stage: run this after ingestion to
turn everything in data/raw/ into feature-ready tables in data/processed/.

Usage:
    python -m src.cleaning.run_cleaning
"""

from __future__ import annotations

from src.cleaning import clean_bootstrap, clean_fixtures


def main() -> None:
    clean_bootstrap.run()
    clean_fixtures.run()


if __name__ == "__main__":
    main()