"""
utils.py

Shared helpers for the cleaning layer.
"""

from __future__ import annotations

import json
from pathlib import Path

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATA_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def latest_raw_file(subfolder: str, prefix: str) -> Path:
    """Finds the most recently modified raw file matching a prefix."""
    folder = DATA_RAW_DIR / subfolder
    matches = sorted(folder.glob(f"{prefix}*.json"), key=lambda p: p.stat().st_mtime)

    if not matches:
        raise FileNotFoundError(
            f"No raw files matching '{prefix}*' found in {folder}. "
            "Run src/ingestion/fetch_fpl_data.py first."
        )

    return matches[-1]


def load_json(path: Path) -> dict | list:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_processed(df, filename: str) -> Path:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    return out_path