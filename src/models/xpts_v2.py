"""
xpts_v2.py

V2 expected-points model: a real regression model (Ridge), trained on
multi-season historical data, using the rolling features from
build_historical_features.py as inputs and actual gameweek points as
the target.

Validation is season-based walk-forward, never random:
    Train:      earliest N-2 seasons
    Validation: second-to-last season
    Test:       most recent season

Usage:
    python -m src.models.xpts_v2 --train
    python -m src.models.xpts_v2 --predict
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.cleaning.utils import DATA_PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = DATA_PROCESSED_DIR.parents[1] / "models"

FEATURE_COLUMNS = [
    "avg_minutes_last_n", "avg_points_last_n", "avg_bonus_last_n",
    "avg_bps_last_n", "xG_per90_last_n", "xA_per90_last_n",
    "games_in_window", "value", "was_home",
]
POSITION_COLUMN = "position"
TARGET_COLUMN = "total_points"


def load_historical_features() -> pd.DataFrame:
    path = DATA_PROCESSED_DIR / "historical_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/features/build_historical_features.py first.")
    return pd.read_csv(path)


def season_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["season"].unique())


def prepare_design_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    df["was_home"] = df["was_home"].astype(int)

    position_dummies = pd.get_dummies(df[POSITION_COLUMN], prefix="pos")
    X = pd.concat([df[FEATURE_COLUMNS], position_dummies], axis=1)
    X = X.fillna(0)

    y = df[TARGET_COLUMN]
    return X, y


def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seasons = season_order(df)
    if len(seasons) < 3:
        raise ValueError(
            f"Need at least 3 seasons for a train/validation/test split, found {len(seasons)}: {seasons}"
        )

    train_seasons = seasons[:-2]
    val_season = seasons[-2]
    test_season = seasons[-1]

    logger.info("Train seasons: %s | Validation: %s | Test: %s", train_seasons, val_season, test_season)

    train_df = df[df["season"].isin(train_seasons)]
    val_df = df[df["season"] == val_season]
    test_df = df[df["season"] == test_season]

    return train_df, val_df, test_df


def train() -> None:
    df = load_historical_features()
    train_df, val_df, test_df = walk_forward_split(df)

    X_train, y_train = prepare_design_matrix(train_df)
    X_val, y_val = prepare_design_matrix(val_df)
    X_test, y_test = prepare_design_matrix(test_df)

    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae = mean_absolute_error(y_test, test_pred)

    naive_pred_val = np.full_like(y_val, fill_value=y_train.mean(), dtype=float)
    naive_mae_val = mean_absolute_error(y_val, naive_pred_val)

    logger.info("Validation MAE: %.3f (naive baseline: %.3f)", val_mae, naive_mae_val)
    logger.info("Test MAE: %.3f", test_mae)

    coef_summary = pd.Series(model.coef_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature coefficients (higher = pushes prediction up):")
    print(coef_summary.to_string())

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "xpts_v2_ridge.joblib"
    joblib.dump({"model": model, "feature_columns": list(X_train.columns)}, model_path)
    logger.info("Saved trained model to %s", model_path)


def predict_current_season() -> pd.DataFrame:
    model_path = MODELS_DIR / "xpts_v2_ridge.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Run 'python -m src.models.xpts_v2 --train' first.")

    bundle = joblib.load(model_path)
    model, feature_columns = bundle["model"], bundle["feature_columns"]

    features = pd.read_csv(DATA_PROCESSED_DIR / "features.csv")

    df = features.copy()
    df["was_home"] = 0
    if "value" not in df.columns:
        df["value"] = df["now_cost"]

    position_dummies = pd.get_dummies(df["position"], prefix="pos")
    X = pd.concat([df[[c for c in FEATURE_COLUMNS if c in df.columns]], position_dummies], axis=1)
    X = X.reindex(columns=feature_columns, fill_value=0)

    df["xPts_v2"] = model.predict(X)

    out_path = DATA_PROCESSED_DIR / "predictions_v2.csv"
    df.sort_values("xPts_v2", ascending=False).to_csv(out_path, index=False)
    logger.info("Saved V2 predictions to %s", out_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or run the V2 regression xPts model.")
    parser.add_argument("--train", action="store_true", help="Train V2 on historical data.")
    parser.add_argument("--predict", action="store_true", help="Predict this season using the trained V2 model.")
    args = parser.parse_args()

    if args.train:
        train()
    if args.predict:
        predict_current_season()
    if not args.train and not args.predict:
        parser.print_help()


if __name__ == "__main__":
    main()