# Vantage Analytics — Full Project Handoff

This document is a complete state dump of an in-progress project, written
so a different AI assistant (or a new session) can pick it up with zero
prior context. Read this fully before making any changes or suggestions.

---

## 1. The project, in one paragraph

An autonomous Fantasy Premier League (FPL) decision-making system that
predicts player performance, builds an optimal squad under real FPL
constraints, and manages transfers — with the eventual goal of running a
season-long "AI vs human" experiment where the AI's decisions are locked
before each Gameweek deadline and never adjusted after seeing results.
It is also intended as an open-source portfolio project (GitHub +
YouTube/Instagram content), branded **Vantage Analytics**.

## 2. Non-negotiable methodology rules

These have been enforced throughout and should not be relaxed:

1. **No data leakage.** Any prediction for Gameweek N must only use data
   that existed before Gameweek N's deadline. Every rolling feature is
   built with `.shift(1)` before any rolling window, so the *current*
   row's own result is never included in its own features.
2. **Time-based validation only.** Never random train/test splits.
   Walk-forward by season: train on earliest seasons, validate on the
   next, test on the most recent.
3. **Decisions are locked, not adjustable.** Once a Gameweek's squad
   decision is written to disk (`decision_log.py`), it must never be
   overwritten — the script raises `FileExistsError` if you try. This is
   what makes the "AI vs human" comparison credible instead of hindsight.
4. **Squad selection is constrained optimization (ILP), not top-N by
   score.** Budget, position counts (2 GK/5 DEF/5 MID/3 FWD), and a
   max-3-players-per-club rule are hard constraints solved via PuLP, not
   heuristics.

## 3. Full architecture / data flow

```
Live season (current year):
  FPL official API (bootstrap-static, fixtures, element-summary)
       ↓
  src/ingestion/fetch_fpl_data.py
       ↓
  src/cleaning/{clean_bootstrap, clean_fixtures, clean_player_history}.py
       ↓
  src/features/build_features.py   (rolling form, fixture-run difficulty)
       ↓
  src/models/{xpts_v1, xpts_v2}.py   (predict expected points)
       ↓
  src/optimization/{squad_selector, lineup_selector}.py   (ILP)
       ↓
  src/evaluation/decision_log.py   (LOCK the decision, timestamped)
       ↓ (once gameweek finishes)
  src/evaluation/gameweek_review.py   (predicted vs actual)
       ↓
  src/evaluation/compare_models.py   (V1 vs V2 head-to-head, running log)

Historical data (multi-season, for training/backtesting):
  vaastav/Fantasy-Premier-League GitHub archive (community dataset)
       ↓
  src/ingestion/fetch_historical_data.py
       ↓
  src/cleaning/clean_historical_data.py
       ↓
  src/features/build_historical_features.py
       ↓
  src/models/xpts_v2.py --train   (Ridge regression, walk-forward validated)
       ↓
  src/evaluation/backtest_v2.py            (unconstrained: fresh squad every week)
  src/evaluation/backtest_v2_transfers.py  (realistic: 1 free transfer/week, -4 hits)

Orchestration:
  scripts/run_gameweek.py   (chains the whole live-season pipeline in one command)
```

## 4. Every file that exists, what it does, and its status

### Ingestion
- `src/ingestion/fetch_fpl_data.py` — pulls current-season data from the
  official FPL API. `bootstrap-static` (all players/teams/gameweeks),
  `fixtures` (with FDR — Fixture Difficulty Rating), and optionally
  `--player-history` (per-player per-gameweek history, ~600+ API calls,
  slow, rate-limited). ✅ Working, tested against real API.
- `src/ingestion/fetch_historical_data.py` — pulls multi-season
  gameweek-by-gameweek data from `vaastav/Fantasy-Premier-League` on
  GitHub (`raw.githubusercontent.com/.../data/{season}/gws/merged_gw.csv`).
  Default: 2020-21 through 2024-25 (5 seasons). ✅ Tested, ~133k rows.

### Cleaning
- `src/cleaning/utils.py` — shared helpers: `latest_raw_file()` (always
  grabs the most recent timestamped raw pull), `load_json()`,
  `save_processed()`. `DATA_RAW_DIR` / `DATA_PROCESSED_DIR` constants.
- `src/cleaning/clean_bootstrap.py` — raw bootstrap JSON →
  `players.csv`, `teams.csv`, `gameweeks.csv`. Joins in position names
  and team names (raw data only has integer IDs).
- `src/cleaning/clean_fixtures.py` — raw fixtures JSON → `fixtures.csv`.
  **Important gotcha discovered**: FPL's `finished` flag lags the real
  result by hours (official BPS/bonus confirmation pass) —
  `finished_provisional` is what should be treated as "the match
  actually happened, real scores are in." Both flags are kept in the
  output; downstream code should prefer `finished_provisional`.
- `src/cleaning/run_cleaning.py` — runs `clean_bootstrap` + `clean_fixtures`.
- `src/cleaning/clean_player_history.py` — raw per-player element-summary
  JSON → `player_history.csv` (one row per player per gameweek, current
  season only).
- `src/cleaning/clean_historical_data.py` — combines multi-season raw
  CSVs → `historical_gameweeks.csv`. **Two real data-quality fixes
  applied here, found by inspection, not assumption**:
  - 2021-22 season used `"GKP"` for part of the season instead of
    `"GK"` — normalized.
  - 2024-25 introduced `"AM"` (Assistant Manager) — a real FPL feature
    where you can pick an actual PL manager as a squad slot, scored on
    entirely different rules (team results, not player stats). These
    rows are **excluded entirely**, not relabeled — they are not
    players and would corrupt a player-points regression.

### Features
- `src/features/build_features.py` — current-season feature table.
  Rolling last-5-played-games form (`avg_minutes_last_n`,
  `xG_per90_last_n`, `xA_per90_last_n`, etc.), a crude `p_start`
  estimate (FPL's own `chance_of_playing` blended with recent minutes
  share), and **`fixture_run_difficulty`**: average FDR across a
  player's team's **next 5 fixtures** (not just the next match) — this
  was a deliberate fix partway through, since a squad is *held* across
  multiple gameweeks, not rebuilt weekly, so picking on next-week's
  matchup alone was myopic. Output: `features.csv`.
- `src/features/build_historical_features.py` — same idea, applied
  across historical multi-season data instead of one live snapshot.
  Every feature per (player, season, gameweek) row uses only **prior**
  gameweeks within that same season (via `.shift(1)` then rolling).
  **Known pandas gotcha found here**: `groupby(...).apply()` silently
  drops the grouping columns in some pandas versions — this file
  deliberately avoids `.apply()` and uses `groupby(...).transform()`
  instead. Output: `historical_features.csv`.

### Models
- `configs/fpl_scoring.py` — FPL's official point values by position
  (goals, assists, clean sheets, etc.) — hardcoded game constants.
- `configs/fpl_constraints.py` — squad budget (£100m), position counts,
  valid formations, max-3-per-club, and **transfer rules**
  (`FREE_TRANSFERS_PER_GAMEWEEK=1`, `MAX_BANKED_FREE_TRANSFERS=5`,
  `TRANSFER_HIT_COST=4` — confirmed via web search: FPL raised the
  banked-transfer cap from 2 to 5 starting 2024/25).
- `src/models/xpts_v1.py` — **V1: rule-based formula, no ML.** Takes
  `p_start`, per-90 xG/xA, and `fixture_run_difficulty`, applies them
  through the actual FPL scoring table (goals/assists/clean
  sheets/bonus, position-weighted). Fully explainable, zero training.
  **Known weak spots, explicitly flagged, not yet fixed:** the
  clean-sheet probability formula is a crude linear proxy off FDR
  alone (no real attack/defense strength model), and the bonus-points
  term is a similarly crude proxy off combined goal+assist involvement.
- `src/models/xpts_v2.py` — **V2: real Ridge regression**, trained on
  5 seasons of historical data. Walk-forward split: train on
  2020-21/21-22/22-23, validate on 2023-24, test on 2024-25. **Real
  results: Validation MAE 1.023, Test MAE 1.061, vs. 1.503 for a naive
  "always predict the mean" baseline** — genuinely beats naive, modest
  but real improvement. `--train` fits and saves the model to
  `models/xpts_v2_ridge.joblib`; `--predict` applies it to the current
  season's `features.csv`. **Known caveat**: `avg_bonus_last_n` has a
  counterintuitive negative coefficient in the trained model, almost
  certainly multicollinearity with `avg_points_last_n` (bonus points
  are a subset of total points) — flagged, not fixed. **Also**: V2's
  feature set doesn't include fixture difficulty at all (the historical
  dataset doesn't carry FDR), so V2 is currently blind to
  opponent/fixture strength — V1 has this, V2 doesn't. This is an
  honest asymmetry between the two models, not a bug.

### Optimization
- `src/optimization/squad_selector.py` — ILP (via PuLP) that picks the
  optimal 15-man squad maximizing total predicted points subject to
  budget/position/club constraints. **Version-aware**: reads
  `predictions_{version}.csv`, writes `squad_{version}.csv`, works
  identically for v1 (`xPts` column) or v2 (`xPts_v2` column).
- `src/optimization/lineup_selector.py` — given a 15-man squad, ILP-picks
  the best valid starting XI formation (from `VALID_FORMATIONS`),
  assigns captain (highest xPts starter, points double) and
  vice-captain (second highest), orders the bench. Version-aware.

### Evaluation
- `src/evaluation/decision_log.py` — locks a timestamped, permanent
  snapshot of a version's squad/XI/captain decision for a gameweek.
  Refuses to overwrite an existing lock (`FileExistsError`) — this is
  the mechanism that enforces "no hindsight" for the whole project.
- `src/evaluation/gameweek_review.py` — compares a locked decision
  against actual results once a gameweek finishes. **Correctly handles
  partial gameweeks**: checks `finished_provisional` per fixture (see
  cleaning gotcha above) and excludes any starter whose match hasn't
  happened yet from the totals, rather than wrongly scoring them as 0.
- `src/evaluation/compare_models.py` — head-to-head V1 vs V2 for a given
  gameweek, plus a running `season_comparison.csv` log (win/loss/tie per
  gameweek, accumulating across the season).
- `src/evaluation/backtest_v2.py` — simulates V2 gameweek-by-gameweek
  across a full historical season (2024-25, the genuine held-out test
  season — V2 never trained on it). **Rebuilds an entirely new optimal
  squad from scratch every week — no transfer limits.** Real result:
  **2260 total actual points across 38 gameweeks.** Explicitly caveated
  in the output as NOT comparable to a real manager's season score,
  since real managers don't get unlimited free squad rebuilds.
- `src/evaluation/backtest_v2_transfers.py` — the realistic version:
  carries the squad forward week to week, allows only 1 free transfer
  (banking up to 5), -4 points per extra transfer. Real result:
  **1760 points after hit deductions** (1796 before), 47 total
  transfers across the season, only 9 hits taken (at GW15, 29, 34) when
  the predicted gain justified the cost. This is ~500 points below the
  unconstrained version — an honest, important finding: most of the
  unconstrained version's advantage was just unlimited squad churn, not
  genuinely better picks. 1760/season sits *below* a solid human
  manager's typical 2000-2200 range — meaning V2's current one-week
  greedy transfer logic likely underperforms a good human right now.
  **Simplification stated in the code**: squad budget is treated as a
  constant £100m each week (assumes sale price == current value); real
  FPL's sell-price-on-profit rule isn't modeled, making this slightly
  more generous than true reality, not less.

### Orchestration
- `scripts/run_gameweek.py` — chains the full live-season pipeline
  (fetch → clean → features → predict both versions → optimize both →
  lock both) into one command. Calls functions directly (not
  subprocess) so failures are clear and fast. `--skip-fetch`,
  `--no-lock`, `--v1-only` flags. Tested: happy path, infeasible-ILP
  error handling, re-lock safety (warns, doesn't crash or overwrite).

## 5. Real results so far (this season, live)

- **GW3** (first fully reviewed gameweek, V1 only — V2 didn't exist
  yet): predicted 92.03, actual 56.00. Big miss, and diagnosable: V1's
  top two picks (Fernandes, Mbeumo — captain/VC) delivered far below
  prediction while the cheapest pick (Isak) outperformed. Pattern:
  **V1 over-extrapolates recent hot streaks and under-values emerging
  form** (a known weakness of pure rolling-average features).
- **GW4**: locked for both V1 and V2 before deadline. Awaiting results
  — this is the first genuine head-to-head data point.

## 6. Known open issues / honest caveats (do not silently "fix" without discussion)

- V1's clean-sheet and bonus-point formulas are crude proxies, flagged
  as weak spots since they were written.
- V2's `avg_bonus_last_n` has a counterintuitive negative coefficient
  (likely multicollinearity) — noted, not resolved.
- V2 has no fixture-difficulty awareness at all (V1 does) — an honest
  asymmetry when comparing the two models' picks.
- The transfer-constrained backtest uses a one-week-greedy transfer
  decision (not true multi-gameweek lookahead planning) — this is
  likely why its season score (1760) underperforms a good human
  manager; smarter transfer planning is the clear next lever.
- Budget/sell-price simplification in both backtest scripts (see above).
- A repeated operational failure mode throughout this project: files
  given in chat sometimes weren't actually saved to disk by the user,
  causing `ModuleNotFoundError`/`ImportError` a few times. Always
  verify a file actually exists (`dir`/`ls`) before assuming code from
  a prior conversation turn is present.

## 7. What's next (in rough priority order, not yet built)

1. **Smarter transfer planning** — multi-gameweek lookahead instead of
   greedy one-week decisions (e.g. "hold this transfer, next week's
   opportunity is better") — the clear lever suggested by the backtest
   result above.
2. **Fix V1's crude clean-sheet/bonus proxies** — replace with a real
   attack/defense-strength (Poisson-style) model for clean sheets.
3. **Give V2 fixture-difficulty awareness** — would need to source
   historical FDR data (the vaastav repo may have a per-season
   `fixtures.csv` with FDR; unconfirmed, needs checking) to bring V2 to
   parity with V1 on this feature.
4. **V3+**: tree-based models (Random Forest/Gradient Boosting), per
   the original planned model progression (rule-based → regression →
   trees → XGBoost/LightGBM → ensembles).
5. **Monte Carlo simulation** — probabilistic outcomes instead of point
   predictions, useful for captain/differential decisions.
6. **Dashboard** (Streamlit or similar) — public-facing view of current
   squad, predictions, AI-vs-human leaderboard.
7. **Content automation** — LLM-generated weekly reports/scripts from
   the locked decisions and reviews (LLM explains decisions, never
   makes the numeric predictions itself — this separation was a
   deliberate original design decision).

## 8. Project structure on disk

```
vantage-analytics/
├── data/
│   ├── raw/              (gitignored — bootstrap/, fixtures/, player_history/, historical/)
│   └── processed/        (gitignored — all cleaned CSVs + decisions/ subfolder)
├── models/                (trained model artifacts, e.g. xpts_v2_ridge.joblib)
├── configs/               fpl_scoring.py, fpl_constraints.py
├── src/
│   ├── ingestion/         fetch_fpl_data.py, fetch_historical_data.py
│   ├── cleaning/          utils.py, clean_bootstrap.py, clean_fixtures.py,
│   │                      run_cleaning.py, clean_player_history.py,
│   │                      clean_historical_data.py
│   ├── features/          build_features.py, build_historical_features.py
│   ├── models/            xpts_v1.py, xpts_v2.py
│   ├── optimization/      squad_selector.py, lineup_selector.py
│   └── evaluation/        decision_log.py, gameweek_review.py,
│                          compare_models.py, backtest_v2.py,
│                          backtest_v2_transfers.py
├── scripts/               run_gameweek.py
├── requirements.txt        pandas, numpy, requests, python-dotenv, pulp,
│                          scikit-learn, joblib
├── AGENTS.md              (context file for AI coding agents — should be
│                          kept in sync with this document)
└── README.md
```

## 9. How to resume from a clean checkout

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Live season data
python -m src.ingestion.fetch_fpl_data
python -m src.ingestion.fetch_fpl_data --player-history
python -m src.cleaning.run_cleaning
python -m src.cleaning.clean_player_history
python -m src.features.build_features

# Historical data (only needed once, or to refresh)
python -m src.ingestion.fetch_historical_data
python -m src.cleaning.clean_historical_data
python -m src.features.build_historical_features
python -m src.models.xpts_v2 --train

# Weekly cycle (before a gameweek deadline)
python -m scripts.run_gameweek --gameweek <N>

# After a gameweek finishes
python -m src.evaluation.gameweek_review --gameweek <N> --version v1
python -m src.evaluation.gameweek_review --gameweek <N> --version v2
python -m src.evaluation.compare_models --gameweek <N>
```

---

**Instructions for whichever AI picks this up next**: preserve the
methodology rules in Section 2 without exception. Don't silently
"clean up" the caveats in Section 6 — they're documented tradeoffs, not
bugs, and some (like the budget simplification) are intentional and
should be discussed with the user before changing. When adding new
model versions or features, follow the existing pattern: build it,
test it against real data (not just made-up synthetic data) before
handing it over, and report actual numbers rather than assuming
correctness.
