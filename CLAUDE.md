# Vantage Analytics: context for Claude Code

Read this whole file before doing anything. It covers the project's history,
its rules, its current state, and what to do next. Last updated 27 Nov 2026,
after GW12 was locked (deadline Fri 27 Nov 18:30 UTC), 2026-27 season.
**GW13 is midweek: deadline Tue 1 Dec 18:30 UTC (23:30 Pakistan time).**

Repo: https://github.com/MunimAkbar/vantage-analytics-fpl (public)
The owner works on Windows, in PowerShell, with a Python venv, at
`D:\WorkANDProjects\vantage-analytics`.

---

## 1. What this project is

An AI system that manages a Fantasy Premier League (FPL) team for a full
season. It predicts expected points (xPts) per player, picks a legal 15-man
squad with an integer linear program, picks the starting XI and captain, and
suggests weekly transfers under real FPL rules. The owner copies its decisions
into a real FPL account and runs it against his own human-managed team.

It is also a public portfolio project: GitHub, plus weekly YouTube and
Instagram shorts (squad reveal before a gameweek, predicted vs actual after)
and a longer architecture explainer video.

## 2. Rules that must never be broken

1. **No data leakage.** A prediction for gameweek N may only use data that
   existed before N's deadline. Rolling features use `.shift(1)` before the
   rolling window.
2. **Time-based validation only.** Never random splits. V2 trains on 2020-21
   to 2022-23, validates on 2023-24, tests on 2024-25.
3. **Locks are permanent.** `decision_log.py` refuses to overwrite
   `gw{N}_decision_{version}.csv`. Never delete, edit or regenerate a lock
   file, and never write code that does. A lock made after a deadline is not
   a valid AI decision.
4. **Squad selection is an ILP, not a sort.** Budget, 2/5/5/3 positions, max 3
   per club, valid formations.
5. **Don't change V1's logic mid-season.** The real account follows V1, so
   changing it changes what the scoreboard means. New models are added as new
   versions (v3, and so on) that run beside V1 and V2.
6. **Test against real data before claiming something works**, and report the
   real numbers. Several bugs in this project were only caught that way.
7. **Ask before deleting or overwriting anything in `data/`.** Much of it is
   the experiment's record.

## 3. Architecture

```
Live season:  FPL public API (bootstrap-static, fixtures, element-summary)
  -> src/ingestion/fetch_fpl_data.py          raw JSON, timestamped, in data/raw/
  -> src/cleaning/run_cleaning.py              players, teams, gameweeks, fixtures CSVs
  -> src/cleaning/clean_player_history.py      player_history.csv (per player per GW)
  -> src/features/build_features.py            features.csv
  -> src/models/xpts_v1.py | xpts_v2.py --predict   predictions_v1.csv / predictions_v2.csv
  -> the weekly decision tools (section 5)
  -> src/evaluation/gameweek_review.py, compare_models.py   after the gameweek

Historical:   vaastav/Fantasy-Premier-League GitHub archive, 5 seasons
  -> src/ingestion/fetch_historical_data.py
  -> src/cleaning/clean_historical_data.py     drops "AM" rows, maps GKP -> GK
  -> src/features/build_historical_features.py uses groupby().transform(), NOT .apply()
  -> src/models/xpts_v2.py --train             models/xpts_v2_ridge.joblib
  -> src/evaluation/backtest_v2.py, backtest_v2_transfers.py
```

Configs:
- `configs/fpl_scoring.py`: points by position.
- `configs/fpl_constraints.py`: budget 100.0, positions, formations, max 3 per
  club, 1 free transfer a week, up to 5 banked, -4 per extra transfer.
- `configs/manual_overrides.py`: planned, **does not exist yet** (see section
  7). Meant to force `p_start` for injured players the API hasn't flagged yet,
  applied inside `build_features.py`. Until then, check flagged players by eye.

Two data gotchas already handled:
- FPL's `finished` flag lags results by hours. Use `finished_provisional`.
- Several players share surnames (Davis, Mitchell). Match on player id, or on
  name plus club or price, never surname alone.

## 4. Models

- **V1, rule-based.** Applies the FPL scoring table to `p_start`, rolling
  per-90 xG/xA (last 5 played games), and `fixture_run_difficulty` (average
  FDR of the next 5 fixtures). No training. Known weak spots: the clean-sheet
  probability is a linear proxy off FDR, the bonus term is crude, and it
  over-extrapolates recent hot streaks.
- **V2, Ridge regression** on about 134k historical player-gameweek rows.
  Validation MAE 1.023, test MAE 1.061, against 1.503 for predicting the mean.
  Known gaps: no fixture or opponent features (the historical data has no
  FDR), no availability signal (it will rate an injured player highly), and a
  negative `avg_bonus_last_n` coefficient, likely from collinearity. In
  `--predict` it hard-codes `was_home = 0`.
- **Backtests on 2024-25, V2:** 2260 points when rebuilding the squad freely
  every week (unrealistic), and **1760 points** under real transfer rules (47
  transfers, 9 hits). That is below a good human manager's typical 2000-2200,
  mostly because transfer planning is greedy, one week at a time.

## 5. The weekly tools and the routine

- `sync_squad.py` reads `data/processed/account_squad.txt` (one player per
  line: `name, price, club`, with plain ASCII spellings like `Gross` and
  `E.Le Fee`) and writes `decisions/current_squad.csv`. It's a snapshot of the
  real account, never a lock. `--budget` is squad value plus money in the bank.
- `suggest_transfers.py` starts from whichever is newer: the latest lock or
  the synced squad. Paid hits are capped at 2.
- `lock_squad.py` picks XI, captain and vice-captain from the synced squad and
  V1 predictions, prints the plan, and locks only after the user types `LOCK`.
  It refuses if the gameweek isn't `is_next`, if the deadline has passed, or if
  a lock already exists.
- `show_squad.py` prints a lock. With no `--gameweek` it shows the latest.
- `weekly_report.py` reviews the previous gameweek and prints next week's
  captain and transfers. It errors if the previous gameweek has no lock.
- `transfer_selector.py` holds the shared transfer ILP used by the suggester,
  the report and the transfers backtest.

The routine, the day before a deadline:

1. Fetch **with `--player-history`**, clean, `build_features`, `xpts_v1`.
   Without the flag, the per-player history files aren't refreshed and recent
   form stops at the last full fetch (this happened before GW11). The full
   fetch takes about 20 minutes.
2. Check `gameweeks.csv` shows the right `is_next` gameweek, and that the
   maximum `gameweek_id` in `player_history.csv` is the gameweek just played.
3. Update `account_squad.txt` to match the app (`name, price, club`, sell
   prices from the app), save it, then run `sync_squad`. The sync reads only
   this file, so check its output lists the app's players.
4. Run `suggest_transfers` with the free-transfer count from the app.
5. Make the transfers in the app, update `account_squad.txt`, run `sync_squad`
   again.
6. Run `lock_squad`, check every name matches the app, type `LOCK`. Lock at
   least an hour before the deadline.
7. Set the same XI, captain and vice-captain in the app.

## 6. The record so far

| GW | What happened |
|---|---|
| 3 | V1 locked. Predicted 92.03, actual 56. Captain and VC (Fernandes, Mbeumo) flopped. |
| 4 | V1 and V2 locked. V1 predicted 75.26, actual 52. The same two players flopped again. |
| 5 | Nothing locked (international break). |
| 6 | V1 locked, but the real account differed in 5 players. Not a clean AI week. |
| 7-9 | Nothing locked. The account played the synced squad. GW9 scored about 50. |
| 10 | Not locked: the LOCK prompt was never confirmed. The owner made the 3 suggested transfers (Van Hecke, Tavernier, Gibbs-White in for Mitchell, Lewis-Potter, Gross) and played the suggested captain (Barry, 32 points). The real account scored 70. Treat it as AI-advised, not a locked AI decision. |
| 11 | V1 locked 20 Nov 18:29 UTC, about 16.5h before the deadline. Transfer: Thiago (suspended until 5 Dec) out, Evanilson in, 1 FT, no hit. Captain Barry, VC Evanilson, 3-4-3. Predicted 59.82 (XI + captain). The lock matches the real account squad. Budget 100.4, bank 3.6 after the transfer. **Actual 39** (review: Tavernier 8 best; Barry (C), Haaland, Gibbs-White, Rushworth 2 each). Human team scored 44. |
| 12 | V1 locked 27 Nov 18:05 UTC, only 25 min before the deadline (the routine was started late). Transfer: Mbeumo out, Barcola (Liverpool, 8.1) in, 1 FT, no hit. Captain Tavernier, VC Evanilson, 3-4-3. Predicted 62.56. Davis on the bench with a calf knock (75%). The lock matches the real account squad. Budget 100.4. |

Files in `data/processed/decisions/` as of 27 Nov: `current_squad.csv`
(synced 27 Nov, includes Barcola); `gw11_review_v1`, `gw11_summary_v1`;
`gw12_transfer_suggestion_v1`, `gw12_decision_v1`; `gw3_decision_v1`, `gw3_review_v1`,
`gw3_summary_v1`; `gw4_decision_v1`, `gw4_decision_v2`, with reviews and
summaries for both; `gw5_transfer_suggestion_v1`; `gw6_decision_v1`;
`gw7_transfer_suggestion_v1` (actually the GW10 suggestion, from the old naming
rule); `gw11_transfer_suggestion_v1`; `gw11_decision_v1`. There is also an
old, unversioned `gw3_review.csv`.

`data/records/` (committed) holds `gw10_ai_transfer_suggestion.csv`,
`gw10_synced_squad_pre_transfers.csv`, and `gw11_ai_transfer_suggestion.csv`.
`gw11_decision_v1.csv` is a copy of the GW11 lock. The GW11 suggestion copy is
the final run (after the GW10 player history was fetched); its xPts match the
lock. Added 27 Nov: `gw11_review_v1.csv`, `gw11_summary_v1.csv`,
`gw12_ai_transfer_suggestion.csv`, `gw12_decision_v1.csv`.

Human team points are only noted in the record table so far (GW11: 44). A
committed `data/records/human_vs_ai.csv` is planned but not started.

The local venv was rebuilt on 20 Nov on Python 3.13 (the old Anaconda base was
removed), with the same package versions as before: pandas 3.0.5, numpy
2.5.2, scikit-learn 1.9.0 (the version `xpts_v2_ridge.joblib` was saved
with), pulp 3.3.2.

## 7. Known problems, roughly in priority order

Checked against the files on 20 Nov:

1. Done: `requirements.txt` pins `pulp>=2.7,<4` (PuLP 4.x breaks every
   optimizer).
2. Done: `suggest_transfers.py` names suggestion files from `is_next` in
   `gameweeks.csv`, falling back to last lock + 1 with a warning.
3. **Open:** `scripts/run_gameweek.py` predates the sync and lock workflow. It
   rebuilds a fresh squad and locks it, which is wrong now that a real squad
   exists. Update it to call the new workflow, or remove it. The README marks
   it as legacy.
4. Done: `lock_squad.py` prints the deadline in the PC's local time.
5. Done: `weekly_report.py` uses `load_current_squad()` and the synced budget.
6. Half done: `build_features.py` has `next_n_fixtures_difficulty`, but
   **`apply_manual_overrides` and `configs/manual_overrides.py` don't exist.**
   Build them before relying on overrides.
7. Budget is simplified: one flat number per week. FPL's sell-price rule (you
   keep half of a price rise) isn't modelled.
8. Chips (wildcard, free hit, bench boost, triple captain) aren't modelled.
9. Captain choice ignores correlated risk, such as captain and VC from one club.
10. Transfers are planned one week at a time.

Before trusting anything above, check which files actually exist and contain
these changes. Code given in chat has sometimes not been saved to disk.

## 8. What to do next

**First, the weekly routine.** GW12 is locked. After GW12 finishes, run
`gameweek_review --gameweek 12`, then run the routine in section 5 for GW13.
GW13 is midweek (deadline Tue 1 Dec 18:30 UTC), so there are only about 4 days
between deadlines. Start the routine early on the deadline day, and lock at
least an hour before. Nothing below should get in its way.

**Then the public app**, in this order:

1. **Prediction contract.** One CSV per model per gameweek with these columns:
   `player_id, web_name, team_name, position, now_cost, xPts, p_start,
   next_opponent, next_fdr, fixture_run_difficulty, model_version, gameweek,
   generated_at_utc`. Write an exporter that produces it from V1 and V2. The
   app reads only this file, so models can change without touching the UI.
2. **Streamlit MVP**, read-only: this week's AI squad and captain; the track
   record including the misses; V1 vs V2 vs human; a methodology page; and a
   transfer helper where a visitor types their 15 players, budget and free
   transfers. Label everything as experimental.
3. **Deploy to Streamlit Community Cloud** (free, public repo, about 1 GB of
   memory, sleeps after 12 idle hours). Commit only the small derived files the
   app needs, never `data/raw` or `data/processed`.
4. **Weekly GitHub Actions job** to refresh and publish the predictions file.
   Cache FPL responses: the API is unofficial and undocumented, so never call it
   per visitor. No Premier League logos.
5. **Team-ID import** (`/api/entry/{id}/...`) to replace the manual sync, but
   only after testing on the owner's machine which fields it returns.
6. **Model v3**, as a new version beside V1 and V2: fixture and opponent
   features, an availability or minutes model, gradient boosting, and
   evaluation by season points under transfer rules, not just MAE. Let the live
   scoreboard decide whether it's better.

## 9. How to work with the owner

- Give runnable PowerShell commands. Don't put `<placeholder>` text in
  commands, because PowerShell treats `<` as an operator.
- Before changing a file, show the plan and say which files will change.
- Never touch lock files, and never relock a past gameweek.
- Be honest about results, including bad ones. They are the content.
