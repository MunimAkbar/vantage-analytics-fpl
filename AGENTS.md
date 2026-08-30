# Vantage Analytics — Agent Context

Read this before starting any task in this repo. It explains what the
project is, why it's structured the way it is, and the rules that must
never be broken regardless of what a specific task asks for.

## What this project is

An autonomous Fantasy Premier League (FPL) decision-making system. It
predicts player performance, selects an FPL squad under real constraints,
and manages transfers/captaincy — with **zero manual override** once a
Gameweek's decisions are locked. It runs against a human-managed team all
season as a head-to-head experiment. It is also an open-source portfolio
project, so code quality, documentation, and auditability all matter as
much as raw predictive performance.

## Non-negotiable rules

1. **No data leakage.** Any prediction for Gameweek N must only use data
   that existed before Gameweek N's deadline. Never let a model or
   feature pull in Gameweek N (or later) results, future fixtures,
   future injury news, or future price changes. When adding a feature,
   agents must be able to explain *when* that data became available.
2. **Time-based validation only.** No random train/test splits. Use
   walk-forward or season-based splits (e.g. train on seasons 1–3,
   validate on season 4, test on season 5).
3. **Explainability over cleverness.** Prefer a simpler model whose
   reasoning can be surfaced to a human/LLM explanation layer over a
   marginally more accurate black box, especially in early phases.
4. **Squad selection is constrained optimization, not top-N by score.**
   Selecting a legal FPL squad (2 GK / 5 DEF / 5 MID / 3 FWD, budget,
   max 3 players per club) is an ILP problem, not a sort-and-slice.
5. **Don't skip ahead of the current phase** (see Roadmap below) unless
   explicitly asked to. Get each phase working end-to-end before adding
   sophistication to it.

## Architecture (data flow)

```
Data Sources (FPL API, historical archives)
     ↓
src/ingestion/      → raw data, saved to data/raw/ (gitignored, timestamped)
     ↓
src/cleaning/       → validated, deduplicated DataFrames
     ↓
src/features/       → engineered features (form, fixture difficulty, etc.)
     ↓
src/ratings/        → dynamic player/team rating system
     ↓
src/models/         → expected-points prediction (V1 rule-based → V5 ensembles)
     ↓
src/optimization/   → squad / starting XI / captain / bench / transfer solver
     ↓
src/simulation/     → Monte Carlo simulation over outcomes
     ↓
src/evaluation/     → backtesting, AI-vs-human tracking, accuracy metrics
```

Each `src/` subfolder should stay independently testable — e.g.
`src/models/` should be callable with a features DataFrame and return
predictions, without needing to know where the features came from.

## Current status / roadmap position

- ✅ Phase 1 (research) — informal, ongoing
- ✅ Phase 2 (data pipeline) — `src/ingestion/fetch_fpl_data.py` exists and
  pulls `bootstrap-static` + `fixtures` from the official FPL API.
  `--player-history` flag pulls full per-player history (slow, opt-in).
- 🚧 **Currently here:** Phase 3 — baseline expected-points model (V1,
  rule-based). See "Baseline model spec" below.
- ⬜ Phase 4 — squad optimizer (ILP)
- ⬜ Phase 5 — backtesting
- ⬜ Phase 6 — Monte Carlo simulation
- ⬜ Phase 7 — transfer planner
- ⬜ Phase 8 — live season (no manual intervention from here on)
- ⬜ Phase 9 — dashboard
- ⬜ Phase 10 — content automation

Don't build Phase 4+ logic until Phase 3 (the baseline model) is
producing sane per-player expected-points output that can be sanity
checked against real FPL data.

## Baseline model spec (V1, current task)

`xPts(player, gameweek)` should be a rule-based formula built directly
from the FPL scoring table (goals/assists/clean sheets/cards/bonus, all
position-weighted), driven by:

- `P(start)` — from rolling minutes over recent games + FPL's own
  `chance_of_playing_this_round` field, not season-long averages.
- `xG_match`, `xA_match` — per-90 rates × expected minutes, adjusted by
  opponent defensive strength.
- `P(clean_sheet)` — from team-level attack/defense strength (Poisson-style).
- `E[bonus_points]` — start with a simple proxy: rank players within a
  match by (xG + xA + clean-sheet contribution), top 3 get an expected
  bonus fraction. This is intentionally crude in V1.

This formula becomes the **feature set for V2** (regression) later —
don't discard the intermediate values (`P_start`, `xG_match`, etc.), they
should be stored as reusable columns, not baked only into a final score.

## Conventions

- Python, type-hinted where reasonable, docstrings on public functions
  explaining *why* not just *what* (especially in `src/models/` and
  `src/optimization/`, where the "why" is a football/stats decision).
- Config values (scoring rules, budget, squad constraints) belong in
  `configs/`, not hardcoded in logic files.
- Raw data lives in `data/raw/`, is timestamped, and is never committed
  (see `.gitignore`). Processed/derived data goes in `data/processed/`.
- Tests go in `tests/`, mirroring the `src/` structure.
- Every model version (V1, V2, ...) should be runnable and comparable —
  don't replace V1's code when building V2, keep both selectable.

## What NOT to do without being asked

- Don't add reinforcement learning — the roadmap explicitly defers RL
  until prediction/optimization/simulation baselines are proven out.
- Don't let an LLM produce the numeric predictions — LLMs are for the
  explanation/content layer only, sitting on top of the ML pipeline.
- Don't commit data files, API keys, or `.env` contents.
