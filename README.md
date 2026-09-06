# Vantage Analytics — AI-Powered Fantasy Premier League (FPL)

An autonomous FPL decision-making system that manages its own team for a full
Premier League season, competing head-to-head against a human-managed team.

> **The experiment:** can a data-driven system — using predictive modeling,
> constrained optimization, and simulation — outperform a human FPL manager
> over 38 Gameweeks?

## Overview

- **Human Team** — manually managed FPL decisions.
- **AI Team** — every squad, transfer, captain, and bench decision made
  entirely by the system below, with no manual override after each
  Gameweek's predictions are locked.

Every AI decision is timestamped and recorded *before* the Gameweek deadline,
so results are auditable rather than reconstructed after the fact.

## Architecture

```
Football Data
     ↓
Data Ingestion → Cleaning → Feature Engineering
     ↓
Player & Team Ratings
     ↓
Prediction Models (expected points)
     ↓
Squad Optimizer (constrained selection)
     ↓
Starting XI / Captain / Bench
     ↓
Transfer Planner (multi-Gameweek)
     ↓
AI Team Decision
     ↓
Results Tracking + Dashboard + Content
```

## Project structure

```
vantage-analytics/
├── data/               # raw/ and processed/ data (gitignored — see below)
├── notebooks/          # exploration and prototyping
├── src/
│   ├── ingestion/      # pulling data from FPL API and other sources
│   ├── cleaning/       # validation, deduplication, type fixing
│   ├── features/       # feature engineering
│   ├── ratings/        # dynamic player/team rating system
│   ├── models/         # expected points prediction models (V1 → V5)
│   ├── optimization/   # squad/XI/captain/transfer optimization
│   ├── simulation/      # Monte Carlo simulation
│   └── evaluation/     # backtesting, AI-vs-human tracking
├── configs/            # config files (model params, FPL rules, etc.)
├── tests/
├── dashboards/         # Streamlit/FastAPI dashboard
├── scripts/            # one-off / scheduled run scripts
└── docs/                # methodology write-ups, weekly reports
```

## Data

Raw and processed data are **not committed to this repo** (see
`.gitignore`) — only the pipeline code that produces them. Data sources
under investigation:

- Official FPL API (`fantasy.premierleague.com/api`) — free, no auth
- Historical FPL data — community archives (season-by-season CSVs)
- xG/xA data — under evaluation for open-source licensing compatibility

## Methodology notes

- All predictions use only information available before the relevant
  Gameweek deadline — strict time-based validation, no lookahead.
- Model progression: rule-based → regression → tree ensembles → advanced
  ensembles, each benchmarked against the previous version and against the
  human team.

## Status

🚧 Early development — Phase 2/3 (data pipeline + baseline model).

## License

MIT — see [LICENSE](LICENSE).
