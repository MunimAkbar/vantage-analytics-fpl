# Vantage Analytics — Autonomous AI Fantasy Premier League (FPL) Engine

An autonomous FPL decision-making system that manages its own squad for a full Premier League season, competing head-to-head against a human-managed team with **zero manual intervention**.

> **The Experiment:** Can a strictly leak-free, data-driven pipeline — combining predictive modeling, integer linear programming (ILP), and walk-forward validation — outperform an experienced human FPL manager over 38 Gameweeks?

---

## 🏗️ System Design & Architecture

The system operates as an end-to-end modular pipeline. Each module in `src/` is decoupled, independently testable, and enforced by strict **time-based validation** (no lookahead data leakage).

```
   +-----------------------------------------------------------------------+
   |                             DATA SOURCES                              |
   |   - Official FPL API (bootstrap-static, fixtures, player summaries)   |
   |   - Multi-Season Historical Archives (Vaastav FPL Dataset)            |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                       INGESTION & CLEANING (src/)                      |
   |  - fetch_fpl_data.py / fetch_historical_data.py                       |
   |  - clean_bootstrap.py / clean_fixtures.py / clean_historical_data.py  |
   |  - Drops lookahead targets (e.g. unshifted xP) & standardizes schemas  |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                     FEATURE ENGINEERING (src/features/)               |
   |  - build_features.py / build_historical_features.py                   |
   |  - Calculates rolling 5-GW form (xG90, xA90, minutes, clean sheets)    |
   |  - Opponent difficulty & expected playing probability P(start)        |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                   EXPECTED POINTS MODELS (src/models/)                |
   |  - xpts_v1.py: Rule-based physics formula from FPL scoring rules       |
   |  - xpts_v2.py: Ridge ML model trained via season walk-forward split    |
   |  - Outputs: data/processed/predictions_v1.csv & predictions_v2.csv    |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                     OPTIMIZATION ENGINE (src/optimization/)            |
   |  - squad_selector.py: PuLP ILP solver for 15-player squad (£100M, 3/club)|
   |  - lineup_selector.py: PuLP ILP solver for XI (formation, C/VC, bench)|
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                   EVALUATION & AUDIT (src/evaluation/)                |
   |  - decision_log.py: Timestamped pre-deadline JSON lock files          |
   |  - gameweek_review.py: MAE/RMSE analytics & AI vs Human head-to-head   |
   +-----------------------------------------------------------------------+
```

---

## 📊 Completed Phases & Roadmap Status

- [x] **Phase 1: Architecture & Requirements** — Modular folder structure, non-negotiable data leakage rules, and configuration primitives.
- [x] **Phase 2: Data Ingestion & Cleaning** — Live FPL API integration + multi-season historical dataset processing with zero lookahead bias.
- [x] **Phase 3: Expected Points (xP) Prediction Models**
  - **V1 (Baseline Physics Engine):** Direct FPL scoring rule modeling driven by $P(\text{start})$, expected goals ($xG$), expected assists ($xA$), clean sheet probability, and expected bonus proxies.
  - **V2 (Ridge Machine Learning Model):** Trained on multi-season rolling historical features with season-based walk-forward validation (Train on season $T-2$, Validate on $T-1$, Test on $T$). Model artifacts serialized to `models/xpts_v2_ridge.joblib`.
- [x] **Phase 4: Squad & Lineup Optimization** — Integer Linear Programming (ILP) solvers using `PuLP`:
  - **15-Player Squad Solver:** Selects 2 GK, 5 DEF, 5 MID, 3 FWD under budget ($\le £100.0\text{M}$) and max 3 players per club.
  - **Starting XI & Captain Solver:** Chooses optimal legal tactical formation (1 GK, 3–5 DEF, 3–5 MID, 1–3 FWD), assigns Captain ($2\times$) & Vice-Captain, and orders the bench.
- [x] **Phase 5: Pre-Gameweek Audit & Review System** — Immutable pre-deadline JSON decision logs for complete auditability, and gameweek post-mortem analysis (MAE, RMSE, rank correlation).
- [ ] **Phase 6: Multi-Gameweek Transfer Planner** — ILP solver for rolling transfers, hit penalties, and wildcard/free-hit chip planning.
- [ ] **Phase 7: Monte Carlo Outcome Simulation** — Variance modeling across match outcomes and opponent risks.
- [ ] **Phase 8: Web Dashboard & Automated Content Engine** — Streamlit tracking UI and automated match reports.

---

## 📁 Repository Structure & Modules

```
vantage-analytics/
├── configs/                    # System rules & scoring parameters
│   ├── fpl_constraints.py      # Budget limits (£100M), squad sizes, club max
│   └── fpl_scoring.py          # Position-weighted scoring rules (goals, CS, cards)
├── data/
│   ├── raw/                    # Gitignored raw API responses & archives
│   └── processed/              # Cleaned CSVs, predictions, decision audit logs
├── models/                     # Saved ML model artifacts (joblib weights)
├── src/
│   ├── ingestion/              # FPL API & historical archive fetchers
│   │   ├── fetch_fpl_data.py
│   │   └── fetch_historical_data.py
│   ├── cleaning/               # Sanitization & leak-free dataset generation
│   │   ├── clean_bootstrap.py
│   │   ├── clean_fixtures.py
│   │   ├── clean_historical_data.py
│   │   ├── clean_player_history.py
│   │   └── utils.py
│   ├── features/               # Rolling statistics & match context features
│   │   ├── build_features.py
│   │   └── build_historical_features.py
│   ├── models/                 # Predictive expected points models
│   │   ├── xpts_v1.py          # Rule-based physics baseline
│   │   └── xpts_v2.py          # Ridge Regression ML engine
│   ├── optimization/           # PuLP ILP mathematical optimization solvers
│   │   ├── squad_selector.py   # 15-player squad optimization
│   │   └── lineup_selector.py  # XI formation, captaincy & bench ordering
│   └── evaluation/             # Pre-deadline audit logging & accuracy reviews
│       ├── decision_log.py     # Timestamped decision audit snapshots
│       └── gameweek_review.py  # Gameweek performance & AI vs Human metrics
└── requirements.txt            # Project dependencies
```

---

## 🚀 Quickstart & Pipeline Execution

### 1. Environment Setup
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Data Pipeline & Predictions
```bash
# Ingest live FPL data
python -m src.ingestion.fetch_fpl_data

# Clean data and build features
python -m src.cleaning.run_cleaning
python -m src.features.build_features

# Run Predictions (V1 Baseline or V2 ML)
python -m src.models.xpts_v1
python -m src.models.xpts_v2 --predict
```

### 3. Run Squad & Lineup Optimization
```bash
# Select initial 15-player squad
python -m src.optimization.squad_selector

# Select starting XI, Captain, Vice-Captain, and Bench
python -m src.optimization.lineup_selector
```

### 4. Lock Decisions & Review Performance
```bash
# Create timestamped pre-deadline audit snapshot
python -m src.evaluation.decision_log --lock --gw 1

# Evaluate predictions post-gameweek
python -m src.evaluation.gameweek_review --gw 1
```

---

## 🛡️ Key System Principles & Non-Negotiables

1. **Zero Data Leakage:** Every feature for Gameweek $N$ strictly uses information known *before* Gameweek $N$'s deadline.
2. **Season-Based Walk-Forward Validation:** No random train/test splits. ML models are trained strictly on past seasons and validated on subsequent seasons.
3. **Constrained ILP Optimization:** Squad selection and tactical lineups use Integer Linear Programming (`PuLP`), guaranteeing strictly valid FPL rules rather than simple top-N sorting.
4. **Full Pre-Deadline Auditability:** AI team choices are written to immutable timestamped logs before kickoff, preventing post-hoc bias.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
