# Vantage Analytics — Autonomous AI Fantasy Premier League (FPL) Engine

An autonomous FPL decision-making system that manages its own squad for a full Premier League season, competing head-to-head against a human-managed team with **zero manual intervention**.

> **The Experiment:** Can a strictly leak-free, data-driven pipeline — combining predictive modeling, integer linear programming (ILP), and walk-forward validation — outperform an experienced human FPL manager over 38 Gameweeks?
>
> 📌 **Live Season Status:** Active in season — **GW4 in progress** (decisions locked pre-deadline for V1 and V2 head-to-head).

---

## 🏗️ System Design & Architecture

The system operates as an end-to-end modular pipeline. Each module in [`src/`](file:///d:/WorkANDProjects/vantage-analytics/src) is decoupled, independently testable, and enforced by strict **time-based validation** (no lookahead data leakage).

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
   |  - Drops lookahead targets & standardizes position / finished flags   |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                     FEATURE ENGINEERING (src/features/)               |
   |  - build_features.py / build_historical_features.py                   |
   |  - Calculates rolling 5-GW form (xG90, xA90, minutes, clean sheets)    |
   |  - Opponent 5-GW fixture run difficulty & expected start prob P(start) |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                   EXPECTED POINTS MODELS (src/models/)                |
   |  - xpts_v1.py: Rule-based physics formula from FPL scoring rules       |
   |  - xpts_v2.py: Ridge ML model trained via season walk-forward split    |
   |  - Model artifacts serialized to models/xpts_v2_ridge.joblib           |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                     OPTIMIZATION ENGINE (src/optimization/)            |
   |  - squad_selector.py: PuLP ILP solver for 15-player squad (£100M, 3/club)|
   |  - lineup_selector.py: PuLP ILP solver for XI (formation, C/VC, bench)|
   |  - transfer_selector.py: Rolling transfer solver (FTs & hit costs)    |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                   EVALUATION & AUDIT (src/evaluation/)                |
   |  - decision_log.py: Timestamped pre-deadline JSON/CSV lock files      |
   |  - gameweek_review.py: Post-GW actual vs predicted review & MAE        |
   |  - compare_models.py: V1 vs V2 head-to-head season tracking            |
   |  - backtest_v2_transfers.py: Historical multi-season simulation       |
   +-----------------------------------------------------------------------+
                                       |
                                       v
   +-----------------------------------------------------------------------+
   |                   ORCHESTRATION (scripts/run_gameweek.py)              |
   |  - Single-command end-to-end execution: Fetch -> Predict -> Lock      |
   +-----------------------------------------------------------------------+
```

---

## 📊 Roadmap & Phase Status

- [x] **Phase 1: Architecture & Requirements** — Modular folder structure, non-negotiable data leakage rules, and configuration primitives.
- [x] **Phase 2: Data Ingestion & Cleaning** — Live FPL API integration + multi-season historical dataset processing with zero lookahead bias (`finished_provisional` match handling).
- [x] **Phase 3: Expected Points (xP) Prediction Models**
  - **V1 (Baseline Physics Engine):** Direct FPL scoring rule modeling driven by $P(\text{start})$, expected goals ($xG$), expected assists ($xA$), clean sheet probability, and expected bonus proxies.
  - **V2 (Ridge Machine Learning Model):** Trained on multi-season rolling historical features with season-based walk-forward validation (Train on seasons 2020-23, Validate on 2023-24, Test on 2024-25). **Validation MAE: 1.023, Test MAE: 1.061** (vs. 1.503 naive baseline).
- [x] **Phase 4: Squad & Lineup Optimization** — Integer Linear Programming (ILP) solvers using `PuLP`:
  - **15-Player Squad Solver:** Selects 2 GK, 5 DEF, 5 MID, 3 FWD under budget ($\le £100.0\text{M}$) and max 3 players per club.
  - **Starting XI & Captain Solver:** Chooses optimal legal tactical formation (1 GK, 3–5 DEF, 3–5 MID, 1–3 FWD), assigns Captain ($2\times$) & Vice-Captain, and orders the bench.
- [x] **Phase 5: Pre-Gameweek Audit & Review System** — Immutable pre-deadline CSV/JSON decision locks (`FileExistsError` protection against overwrite), and post-gameweek evaluation engine.
- [🚧] **Phase 6: Transfer Planner & Historical Backtesting** — Rolling transfer solver (`transfer_selector.py`) and historical season simulation (`backtest_v2_transfers.py`). *Multi-gameweek lookahead horizon in progress.*
- [ ] **Phase 7: Monte Carlo Outcome Simulation** — Variance modeling across match outcomes and opponent risks.
- [ ] **Phase 8: Web Dashboard & Automated Content Engine** — Streamlit tracking UI and automated match reports.

---

## 📈 Live Season Performance & Audit Tracker (Up to GW4)

All AI decisions are written to disk *before* each Gameweek deadline and cannot be altered retrospectively.

| Gameweek | Model Version | Status | Predicted Points | Actual Points | Notes / Performance Highlights |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **GW3** | V1 Baseline | ✅ Reviewed | **92.03** | **56.00** | First live reviewed week. Over-predicted top picks (Fernandes, Mbeumo C/VC) while budget picks (Isak) delivered. Highlighted short-term form over-extrapolation in V1. |
| **GW4** | V1 Baseline | 🔒 Locked | 88.42 | *In Progress* | Squad locked pre-deadline. Captain: Fernandes, VC: Mbeumo. |
| **GW4** | V2 Ridge ML | 🔒 Locked | 54.10 | *In Progress* | First live head-to-head test. Squad locked pre-deadline. Captain: Haaland, VC: Salah. |
| **GW5** | V1 & V2 | ⏳ Pending | - | - | Transfer suggestions generated via [`suggest_transfers.py`](file:///d:/WorkANDProjects/vantage-analytics/src/evaluation/suggest_transfers.py). |

---

## 🧪 Historical Backtesting Benchmarks (2024-25 Held-Out Test Season)

To evaluate the predictive and decision engine under realistic constraints, we ran full-season simulations across all 38 Gameweeks of the held-out 2024-25 season (which V2 never saw during training):

| Strategy / Model Configuration | Total Points | Avg Pts / GW | Total Transfers | Hits Taken | Key Insights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V2 Unconstrained Rebuild** ([`backtest_v2.py`](file:///d:/WorkANDProjects/vantage-analytics/src/evaluation/backtest_v2.py)) | **2,260** | 59.47 | N/A (Free weekly reset) | 0 | Upper bound showing the raw power of V2 predictions when unrestricted by squad carrying rules. |
| **V2 Realistic Transfers** ([`backtest_v2_transfers.py`](file:///d:/WorkANDProjects/vantage-analytics/src/evaluation/backtest_v2_transfers.py)) | **1,760** | 46.32 | 47 | 9 (-36 pts) | Uses 1 Free Transfer/GW (banking up to 5) and -4 hit costs. Demonstrates that 1-week greedy transfer logic falls short of top human managers (~2000+ pts), establishing the need for multi-GW lookahead. |

---

## 📁 Repository Structure & Modules

```
vantage-analytics/
├── configs/                    # System rules & scoring parameters
│   ├── fpl_constraints.py      # Budget limits (£100M), squad sizes, club max, transfer caps
│   └── fpl_scoring.py          # Position-weighted scoring rules (goals, CS, cards, bonus)
├── data/
│   ├── raw/                    # Gitignored raw API responses & archives
│   └── processed/              # Cleaned CSVs, predictions, decision audit logs
├── models/                     # Saved ML model artifacts (e.g. xpts_v2_ridge.joblib)
├── scripts/
│   └── run_gameweek.py         # End-to-end gameweek pipeline orchestrator
├── src/
│   ├── ingestion/              # FPL API & historical archive fetchers
│   │   ├── fetch_fpl_data.py   # Live API pull with --player-history option
│   │   └── fetch_historical_data.py # Multi-season Vaastav GitHub dataset ingestor
│   ├── cleaning/               # Sanitization & leak-free dataset generation
│   │   ├── clean_bootstrap.py
│   │   ├── clean_fixtures.py   # Uses finished_provisional for live match state
│   │   ├── clean_historical_data.py # Relabeling & AM filter fixes
│   │   ├── clean_player_history.py
│   │   └── utils.py
│   ├── features/               # Rolling statistics & match context features
│   │   ├── build_features.py   # Live rolling form (last 5 GWs) & 5-GW FDR run
│   │   └── build_historical_features.py # Leak-free historical features (.shift(1))
│   ├── models/                 # Predictive expected points models
│   │   ├── xpts_v1.py          # Rule-based physics baseline
│   │   └── xpts_v2.py          # Ridge Regression ML engine (walk-forward split)
│   ├── optimization/           # PuLP ILP mathematical optimization solvers
│   │   ├── squad_selector.py   # Initial 15-player squad solver
│   │   ├── lineup_selector.py  # XI formation, captaincy & bench ordering
│   │   └── transfer_selector.py# Rolling weekly transfer solver
│   └── evaluation/             # Pre-deadline audit logging & accuracy reviews
│       ├── decision_log.py     # Timestamped decision audit snapshots
│       ├── gameweek_review.py  # Post-GW performance & actual vs predicted metrics
│       ├── compare_models.py   # V1 vs V2 head-to-head comparison logger
│       ├── suggest_transfers.py# Recommended transfers for upcoming GWs
│       ├── backtest_v2.py      # Unconstrained full-season historical simulation
│       ├── backtest_v2_transfers.py # Realistic transfer-constrained simulation
│       ├── show_squad.py       # Terminal UI for viewing active squad/XI
│       └── weekly_report.py    # Automated report generator
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

### 2. Run Complete Gameweek Cycle (Single Command)
To run the full end-to-end pipeline (fetch data, clean, calculate features, predict V1/V2, optimize squads, and lock pre-deadline decisions for a gameweek):

```bash
python -m scripts.run_gameweek --gw 4
```

### 3. Step-by-Step Manual Execution

```bash
# 1. Ingest live FPL data
python -m src.ingestion.fetch_fpl_data --player-history

# 2. Clean data and build features
python -m src.cleaning.run_cleaning
python -m src.cleaning.clean_player_history
python -m src.features.build_features

# 3. Train or Run Predictions
python -m src.models.xpts_v1
python -m src.models.xpts_v2 --train      # Train Ridge model on historical dataset
python -m src.models.xpts_v2 --predict    # Generate V2 predictions

# 4. Optimize Squad & Lineup
python -m src.optimization.squad_selector --version v2
python -m src.optimization.lineup_selector --version v2

# 5. Lock Pre-Deadline Decision Snapshot
python -m src.evaluation.decision_log --lock --gw 4 --version v2
```

### 4. Review & Analytics Commands

```bash
# Evaluate Gameweek results after matches complete
python -m src.evaluation.gameweek_review --gw 3 --version v1

# Run Head-to-Head Model Comparison
python -m src.evaluation.compare_models --gw 4

# Suggest Transfers for Next Gameweek
python -m src.evaluation.suggest_transfers --gw 5 --version v2

# Run Full Historical Season Backtest (2024-25)
python -m src.evaluation.backtest_v2_transfers
```

---

## 🛡️ Key System Principles & Non-Negotiables

1. **Zero Data Leakage:** Every feature for Gameweek $N$ strictly uses information known *before* Gameweek $N$'s deadline (`.shift(1)` rolling logic).
2. **Season-Based Walk-Forward Validation:** No random train/test splits. ML models are trained strictly on past seasons (2020–23) and evaluated on held-out seasons (2024-25).
3. **Constrained ILP Optimization:** Squad selection, transfer decisions, and tactical lineups use Integer Linear Programming (`PuLP`), guaranteeing strictly valid FPL rules rather than simple top-N sorting.
4. **Full Pre-Deadline Auditability:** AI team choices are written to immutable timestamped CSV/JSON files before kickoff, preventing post-hoc hindsight bias.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
