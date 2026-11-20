# Vantage Analytics — Autonomous AI Fantasy Premier League (FPL) Engine

An FPL decision-making system that manages a real squad for the full 2026-27 Premier League season, competing head-to-head against a human-managed team. The AI makes the decisions (transfers, starting XI, captain); the owner copies them into a real FPL account by hand, because FPL has no public API for making transfers.

> **The Experiment:** Can a strictly leak-free, data-driven pipeline — combining predictive modeling, integer linear programming (ILP), and walk-forward validation — outperform an experienced human FPL manager over 38 Gameweeks?
>
> 📌 **Live Season Status (20 Nov 2026):** **GW11 locked** before the deadline (V1). Every result is reported below, including the misses. This is an experiment, not betting or FPL advice.

---

## 🏗️ System Design & Architecture

The system operates as an end-to-end modular pipeline. Each module in [`src/`](src) is decoupled, independently testable, and enforced by strict **time-based validation** (no lookahead data leakage).

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
   |                  WEEKLY DECISION TOOLS (src/evaluation/)               |
   |  - sync_squad.py: snapshot of the squad held in the real FPL account  |
   |  - suggest_transfers.py: transfer ILP from the real squad (FTs, hits) |
   |  - lock_squad.py: picks XI/C/VC, locks only before the deadline       |
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

## 📈 Live Season Record (up to GW11)

A locked decision is written to disk *before* the gameweek deadline and is never edited afterwards. Weeks without a lock are listed too, with the reason. Predicted points are the starting XI plus the captain's double.

| Gameweek | Model | Status | Predicted | Actual | Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **GW3** | V1 | ✅ Locked & reviewed | 92.03 | 56 | Captain Fernandes and VC Mbeumo both flopped. Showed V1 over-extrapolating recent form. |
| **GW4** | V1 | ✅ Locked & reviewed | 75.26 | 52 | Same captain and VC flopped again. |
| **GW4** | V2 | ✅ Locked & reviewed | 42.89 | 42 | First V2 lock. Captain Hinshelwood, VC Mundle. Closest prediction so far. |
| **GW5** | — | Not locked | — | — | International break. |
| **GW6** | V1 | 🔒 Locked | 69.34 | — | The real account differed from the lock in 5 players, so not a clean AI week. |
| **GW7–9** | — | Not locked | — | ~50 (GW9) | The account played the synced squad without new locks. |
| **GW10** | V1 (advised) | Not locked | — | 70 | The lock was never confirmed. The 3 suggested transfers and the suggested captain (Barry) were played. AI-advised, not a locked AI decision. |
| **GW11** | V1 | 🔒 Locked | 59.82 | *pending* | Suspended Thiago → Evanilson (1 free transfer, no hit). Captain Barry, VC Evanilson. |

Records the experiment needs to keep are copied to [`data/records/`](data/records).

---

## 🧪 Historical Backtesting Benchmarks (2024-25 Held-Out Test Season)

To evaluate the predictive and decision engine under realistic constraints, we ran full-season simulations across all 38 Gameweeks of the held-out 2024-25 season (which V2 never saw during training):

| Strategy / Model Configuration | Total Points | Avg Pts / GW | Total Transfers | Hits Taken | Key Insights |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **V2 Unconstrained Rebuild** ([`backtest_v2.py`](src/evaluation/backtest_v2.py)) | **2,260** | 59.47 | N/A (Free weekly reset) | 0 | Upper bound showing the raw power of V2 predictions when unrestricted by squad carrying rules. |
| **V2 Realistic Transfers** ([`backtest_v2_transfers.py`](src/evaluation/backtest_v2_transfers.py)) | **1,760** | 46.32 | 47 | 9 (-36 pts) | Uses 1 Free Transfer/GW (banking up to 5) and -4 hit costs. Demonstrates that 1-week greedy transfer logic falls short of top human managers (~2000+ pts), establishing the need for multi-GW lookahead. |

---

## 📁 Repository Structure & Modules

```
vantage-analytics/
├── configs/                    # System rules & scoring parameters
│   ├── fpl_constraints.py      # Budget limits (£100M), squad sizes, club max, transfer caps
│   └── fpl_scoring.py          # Position-weighted scoring rules (goals, CS, cards, bonus)
├── data/
│   ├── raw/                    # Gitignored raw API responses & archives
│   ├── processed/              # Gitignored cleaned CSVs, predictions, decision locks
│   └── records/                # Committed copies of suggestions and squads kept for the record
├── models/                     # Saved ML model artifacts (e.g. xpts_v2_ridge.joblib)
├── scripts/
│   └── run_gameweek.py         # Legacy: builds a fresh squad; predates the real-squad workflow
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
│       ├── sync_squad.py       # Snapshot of the real account squad (not a lock)
│       ├── suggest_transfers.py# Transfer suggestion for the next GW from the real squad
│       ├── lock_squad.py       # Pre-deadline XI/captain lock, confirmed by typing LOCK
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

### 2. Refresh Data and Predictions (the day before a deadline)

```bash
# Fetch live FPL data. --player-history is required: without it, recent form stops at the last run.
python -m src.ingestion.fetch_fpl_data --player-history

# Clean data and build features
python -m src.cleaning.run_cleaning
python -m src.cleaning.clean_player_history
python -m src.features.build_features

# Predict
python -m src.models.xpts_v1
python -m src.models.xpts_v2 --predict    # optional; --train retrains the Ridge model on history
```

### 3. Weekly Decision Routine (real squad)

```bash
# 1. Write the squad held in the app to data/processed/account_squad.txt
#    (one player per line: name, price, club), then snapshot it.
#    --budget is squad sell value plus money in the bank.
python -m src.evaluation.sync_squad --budget 100.4

# 2. Suggest transfers; set --free-transfers to what the app shows
python -m src.evaluation.suggest_transfers --version v1 --free-transfers 1

# 3. Make the transfers in the app, update account_squad.txt, then sync again
python -m src.evaluation.sync_squad --budget 100.4

# 4. Lock the XI, captain and VC (asks you to type LOCK; refuses after the deadline)
python -m src.evaluation.lock_squad --gameweek 11

# 5. Show the locked squad, then set the same lineup in the app
python -m src.evaluation.show_squad
```

### 4. Review & Analytics Commands

```bash
# Evaluate a gameweek after its matches finish
python -m src.evaluation.gameweek_review --gameweek 10 --version v1

# Review last week and preview next week's captain and transfers
python -m src.evaluation.weekly_report --completed-gameweek 10 --version v1 --free-transfers 1

# V1 vs V2 head-to-head
python -m src.evaluation.compare_models --gameweek 4

# Full historical season backtest with real transfer rules
python -m src.evaluation.backtest_v2_transfers
```

---

## 🛡️ Key System Principles & Non-Negotiables

1. **Zero Data Leakage:** Every feature for Gameweek $N$ strictly uses information known *before* Gameweek $N$'s deadline (`.shift(1)` rolling logic).
2. **Season-Based Walk-Forward Validation:** No random train/test splits. ML models are trained strictly on past seasons (2020–23) and evaluated on held-out seasons (2024-25).
3. **Constrained ILP Optimization:** Squad selection, transfer decisions, and tactical lineups use Integer Linear Programming (`PuLP`), guaranteeing strictly valid FPL rules rather than simple top-N sorting.
4. **Full Pre-Deadline Auditability:** AI team choices are written to timestamped CSV files before the deadline and are never overwritten (`FileExistsError`), preventing post-hoc hindsight bias.
5. **Honest Reporting:** Bad weeks and weeks without a clean lock stay in the record.

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
