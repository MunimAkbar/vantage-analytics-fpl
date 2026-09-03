"""
fpl_constraints.py

FPL's official squad-building rules. Stable game constants — the
optimizer should read these, never hardcode them inline.
"""

SQUAD_BUDGET = 100.0  # in £m, matches now_cost units after /10 conversion

# Full 15-man squad composition
SQUAD_POSITION_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}

# Valid starting XI formations: (GK, DEF, MID, FWD), GK always 1
VALID_FORMATIONS = [
    (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 3, 3), (1, 4, 4, 2),
    (1, 4, 5, 1), (1, 5, 2, 3), (1, 5, 3, 2), (1, 5, 4, 1),
]

MAX_PLAYERS_PER_CLUB = 3