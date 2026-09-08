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

# Transfer rules (2024/25 season onward — confirmed via official FPL rule change
# announcements: banked free transfers raised from 2 to 5 that season).
FREE_TRANSFERS_PER_GAMEWEEK = 1
MAX_BANKED_FREE_TRANSFERS = 5
TRANSFER_HIT_COST = 4  # points deducted per transfer beyond the free allowance