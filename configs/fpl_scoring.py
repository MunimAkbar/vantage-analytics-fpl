"""
fpl_scoring.py

FPL's official scoring rules, by position. These are stable game constants,
not something the model should ever "learn" or override.
"""

GOAL_POINTS = {"GK": 10, "DEF": 6, "MID": 5, "FWD": 4}
ASSIST_POINTS = 3  # same for every position
CLEAN_SHEET_POINTS = {"GK": 4, "DEF": 4, "MID": 1, "FWD": 0}
MINUTES_POINTS_60_PLUS = 2
MINUTES_POINTS_UNDER_60 = 1