"""Pure helpers to express baseline strategies as controller plans.

A plan is (start_compound, [(lap_fraction, action_int), ...]). action_int matches
ma_obs: 1=SOFT, 2=MEDIUM, 3=HARD.
"""
from __future__ import annotations

import json
import re

ACTION_FOR = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}

# Cars start on MEDIUM (training default); a 1-stop must switch compound.
_ANCHORS = {
    "onestop": ("MEDIUM", [(0.55, ACTION_FOR["HARD"])]),
    "twostop": ("MEDIUM", [(0.30, ACTION_FOR["SOFT"]), (0.65, ACTION_FOR["HARD"])]),
}


def anchor_plan(kind: str):
    return _ANCHORS[kind]


def parse_mc_plan(name: str, compound_sequence: str, total_laps: int):
    """Parse a scenarios default_plan into (start_compound, plan).

    name carries stint lengths in parens, e.g. '... (18/18/21)';
    compound_sequence is 'A -> B -> C' (arrow may be unicode →).
    """
    compounds = [c.strip().upper()
                 for c in compound_sequence.replace("→", "->").split("->")]
    m = re.search(r"\(([\d/]+)\)", name)
    laps = [int(x) for x in m.group(1).split("/")] if m else []
    plan = []
    cum = 0
    for k in range(len(laps) - 1):                  # final stint has no pit
        cum += laps[k]
        plan.append((cum / total_laps, ACTION_FOR[compounds[k + 1]]))
    return compounds[0], plan


def load_mc_plan(scenarios_path: str, total_laps: int):
    """Load and parse the MC default_plan from a scenarios_<circuit>_<season>.json."""
    d = json.load(open(scenarios_path))
    dp = d["default_plan"]
    return parse_mc_plan(dp["name"], dp["compound_sequence"], total_laps)
