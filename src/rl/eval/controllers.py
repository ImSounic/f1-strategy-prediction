"""Per-car controllers for evaluation races.

A controller maps a (normalised) observation -> Discrete(4) action int
(0 stay, 1 pit-SOFT, 2 pit-MED, 3 pit-HARD) and declares a start_compound.
ScriptedController is pure; RLController wraps a trained module (torch, HPC only).
"""
from __future__ import annotations

_PIT_WINDOW = 0.03   # ~1.7 laps at 57 laps; the legality mask blocks a double-pit


class ScriptedController:
    def __init__(self, start_compound: str, plan):
        self.start_compound = start_compound
        self.plan = list(plan)            # [(lap_fraction, action_int), ...]

    def decide(self, obs) -> int:
        lap_frac = float(obs[0])
        for frac, action in self.plan:
            if frac <= lap_frac < frac + _PIT_WINDOW:
                return int(action)
        return 0


class RLController:
    """Controller wrapping a trained RLModule. SAMPLES from the policy (matching how
    it was trained/acted): pitting is a rare per-lap event, so greedy argmax would
    pick 'stay' almost every lap and under-pit. start_compound matches training
    (cars started on MEDIUM)."""

    def __init__(self, module, start_compound: str = "MEDIUM"):
        self.module = module
        self.start_compound = start_compound

    def decide(self, obs) -> int:
        import torch
        with torch.no_grad():
            batch = {"obs": torch.tensor(obs, dtype=torch.float32).unsqueeze(0)}
            out = self.module.forward_inference(batch)
            if "actions" in out:
                return int(out["actions"][0])
            logits = out["action_dist_inputs"][0]         # Discrete logits -> sample
            probs = torch.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())
