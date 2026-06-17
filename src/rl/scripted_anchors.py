"""
Scripted anchor policies for the RL-2b league
=============================================
Deterministic, frozen RLModules that play a fixed pit plan. They race in-env as
opponents and form the absolute yardstick for the RL-2b gate. New API stack.

The obs vector (see ma_obs.build_obs) has lap-fraction at index 0. A plan is a
list of (pit_lap_fraction, action_int); the module fires the pit action on the
first step at/after each planned fraction, otherwise stays out (action 0).
Action ints match ma_obs._ACTION_COMPOUND: 0 stay, 1 SOFT, 2 MED, 3 HARD.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI

ANCHOR_PLANS = {
    "anchor_onestop": [(0.55, 2)],                 # one stop ~55% distance to MEDIUM
    "anchor_twostop": [(0.35, 2), (0.70, 3)],      # two stops: MED then HARD
}


class ScriptedAnchorModule(TorchRLModule, InferenceOnlyAPI):
    """Emits a planned pit action when lap-fraction first crosses a pit point."""

    def setup(self):
        # model_config carries the plan; default to a no-op (always stay).
        self.plan = list(self.model_config.get("plan", [])) if self.model_config else []
        # RLlib's learner builds an optimizer per module over its parameters; a
        # parameter-less module yields an empty-optimizer error. This single unused
        # param keeps the optimizer non-empty. The anchor is not in policies_to_train,
        # so it is never updated and the param stays frozen; the forward ignores it.
        self._unused = nn.Parameter(torch.zeros(1))

    def get_non_inference_attributes(self):
        # InferenceOnlyAPI hook: no training-only state to drop (scripted/frozen).
        return []

    def _action_for(self, obs_row) -> int:
        lap_frac = float(obs_row[0])
        act = 0
        for frac, a in self.plan:
            if frac <= lap_frac < frac + 0.03:
                act = a
        return act

    def _forward_inference(self, batch, **kwargs):
        obs = batch["obs"]
        acts = torch.tensor([self._action_for(row) for row in obs], dtype=torch.int64)
        return {"actions": acts}

    def _forward_exploration(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)

    def _forward_train(self, batch, **kwargs):
        return self._forward_inference(batch, **kwargs)
