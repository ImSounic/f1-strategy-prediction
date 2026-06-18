"""Load the trained main module from a league checkpoint for inference.

Loads ONLY the RLModule from its checkpoint subdirectory — no Algorithm, no env,
no env-runners, no Ray. (Algorithm.from_checkpoint would rebuild env-runners that
call gym.make('f1_ma'), which isn't registered in the eval process.)
"""
from __future__ import annotations

from pathlib import Path


def _module_path(checkpoint: str, module_id: str) -> Path:
    base = Path(checkpoint).resolve()
    try:
        from ray.rllib.core import (
            COMPONENT_LEARNER_GROUP, COMPONENT_LEARNER, COMPONENT_RL_MODULE,
        )
        p = base / COMPONENT_LEARNER_GROUP / COMPONENT_LEARNER / COMPONENT_RL_MODULE / module_id
        if p.exists():
            return p
    except Exception:  # noqa: BLE001
        pass
    p = base / "learner_group" / "learner" / "rl_module" / module_id
    if p.exists():
        return p
    # Fallback: search for the module_id directory anywhere under the checkpoint.
    matches = [d for d in base.rglob(module_id) if d.is_dir()]
    if not matches:
        raise FileNotFoundError(f"module '{module_id}' not found under {base}")
    return matches[0]


def load_main_module(checkpoint: str, module_id: str = "main_1"):
    """Return the trained RLModule for module_id (default the stronger main agent)."""
    from ray.rllib.core.rl_module.rl_module import RLModule
    return RLModule.from_checkpoint(_module_path(checkpoint, module_id))
