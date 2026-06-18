"""Load the trained main module from a league checkpoint for inference."""
from __future__ import annotations

from pathlib import Path


def load_main_module(checkpoint: str, module_id: str = "main_1"):
    """Return (module, algo) for module_id (default the stronger main agent).
    Uses the validated Algorithm.from_checkpoint, then extracts one module. Keep the
    returned algo alive (caller stops it) so the module isn't garbage-collected."""
    from ray.rllib.algorithms.algorithm import Algorithm
    algo = Algorithm.from_checkpoint(str(Path(checkpoint).resolve()))
    module = algo.get_module(module_id)
    return module, algo
