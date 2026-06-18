"""Pure tests for the scripted controller. Run: python tests/test_eval_controllers.py"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.eval.controllers import ScriptedController


def _obs(lap_frac):
    v = np.zeros(18, dtype=np.float32)
    v[0] = lap_frac
    return v


def test_scripted_fires_in_window_else_stays():
    c = ScriptedController(start_compound="MEDIUM", plan=[(0.55, 3)])
    assert c.decide(_obs(0.10)) == 0
    assert c.decide(_obs(0.56)) == 3
    assert c.decide(_obs(0.90)) == 0
    assert c.start_compound == "MEDIUM"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn(); print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return failed


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
