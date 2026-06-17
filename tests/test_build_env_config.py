"""Pure test for the self-play policy mapping. Run: python tests/test_build_env_config.py"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rl.build_env_config import policy_mapping_fn


def test_policy_mapping_is_shared_main():
    # Every car maps to the single shared 'main' policy (self-play).
    assert policy_mapping_fn("car_0") == "main"
    assert policy_mapping_fn("car_20", None) == "main"
    assert policy_mapping_fn("car_5", episode=None, worker=None) == "main"


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
