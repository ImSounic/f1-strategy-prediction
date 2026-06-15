#!/usr/bin/env python
"""
Post-clone sanity check
=======================
Confirms a fresh clone has everything wired up to run the pipeline:

  1. Python version
  2. Core dependencies importable (and RL deps, reported separately)
  3. Committed data / model / config artifacts present
  4. The production tyre model actually loads

Run from the repo root:   python scripts/verify_setup.py
Exit code 0 = ready to run the committed pipeline (make all).
Non-zero  = something required is missing (details printed).

Note: re-ingesting raw data (make ingest) is NOT required for the core
pipeline — the engineered feature parquets are committed.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Make output safe on legacy Windows consoles (cp1252) and pipes.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# Use unicode glyphs only if the active stdout encoding can render them.
_enc = (getattr(sys.stdout, "encoding", "") or "ascii").lower()
_UNICODE_OK = "utf" in _enc
_TICK = "✓" if _UNICODE_OK else "OK"
_CROSS = "✗" if _UNICODE_OK else "X"
_BANG = "!"

# (import name, pip name) — core deps required by the non-RL pipeline
CORE_DEPS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("pyarrow", "pyarrow"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("shap", "shap"),
    ("matplotlib", "matplotlib"),
    ("fastf1", "fastf1"),
    ("requests", "requests"),
    ("yaml", "pyyaml"),
]

# RL deps — only needed for src/rl/* (reported but not fatal by default)
RL_DEPS = [
    ("torch", "torch"),
    ("gymnasium", "gymnasium"),
    ("stable_baselines3", "stable-baselines3"),
    ("sb3_contrib", "sb3-contrib"),
]

# Committed artifacts the pipeline/frontend depend on (relative to root)
REQUIRED_FILES = [
    "configs/config.yaml",
    "configs/drivers_2025.json",
    "data/processed/clean_laps.parquet",
    "data/features/stint_features.parquet",
    "data/features/lap_features.parquet",
    "data/raw/supplementary/pirelli_circuit_characteristics.csv",
    "models/tyre_deg_production.json",
    "models/best_xgboost_model.json",
    "models/comparison_results.json",
    "models/safety_car_priors.json",
]

GREEN, RED, YEL, DIM, RST = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def ok(msg: str) -> None:
    print(f"  {GREEN}{_TICK}{RST} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}{_CROSS}{RST} {msg}")


def warn(msg: str) -> None:
    print(f"  {YEL}{_BANG}{RST} {msg}")


def check_imports(deps, label):
    print(f"\n{label}")
    missing = []
    for mod, pip_name in deps:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok(f"{mod:<20} {DIM}{ver}{RST}")
        except Exception as e:  # noqa: BLE001
            fail(f"{mod:<20} missing  ({type(e).__name__})  →  pip install {pip_name}")
            missing.append(pip_name)
    return missing


def main() -> int:
    print("=" * 60)
    print("  F1 Strategy Optimizer — setup verification")
    print("=" * 60)

    problems = 0

    # 1. Python version
    print("\nPython")
    v = sys.version_info
    if (v.major, v.minor) >= (3, 10):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor} — project needs 3.10+")
        problems += 1

    # 2. Dependencies
    missing_core = check_imports(CORE_DEPS, "Core dependencies")
    problems += len(missing_core)

    missing_rl = check_imports(RL_DEPS, "Reinforcement-learning dependencies (optional)")
    if missing_rl:
        warn("RL deps missing — fine unless you run src/rl/*. "
             "Install with: pip install -r requirements-rl.txt")

    # 3. Committed artifacts
    print("\nCommitted artifacts")
    for rel in REQUIRED_FILES:
        p = ROOT / rel
        if p.exists():
            ok(rel)
        else:
            fail(f"{rel}  — MISSING")
            problems += 1

    # 4. Production model actually loads
    print("\nProduction model load test")
    prod = ROOT / "models/tyre_deg_production.json"
    if "xgboost" in missing_core:
        warn("skipped (xgboost not installed)")
    elif not prod.exists():
        fail("models/tyre_deg_production.json missing — run `make model`")
        problems += 1
    else:
        try:
            import xgboost as xgb

            m = xgb.XGBRegressor()
            m.load_model(str(prod))
            ok(f"loaded tyre_deg_production.json ({m.n_features_in_} features)")
        except Exception as e:  # noqa: BLE001
            fail(f"failed to load production model: {e}")
            problems += 1

    # Summary
    print("\n" + "=" * 60)
    if problems == 0:
        print(f"  {GREEN}READY{RST} — committed pipeline can run. Try: make all")
        if missing_rl:
            print(f"  {DIM}(install requirements-rl.txt before RL work){RST}")
        return 0
    print(f"  {RED}{problems} problem(s) found{RST} — see above.")
    if missing_core:
        print(f"  Fix deps:  pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
