"""
Strategy-matching metrics for validation
========================================
Pure, dependency-free helpers (no ML imports) so they are unit-testable in
isolation. Two distinct notions of a "correct" prediction:

  - **stop-count match**: did we predict the right NUMBER of pit stops?
  - **full-strategy match**: did we predict the right ordered COMPOUND SEQUENCE
    (e.g. SOFT → MEDIUM → HARD)?

Top-k is computed over **distinct compound sequences**, ranked best-first.
This matters: the simulator emits many candidate strategies that share a
compound sequence and differ only by pit-lap split. Without de-duplicating by
sequence, the top-5 fastest *results* can all be variants of a single sequence,
collapsing top-5 back down to top-1 (the original bug).
"""
from __future__ import annotations


def normalize_sequence(seq) -> tuple:
    """Canonicalize a compound sequence to a tuple of upper-case tokens.

    Accepts either a list (``["SOFT", "HARD"]``) or a string
    (``"SOFT → HARD"`` / ``"SOFT->HARD"``). Blank tokens and ``"UNKNOWN"``
    are dropped, so a wet/unknown actual strategy normalizes to ``()`` and can
    never produce a false-positive match.
    """
    if seq is None:
        return ()
    if isinstance(seq, str):
        parts = seq.replace("->", "→").split("→")
    else:
        parts = list(seq)

    out = []
    for p in parts:
        token = str(p).strip().upper()
        if token and token != "UNKNOWN":
            out.append(token)
    return tuple(out)


def _distinct_sequences(sim_results: list) -> list:
    """Ordered list of unique compound sequences (best-first), de-duplicated.

    Assumes ``sim_results`` is already ranked best-first (lowest median time,
    after any compound-prior reranking).
    """
    seen = set()
    distinct = []
    for r in sim_results:
        key = normalize_sequence(r.get("compound_sequence"))
        if not key or key in seen:
            continue
        seen.add(key)
        distinct.append(key)
    return distinct


def score_race(sim_results: list, real: dict, top_ks=(3, 5)) -> dict:
    """Score one race's ranked candidate strategies against the actual strategy.

    Parameters
    ----------
    sim_results : list of dict
        Candidate strategies, ranked best-first. Each needs ``"num_stops"`` and
        ``"compound_sequence"`` (both are returned by ``run_monte_carlo``).
    real : dict
        Actual strategy, with ``"n_stops"`` and ``"compounds"`` (a list) or
        ``"compound_sequence"`` (a string).
    top_ks : tuple of int
        Which top-k cutoffs to report for full-strategy match (default 3 and 5).

    Returns
    -------
    dict with keys:
        stop_match            — top pick has the right number of stops
        strategy_exact        — top pick is the exact compound sequence
        strategy_top{k}       — actual sequence is within the top-k distinct picks
        recommended_stops     — stop count of the top pick
        recommended_sequence  — compound sequence string of the top pick
    """
    base = {
        "stop_match": False,
        "strategy_exact": False,
        "recommended_stops": None,
        "recommended_sequence": None,
    }
    for k in top_ks:
        base[f"strategy_top{k}"] = False

    if not sim_results:
        return base

    top = sim_results[0]
    real_seq = normalize_sequence(real.get("compounds") or real.get("compound_sequence"))
    top_seq = normalize_sequence(top.get("compound_sequence"))

    base["recommended_stops"] = top.get("num_stops")
    base["recommended_sequence"] = top.get("compound_sequence")
    base["stop_match"] = top.get("num_stops") == real.get("n_stops")
    base["strategy_exact"] = bool(real_seq) and top_seq == real_seq

    distinct = _distinct_sequences(sim_results)
    for k in top_ks:
        base[f"strategy_top{k}"] = bool(real_seq) and real_seq in distinct[:k]

    return base
