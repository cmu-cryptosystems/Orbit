"""Compressed DAG → orbit_core smoke test (Gurobi)."""

from __future__ import annotations

import pytest

from scripts.latency_estimator.latency_estimator import LatencyEstimator
from scripts.optimizer.orbit.orbit_core import orbit_core
from scripts.params.params import Params
from scripts.tdag import addition_squash, auto_compress, build_from_mlir


def test_orbit_core_motivation(motivation_mlir: str, toy_cost_json: str):
    # Orbit solves with Gurobi. The motivation DAG is tiny, so it fits well
    # within Gurobi's bundled size-limited license (no academic license needed).
    pytest.importorskip("gurobipy")

    params = Params(
        toy_cost_json,
        "Orbit",
        "compile",
        threads=2,
        bpsdepth=15,
    )
    le = LatencyEstimator(params)

    og = build_from_mlir(motivation_mlir, params)
    og.squash_ops(["rescale", "upscale", "modswitch", "bootstrap"])
    addition_squash(og)
    comp_dag, _ = auto_compress(og, is_ignore_weight=True)

    assign, times = orbit_core(comp_dag, le, params)
    assert assign is not None
    assert times is not None
    assert "ILP-QBP Time" in times
