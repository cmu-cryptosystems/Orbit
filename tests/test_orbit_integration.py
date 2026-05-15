"""Compressed DAG → orbit_core (PuLP) smoke test."""

from __future__ import annotations

import pytest

from scripts.latency_estimator.latency_estimator import LatencyEstimator
from scripts.optimizer.orbit.orbit_core import orbit_core
from scripts.params.params import Params
from scripts.tdag import addition_squash, auto_compress, build_from_mlir


def test_orbit_core_motivation_pulp(motivation_mlir: str, toy_cost_json: str):
    pytest.importorskip("pulp")
    params = Params(
        toy_cost_json,
        "Orbit",
        "compile",
        placement_backend="ilp",
        ilp_solver="pulp",
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
