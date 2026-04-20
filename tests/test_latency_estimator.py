"""LatencyEstimator loads with toy cost JSON."""

from __future__ import annotations

from scripts.latency_estimator.latency_estimator import LatencyEstimator
from scripts.params.params import Params


def test_latency_estimator_toy(toy_cost_json: str):
    params = Params(toy_cost_json, "Orbit", "compile")
    le = LatencyEstimator(params)
    assert "add_single" in le.lin_op_lmaps
