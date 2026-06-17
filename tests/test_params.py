"""Params loading."""

from __future__ import annotations

from scripts.params.params import Params


def test_params_toy_runtime(toy_cost_json: str):
    p = Params(toy_cost_json, "Orbit", "compile")
    assert p.backend == "Toy"
    assert p.mode == "compile"
