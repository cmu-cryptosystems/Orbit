"""Params loading and ilp_solver validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.params.params import Params


def test_params_toy_runtime(toy_cost_json: str):
    p = Params(toy_cost_json, "Orbit", "compile")
    assert p.backend == "Toy"
    assert p.ilp_solver == "gurobi"
    assert p.mode == "compile"


def test_params_ilp_solver_override(toy_cost_json: str):
    p = Params(toy_cost_json, "Orbit", "compile", ilp_solver="pulp")
    assert p.ilp_solver == "pulp"


def test_params_ilp_solver_from_json(tmp_path: Path, toy_cost_json: str):
    with open(toy_cost_json) as f:
        data = json.load(f)
    data["ilp_solver"] = "pulp"
    jf = tmp_path / "cfg.json"
    with open(jf, "w") as f:
        json.dump(data, f)
    p = Params(str(jf), "Orbit", "compile")
    assert p.ilp_solver == "pulp"


def test_params_rejects_bad_ilp_solver(toy_cost_json: str):
    with pytest.raises(ValueError, match="ilp_solver"):
        Params(toy_cost_json, "Orbit", "compile", ilp_solver="not_a_solver")
