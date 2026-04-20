"""Unit tests for Params.check_res and Params.check_resbts (pure numeric invariants)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.params.params import Params


def _params_with(sf: int, sw: int, bts_lb: int, bts_ub: int, lvl_lb: int, lvl_ub: int, tmp_path: Path) -> Params:
    cfg = {
        "runtime": "Toy",
        "poly_deg": 8192,
        "bootstrapLevelLowerBound": bts_lb,
        "bootstrapLevelUpperBound": bts_ub,
        "levelLowerBound": lvl_lb,
        "levelUpperBound": lvl_ub,
        "rescalingFactor": sf,
        "latencyTable": {},
    }
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return Params(str(p), "Orbit", "compile", Sw=sw)


def test_check_res_same_point_ok(tmp_path: Path):
    p = _params_with(sf=51, sw=40, bts_lb=3, bts_ub=16, lvl_lb=1, lvl_ub=16, tmp_path=tmp_path)
    assert p.check_res(10, 51, 10, 51) is True


def test_check_res_invalid_level_order(tmp_path: Path):
    p = _params_with(sf=51, sw=40, bts_lb=3, bts_ub=16, lvl_lb=1, lvl_ub=16, tmp_path=tmp_path)
    assert p.check_res(5, 51, 10, 51) is False


def test_check_res_invalid_scale_budget(tmp_path: Path):
    p = _params_with(sf=51, sw=40, bts_lb=3, bts_ub=16, lvl_lb=1, lvl_ub=16, tmp_path=tmp_path)
    # Same level but output scale demands more budget than available at this level
    assert p.check_res(10, 300, 10, 200) is False


def test_check_resbts_falls_back_to_bootstrap_path(tmp_path: Path):
    p = _params_with(sf=51, sw=40, bts_lb=3, bts_ub=16, lvl_lb=1, lvl_ub=16, tmp_path=tmp_path)
    # Level increases, so plain check_res fails, but bootstrap can lift the budgeted level first
    assert p.check_res(6, 51, 8, 51) is False
    assert p.check_resbts(6, 51, 8, 51) is True


def test_check_resbts_false_when_bootstrap_window_invalid(tmp_path: Path):
    p = _params_with(sf=51, sw=40, bts_lb=3, bts_ub=10, lvl_lb=1, lvl_ub=16, tmp_path=tmp_path)
    # Feasible to bts_lb, but post-bootstrap target level leaves out_lvl+r outside (bts_lb, bts_ub]
    assert p.check_res(6, 51, 20, 51) is False
