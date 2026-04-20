"""Pytest fixtures: run from Orbit repo root (see pytest.ini pythonpath)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

ORBIT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def orbit_root() -> Path:
    return ORBIT_ROOT


@pytest.fixture
def toy_cost_json(orbit_root: Path) -> str:
    p = orbit_root / "cost_models" / "toy_backend.json"
    assert p.is_file(), f"Missing {p}"
    return str(p)


@pytest.fixture
def motivation_mlir(orbit_root: Path) -> str:
    p = orbit_root / "mlirs_input" / "motivation.mlir"
    assert p.is_file(), f"Missing {p}"
    return str(p)
