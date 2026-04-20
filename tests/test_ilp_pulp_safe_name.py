"""PuLP name sanitization used for MILP problem/constraint names."""

from __future__ import annotations

from scripts.optimizer.orbit.ilp_core import _pulp_safe_name


def test_pulp_safe_name_strips_spaces():
    assert _pulp_safe_name("a b c") == "a_b_c"
    assert " " not in _pulp_safe_name("Task (Partition-x y_z)")


def test_pulp_safe_name_non_empty():
    assert _pulp_safe_name("   ") == "orbit_ilp"


def test_pulp_safe_name_truncates():
    long = "x" * 500
    out = _pulp_safe_name(long, max_len=50)
    assert len(out) == 50
