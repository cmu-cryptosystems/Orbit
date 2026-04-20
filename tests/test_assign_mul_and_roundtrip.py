"""Assign deduce / check_assign for ``mul``, and ``to_dict`` / ``from_dict``."""

from __future__ import annotations

import pytest

from scripts.assignment.assignment import Assign
from scripts.params.params import Params
from scripts.tdag.tdag import Tdag


def _p(toy_cost_json: str) -> Params:
    return Params(toy_cost_json, "Orbit", "compile", Sw=40)


def _mul_two_preds_graph(p: Params) -> Tdag:
    g = Tdag(p, name="m2")
    g.add_node("a", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_node("b", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_node("m", op="mul", weight=1, level=None, scale=None, op_descr={"single": 0, "double": 1}, comment="")
    g.add_edge("a", "m", weight=1)
    g.add_edge("b", "m", weight=1)
    g.inputs = {"a", "b"}
    g.outputs = {"m"}
    return g


def test_deduce_mul_two_predecessors_scales_sum(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    a.v_lvl_out.update({"a": 10, "b": 10, "m": 10})
    a.v_scl_out.update({"a": 10, "b": 10, "m": 20})
    a.e_lvl_out[("a", "m")] = 10
    a.e_scl_out[("a", "m")] = 10
    a.e_lvl_out[("b", "m")] = 10
    a.e_scl_out[("b", "m")] = 10
    assert a._deduce_in_lvl_scl("m") == (10, 20)


def test_deduce_mul_single_predecessor_doubles_scale(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = Tdag(p, name="m1")
    g.add_node("a", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_node("m", op="mul", weight=1, level=None, scale=None, op_descr={"single": 0, "double": 1}, comment="")
    g.add_edge("a", "m", weight=1)
    g.inputs = {"a"}
    g.outputs = {"m"}
    a = Assign(g)
    a.v_lvl_out.update({"a": 8, "m": 8})
    a.v_scl_out.update({"a": 40, "m": 80})
    a.e_lvl_out[("a", "m")] = 8
    a.e_scl_out[("a", "m")] = 40
    assert a._deduce_in_lvl_scl("m") == (8, 80)


def test_deduce_mul_inconsistent_levels_raises(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    a.v_lvl_out.update({"a": 10, "b": 10, "m": 10})
    a.v_scl_out.update({"a": 10, "b": 10, "m": 20})
    a.e_lvl_out[("a", "m")] = 10
    a.e_scl_out[("a", "m")] = 10
    a.e_lvl_out[("b", "m")] = 11
    a.e_scl_out[("b", "m")] = 10
    with pytest.raises(ValueError, match="inconsistent input levels"):
        a._deduce_in_lvl_scl("m")


def test_deduce_mul_three_predecessors_raises(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = Tdag(p, name="m3")
    for lab in ("x", "y", "z"):
        g.add_node(lab, op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_node("m", op="mul", weight=1, level=None, scale=None, op_descr={"single": 0, "double": 1}, comment="")
    for lab in ("x", "y", "z"):
        g.add_edge(lab, "m", weight=1)
    g.inputs = {"x", "y", "z"}
    g.outputs = {"m"}
    a = Assign(g)
    for lab in ("x", "y", "z", "m"):
        a.v_lvl_out[lab] = 10
        a.v_scl_out[lab] = 30
    for lab in ("x", "y", "z"):
        a.e_lvl_out[(lab, "m")] = 10
        a.e_scl_out[(lab, "m")] = 10
    with pytest.raises(ValueError, match="must have 1 or 2 predecessors"):
        a._deduce_in_lvl_scl("m")


def test_get_v_in_lvl_scl_rejects_mul_override_mismatch(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    a.v_lvl_out.update({"a": 10, "b": 10, "m": 10})
    a.v_scl_out.update({"a": 10, "b": 10, "m": 20})
    a.e_lvl_out[("a", "m")] = 10
    a.e_scl_out[("a", "m")] = 10
    a.e_lvl_out[("b", "m")] = 10
    a.e_scl_out[("b", "m")] = 10
    a.v_scl_in["m"] = 99
    with pytest.raises(ValueError, match="inconsistent input scale"):
        a.get_v_in_lvl_scl("m")


def test_check_assign_mul_valid(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    lvl, es = 10, 51
    a.v_lvl_in.update({"a": lvl, "b": lvl})
    a.v_scl_in.update({"a": es, "b": es})
    a.v_lvl_out.update({"a": lvl, "b": lvl, "m": lvl})
    a.v_scl_out.update({"a": es, "b": es, "m": es + es})
    a.e_lvl_out[("a", "m")] = lvl
    a.e_scl_out[("a", "m")] = es
    a.e_lvl_out[("b", "m")] = lvl
    a.e_scl_out[("b", "m")] = es
    assert a.check_assign() is True


def test_check_assign_missing_node_output_raises(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    lvl, es = 10, 51
    a.v_lvl_in.update({"a": lvl, "b": lvl})
    a.v_scl_in.update({"a": es, "b": es})
    a.v_lvl_out.update({"a": lvl, "b": lvl})  # omit "m"
    a.v_scl_out.update({"a": es, "b": es})
    a.e_lvl_out[("a", "m")] = lvl
    a.e_scl_out[("a", "m")] = es
    a.e_lvl_out[("b", "m")] = lvl
    a.e_scl_out[("b", "m")] = es
    with pytest.raises(ValueError, match="missing output level/scale assignment"):
        a.check_assign()


def test_check_assign_mul_missing_edge_assignment_raises(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    lvl, es = 10, 51
    a.v_lvl_in.update({"a": lvl, "b": lvl})
    a.v_scl_in.update({"a": es, "b": es})
    a.v_lvl_out.update({"a": lvl, "b": lvl, "m": lvl})
    a.v_scl_out.update({"a": es, "b": es, "m": es + es})
    a.e_lvl_out[("a", "m")] = lvl
    a.e_scl_out[("a", "m")] = es
    # Deliberately omit assignment for edge (b -> m) — deduce path raises KeyError first
    with pytest.raises(KeyError):
        a.check_assign()


def test_check_assign_mul_invalid_transition_via_stub(toy_cost_json: str, monkeypatch: pytest.MonkeyPatch):
    """``check_resbts`` is permissive with bootstrap; stub it to assert the guard is wired."""
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    lvl, es = 10, 51
    a.v_lvl_in.update({"a": lvl, "b": lvl})
    a.v_scl_in.update({"a": es, "b": es})
    a.v_lvl_out.update({"a": lvl, "b": lvl, "m": lvl})
    a.v_scl_out.update({"a": es, "b": es, "m": es + es})
    a.e_lvl_out[("a", "m")] = lvl
    a.e_scl_out[("a", "m")] = es
    a.e_lvl_out[("b", "m")] = lvl
    a.e_scl_out[("b", "m")] = es

    def bad(*_args, **_kwargs):
        return False

    monkeypatch.setattr(type(p), "check_resbts", bad)
    with pytest.raises(ValueError, match="invalid level/scale transition"):
        a.check_assign()


def test_assign_to_dict_from_dict_roundtrip(toy_cost_json: str):
    p = _p(toy_cost_json)
    g = _mul_two_preds_graph(p)
    a = Assign(g)
    a.v_lvl_out = {"a": 1, "m": 2}
    a.v_scl_out = {"a": 3, "m": 6}
    a.v_lvl_in = {"m": 2}
    a.v_scl_in = {"m": 6}
    a.e_lvl_out = {("a", "m"): 2, ("b", "m"): 2}
    a.e_scl_out = {("a", "m"): 3, ("b", "m"): 3}

    d = a.to_dict()
    b = Assign.from_dict(g, d)
    assert b.v_lvl_out == a.v_lvl_out
    assert b.v_scl_out == a.v_scl_out
    assert b.v_lvl_in == a.v_lvl_in
    assert b.v_scl_in == a.v_scl_in
    assert b.e_lvl_out == a.e_lvl_out
    assert b.e_scl_out == a.e_scl_out
    assert b.e_lvl_out[("a", "m")] == 2
