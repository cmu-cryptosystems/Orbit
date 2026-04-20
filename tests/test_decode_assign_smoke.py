"""Minimal decode_assign → check_tdag smoke (no ILP / no benchmark MLIR)."""

from __future__ import annotations

import pytest

from scripts.assignment.assignment import Assign
from scripts.assignment.decode_assign import decode_assign
from scripts.params.params import Params
from scripts.tdag.tdag import Tdag


def _tiny_linear_negate_graph(toy_cost_json: str) -> tuple[Tdag, Assign, dict[str, str]]:
    """arg0 --negate--> n1 (single predecessor, same level/scale in/out)."""
    p = Params(toy_cost_json, "Orbit", "compile", Sw=40)
    g = Tdag(p, name="tiny")
    g.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_node("n1", op="negate", weight=1, level=None, scale=None, op_descr={}, comment="")
    g.add_edge("arg0", "n1", weight=1)
    g.inputs = {"arg0"}
    g.outputs = {"n1"}

    a = Assign(g)
    lvl, scl = 10, 51
    a.v_lvl_in["arg0"] = lvl
    a.v_scl_in["arg0"] = scl
    a.v_lvl_out["arg0"] = lvl
    a.v_scl_out["arg0"] = scl
    a.e_lvl_out[("arg0", "n1")] = lvl
    a.e_scl_out[("arg0", "n1")] = scl
    a.v_lvl_out["n1"] = lvl
    a.v_scl_out["n1"] = scl

    og_to_comp = {"arg0": "arg0", "n1": "n1"}
    return g, a, og_to_comp


def test_decode_assign_linear_negate_passes_check_tdag(toy_cost_json: str):
    og, assign, og_to_comp = _tiny_linear_negate_graph(toy_cost_json)
    fdag = decode_assign(assign, og, og_to_comp)
    assert fdag.has_node("n1_out") or any("n1" in n for n in fdag.nodes)
    assert len(fdag.nodes) >= 2


def test_decode_assign_rejects_inconsistent_edge_assignments(toy_cost_json: str):
    og, assign, og_to_comp = _tiny_linear_negate_graph(toy_cost_json)
    # Break the edge scale so rescale/bootstrap insertion cannot reconcile preds
    assign.e_scl_out[("arg0", "n1")] = 10
    assign.v_scl_out["arg0"] = 51
    with pytest.raises((AssertionError, ValueError)):
        decode_assign(assign, og, og_to_comp)
