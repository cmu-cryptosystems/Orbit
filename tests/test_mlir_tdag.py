"""MLIR → Tdag pipeline (compile mode)."""

from __future__ import annotations

import networkx as nx

from scripts.tdag import build_from_mlir


def test_build_from_mlir_motivation(motivation_mlir: str, toy_cost_json: str):
    from scripts.params.params import Params

    params = Params(toy_cost_json, "Orbit", "compile")
    tdag = build_from_mlir(motivation_mlir, params)
    assert len(tdag.nodes) >= 1
    assert nx.is_directed_acyclic_graph(tdag)
    assert len(tdag.inputs) >= 1
    assert len(tdag.outputs) >= 1
