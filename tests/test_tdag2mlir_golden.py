"""Stable substring checks for ``tdag_to_mlir`` emission (no full MLIR parser)."""

from __future__ import annotations

from scripts.params.params import Params
from scripts.tdag.tdag import Tdag
from scripts.tdag.tdag2mlir import tdag_to_mlir


def _params(toy_cost_json: str) -> Params:
    return Params(toy_cost_json, "Orbit", "compile", Sw=40)


def test_tdag_to_mlir_negate_snippet(toy_cost_json: str, tmp_path):
    p = _params(toy_cost_json)
    g = Tdag(p, name="golden_neg")
    g.add_node("arg0", op="input", weight=1, level=10, scale=51, op_descr={}, comment="// in")
    g.add_node("n1", op="negate", weight=1, level=10, scale=51, op_descr={}, comment="// neg")
    g.add_edge("arg0", "n1", weight=1)
    g.inputs = {"arg0"}
    g.outputs = {"n1"}

    out = tmp_path / "out.mlir"
    tdag_to_mlir(g, str(out))
    text = out.read_text()
    # Dacapo level flip on output: bts_ub - 10 == 6 for toy cost model
    assert "sym_name = \"mlirs/.mlir\"" in text
    assert "^bb0(" in text
    assert "%arg0:" in text
    assert "earth.negate" in text
    assert "tensor<1x!earth.ci<51 * 6>>" in text
    assert "\"func.return\"" in text


def test_tdag_to_mlir_rotate_and_bootstrap_attrs(toy_cost_json: str, tmp_path):
    p = _params(toy_cost_json)
    g = Tdag(p, name="golden_rb")
    g.add_node("arg0", op="input", weight=1, level=10, scale=51, op_descr={}, comment="")
    g.add_node("r1", op="rotate", weight=1, level=10, scale=51, op_descr={"offset": -2, "weight": 3}, comment="// rot")
    g.add_node("b1", op="bootstrap", weight=1, level=12, scale=51, op_descr={"targetLevel": 12}, comment="// bts")
    g.add_edge("arg0", "r1", weight=1)
    g.add_edge("r1", "b1", weight=1)
    g.inputs = {"arg0"}
    g.outputs = {"b1"}

    out = tmp_path / "out.mlir"
    tdag_to_mlir(g, str(out))
    text = out.read_text()
    assert "earth.rotate" in text and "offset = array<i64: -2>" in text
    assert "earth.bootstrap" in text
    # Written targetLevel is mirrored for Dacapo output
    assert f"targetLevel = {p.bts_ub - 12} : i64" in text


def test_tdag_to_mlir_constant_uses_plaintext_type(toy_cost_json: str, tmp_path):
    p = _params(toy_cost_json)
    g = Tdag(p, name="golden_cst")
    g.add_node("c0", op="constant", weight=1, level=5, scale=45, op_descr={"rms_var": 0.25, "value": 3}, comment="// c")
    g.add_node("n1", op="negate", weight=1, level=5, scale=45, op_descr={}, comment="")
    g.add_edge("c0", "n1", weight=1)
    g.inputs = {"c0"}  # constant as graph entry (in_degree 0)
    g.outputs = {"n1"}

    out = tmp_path / "out.mlir"
    tdag_to_mlir(g, str(out))
    text = out.read_text()
    assert "earth.constant" in text
    assert "rms_var = 0.25 : f64" in text and "value = 3 : i64" in text
    assert "!earth.pl<" in text


def test_tdag_to_mlir_allows_mixed_compressed_add_descriptor(toy_cost_json: str, tmp_path):
    p = _params(toy_cost_json)
    g = Tdag(p, name="mixed_add")
    g.add_node("arg0", op="input", weight=1, level=10, scale=51, op_descr={}, comment="")
    g.add_node("arg1", op="input", weight=1, level=10, scale=51, op_descr={}, comment="")
    g.add_node(
        "add",
        op="add",
        weight=1,
        level=10,
        scale=51,
        op_descr={"single": 1, "double": 1},
        comment="// mixed compressed add",
    )
    g.add_edge("arg0", "add", weight=1)
    g.add_edge("arg1", "add", weight=1)
    g.inputs = {"arg0", "arg1"}
    g.outputs = {"add"}

    out = tmp_path / "out.mlir"
    tdag_to_mlir(g, str(out))
    text = out.read_text()

    assert "earth.add" in text
    assert "mixed compressed add" in text
