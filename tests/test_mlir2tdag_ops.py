"""MLIR → Tdag parsing for individual earth ops (small synthetic snippets)."""

from __future__ import annotations

import textwrap

import pytest

from scripts.params.params import Params
from scripts.tdag import build_from_mlir


def _write_mlir(tmp_path, body: str) -> str:
    path = tmp_path / "snippet.mlir"
    path.write_text(
        textwrap.dedent(
            f'''
            "builtin.module"() <{{sym_name = "test"}}> ({{
              "func.func"() <{{function_type = (tensor<1x!earth.ci<40 * 10>>) -> tensor<1x!earth.ci<40 * 10>>, sym_name = "f"}}> ({{
            {body}
              }}) : () -> () loc(unknown)
            }}) : () -> () loc(unknown)
            '''
        ).strip()
    )
    return str(path)


def test_build_from_mlir_negate(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.negate"(%arg0) : (tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%0) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    g = build_from_mlir(mlir, p)
    assert "0" in g.nodes
    assert g.nodes["0"]["op"] == "negate"


def test_build_from_mlir_constant(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.constant"() <{rms_var = 1.0 : f64, value = 7 : i64}> : () -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%0) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    g = build_from_mlir(mlir, p)
    assert g.nodes["0"]["op"] == "constant"
    assert g.nodes["0"]["op_descr"]["value"] == 7
    assert g.nodes["0"]["op_descr"]["rms_var"] == 1.0


def test_build_from_mlir_rotate_offset(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.rotate"(%arg0) <{offset = array<i64: -3>}> : (tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%0) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    g = build_from_mlir(mlir, p)
    assert g.nodes["0"]["op"] == "rotate"
    assert g.nodes["0"]["op_descr"]["offset"] == -3
    assert "weight" in g.nodes["0"]["op_descr"]


def test_build_from_mlir_mul_ci_ci(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.mul"(%arg0, %arg0) : (tensor<1x!earth.ci<40 * 10>>, tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%0) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    g = build_from_mlir(mlir, p)
    assert g.nodes["0"]["op"] == "mul"
    assert g.nodes["0"]["op_descr"]["double"] == 1


def test_build_from_mlir_modswitch_upscale_bootstrap(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.modswitch"(%arg0) <{downFactor = 2 : i64}> : (tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      %1 = "earth.upscale"(%0) <{upFactor = 5 : i64}> : (tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      %2 = "earth.bootstrap"(%1) <{targetLevel = 8 : i64}> : (tensor<1x!earth.ci<40 * 10>>) -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%2) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    g = build_from_mlir(mlir, p)
    assert g.nodes["0"]["op_descr"]["downFactor"] == 2
    assert g.nodes["1"]["op_descr"]["upFactor"] == 5
    # Dacapo bootstrap target is mirrored against bts_ub in mlir2tdag
    assert g.nodes["2"]["op_descr"]["targetLevel"] == p.bts_ub - 8


def test_build_from_mlir_malformed_constant_raises(toy_cost_json: str, tmp_path):
    mlir = _write_mlir(
        tmp_path,
        '''
    ^bb0(%arg0: tensor<1x!earth.ci<40 * 10>> loc("t")):
      %0 = "earth.constant"() <{oops = 1 : i64}> : () -> (tensor<1x!earth.ci<40 * 10>>) loc("t")
      "func.return"(%0) : (tensor<1x!earth.ci<40 * 10>>) -> () loc("t")
    ''',
    )
    p = Params(toy_cost_json, "Orbit", "compile")
    with pytest.raises(Exception, match="constant"):
        build_from_mlir(mlir, p)
