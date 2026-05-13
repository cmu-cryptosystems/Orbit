"""
Integration test for the Rotom → Orbit pipeline.

Generates a synthetic Rotom-format circuit (manifest + kernel files),
deserializes it with build_from_rotom, and validates the resulting Tdag
against the invariants that Orbit's optimizer expects.

Run from the Orbit project root:
    python -m scripts.tdag.test_rotom2tdag
"""

import json
import os
import tempfile
import networkx as nx

from .rotom2tdag import build_from_rotom
from ..params.params import Params


# ---------------------------------------------------------------------------
# helpers to synthesise Rotom-format files without needing Rotom installed
# ---------------------------------------------------------------------------

def _write_kernel_file(path, instructions):
    """Write a list of raw instruction strings to a kernel .txt file."""
    with open(path, 'w') as f:
        f.write("# HE Kernel Instruction File\n")
        f.write("# Kernel Index: 0\n")
        f.write("# Format: {index} {is_secret}: {operation}\n")
        f.write("#" + "=" * 70 + "\n\n")
        for line in instructions:
            f.write(line + "\n")


def _write_manifest(path, circuit_name, kernels):
    """Write a manifest JSON matching CircuitSerializer's output."""
    manifest = {"circuit_name": circuit_name, "kernels": kernels}
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)


def _make_params():
    """Create a minimal Params for compile mode (no cost JSON needed)."""
    p = Params(le_json=None, sysname="test", mode="compile")
    p.poly_deg = 32768
    p.max_slot = p.poly_deg // 2
    p.bts_ub = 14
    p.bts_lb = 1
    p.lvl_ub = 14
    p.lvl_lb = 1
    p.Sf = 51
    p.Sw = 40
    p.Csw = 40
    p.backend = ""
    p.latency_table = {}
    p.bpsdepth = None
    p.threads = 1
    p.comp = True
    p.part = True
    p.reqbp = False
    p.netname = "test"
    p.trunc_val = 1
    p.dacapo_mlir_in = False
    p.dacapo_mlir_out = False
    return p


# ---------------------------------------------------------------------------
# test cases
# ---------------------------------------------------------------------------

def test_basic_add_circuit():
    """Two packed inputs → add → single output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_add"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "# Ciphertext 0",
            "0 True: pack (a[0:64])",
            "1 True: pack (b[0:64])",
            "2 True: (+ 0 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "add",
                "layout": "test_layout",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert len(tdag.nodes) == 3, f"expected 3 nodes, got {len(tdag.nodes)}"
        assert len(tdag.edges) == 2, f"expected 2 edges, got {len(tdag.edges)}"
        assert tdag.inputs == {'0', '1'}
        assert tdag.outputs == {'2'}
        assert tdag.nodes['2']['op'] == 'add'
        assert tdag.nodes['2']['op_descr'] == {'single': 0, 'double': 1}
        assert tdag.nodes['0']['op'] == 'input'
        assert tdag.nodes['1']['op'] == 'input'
        assert nx.is_directed_acyclic_graph(tdag)
    print("  PASS  test_basic_add_circuit")


def test_mul_ct_pt():
    """ct * pt → single mul."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_mul"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x[0:64])",
            "1 False: mask [1,0,1,0]",
            "2 True: (* 0 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "mul",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['2']['op'] == 'mul'
        assert tdag.nodes['2']['op_descr'] == {'single': 1, 'double': 0}, \
            f"expected single mul, got {tdag.nodes['2']['op_descr']}"
        assert tdag.nodes['1']['op'] == 'constant'
        assert '1' not in tdag.inputs, "pl mask should not be in tdag.inputs"
        assert tdag.inputs == {'0'}, "only ci pack should be an input"
    print("  PASS  test_mul_ct_pt")


def test_rotation():
    """Rotation carries offset and NAF weight."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_rot"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x[0:64])",
            "1 True: (<< 0 5)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "rot",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1],
                "dependencies": [],
                "outputs": [1],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['1']['op'] == 'rotate'
        assert tdag.nodes['1']['op_descr']['offset'] == 5
        assert tdag.nodes['1']['op_descr']['weight'] >= 1
    print("  PASS  test_rotation")


def test_sub_maps_to_add():
    """SUB in Rotom maps to 'add' in Tdag (negate is free in CKKS)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_sub"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (a)",
            "1 True: pack (b)",
            "2 True: (- 0 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "sub",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['2']['op'] == 'add'
    print("  PASS  test_sub_maps_to_add")


def test_rescale():
    """Rescale instruction becomes 'rescale' op with empty op_descr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_res"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x)",
            "1 True: (rescale 0 / 2^40)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "rescale",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1],
                "dependencies": [],
                "outputs": [1],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['1']['op'] == 'rescale'
        assert tdag.nodes['1']['op_descr'] == {}
    print("  PASS  test_rescale")


def test_v2_format_ci_pl():
    """Deserializer handles v2 ci/pl tokens from improved serializer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_v2"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 ci: pack (x)",
            "1 pl: mask [1,1,1]",
            "2 ci: (* 0 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "mul",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['2']['op'] == 'mul'
        assert tdag.nodes['2']['op_descr'] == {'single': 1, 'double': 0}
    print("  PASS  test_v2_format_ci_pl")


def test_multi_kernel():
    """Cross-kernel references via global instruction indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_multi"
        # kernel 0: two packs and an add
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (a)",
            "1 True: pack (b)",
            "2 True: (+ 0 1)",
        ])
        # kernel 1: references kernel 0's output via cs_ref, rotates it
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_1.txt"), [
            "3 True: 2",
            "4 True: (<< 3 -7)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "add",
                "layout": "l0",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            },
            {
                "kernel_idx": 1,
                "operation": "rot",
                "layout": "l1",
                "num_ciphertexts": 1,
                "instructions": [3, 4],
                "dependencies": ["l0"],
                "outputs": [4],
            },
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert len(tdag.nodes) == 5
        assert tdag.has_edge('2', '3'), "cs_ref should create edge from referenced node"
        assert tdag.has_edge('3', '4')
        # kernel 0 layout "l0" is consumed by kernel 1, so only kernel 1
        # is terminal — its outputs are the real circuit outputs.
        assert tdag.outputs == {'4'}, \
            f"only terminal kernel outputs should be tracked, got {tdag.outputs}"
        assert tdag.nodes['4']['op'] == 'rotate'
        assert tdag.nodes['4']['op_descr']['offset'] == -7
        assert nx.is_directed_acyclic_graph(tdag)
    print("  PASS  test_multi_kernel")


def test_fallback_outputs():
    """When manifest has no outputs, sink nodes become outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_sink"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x)",
            "1 True: (<< 0 3)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "rot",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1],
                "dependencies": [],
                "outputs": [],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.outputs == {'1'}, \
            f"sink node '1' should be output, got {tdag.outputs}"
    print("  PASS  test_fallback_outputs")


def test_compile_mode_levels_none():
    """All level/scale values should be None in compile mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_none"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x)",
            "1 True: pack (y)",
            "2 True: (* 0 1)",
            "3 True: (<< 2 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "matmul",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2, 3],
                "dependencies": [],
                "outputs": [3],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        for v in tdag.nodes:
            assert tdag.nodes[v]['level'] is None, \
                f"node {v} level should be None in compile mode"
            assert tdag.nodes[v]['scale'] is None, \
                f"node {v} scale should be None in compile mode"
    print("  PASS  test_compile_mode_levels_none")


def test_metadata_preserved():
    """Trailing # comments are stored in the node's comment field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_meta"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (x) # conv2d_layer1",
            "1 True: (<< 0 3) # rotate_step",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "conv",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1],
                "dependencies": [],
                "outputs": [1],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['0']['comment'] == 'conv2d_layer1'
        assert tdag.nodes['1']['comment'] == 'rotate_step'
    print("  PASS  test_metadata_preserved")


def test_pl_pack_is_constant():
    """Plaintext packs (pl/False) should become 'constant', not 'input'."""
    with tempfile.TemporaryDirectory() as tmpdir:
        name = "test_plpack"
        _write_kernel_file(os.path.join(tmpdir, f"{name}_kernel_0.txt"), [
            "0 True: pack (ct_data)",
            "1 False: pack (weight_matrix)",
            "2 True: (* 0 1)",
        ])
        _write_manifest(os.path.join(tmpdir, f"{name}_manifest.json"), name, [
            {
                "kernel_idx": 0,
                "operation": "matmul",
                "layout": "l",
                "num_ciphertexts": 1,
                "instructions": [0, 1, 2],
                "dependencies": [],
                "outputs": [2],
            }
        ])

        params = _make_params()
        tdag = build_from_rotom(
            os.path.join(tmpdir, f"{name}_manifest.json"), params)

        assert tdag.nodes['0']['op'] == 'input'
        assert tdag.nodes['1']['op'] == 'constant'
        assert tdag.nodes['1']['op_descr'] == {'value': 0, 'rms_var': 0.0}
        assert tdag.inputs == {'0'}, \
            f"only ci pack should be input, got {tdag.inputs}"
        assert tdag.nodes['2']['op_descr'] == {'single': 1, 'double': 0}, \
            "ct * pt should be single mul"
    print("  PASS  test_pl_pack_is_constant")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def run_all():
    print("Running rotom2tdag integration tests...\n")
    test_basic_add_circuit()
    test_mul_ct_pt()
    test_rotation()
    test_sub_maps_to_add()
    test_rescale()
    test_v2_format_ci_pl()
    test_multi_kernel()
    test_fallback_outputs()
    test_compile_mode_levels_none()
    test_metadata_preserved()
    test_pl_pack_is_constant()
    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all()
