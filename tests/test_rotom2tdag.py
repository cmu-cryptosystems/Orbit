from __future__ import annotations

import json
from pathlib import Path

from scripts.params.params import Params
from scripts.tdag.rotom2tdag import build_from_rotom


def test_rotom_plaintext_pack_preserves_constant_table_index(
    toy_cost_json: str, tmp_path: Path
):
    manifest = {
        "format_version": 2,
        "circuit_name": "toy_rotom",
        "kernels": [
            {
                "kernel_idx": 0,
                "layout": "direct:toy",
                "instructions": [0, 1, 2],
                "outputs": [2],
            }
        ],
    }
    manifest_path = tmp_path / "toy_rotom_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "toy_rotom_kernel_0.txt").write_text(
        "\n".join(
            [
                "# Format: {index} {ci|pl}: {operation}",
                "0 ci: pack ([0:1:1]) # scope=input;op=input",
                "1 pl: pack ([C]) # scope=classifier.weight;op=weight",
                "2 ci: (* 0 1) # scope=classifier;op=linear",
            ]
        ),
        encoding="utf-8",
    )
    params = Params(toy_cost_json, "Orbit", "compile")

    tdag = build_from_rotom(str(manifest_path), params)

    assert tdag.nodes["1"]["op"] == "constant"
    assert tdag.nodes["1"]["op_descr"] == {"value": 1, "rms_var": 0.0}


def test_rotom_plaintext_cross_kernel_reference_uses_referenced_index(
    toy_cost_json: str, tmp_path: Path
):
    manifest = {
        "format_version": 2,
        "circuit_name": "toy_rotom",
        "kernels": [
            {
                "kernel_idx": 0,
                "layout": "direct:toy",
                "instructions": [0, 1, 2],
                "outputs": [2],
            }
        ],
    }
    manifest_path = tmp_path / "toy_rotom_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "toy_rotom_kernel_0.txt").write_text(
        "\n".join(
            [
                "# Format: {index} {ci|pl}: {operation}",
                "0 ci: pack ([0:1:1]) # scope=input;op=input",
                "1 pl: pack ([C]) # scope=shared.weight;op=weight",
                "2 pl: 1 # scope=shared.weight.copy;op=weight",
            ]
        ),
        encoding="utf-8",
    )
    params = Params(toy_cost_json, "Orbit", "compile")

    tdag = build_from_rotom(str(manifest_path), params)

    assert tdag.nodes["2"]["op"] == "constant"
    assert tdag.nodes["2"]["op_descr"] == {"value": 1, "rms_var": 0.0}
