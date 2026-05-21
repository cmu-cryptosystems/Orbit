from __future__ import annotations

import re
from typing import Any

from lower.layout_cts import LayoutCiphertexts

from ..params.params import Params
from .rotom2tdag import (
    _classify_operation,
    _parse_instruction,
    mark_tdag_outputs_from_manifest,
    populate_tdag_from_all_instr,
)
from .tdag import Tdag


def _canonicalize_operation(operation: str) -> str:
    for symbol in ("+", "*"):
        match = re.match(rf"^\({re.escape(symbol)}\s+(\d+)\s+(\d+)\)$", operation)
        if match:
            a, b = int(match.group(1)), int(match.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            return f"({symbol} {lo} {hi})"
    return operation


def _ingest_he_terms_instruction_stream(
    he_terms_dict: dict[int, Any],
    global_env: dict,
    all_instr: dict,
    kernel_idx: int,
) -> dict:
    for _ct_idx, he_term in sorted(he_terms_dict.items(), key=lambda x: x[0]):
        instrs, global_env = he_term.instrs(env=global_env)
        for line in instrs:
            parsed = _parse_instruction(line)
            if parsed is None:
                continue
            parsed["operation"] = _canonicalize_operation(parsed["operation"])
            op_type, op_info = _classify_operation(parsed["operation"])
            all_instr[parsed["index"]] = {
                **parsed,
                "op_type": op_type,
                "op_info": op_info,
                "kernel_idx": kernel_idx,
            }
    return global_env


def build_all_instr_and_manifest_from_circuit_ir(
    circuit_ir: dict,
    circuit_name: str,
) -> tuple[dict, dict]:
    global_env: dict = {}
    all_instr: dict = {}
    kernels_meta: list[dict] = []

    for kernel_idx, (kernel_term, layout_cts) in enumerate(circuit_ir.items()):
        he_terms_dict = layout_cts.cts if isinstance(layout_cts, LayoutCiphertexts) else layout_cts
        kernel_metadata: dict[str, Any] = {
            "kernel_idx": kernel_idx,
            "operation": str(kernel_term.op),
            "layout": str(kernel_term.layout),
            "num_ciphertexts": len(he_terms_dict),
            "instructions": [],
            "dependencies": [],
            "outputs": [],
        }
        if hasattr(kernel_term, "cs") and kernel_term.cs:
            for child in kernel_term.cs:
                if hasattr(child, "layout"):
                    kernel_metadata["dependencies"].append(str(child.layout))

        global_env = _ingest_he_terms_instruction_stream(
            he_terms_dict, global_env, all_instr, kernel_idx
        )

        for _ct_idx, he_term in sorted(he_terms_dict.items(), key=lambda x: x[0]):
            if he_term in global_env:
                kernel_metadata["outputs"].append(global_env[he_term])

        kernels_meta.append(kernel_metadata)

    return all_instr, {"circuit_name": circuit_name, "kernels": kernels_meta}


def build_from_circuit_ir(
    circuit_ir: dict,
    params: Params,
    circuit_name: str = "circuit",
) -> Tdag:
    all_instr, manifest = build_all_instr_and_manifest_from_circuit_ir(
        circuit_ir, circuit_name
    )
    tdag = Tdag(params, name=circuit_name)
    populate_tdag_from_all_instr(tdag, all_instr, params)
    mark_tdag_outputs_from_manifest(tdag, manifest)
    return tdag


def build_kernel_tdags_from_circuit_ir(
    circuit_ir: dict,
    params: Params,
    circuit_name: str = "circuit",
) -> list[Tdag]:
    all_instr, manifest = build_all_instr_and_manifest_from_circuit_ir(
        circuit_ir, circuit_name
    )
    kernel_tdags = []
    for kernel_info in manifest["kernels"]:
        kernel_idx = kernel_info["kernel_idx"]
        kernel_instr = {
            idx: instr
            for idx, instr in all_instr.items()
            if instr.get("kernel_idx") == kernel_idx
        }
        tdag = Tdag(params, name=f"{circuit_name}_kernel_{kernel_idx}")
        populate_tdag_from_all_instr(tdag, kernel_instr, params)
        for out_idx in kernel_info.get("outputs", []):
            label = str(out_idx)
            if label in tdag.nodes:
                tdag.outputs.add(label)
        if not tdag.outputs:
            for v in tdag.nodes:
                if tdag.out_degree(v) == 0:
                    tdag.outputs.add(v)
        kernel_tdags.append(tdag)
    return kernel_tdags


def lower_heterm_to_orbit_ir(
    circuit_ir: dict,
    params: Params,
    circuit_name: str = "circuit",
) -> Tdag:
    return build_from_circuit_ir(circuit_ir, params, circuit_name=circuit_name)
