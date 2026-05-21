from __future__ import annotations

import json
import os
import re

from ..params.params import Params
from ..utils.rot_decompose import get_naf_weight
from .tdag import Tdag


def _parse_instruction(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    metadata = ""
    if " # " in line:
        line, metadata = line.rsplit(" # ", 1)

    match = re.match(r"^(\d+)\s+(True|False|true|false|ci|pl)\s*:\s*(.+)$", line)
    if not match:
        return None

    token = match.group(2)
    return {
        "index": int(match.group(1)),
        "secret": token in ("True", "true", "ci"),
        "operation": match.group(3).strip(),
        "metadata": metadata,
    }


def _classify_operation(operation: str) -> tuple[str, dict]:
    m = re.match(r"^\(\+\s+(\d+)\s+(\d+)\)$", operation)
    if m:
        return "add", {"operands": [int(m.group(1)), int(m.group(2))]}

    m = re.match(r"^\(-\s+(\d+)\s+(\d+)\)$", operation)
    if m:
        return "sub", {"operands": [int(m.group(1)), int(m.group(2))]}

    m = re.match(r"^\(\*\s+(\d+)\s+(\d+)\)$", operation)
    if m:
        return "mul", {"operands": [int(m.group(1)), int(m.group(2))]}

    m = re.match(r"^\(<<\s+(\d+)\s+(-?\d+)\)$", operation)
    if m:
        return "rotate", {"operands": [int(m.group(1))], "offset": int(m.group(2))}

    m = re.match(r"^\(poly\s+(\d+)\)$", operation)
    if m:
        return "poly", {"operands": [int(m.group(1))]}

    m = re.match(r"^\(rescale\s+(\d+)\s*/\s*2\^(\d+)\)$", operation)
    if m:
        return "rescale", {"operands": [int(m.group(1))], "factor": int(m.group(2))}

    m = re.match(r"^punctured pack\s+\((.+)\)$", operation)
    if m:
        return "pack", {"layout": m.group(1)}

    m = re.match(r"^pack\s+\((.+)\)$", operation)
    if m:
        return "pack", {"layout": m.group(1)}

    if operation == "zero mask":
        return "zero_mask", {}

    m = re.match(r"^mask\s+(.+)$", operation)
    if m:
        return "mask", {"mask_data": m.group(1)}

    m = re.match(r"^(\d+)$", operation)
    if m:
        return "cs_ref", {"ref_index": int(m.group(1))}

    return "unknown", {"raw": operation}


def _read_kernel_instructions(filepath: str) -> list[dict]:
    instructions = []
    with open(filepath) as f:
        for line in f:
            parsed = _parse_instruction(line)
            if parsed is not None:
                instructions.append(parsed)
    return instructions


def _map_op_to_tdag(op_type: str, secret: bool) -> str:
    if op_type in ("pack", "cs_ref"):
        return "input" if secret else "constant"
    return {
        "add": "add",
        "sub": "add",
        "mul": "mul",
        "rotate": "rotate",
        "rescale": "rescale",
        "mask": "constant",
        "zero_mask": "constant",
        "poly": "mul",
    }.get(op_type, op_type)


def _build_op_descr(
    op_type: str, op_info: dict, operand_secrets: list[bool], params: Params
) -> dict:
    if op_type in ("add", "sub", "mul"):
        pl_cnt = sum(1 for secret in operand_secrets if not secret)
        return {"single": min(pl_cnt, 1), "double": 1 - min(pl_cnt, 1)}

    if op_type == "rotate":
        offset = op_info["offset"]
        return {"offset": offset, "weight": get_naf_weight(offset, params.max_slot)}

    if op_type in ("rescale", "pack", "cs_ref"):
        return {}

    if op_type in ("mask", "zero_mask"):
        return {"value": 0, "rms_var": 0.0}

    if op_type == "poly":
        return {"single": 1, "double": 0}

    return {}


def populate_tdag_from_all_instr(tdag: Tdag, all_instr: dict, params: Params) -> None:
    for idx in sorted(all_instr.keys()):
        instr = all_instr[idx]
        op_type = instr["op_type"]
        op_info = instr["op_info"]
        label = str(idx)

        operand_indices = op_info.get("operands", [])
        operand_secrets = [
            all_instr[oi]["secret"] if oi in all_instr else False
            for oi in operand_indices
        ]

        tdag_op = _map_op_to_tdag(op_type, instr["secret"])
        op_descr = _build_op_descr(op_type, op_info, operand_secrets, params)
        if tdag_op == "constant" and "value" not in op_descr:
            op_descr = {"value": 0, "rms_var": 0.0}

        tdag.add_node(
            label,
            op=tdag_op,
            level=None,
            scale=None,
            weight=1,
            op_descr=op_descr,
            comment=instr.get("metadata", ""),
        )

        if tdag_op == "input":
            tdag.inputs.add(label)

        if op_type == "cs_ref":
            ref_label = str(op_info["ref_index"])
            if ref_label in tdag.nodes:
                tdag.add_edge(ref_label, label, weight=1)

        for oi in operand_indices:
            src_label = str(oi)
            if src_label not in tdag.nodes:
                tdag.add_node(
                    src_label,
                    op="input",
                    level=None,
                    scale=None,
                    weight=1,
                    op_descr={},
                    comment="(placeholder) missing operand definition",
                )
                tdag.inputs.add(src_label)
            tdag.add_edge(src_label, label, weight=1)


def mark_tdag_outputs_from_manifest(tdag: Tdag, manifest: dict) -> None:
    consumed_layouts = set()
    for kernel_info in manifest["kernels"]:
        consumed_layouts.update(kernel_info.get("dependencies", []))

    for kernel_info in manifest["kernels"]:
        if kernel_info.get("layout", "") not in consumed_layouts:
            for out_idx in kernel_info.get("outputs", []):
                label = str(out_idx)
                if label in tdag.nodes:
                    tdag.outputs.add(label)

    if not tdag.outputs:
        for v in tdag.nodes:
            if tdag.out_degree(v) == 0:
                tdag.outputs.add(v)


def build_from_rotom(manifest_path: str, params: Params) -> Tdag:
    with open(manifest_path) as f:
        manifest = json.load(f)

    circuit_name = manifest.get("circuit_name", "rotom_circuit")
    circuit_dir = os.path.dirname(manifest_path)
    tdag = Tdag(params, name=circuit_name)
    all_instr = {}

    for kernel_info in manifest["kernels"]:
        kernel_idx = kernel_info["kernel_idx"]
        kernel_path = os.path.join(circuit_dir, f"{circuit_name}_kernel_{kernel_idx}.txt")
        if not os.path.isfile(kernel_path):
            raise FileNotFoundError(
                f"Kernel file {kernel_path} referenced in manifest not found."
            )

        for instr in _read_kernel_instructions(kernel_path):
            op_type, op_info = _classify_operation(instr["operation"])
            all_instr[instr["index"]] = {
                **instr,
                "op_type": op_type,
                "op_info": op_info,
                "kernel_idx": kernel_idx,
            }

    populate_tdag_from_all_instr(tdag, all_instr, params)
    mark_tdag_outputs_from_manifest(tdag, manifest)
    return tdag
