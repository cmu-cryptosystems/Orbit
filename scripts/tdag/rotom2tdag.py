from .tdag import Tdag
from ..utils.rot_decompose import get_naf_weight
from ..params.params import Params
import re
import os
import json

def _parse_instruction(line: str) -> dict | None:
    """Parse a single Rotom instruction line.

    Format: {index} {True|False}: {operation}  [# metadata]
    Returns dict with keys: index, secret, operation, metadata
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    # Split off trailing metadata comment
    metadata = ''
    if ' # ' in line:
        line, metadata = line.rsplit(' # ', 1)

    # Accept both v1 (True/False) and v2 (ci/pl) secrecy tokens
    match = re.match(r'^(\d+)\s+(True|False|ci|pl):\s+(.+)$', line)
    if not match:
        return None

    token = match.group(2)
    secret = token in ('True', 'ci')

    return {
        'index': int(match.group(1)),
        'secret': secret,
        'operation': match.group(3).strip(),
        'metadata': metadata,
    }

def _classify_operation(operation: str) -> tuple[str, dict]:
    """Classify a Rotom operation string into (op_type, info_dict).

    Recognized formats from HETerm.instrs():
        (+ a b)                 -> ('add', {'operands': [a, b]})
        (- a b)                 -> ('sub', {'operands': [a, b]})
        (* a b)                 -> ('mul', {'operands': [a, b]})
        (<< a offset)           -> ('rotate', {'operands': [a], 'offset': offset})
        (poly a)                -> ('poly', {'operands': [a]})
        (rescale a / 2^b)       -> ('rescale', {'operands': [a], 'factor': b})
        pack (layout_str)       -> ('pack', {'layout': layout_str})
        punctured pack (...)    -> ('pack', {'layout': layout_str})
        mask ...                -> ('mask', {'mask_data': ...})
        zero mask               -> ('zero_mask', {})
        bare integer            -> ('cs_ref', {'ref_index': int})
    """
    # ADD
    m = re.match(r'^\(\+\s+(\d+)\s+(\d+)\)$', operation)
    if m:
        return 'add', {'operands': [int(m.group(1)), int(m.group(2))]}

    # SUB
    m = re.match(r'^\(-\s+(\d+)\s+(\d+)\)$', operation)
    if m:
        return 'sub', {'operands': [int(m.group(1)), int(m.group(2))]}

    # MUL
    m = re.match(r'^\(\*\s+(\d+)\s+(\d+)\)$', operation)
    if m:
        return 'mul', {'operands': [int(m.group(1)), int(m.group(2))]}

    # ROT
    m = re.match(r'^\(<<\s+(\d+)\s+(-?\d+)\)$', operation)
    if m:
        return 'rotate', {'operands': [int(m.group(1))], 'offset': int(m.group(2))}

    # POLY
    m = re.match(r'^\(poly\s+(\d+)\)$', operation)
    if m:
        return 'poly', {'operands': [int(m.group(1))]}

    # RESCALE
    m = re.match(r'^\(rescale\s+(\d+)\s*/\s*2\^(\d+)\)$', operation)
    if m:
        return 'rescale', {'operands': [int(m.group(1))], 'factor': int(m.group(2))}

    # PUNCTURED PACK
    m = re.match(r'^punctured pack\s+\((.+)\)$', operation)
    if m:
        return 'pack', {'layout': m.group(1)}

    # PACK
    m = re.match(r'^pack\s+\((.+)\)$', operation)
    if m:
        return 'pack', {'layout': m.group(1)}

    # ZERO MASK
    if operation.strip() == 'zero mask':
        return 'zero_mask', {}

    # MASK
    m = re.match(r'^mask\s+(.+)$', operation)
    if m:
        return 'mask', {'mask_data': m.group(1)}

    # CS_REF (bare integer from cross-kernel env reference)
    m = re.match(r'^(\d+)$', operation)
    if m:
        return 'cs_ref', {'ref_index': int(m.group(1))}

    return 'unknown', {'raw': operation}

def _read_kernel_instructions(filepath: str) -> list[dict]:
    """Read and parse all instruction lines from a Rotom kernel file."""
    instructions = []
    with open(filepath, 'r') as f:
        for line in f:
            parsed = _parse_instruction(line)
            if parsed is not None:
                instructions.append(parsed)
    return instructions

def _map_op_to_tdag(op_type: str, secret: bool) -> str:
    """Map a Rotom operation type to its Tdag op string.

    Rotom SUB maps to Tdag 'add' because in CKKS the cost is identical
    (negate is essentially free) and Orbit only needs cost-accurate ops.
    Rotom POLY maps to 'mul' as the dominant cost is ct-pt multiplication.

    For pack nodes the secret flag determines the Tdag type:
        ci (secret=True)  -> 'input'    (encrypted circuit input)
        pl (secret=False) -> 'constant' (plaintext weight / mask)
    """
    if op_type == 'pack':
        return 'input' if secret else 'constant'
    if op_type == 'cs_ref':
        return 'input' if secret else 'constant'
    mapping = {
        'add':       'add',
        'sub':       'add',
        'mul':       'mul',
        'rotate':    'rotate',
        'rescale':   'rescale',
        'mask':      'constant',
        'zero_mask': 'constant',
        'poly':      'mul',
    }
    return mapping.get(op_type, op_type)

def _build_op_descr(op_type: str, op_info: dict,
                    operand_secrets: list[bool], params: Params) -> dict:
    """Build a Tdag-compatible op_descr dict from a parsed Rotom operation.

    For add/mul the single/double distinction follows Orbit's convention:
        single = 1  when at least one operand is plaintext (ct-pt)
        double = 1  when both operands are ciphertext   (ct-ct)
    """
    if op_type in ['add', 'sub']:
        pl_cnt = sum(1 for s in operand_secrets if not s)
        return {'single': min(pl_cnt, 1), 'double': 1 - min(pl_cnt, 1)}

    if op_type == 'mul':
        pl_cnt = sum(1 for s in operand_secrets if not s)
        return {'single': min(pl_cnt, 1), 'double': 1 - min(pl_cnt, 1)}

    if op_type == 'rotate':
        offset = op_info['offset']
        weight = get_naf_weight(offset, params.max_slot)
        return {'offset': offset, 'weight': weight}

    if op_type == 'rescale':
        return {}

    if op_type in ['pack', 'cs_ref']:
        return {}

    if op_type in ['mask', 'zero_mask']:
        return {'value': 0, 'rms_var': 0.0}

    if op_type == 'poly':
        # poly is ct-pt multiplication (plaintext polynomial coefficients)
        return {'single': 1, 'double': 0}

    return {}

def build_from_rotom(manifest_path: str, params: Params) -> Tdag:
    """Build a Tdag from a Rotom serialized circuit directory.

    Expects a manifest JSON produced by Rotom's CircuitSerializer alongside
    per-kernel instruction .txt files in the same directory.

    Args:
        manifest_path: path to the *_manifest.json file
        params:        Orbit Params (mode should be 'compile')

    Returns:
        Tdag with level=None, scale=None on every node (to be assigned by
        Orbit's ILP optimizer).
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    circuit_name = manifest.get('circuit_name', 'rotom_circuit')
    circuit_dir = os.path.dirname(manifest_path)
    tdag = Tdag(params, name=circuit_name)

    # ---- pass 1: collect every instruction across all kernels ----
    all_instr = {}  # global_index -> parsed + classified info

    for kernel_info in manifest['kernels']:
        kernel_idx = kernel_info['kernel_idx']
        kernel_file = f"{circuit_name}_kernel_{kernel_idx}.txt"
        kernel_path = os.path.join(circuit_dir, kernel_file)
        if not os.path.isfile(kernel_path):
            raise FileNotFoundError(
                f"Kernel file {kernel_path} referenced in manifest not found.")

        for instr in _read_kernel_instructions(kernel_path):
            idx = instr['index']
            op_type, op_info = _classify_operation(instr['operation'])
            all_instr[idx] = {
                **instr,
                'op_type':    op_type,
                'op_info':    op_info,
                'kernel_idx': kernel_idx,
            }

    # ---- pass 2: build nodes in index order (topological) ----
    for idx in sorted(all_instr.keys()):
        instr = all_instr[idx]
        op_type  = instr['op_type']
        op_info  = instr['op_info']
        label    = str(idx)

        operand_indices = op_info.get('operands', [])
        operand_secrets = [
            all_instr[oi]['secret'] if oi in all_instr else False
            for oi in operand_indices
        ]

        secret   = instr['secret']
        tdag_op  = _map_op_to_tdag(op_type, secret)
        op_descr = _build_op_descr(op_type, op_info, operand_secrets, params)

        # Plaintext packs become constants; Lattigo indexes the .cst file by
        # MLIR constant value, so preserve the Rotom instruction ID here.
        if tdag_op == 'constant' and 'value' not in op_descr:
            op_descr = {'value': idx, 'rms_var': 0.0}

        # In compile mode Orbit sets level/scale to None and solves for them.
        level = None
        scale = None

        tdag.add_node(label, op=tdag_op, level=level, scale=scale,
                      weight=1, op_descr=op_descr,
                      comment=instr.get('metadata', ''))

        # Only ciphertext leaf nodes are circuit inputs;
        # plaintext packs / masks are constants (not in tdag.inputs).
        if tdag_op == 'input':
            tdag.inputs.add(label)

        # cs_ref nodes depend on the instruction they reference
        if op_type == 'cs_ref':
            ref_label = str(op_info['ref_index'])
            if ref_label in tdag.nodes:
                tdag.add_edge(ref_label, label, weight=1)

        for oi in operand_indices:
            src_label = str(oi)
            assert src_label in tdag.nodes, \
                f"Operand {oi} referenced before definition (instruction {idx})."
            tdag.add_edge(src_label, label, weight=1)

    # ---- pass 3: determine output nodes ----
    # Only terminal kernels (whose layouts are not consumed by any other
    # kernel) produce real circuit outputs.  Intermediate kernel outputs
    # are just internal data-flow edges already captured by the graph.
    consumed_layouts = set()
    for kernel_info in manifest['kernels']:
        for dep in kernel_info.get('dependencies', []):
            consumed_layouts.add(dep)

    for kernel_info in manifest['kernels']:
        layout = kernel_info.get('layout', '')
        is_terminal = layout not in consumed_layouts
        if is_terminal:
            for out_idx in kernel_info.get('outputs', []):
                tdag.outputs.add(str(out_idx))

    # Fallback: if no terminal outputs found, use all sink nodes
    if not tdag.outputs:
        for v in tdag.nodes:
            if tdag.out_degree(v) == 0:
                tdag.outputs.add(v)

    return tdag
