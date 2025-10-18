from .tdag import Tdag
from ..utils.rot_decompose import get_naf_weight
from ..params.params import Params
import re

def _extract_op_descr(raw_line: str, params: Params) -> tuple[str, dict]:
    match = re.findall(r'earth\.(\w+)', raw_line)
    if match:
        op = match[0]
        if op in ['negate', 'rescale', 'input']:
            return (op, dict())
        elif op == 'constant':
            pattern = r'rms_var\s*=\s*([\d.eE+-]+)\s*:\s*f64,\s*value\s*=\s*(\d+)\s*:\s*i64'
            submatch = re.search(pattern, raw_line)
            if submatch:
                rms_var = float(submatch.group(1))
                value = int(submatch.group(2))
                return (op, {'rms_var': rms_var, 'value': value})
            else:
                raise Exception(f"Cannot parse constant description from line: {raw_line}")
        elif op in ['add', 'mul']:
            assert len(match) == 3 or len(match) == 4, f"Unexpected format for {op} operation in line: {raw_line}"
            assert match[1] in ['ci', 'pl']
            assert match[2] in ["ci", "pl"]
            if len(match) == 4:
                assert match[3] in ['ci', 'pl']
            pl_cnt = sum(1 for m in match[1:-1] if m == 'pl')
            ci_cnt = sum(1 for m in match[1:-1] if m == 'ci')
            assert ci_cnt >= 1, f"At least one input should be ciphertext for {op} operation in line: {raw_line}"
            return (op, {'single': pl_cnt, 'double': 1-pl_cnt})
        elif op == 'rotate':
            pattern = r'offset\s*=\s*array<i64:\s*(-?\d+)>'
            submatch = re.search(pattern, raw_line)
            if submatch:
                offset = int(submatch.group(1))
                weight = get_naf_weight(offset, params.max_slot)
                return (op, {'offset': offset, 'weight': weight})
            else:
                raise Exception(f"Cannot parse rotate description from line: {raw_line}")
        elif op == 'modswitch':
            pattern = r'downFactor\s*=\s*(\d+)\s*:\s*i64'
            submatch = re.search(pattern, raw_line)
            if submatch:
                downFactor = int(submatch.group(1))
                return (op, {'downFactor': downFactor})
            else:
                raise Exception(f"Cannot parse modswitch description from line: {raw_line}")
        elif op == 'upscale':
            pattern = r'upFactor\s*=\s*(\d+)\s*:\s*i64'
            submatch = re.search(pattern, raw_line)
            if submatch:
                upFactor = int(submatch.group(1))
                return (op, {'upFactor': upFactor})
            else:
                raise Exception(f"Cannot parse upscale description from line: {raw_line}")
        elif op == 'bootstrap':
            pattern = r'targetLevel\s*=\s*(\d+)\s*:\s*i64'
            submatch = re.search(pattern, raw_line)
            if submatch:
                targetLevel = int(submatch.group(1))
                return (op, {'targetLevel': targetLevel})
            else:
                raise Exception(f"Cannot parse bootstrap description from line: {raw_line}")
        else:
            raise Exception(f"Unsupported operation {op} in line: {raw_line}")
    else:
        raise Exception(f"Cannot find operation in line: {raw_line}")

def _add_v_from_mlir_line(tdag: Tdag, raw_line: str):
    varname_pattern = r"%(\d+|arg\d+)"
    var_nums = re.findall(varname_pattern, raw_line)
    assert var_nums, f"Cannot find variable name in line: {raw_line}"
    label = var_nums[0] if var_nums else None

    in_labels = var_nums[1:] if len(var_nums) > 1 else []

    op, op_descr = _extract_op_descr(raw_line, tdag.params)
    if op_descr.get('targetLevel') is not None and tdag.params.dacapo_mlir_in:
        op_descr['targetLevel'] = tdag.params.bts_ub - op_descr['targetLevel']

    type_pattern = r'tensor<(\d+)x!earth\.(ci|pl)<(\d+)\s*\*\s*(\d+)>>'
    type_matches = re.findall(type_pattern, raw_line)
    assert type_matches, f"Cannot find type in line: {raw_line}"
    # get last type match, tensor<(weight)x!earth\.(ci|pl)<(scale)\s*\*\s*(level)>>
    weight = int(type_matches[-1][0])
    scale = int(type_matches[-1][2])
    level = int(type_matches[-1][3])
    if tdag.params.dacapo_mlir_in:
        level = tdag.params.bts_ub - level
    if tdag.params.mode == "compile":
        level = None
        scale = None
    
    # get comments
    comment_pattern = r'loc\([^)]*\)\s*(.+?)$'
    comment_match = re.search(comment_pattern, raw_line)
    comment = ''
    if comment_match:
        metadata = comment_match.group(1).strip()
        if metadata:
            comment = metadata
    
    assert label not in tdag.nodes, f"Variable {label} already exists in Tdag."
    tdag.add_node(label, op=op, level=level, scale=scale, weight=weight, op_descr=op_descr, comment=comment)
    
    for i in range(len(in_labels)):
        in_label = in_labels[i]
        in_type = type_matches[i]
        
        in_level = int(in_type[3])
        in_scale = int(in_type[2])
        in_weight = int(in_type[0])
        if tdag.params.dacapo_mlir_in:
            in_level = tdag.params.bts_ub - in_level
        if tdag.params.mode == "compile":
            in_level = None
            in_scale = None
        if in_label not in tdag.nodes:
            tdag.add_node(in_label, op='input', level=in_level, scale=in_scale, weight=in_weight, op_descr={}, comment='' )
            tdag.inputs.add(in_label)
        else:
            assert tdag.nodes[in_label]['level'] == in_level, f"Level mismatch for variable {in_label}, expected {tdag.nodes[in_label]['level']}, got {in_level}."
            assert tdag.nodes[in_label]['scale'] == in_scale, f"Scale mismatch for variable {in_label}, expected {tdag.nodes[in_label]['scale']}, got {in_scale}."
            assert tdag.nodes[in_label]['weight'] == in_weight, f"Weight mismatch for variable {in_label}, expected {tdag.nodes[in_label]['weight']}, got {in_weight}."
        tdag.add_edge(in_label, label, weight=1)

def _add_inputs_from_mlir(tdag: Tdag, raw_line: str):
    match = re.search(r'^\^bb0\((.*?)\):', raw_line)
    if match:
        args = match.group(1).split(", ")
        for arg in args:
            label = arg.split(":")[0].strip()
            assert label.startswith("%arg"), f"Input argument label {label} must start with '%arg'."
            label = label[1:]  # remove leading '%'
            
            type_pattern = r'tensor<(\d+)x!earth\.(ci|pl)<(\d+)\s*\*\s*(\d+)>>'
            type_matches = re.search(type_pattern, arg)
            assert type_matches, f"Cannot find type in line: {arg}"
            weight = int(type_matches.group(1))
            scale = int(type_matches.group(3))
            level = int(type_matches.group(4))
            if tdag.params.dacapo_mlir_in:
                level = tdag.params.bts_ub - level
            if tdag.params.mode == "compile":
                level = None
                scale = None
            
            tdag.add_node(label, op='input', level=level, scale=scale, weight=weight, op_descr={}, comment='')
            tdag.inputs.add(label)
    else:
        raise ValueError("Input line does not match expected format for module inputs.")

def build_from_mlir(mlir_file: str, params: Params) -> Tdag:
    tdag = Tdag(params, name=mlir_file)
    with open(mlir_file, 'r') as f:
        for line in f:
            if line.strip().startswith('^bb0('):
                _add_inputs_from_mlir(tdag, line.strip())
            elif line.strip().startswith("return") or line.strip().startswith("\"func.return\""):
                match = re.findall(r'%(\d+|arg\d+)', line.strip())
                for output_node in match:
                    tdag.outputs.add(output_node)
            elif line.strip().startswith('%'):
                _add_v_from_mlir_line(tdag, line.strip())
    return tdag