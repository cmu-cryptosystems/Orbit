from .tdag import Tdag
from ..params.params import Params
import networkx as nx
import itertools

def _get_type_string(v: dict, params: Params) -> str:
    level = v['level']
    if params.dacapo_mlir_out:
        level = params.bts_ub - level
    cipl = 'pl' if v['op'] == 'constant' else 'ci'
    return f"tensor<{v['weight']}x!earth.{cipl}<{v['scale']} * {level}>>"

def _get_op_descr(v: dict, params: Params) -> str:
    op = v['op']
    descr = v['op_descr']
    assert op != 'input', "Input node should not have operation description."
    if op in ['negate', 'rescale']:
        return ""
    elif op == 'constant':
        return f"<{{rms_var = {descr['rms_var']} : f64, value = {descr['value']} : i64}}>"
    elif op in ['add', 'mul']:
        return ""
    elif op == 'rotate':
        return f"<{{offset = array<i64: {descr['offset']}>}}>"
    elif op == 'modswitch':
        return f"<{{downFactor = {descr['downFactor']} : i64}}>"
    elif op == 'upscale':
        return f"<{{upFactor = {descr['upFactor']} : i64}}>"
    elif op == 'bootstrap':
        targetLevel = descr['targetLevel']
        if params.dacapo_mlir_out:
            targetLevel = params.bts_ub - targetLevel
        return f"<{{targetLevel = {targetLevel} : i64}}>"
    else:
        raise Exception(f"Unsupported operation {op} in node description.")

def tdag_to_mlir(tdag: Tdag, filename: str):
    mapping = {}
    cnter = 0
    params = tdag.params
    
    def convert(key: str):
        nonlocal cnter
        if key not in mapping:
            mapping[key] = str(cnter)
            cnter += 1
        return f"%{mapping[key]}"
    
    def force_mapping(key: str, converted: str):
        mapping[key] = converted
        
    def gen_mlir_line(vlabel: str) -> str:
        v = tdag.nodes[vlabel]
        contents = []
        conv_v = convert(vlabel)
        conv_inp = ", ".join([convert(u) for u in tdag.predecessors(vlabel)])
        contents.append(f"{conv_v} = \"earth.{v['op']}\"({conv_inp})")
        contents.append(_get_op_descr(v, params))
        inp_types = ", ".join([_get_type_string(tdag.nodes[u], params) for u in tdag.predecessors(vlabel)])
        contents.append(f": ({inp_types}) -> {_get_type_string(v, params)}")
        contents.append("loc(unknown)")
        contents.append(f"{v['comment']}")
        return "    " + " ".join(contents)
    
    real_input_nodes = [n for n in tdag.inputs if tdag.in_degree(n) == 0]
    for i, n in enumerate(real_input_nodes):
        force_mapping(n, f"arg{i}")

    input_str1 = ", ".join([_get_type_string(tdag.nodes[node], params) for node in real_input_nodes])
    input_str2 = ", ".join([f"{convert(node)}: {_get_type_string(tdag.nodes[node], params)} loc(unknown)" for node in real_input_nodes])
    output_str = ", ".join([_get_type_string(tdag.nodes[node], params) for node in tdag.outputs])

    headers = [
        "\"builtin.module\"() <{sym_name = \""+"mlirs/.mlir"+"\"}> ({",
        "  \"func.func\"() <{function_type = ("+input_str1+") -> ("+output_str+"), sym_name = \"constants"+"\"}> ({",
        "^bb0("+input_str2+"):"
    ]
    
    topo_order = list(nx.topological_sort(tdag))
    lines = []
    for node in topo_order:
        if tdag.nodes[node]["op"] != "input":
            lines.append(gen_mlir_line(node))
    
    output_lab = ", ".join([f"{convert(node)}" for node in tdag.outputs])
    footers = [
        "    \"func.return\"("+output_lab+") : ("+output_str+") -> () loc(unknown)",
        "  }) : () -> () loc(unknown)",
        "}) : () -> () loc(unknown)"
    ]

    with open(filename, 'w') as f:
        for line in itertools.chain(headers, lines, footers):
            f.write(line + "\n")
