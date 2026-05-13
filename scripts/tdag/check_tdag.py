from .tdag import Tdag
import networkx as nx

def check_tdag(tdag: Tdag):
    # Check for cycles and connectivity
    topo_order = list(nx.topological_sort(tdag))
    if len(topo_order) != len(tdag.nodes):
        raise ValueError("Graph has cycles or is not fully connected.")
    
    # Check level and scale ranges
    for v in tdag.nodes:
        lvl = tdag.nodes[v].get('level', None)
        scale = tdag.nodes[v].get('scale', None)
        if lvl is None or scale is None:
            raise ValueError(f"Node {v} missing level or scale attribute.")
        # Check scale bounds
        scale_lb = tdag.params.scale_lower_bound(v, tdag.nodes[v], "out")
        if not scale_lb <= scale <= tdag.params.Sf + 2 * tdag.params.Sw:
            if tdag.nodes[v]['op'] == 'constant' and tdag.params.Csw <= scale < scale_lb:
                continue
            raise ValueError(f"Node {v} has scale {scale} out of bounds [{scale_lb}, {tdag.params.Sf + 2 * tdag.params.Sw}].")
        # Check level bounds
        if not tdag.params.lvl_lb <= lvl <= tdag.params.lvl_ub:
            raise ValueError(f"Node {v} has level {lvl} out of bounds [{tdag.params.lvl_lb}, {tdag.params.lvl_ub}].")
        op = tdag.nodes[v].get('op', None)
        if op is None:
            raise ValueError(f"Node {v} missing operation attribute.")
        if op in ['modswitch_single', 'rescale_single'] and lvl >= tdag.params.lvl_ub:
            raise ValueError(f"Node {v} with operation {op} has level {lvl} out of bounds [{tdag.params.lvl_lb}, {tdag.params.lvl_ub-1}].")
        if op == 'bootstrap_single' and not (tdag.params.bts_lb < lvl <= tdag.params.bts_ub):
            raise ValueError(f"Node {v} with operation {op} has level {lvl} out of bounds [{tdag.params.bts_lb+1}, {tdag.params.bts_ub}].")
        # check decryptable: scale <= Sf * (lvl - lvl_lb + 1) + Sf - 7
        if scale > tdag.params.Sf * (lvl - tdag.params.lvl_lb + 2) - 7:
            raise ValueError(f"Node {v} has scale {scale} exceeding decryptable limit {tdag.params.Sf * (lvl - tdag.params.lvl_lb + 2)  -7} for level {lvl}.")
    
    # Check level and scale relations for each operation
    for v in tdag.nodes:
        op = tdag.nodes[v]['op']
        if op in ['constant', 'input']:
            continue
        lvl = tdag.nodes[v]['level']
        scale = tdag.nodes[v]['scale']
        preds = list(tdag.predecessors(v))
        if op in ['add', 'rotate', 'negate']:
            for p in preds:
                if tdag.nodes[p]['level'] != lvl:
                    raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched level {tdag.nodes[p]['level']} (expected {lvl}).")
                if tdag.nodes[p]['scale'] != scale:
                    raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched scale {tdag.nodes[p]['scale']} (expected {scale}).")
        elif op == 'mul':
            if len(preds) > 2 or len(preds) == 0:
                raise ValueError(f"Node {v} with operation {op} must have 1 or 2 predecessors.")
            # level: p1.lvl == p2.lvl == v.lvl
            # scale: p1.scale + p2.scale == v.scale
            if len(preds) == 2:
                p1, p2 = preds[0], preds[1]
            else:
                p1, p2 = preds[0], preds[0]
            if tdag.nodes[p1]['level'] != lvl or tdag.nodes[p2]['level'] != lvl:
                raise ValueError(f"Node {v} with operation {op} has predecessors {p1}, {p2} with mismatched levels {tdag.nodes[p1]['level']}, {tdag.nodes[p2]['level']} (expected {lvl}).")
            if tdag.nodes[p1]['scale'] + tdag.nodes[p2]['scale'] != scale:
                raise ValueError(f"Node {v} with operation {op} has predecessors {p1}, {p2} with scales {tdag.nodes[p1]['scale']}, {tdag.nodes[p2]['scale']} not summing to {scale}.")
        elif op == 'rescale':
            if len(preds) != 1:
                raise ValueError(f"Node {v} with operation {op} must have exactly 1 predecessor.")
            p = preds[0]
            if tdag.nodes[p]['level'] != lvl + 1:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched level {tdag.nodes[p]['level']} (expected {lvl+1}).")
            if tdag.nodes[p]['scale'] != scale + tdag.params.Sf:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched scale {tdag.nodes[p]['scale']} (expected {scale + tdag.params.Sf}).")
        elif op == 'modswitch':
            if len(preds) != 1:
                raise ValueError(f"Node {v} with operation {op} must have exactly 1 predecessor.")
            p = preds[0]
            dropFactor = tdag.nodes[v]['op_descr'].get('downFactor', None)
            if dropFactor is None:
                raise ValueError(f"Node {v} with operation {op} missing downFactor in op_descr.")
            if tdag.nodes[p]['level'] != lvl + dropFactor:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched level {tdag.nodes[p]['level']} (expected {lvl + dropFactor}).")
            if tdag.nodes[p]['scale'] != scale:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched scale {tdag.nodes[p]['scale']} (expected {scale}).")
        elif op == 'upscale':
            if len(preds) != 1:
                raise ValueError(f"Node {v} with operation {op} must have exactly 1 predecessor.")
            p = preds[0]
            upFactor = tdag.nodes[v]['op_descr'].get('upFactor', None)
            if upFactor is None:
                raise ValueError(f"Node {v} with operation {op} missing upFactor in op_descr.")
            if tdag.nodes[p]['level'] != lvl:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched level {tdag.nodes[p]['level']} (expected {lvl}).")
            if tdag.nodes[p]['scale'] != scale - upFactor:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched scale {tdag.nodes[p]['scale']} (expected {scale - upFactor}).")
        elif op == 'bootstrap':
            if len(preds) != 1:
                raise ValueError(f"Node {v} with operation {op} must have exactly 1 predecessor.")
            p = preds[0]
            targetLevel = tdag.nodes[v]['op_descr'].get('targetLevel', None)
            if targetLevel is None:
                raise ValueError(f"Node {v} with operation {op} missing targetLevel in op_descr.")
            if tdag.nodes[p]['level'] != tdag.params.bts_lb:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with level {tdag.nodes[p]['level']} (expected {tdag.params.bts_lb}).")
            if tdag.nodes[p]['scale'] != tdag.params.Sf:
                raise ValueError(f"Node {v} with operation {op} has predecessor {p} with mismatched scale {tdag.nodes[p]['scale']} (expected {tdag.params.Sf}).")
            if lvl != targetLevel:
                raise ValueError(f"Node {v} with operation {op} has targetLevel {targetLevel} not matching its level {lvl}.")
            if scale != tdag.params.Sf:
                raise ValueError(f"Node {v} with operation {op} has scale {scale} not matching expected {tdag.params.Sf}.")
        else:
            raise ValueError(f"Node {v} has unknown operation {op}.")
    return True
