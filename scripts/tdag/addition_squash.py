from .tdag import Tdag
import networkx as nx

def split_constants(tdag: Tdag):
    """Give each plaintext constant consumer its own constant node.

    Orbit's ILP treats a plaintext constant as a value adapted to exactly one
    consumer's level/scale. Rotom can legitimately reuse a packed plaintext
    weight or mask across multiple operations, so split those fan-out nodes
    before assignment.
    """
    plain_nodes = list(tdag.nodes())
    for v in plain_nodes:
        if tdag.nodes[v]['op'] == 'constant':
            assert tdag.in_degree(v) == 0
            children = list(tdag.successors(v))
            if not children:
                tdag.inputs.discard(v)
                tdag.outputs.discard(v)
                tdag.remove_node(v)
                continue
            if len(children) == 1:
                continue
            for i, child in enumerate(children):
                eweight = tdag.edges[v, child]['weight']
                new_v = f"{v}_c{i}"
                vattr = tdag.nodes[v]
                tdag.add_node(new_v, op='constant', weight=eweight,
                              level=vattr['level'], scale=vattr['scale'],
                              op_descr=vattr['op_descr'].copy(),
                              comment=vattr['comment'])
                tdag.add_edge(new_v, child, weight=eweight)
            tdag.remove_node(v)
            

def addition_squash(tdag: Tdag):
    split_constants(tdag)
    dep_to_vs = tdag.get_depth_traversal()
    squash_ranges = dict()
    squashed_vs = set()
    for dep, vs in sorted(dep_to_vs.items(), reverse=True):
        for v in sorted(vs):
            if tdag.nodes[v]['op'] == 'add' and v not in squashed_vs:
                cur_squash_set = {v}
                end_dep = dep - 1

                while end_dep >= 0:
                    prev_squash_size = len(cur_squash_set)
                    for squash_cand in dep_to_vs[end_dep]:
                        if tdag.nodes[squash_cand]['op'] != 'add':
                            continue

                        is_squashable = (tdag.out_degree(squash_cand) != 0)
                        for child in tdag.successors(squash_cand):
                            if child not in cur_squash_set:
                                is_squashable = False
                        
                        if is_squashable:
                            cur_squash_set.add(squash_cand)
                    
                    if len(cur_squash_set) == prev_squash_size:
                        break

                    end_dep -= 1

                if len(cur_squash_set) > 1:
                    squash_ranges[v] = cur_squash_set.copy()
                    squashed_vs |= cur_squash_set
    
    for head, squashed_vertex_set in squash_ranges.items():
        # print(f"[OUT->IN Squash] center {head}:")
        # for v in squashed_vertex_set:
        #     print(f"  -{v}")
        removal_set = squashed_vertex_set - {head}
        
        inputs_of_removal_set = set()
        inputs_edges_weight = dict()
        for v in squashed_vertex_set:
            for inp in tdag.predecessors(v):
                inputs_of_removal_set.add(inp)
                inputs_edges_weight[inp] = tdag.edges[inp, v]['weight']
        
        inputs_of_removal_set -= removal_set
        
        for v in removal_set:
            tdag.nodes[head]['op_descr']['single'] += tdag.nodes[v]['op_descr']['single']
            tdag.nodes[head]['op_descr']['double'] += tdag.nodes[v]['op_descr']['double']
            tdag.remove_node(v)
        
        for new_input in inputs_of_removal_set:
            if not tdag.has_edge(new_input, head):
                tdag.add_edge(new_input, head, weight=inputs_edges_weight[new_input])
        
def addition_desquash(tdag: Tdag):
    plain_nodes = list(tdag.nodes())
    for v in plain_nodes:
        if tdag.nodes[v]['op'] != 'add':
            continue
        if tdag.in_degree(v) <= 2:
            continue
        vattr = tdag.nodes[v]
        inputs = list(tdag.predecessors(v))
        for i in range(1, len(inputs)):
            tdag.remove_edge(inputs[i], v)
        cur_v = f"{v}_add_{len(inputs)-2}"
        tdag.add_node(cur_v, op='add', weight=vattr['weight'],
                      level=vattr['level'], scale=vattr['scale'],
                      op_descr={'single':0, 'double':vattr['weight']},
                      comment=vattr['comment'])
        tdag.add_edge(inputs[-1], cur_v, weight=vattr['weight'])
        tdag.add_edge(inputs[-2], cur_v, weight=vattr['weight'])
        for i in range(len(inputs)-3, 0, -1):
            new_v = f"{v}_add_{i}"
            op_descr = {'single':vattr['weight'], 'double':0} if tdag.nodes[inputs[i]]['op']=='constant' else {'single':0, 'double':vattr['weight']}
            tdag.add_node(new_v, op='add', weight=vattr['weight'],
                            level=vattr['level'], scale=vattr['scale'],
                            op_descr=op_descr,
                            comment=vattr['comment'])
            tdag.add_edge(inputs[i], new_v, weight=vattr['weight'])
            tdag.add_edge(cur_v, new_v, weight=vattr['weight'])
            cur_v = new_v
        tdag.add_edge(cur_v, v, weight=vattr['weight'])
        is_v_single = tdag.nodes[inputs[0]]['op']=='constant'
        tdag.nodes[v]['op_descr'] = {'single':vattr['weight'], 'double': 0} if is_v_single else {'single':0, 'double':vattr['weight']}

def addition_desquash_bypass(tdag: Tdag):
    params = tdag.params
    if params.bpsdepth is None:
        return
    plain_nodes = list(nx.topological_sort(tdag))
    mul_depth = dict()
    for v in plain_nodes:
        if tdag.in_degree(v) == 0:
            mul_depth[v] = 0
        else:
            inp_mul = max(mul_depth[inp] for inp in tdag.predecessors(v))
            mul_depth[v] = inp_mul + (1 if tdag.nodes[v]['op']=='mul' else 0)
    
    def _split_one_v_from_add(v: str, vinp: str):
        vattr = tdag.nodes[v]
        new_v = f"{v}_add_({vinp})"
        # must be a double addition (vinp is not constant)
        new_v_op_descr = {'single': 0, 'double': vattr['weight']}
        tdag.add_node(new_v, op='add', weight=vattr['weight'],
                      level=vattr['level'], scale=vattr['scale'],
                      op_descr=new_v_op_descr,
                      comment=vattr['comment'])
        children = list(tdag.successors(v))
        for oup in children:
            eweight = tdag.edges[v, oup]['weight']
            tdag.remove_edge(v, oup)
            tdag.add_edge(new_v, oup, weight=eweight)
        tdag.remove_edge(vinp, v)
        tdag.add_edge(vinp, new_v, weight=vattr['weight'])
        tdag.add_edge(v, new_v, weight=vattr['weight'])
    
    for v in plain_nodes:
        if tdag.nodes[v]['op'] == 'add' and tdag.in_degree(v) > 2:
            inp_wo_const = [(inp, mul_depth[inp]) for inp in tdag.predecessors(v) if tdag.nodes[inp]['op'] != 'constant']
            if len(inp_wo_const) <= 2:
                continue
            sorted_inp = sorted(inp_wo_const, key=lambda x: x[1])
            while len(sorted_inp) >= 2 and sorted_inp[-1][1] - sorted_inp[0][1] >= params.bpsdepth:
                vinp = sorted_inp[0][0]
                _split_one_v_from_add(v, vinp)
                sorted_inp = sorted_inp[1:]
