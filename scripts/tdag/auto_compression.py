from .tdag import Tdag
from .addition_squash import addition_squash
from collections import defaultdict
import networkx as nx

def _num_unique_vals(D) -> int:
    if not isinstance(D, dict):
        return -1
    return len(set(D.values()))

def _initial_grouping(tdag: Tdag, is_ignore_weight: bool, reps=100) -> dict[str, str]:
    depth_traversal = tdag.get_depth_traversal()
    # depth-0 nodes are inputs and standalone constants; skip them
    dep_to_vs = sorted((d, vs) for d, vs in depth_traversal.items() if d > 0)

    prev_grouping = None
    for r in range(reps):
        grouping = dict()

        # Assign all depth-0 nodes (inputs + standalone constants) to root groups
        for v in depth_traversal.get(0, []):
            grouping[v] = f"{v}_root"

        for (dep, vs) in dep_to_vs:
            for v in vs:
                for u in tdag.predecessors(v):
                    if tdag.nodes[u]['op'] == 'constant':
                        assert tdag.out_degree(u) == 1
                        assert list(tdag.successors(u))[0] == v
                        grouping[u] = f"tdag_constants_depth{dep}"
        
        for (dep, vs) in dep_to_vs:
            pp_groups = dict()
            for v in vs:
                if tdag.nodes[v]['op'] == 'constant':
                    continue
                ituple = tuple(sorted({grouping[u] for u in tdag.predecessors(v)}))
                ctuple = None if not prev_grouping else tuple(sorted({prev_grouping[w] for w in tdag.successors(v)}))
                
                if is_ignore_weight:
                    descriptor = (ituple, ctuple, tdag.nodes[v]['op'], tdag.nodes[v]['level'], tdag.nodes[v]['scale'])
                else:
                    weight_single = tdag.nodes[v]['op_descr'].get('single', 0)
                    weight_double = tdag.nodes[v]['op_descr'].get('double', 0)
                    weight_rotate = tdag.nodes[v]['op_descr'].get('weight', 0)
                    weight_node = tdag.nodes[v]['weight']
                    weight_inputs = len(list(tdag.predecessors(v)))
                    descriptor = (ituple, ctuple, tdag.nodes[v]['op'], tdag.nodes[v]['level'], tdag.nodes[v]['scale'], 
                                  weight_single, weight_double, weight_rotate, weight_node, weight_inputs)
                
                if descriptor not in pp_groups:
                    pp_groups[descriptor] = [v]
                else:
                    pp_groups[descriptor].append(v)
            
            for i, (_, vlist) in enumerate(pp_groups.items()):
                for v in vlist:
                    assert tdag.nodes[v]['op'] != 'constant'
                    grouping[v] = f"dep_{dep}_grp_{i}"

        if _num_unique_vals(grouping) == _num_unique_vals(prev_grouping):
            return grouping
        prev_grouping = grouping
    
    return prev_grouping

def auto_compress(og_dag: Tdag, is_ignore_weight: bool):
    ini_groups = _initial_grouping(og_dag, is_ignore_weight)
    og_to_comp = dict()
    comp_dag = Tdag(og_dag.params, og_dag.name + "_comp")

    depth_traversal = og_dag.get_depth_traversal()
    dep_to_vs = sorted((d, vs) for d, vs in depth_traversal.items() if d > 0)

    # handling input nodes
    for og_inp in og_dag.inputs:
        comp_inp = f"{og_inp}_comp"
        comp_dag.add_node(comp_inp, op='input', weight=og_dag.nodes[og_inp]['weight'],
                          level=og_dag.nodes[og_inp]['level'], scale=og_dag.nodes[og_inp]['scale'],
                          op_descr=og_dag.nodes[og_inp]['op_descr'],
                          comment=og_dag.nodes[og_inp]['comment'])
        comp_dag.inputs.add(comp_inp)
        og_to_comp[og_inp] = comp_inp
    
    # handling constant nodes
    const_dep = dict()
    for (dep, vs) in dep_to_vs:
        for v in vs:
            for u in og_dag.predecessors(v):
                if og_dag.nodes[u]['op'] == 'constant':
                    assert og_dag.out_degree(u) == 1
                    assert list(og_dag.successors(u))[0] == v
                    if dep not in const_dep:
                        const_node_dep = f"const_cp_{dep}"
                        comp_dag.add_node(const_node_dep, op='constant', weight=1,
                                          level=None, scale=None,
                                          op_descr = dict(),
                                          comment = '')
                        # will be splitted after compression
                        const_dep[dep] = const_node_dep
                    og_to_comp[u] = const_dep[dep]
    
    for (dep, vs) in dep_to_vs:
        groups = defaultdict(list)
        for v in vs:
            if og_dag.nodes[v]['op'] == 'constant':
                continue
            ituple = tuple(sorted({og_to_comp[u] for u in og_dag.predecessors(v)}))
            descriptor = (ituple, og_dag.nodes[v]['op'], ini_groups[v])
            groups[descriptor].append(v)
        for i, (descr, vlist) in enumerate(groups.items()):
            new_weight = 0
            new_level = None
            new_scale = None
            new_op_descr = defaultdict(int)
            new_comment = ''
            for v in vlist:
                assert og_dag.nodes[v]['op'] != 'constant'
                new_weight += og_dag.nodes[v]['weight']
                if new_level is None:
                    new_level = og_dag.nodes[v]['level']
                else:
                    assert new_level == og_dag.nodes[v]['level'], "Level mismatch in grouped nodes"
                if new_scale is None:
                    new_scale = og_dag.nodes[v]['scale']
                else:
                    assert new_scale == og_dag.nodes[v]['scale'], "Scale mismatch in grouped nodes"
                if og_dag.nodes[v]['op'] == 'add':
                    new_op_descr['single'] += og_dag.nodes[v]['op_descr'].get('single', 0)
                    new_op_descr['double'] += og_dag.nodes[v]['op_descr'].get('double', 0)
                elif og_dag.nodes[v]['op'] == 'rotate':
                    new_op_descr['weight'] += og_dag.nodes[v]['op_descr'].get('weight', 0)
                elif og_dag.nodes[v]['op'] == 'mul':
                    new_op_descr['single'] = og_dag.nodes[v]['op_descr']['single']
                    new_op_descr['double'] = og_dag.nodes[v]['op_descr']['double']

                if og_dag.nodes[v]['comment'] != '' and new_comment == '':
                    new_comment = og_dag.nodes[v]['comment']
            
            new_op_descr = dict(new_op_descr)
            new_v = f"dep_{dep}_grp_{i}_cp"
            comp_dag.add_node(new_v, op=descr[1], weight=new_weight,
                                level=new_level, scale=new_scale,
                                op_descr = new_op_descr,
                                comment = new_comment)
            for comp_inp in descr[0]:
                comp_dag.add_edge(comp_inp, new_v, weight=1)
                # weight will be assigned after compression
            
            for v in vlist:
                og_to_comp[v] = new_v
    
    for og_out in og_dag.outputs:
        comp_dag.outputs.add(og_to_comp[og_out])

    _check_valid_compress(og_dag, comp_dag, og_to_comp)
    _split_const_wcomp(og_dag, comp_dag, og_to_comp)
    _update_edge_weights(og_dag, comp_dag, og_to_comp)
    return comp_dag, og_to_comp

def _get_comp_to_og(og_to_comp: dict[str, str]) -> dict[str, set[str]]:
    comp_to_og = defaultdict(set)
    for og_v, comp_v in og_to_comp.items():
        comp_to_og[comp_v].add(og_v)
    return dict(comp_to_og)

def _print_invalid_compress_info(og_vs, comp_inp_og_vs, og_inp):
    print("Invalid compression detected!")
    print("OG vertex group:")
    for v in og_vs:
        print(v, end=", ")
    print()
    
    print("Compressed Inputs Expanded:")
    for ov in comp_inp_og_vs:
        print(ov, end=", ")
    print()
    
    print("Inputs from OG vertex group:")
    for ov in og_inp:
        print(ov, end=", ")
    print()

def _check_valid_compress(og_dag: Tdag, comp_dag: Tdag, og_to_comp: dict[str, str]):
    comp_to_og = _get_comp_to_og(og_to_comp)
    
    for comp_v, og_vs in comp_to_og.items():
        og_inp_set = set()
        for og_v in og_vs:
            for og_u in og_dag.predecessors(og_v):
                og_inp_set.add(og_u)
        
        for comp_inp in comp_dag.predecessors(comp_v):
           if not comp_to_og[comp_inp].issubset(og_inp_set):
                set_minus = comp_to_og[comp_inp] - og_inp_set
                if all(og_dag.nodes[ov]['op'] == 'constant' for ov in set_minus):
                    continue
                
                _print_invalid_compress_info(og_vs, comp_to_og[comp_inp], og_inp_set)
                raise Exception("Invalid compression detected!")
                
def _split_const_wcomp(og_dag: Tdag, comp_dag: Tdag, og_to_comp: dict[str, str]):
    comp_to_og = _get_comp_to_og(og_to_comp)
    
    plain_vs = list(comp_dag.nodes())
    for v in plain_vs:
        if comp_dag.nodes[v]['op'] == 'constant':
            assert comp_dag.in_degree(v) == 0
            for i, child in enumerate(comp_dag.successors(v)):
                new_const = f"{v}_c{i}"
                comp_dag.add_node(new_const, op='constant', weight=comp_dag.nodes[child]['weight'],
                                  level=comp_dag.nodes[v]['level'], scale=comp_dag.nodes[v]['scale'],
                                  op_descr = dict(),
                                  comment = comp_dag.nodes[child]['comment'])
                comp_dag.add_edge(new_const, child, weight=1)
                for og_v in comp_to_og[child]:
                    for og_u in og_dag.predecessors(og_v):
                        if og_dag.nodes[og_u]['op'] == 'constant' and og_to_comp[og_u] == v:
                            og_to_comp[og_u] = new_const
            comp_dag.remove_node(v)
            
def _update_edge_weights(og_dag: Tdag, comp_dag: Tdag, og_to_comp: dict[str, str]):
    for cu, cv in comp_dag.edges():
        comp_dag.edges[cu, cv]['weight'] = 0

    for og_u, og_v in og_dag.edges():
        cu = og_to_comp[og_u]
        cv = og_to_comp[og_v]
        assert comp_dag.has_edge(cu, cv), f"Compressed edge ({cu}, {cv}) does not exist"
        comp_dag.edges[cu, cv]['weight'] += og_dag.edges[og_u, og_v]['weight']

def tdag_equal_biject(dag1: Tdag, dag2: Tdag) -> dict[str, str] | None:
    if len(dag1.nodes) != len(dag2.nodes):
        return None
    if len(dag1.get_depth_traversal()) != len(dag2.get_depth_traversal()):
        return None
    
    new_dag = Tdag(dag1.params, "combined_dag")
    vin = "ALL_in"
    vout = "ALL_out"
    new_dag.add_node(vin, op='input', weight=1, level=None, scale=None, op_descr=dict(), comment='')
    new_dag.add_node(vout, op='output', weight=1, level=None, scale=None, op_descr=dict(), comment='')
    new_dag.inputs.add(vin)
    new_dag.outputs.add(vout)
    
    def _merge_dag_in(new_dag: Tdag, tdag: Tdag, adlabel: str):
        for v in tdag.nodes:
            new_dag.add_node(v+adlabel, **tdag.nodes[v])
        for u, v in tdag.edges:
            new_dag.add_edge(u+adlabel, v+adlabel, **tdag.edges[u, v])
        new_dag.add_edge(vin, list(tdag.inputs)[0]+adlabel, weight=1)
        new_dag.add_edge(list(tdag.outputs)[0]+adlabel, vout, weight=1)
    
    _merge_dag_in(new_dag, dag1, "_dag1")
    _merge_dag_in(new_dag, dag2, "_dag2")
    
    _, og_to_comp = auto_compress(new_dag, is_ignore_weight=False)
    
    if og_to_comp[list(dag1.inputs)[0]+"_dag1"] != og_to_comp[list(dag2.inputs)[0]+"_dag2"]:
        return None
    
    comp_to_og = _get_comp_to_og(og_to_comp)
    bijection = dict()
    # handle non-constant bijection
    for _, og_vs in comp_to_og.items():
        og_dag1 = []
        og_dag2 = []
        for og_v in og_vs:
            if og_v.endswith("_dag1"):
                og_dag1.append(og_v[:-5])
            elif og_v.endswith("_dag2"):
                og_dag2.append(og_v[:-5])
            else:
                assert og_v == "ALL_in" or og_v == "ALL_out", "Unexpected vertex label"
        assert len(og_dag1) == len(og_dag2), "Different number of vertices in the two DAGs"
        
        for i in range(len(og_dag1)):
            if dag1.nodes[og_dag1[i]]['op'] == 'constant':
                continue
            bijection[og_dag1[i]] = og_dag2[i]
    # handle constant bijection
    for v1 in dag1.nodes:
        if dag1.nodes[v1]['op'] != 'constant':
            assert v1 in bijection, f"Vertex {v1} not found in bijection"
        else:
            assert dag1.out_degree(v1) == 1, f"Constant vertex {v1} has more than one child"
            child = list(dag1.successors(v1))[0]
            assert child in bijection, f"Child {child} of constant vertex {v1} not found in bijection"
            child2 = bijection[child]
            c2inp = [c for c in dag2.predecessors(child2) if dag2.nodes[c]['op'] == 'constant']
            assert len(c2inp) == 1, f"Child {child2} of constant vertex {v1} has more than one constant input"
            bijection[v1] = c2inp[0]
    
    return bijection

def check_tdag_equal(raw_dag1: Tdag, raw_dag2: Tdag) -> bool:
    tdag1 = raw_dag1.copy_tdag()
    tdag2 = raw_dag2.copy_tdag()
    tdag1.squash_ops(['bootstrap', 'modswitch', 'rescale', 'upscale'])
    tdag2.squash_ops(['bootstrap', 'modswitch', 'rescale', 'upscale'])
    addition_squash(tdag1)
    addition_squash(tdag2)
    return tdag_equal_biject(tdag1, tdag2) is not None