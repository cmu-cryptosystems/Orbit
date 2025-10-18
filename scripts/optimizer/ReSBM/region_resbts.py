from ...tdag import *
from ...assignment import *
from ...params.params import Params
from ...latency_estimator import *
from ...visualize import *
import networkx as nx
from collections import deque

def _assign_rdag_lvlscl(rdag: Tdag, tdag: Tdag, in_lvl: int, in_scl: int):
    def _get_ls(v):
        if v in rdag.nodes:
            return rdag.nodes[v]['level'], rdag.nodes[v]['scale']
        else:
            return in_lvl, in_scl
    
    topo_order = list(nx.topological_sort(rdag))
    for v in topo_order:
        rdag.nodes[v]['level'] = None
        rdag.nodes[v]['scale'] = None
        if rdag.nodes[v]['op'] == 'mul':
            assert 1 <= tdag.in_degree(v) <= 2, f"Mul node {v} has in_degree {tdag.in_degree(v)}, expected 1 or 2"
            if tdag.in_degree(v) == 1:
                u0, u1 = list(tdag.predecessors(v))[0], list(tdag.predecessors(v))[0]
            else:
                u0, u1 = list(tdag.predecessors(v))[0], list(tdag.predecessors(v))[1]
            u0_lvl, u0_scl = _get_ls(u0)
            u1_lvl, u1_scl = _get_ls(u1)
            assert u0_lvl == u1_lvl, f"Mul node {v} has input levels {u0_lvl} and {u1_lvl}, expected equal"
            rdag.nodes[v]['level'] = u0_lvl
            rdag.nodes[v]['scale'] = u0_scl + u1_scl
        else:
            for u in tdag.predecessors(v):
                if rdag.nodes[v]['op'] == 'output' and (u not in rdag.nodes or rdag.nodes[u]['op'] == 'output'):
                    continue
                u_lvl, u_scl = _get_ls(u)
                if rdag.nodes[v]['level'] is None:
                    rdag.nodes[v]['level'] = u_lvl
                    rdag.nodes[v]['scale'] = u_scl
                else:
                    assert rdag.nodes[v]['level'] == u_lvl, f"Node {v} has input levels {rdag.nodes[v]['level']} and {u_lvl}, expected equal."
                    rdag.nodes[v]['scale'] = max(rdag.nodes[v]['scale'], u_scl)
            if rdag.nodes[v]['level'] is None:
                # totally no input (should be input node of the whole tdag, or constant nodes)
                assert rdag.nodes[v]['op'] in ['input', 'constant'], f"Node {v} has no input, but op is {rdag.nodes[v]['op']}, expected 'input' or 'constant'"
                if rdag.nodes[v]['op'] == 'input':
                    rdag.nodes[v]['level'] = in_lvl
                    rdag.nodes[v]['scale'] = in_scl
                else:
                    rdag.nodes[v]['level'] = in_lvl
                    rdag.nodes[v]['scale'] = tdag.params.Sw


def _rescale_mincut(rdag: Tdag, this_lvl: int, le: LatencyEstimator) -> tuple[Tdag, set[int]]:
    params = rdag.params
    topo_order = list(nx.topological_sort(rdag))
    
    srcs = set()
    sinks = set()
    for v in rdag.nodes:
        if rdag.nodes[v]['op'] == 'mul':
            srcs.add(v)
        if v in rdag.outputs:
            sinks.add(v)
    
    mc_dag = nx.DiGraph()
    
    lat_sum = dict() # u -> prev cut cost
    for v in topo_order:
        lat_res = le.op_lmaps['rescale_single'][this_lvl-1] * rdag.nodes[v]['weight']
        if v in srcs:
            lat_sum[v] = 0
        else:
            lat_sum[v] = sum([lat_sum[u] for u in rdag.predecessors(v)])
            lat_sum[v] += get_tdag_vertex_lat(rdag, v, this_lvl, le)
            lat_sum[v] -= get_tdag_vertex_lat(rdag, v, this_lvl - 1, le)
        
        child_num = rdag.out_degree(v)
        for u in rdag.successors(v):
            mc_dag.add_edge(v, u, capacity=(lat_sum[v] + lat_res) / child_num)
    
    super_source = "super_source"
    super_sink = "super_sink"
    mc_dag.add_node(super_source)
    mc_dag.add_node(super_sink)
    for v in srcs:
        mc_dag.add_edge(super_source, v, capacity=float('inf'))
    for v in sinks:
        mc_dag.add_edge(v, super_sink, capacity=float('inf'))
    
    _, (reachable, non_reachable) = nx.minimum_cut(mc_dag, super_source, super_sink)
    
    reachable = set(reachable) - {super_source, super_sink}
    non_reachable = set(non_reachable) - {super_source, super_sink}
    
    res_vs = set()
    for v in reachable:
        for u in rdag.successors(v):
            if u in non_reachable:
                res_vs.add(v)
                break
    # print(f" reachable nodes: {reachable}\n non_reachable nodes: {non_reachable}\n rescale nodes: {res_vs}")
    
    real_non_reachable = set()
    real_non_reachable.add(super_sink)
    queue = deque([super_sink])
    while queue:
        v = queue.popleft()
        for u in mc_dag.predecessors(v):
            if u in res_vs:
                continue
            if u not in real_non_reachable:
                real_non_reachable.add(u)
                queue.append(u)
    non_reachable = real_non_reachable - {super_sink}
    # print(f" non_reachable nodes for rescale cut: {non_reachable}")
    
    for v in non_reachable:
        rdag.nodes[v]['level'] -= 1
        rdag.nodes[v]['scale'] = max(rdag.nodes[v]['scale'] - params.Sf, params.Sw)

    return rdag, res_vs

def _bts_mincut(rdag: Tdag, in_lvl: int, target_lvl: int, res_vs: set[int], le: LatencyEstimator) -> tuple[Tdag, set[int]]:
    params = rdag.params
    remain_vs = res_vs.copy()
    for v in rdag.nodes:
        if rdag.nodes[v]['level'] == in_lvl - 1:
            remain_vs.add(v)
    # checking scale for remain_vs
    for v in remain_vs:
        vl = rdag.nodes[v]['level']
        vs = rdag.nodes[v]['scale']
        if v in res_vs:
            vl -= 1
            vs = max(vs - params.Sf, params.Sw)
        assert vs <= params.Sf, f"Node {v} to bootstrap has scale {vs}, expected <= {params.Sf}"
        assert vl == params.bts_lb, f"Node {v} to bootstrap has level {vl}, expected == {params.bts_lb}"
    # build remain graph
    bts_dag = nx.DiGraph()
    for v in remain_vs:
        for u in rdag.successors(v):
            if u in remain_vs:
                bts_dag.add_edge(v, u)
    srcs = [v for v in remain_vs if bts_dag.in_degree(v) == 0]
    sinks = [v for v in remain_vs if bts_dag.out_degree(v) == 0]
    revtopo_order = list(reversed(list(nx.topological_sort(bts_dag))))
    
    mc_dag = nx.DiGraph()
    lat_sum = dict() # u -> succ cut cost
    for v in revtopo_order:
        lat_bts = rdag.nodes[v]['weight'] * le.op_lmaps['bootstrap_single'][target_lvl]
        if v in sinks:
            lat_sum[v] = 0
        else:
            lat_sum[v] = sum([lat_sum[u] for u in bts_dag.successors(v)])
            lat_sum[v] += get_tdag_vertex_lat(rdag, v, target_lvl, le)
            lat_sum[v] -= get_tdag_vertex_lat(rdag, v, params.bts_lb, le)
        
        parent_num = rdag.in_degree(v)
        for u in bts_dag.predecessors(v):
            mc_dag.add_edge(u, v, capacity=(lat_sum[v] + lat_bts) / parent_num)
    
    super_source = "super_source"
    super_sink = "super_sink"
    mc_dag.add_node(super_source)
    mc_dag.add_node(super_sink)
    for v in srcs:
        mc_dag.add_edge(super_source, v, capacity=float('inf'))
    for v in sinks:
        mc_dag.add_edge(v, super_sink, capacity=float('inf'))
    
    _, (reachable, non_reachable) = nx.minimum_cut(mc_dag, super_source, super_sink)

    reachable = set(reachable) - {super_source, super_sink}
    non_reachable = set(non_reachable) - {super_source, super_sink}

    bts_vs = set()
    for v in reachable:
        for u in rdag.successors(v):
            if u in non_reachable:
                bts_vs.add(v)
                break
    
    for v in non_reachable:
        rdag.nodes[v]['level'] = target_lvl
        rdag.nodes[v]['scale'] = params.Sf
        
    return rdag, bts_vs

def _get_rdag_assign(og_rdag: Tdag, res_vs: set[int], bts_vs: set[int], reg_in_lvl: int, reg_in_scl: int, target_lvl: int, tdag: Tdag, edge_to_region_ids: dict[tuple[str, str], int], prev_regionIO, le: LatencyEstimator|None) -> tuple[float, Assign]:
    rdag = og_rdag.copy_tdag()
    for v in rdag.inputs:
        for u in tdag.predecessors(v):
            if u in rdag.nodes and rdag.nodes[u]['op'] != 'input':
                continue
            if u not in rdag.nodes:
                reg_u = edge_to_region_ids[(u, v)]
                reg_u_out_lvl, reg_u_out_scl = prev_regionIO[reg_u][2], prev_regionIO[reg_u][3]
                rdag.add_node(u, op='input', weight=tdag.edges[u, v]['weight'], op_descr = dict(),
                            level=reg_u_out_lvl, scale=reg_u_out_scl, comment=f"from region #{reg_u}")
            rdag.add_edge(u, v, weight = tdag.edges[u, v]['weight'])
    
    assign = Assign(rdag)  
    
    for v in rdag.nodes:
        v_il = rdag.nodes[v]['level']
        v_is = rdag.nodes[v]['scale']
        assign.v_lvl_in[v] = v_il
        assign.v_scl_in[v] = v_is
        if rdag.nodes[v]['op'] == 'input':
            assign.v_lvl_out[v] = reg_in_lvl
            assign.v_scl_out[v] = reg_in_scl
        else:
            v_ol, v_os = v_il, v_is
            if v in res_vs:
                v_ol = v_il - 1
                v_os = max(v_is - rdag.params.Sf, rdag.params.Sw)
            if v in bts_vs:
                assert v_ol == rdag.params.bts_lb, f"Node {v} to bootstrap has level {v_il}, expected == {rdag.params.bts_lb}"
                assert v_os <= rdag.params.Sf, f"Node {v} to bootstrap has scale {v_is}, expected <= {rdag.params.Sf}"
                v_ol = target_lvl
                v_os = rdag.params.Sf
            assign.v_lvl_out[v] = v_ol
            assign.v_scl_out[v] = v_os
    
    # update constant nodes' lvl/scl
    for v in rdag.nodes:
        if rdag.nodes[v]['op'] == 'constant':
            assert rdag.out_degree(v) == 1, f"Constant node {v} has out_degree {rdag.out_degree(v)}, expected 1"
            chd = list(rdag.successors(v))[0]
            if rdag.nodes[chd]['op'] == 'add':
                assign.v_lvl_out[v] = assign.v_lvl_in[chd]
                assign.v_scl_out[v] = assign.v_scl_in[chd]
            else:
                assert rdag.nodes[chd]['op'] == 'mul', f"Constant node {v}'s child {chd} has op {rdag.nodes[chd]['op']}, expected 'add' or 'mul'"
                other_v = [u for u in rdag.predecessors(chd) if u != v][0]
                assign.v_lvl_out[v] = assign.v_lvl_in[chd]
                assign.v_scl_out[v] = assign.v_scl_in[chd] - assign.v_scl_out[other_v]
            assign.v_lvl_in[v] = assign.v_lvl_out[v]
            assign.v_scl_in[v] = assign.v_scl_out[v]
            
    for v in rdag.nodes:
        for u in rdag.predecessors(v):
            assign.e_lvl_out[(u, v)] = assign.v_lvl_out[u]
            assign.e_scl_out[(u, v)] = assign.v_scl_out[u]
            if assign.e_lvl_out[(u, v)] != assign.v_lvl_in[v] or (rdag.nodes[v]['op']!='mul' and assign.e_scl_out[(u, v)] != assign.v_scl_in[v]):
                assert rdag.nodes[v]['op'] != 'mul'
                assign.e_lvl_out[(u, v)] = assign.v_lvl_in[v]
                assign.e_scl_out[(u, v)] = assign.v_scl_in[v]
    
    if le is None:
        return None, assign
    
    try:
        total_lat = estimate_assign(assign, le)
    except Exception as e:
        visualize(rdag, "error_region_dag.svg")
        for k in assign.v_lvl_out:
            in_lvl = assign.v_lvl_in.get(k, None)
            in_scl = assign.v_scl_in.get(k, None)
            out_lvl = assign.v_lvl_out.get(k, None)
            out_scl = assign.v_scl_out.get(k, None)
            print(f"Node {k}: in ({in_lvl}, {in_scl}), out ({out_lvl}, {out_scl})")
        for k in assign.e_lvl_out:
            out_lvl = assign.e_lvl_out.get(k, None)
            out_scl = assign.e_scl_out.get(k, None)
            print(f"Edge {k}: out ({out_lvl}, {out_scl})")
        print(f" bts vs: {bts_vs}")
        print(f" res vs: {res_vs}")
        raise e
    
    return total_lat, assign


def tryRegionResBts(region: Tdag, in_lvl, in_scl, target_lvl: int, tdag: Tdag, edge_to_region_ids, prev_regionIO, le: LatencyEstimator, is_rescale: bool, is_bootstrap: bool):
    rdag = region.copy_tdag()
    
    _assign_rdag_lvlscl(rdag, tdag, in_lvl, in_scl)
    res_vs = set()
    bts_vs = set()
    if is_rescale:
        rdag, res_vs = _rescale_mincut(rdag, in_lvl, le)
    if is_bootstrap:
        rdag, bts_vs = _bts_mincut(rdag, in_lvl, target_lvl, res_vs, le)
    
    try:
        rdag_lat, rdag_assign = _get_rdag_assign(rdag, res_vs, bts_vs, in_lvl, in_scl, target_lvl, tdag, edge_to_region_ids, prev_regionIO, le)
    except Exception as e:
        print(f"Error in getting assignment for region with in ({in_lvl}, {in_scl}) and target level {target_lvl}")
        raise e
    out_lvl = None
    out_scl = None
    for v in rdag.outputs:
        if out_lvl is None:
            out_lvl = rdag_assign.v_lvl_out[v]
            out_scl = rdag_assign.v_scl_out[v]
        else:
            assert out_lvl == rdag_assign.v_lvl_out[v], f"Region output node {v} has level {rdag_assign.v_lvl_out[v]}, expected equal to other output nodes' level {out_lvl}"
            out_scl = max(out_scl, rdag_assign.v_scl_out[v])
    return rdag_lat, out_lvl, out_scl


def assignRegionResBts(region: Tdag, tdag: Tdag, edge_to_region_ids, all_regionIO, regionIO, le: LatencyEstimator) -> tuple[dict, dict, dict, dict]:
    in_lvl, in_scl = regionIO[0], regionIO[1]
    target_lvl = regionIO[4]
    is_rescale = regionIO[5]
    is_bootstrap = regionIO[6]
    
    rdag = region.copy_tdag()
    _assign_rdag_lvlscl(rdag, tdag, in_lvl, in_scl)
    res_vs = set()
    bts_vs = set()
    if is_rescale:
        rdag, res_vs = _rescale_mincut(rdag, in_lvl, le)
    if is_bootstrap:
        rdag, bts_vs = _bts_mincut(rdag, in_lvl, target_lvl, res_vs, le)

    _, rdag_assign = _get_rdag_assign(rdag, res_vs, bts_vs, in_lvl, in_scl, target_lvl, tdag, edge_to_region_ids, all_regionIO, None)
    return rdag_assign.v_lvl_out, rdag_assign.v_scl_out, rdag_assign.e_lvl_out, rdag_assign.e_scl_out