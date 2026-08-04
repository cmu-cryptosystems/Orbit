"""Single-input/single-output (SISO) partitioning and bypass splitting.

:func:`rdag_siso_partition` cuts a DAG at depths where exactly one value is live,
yielding a sequence of SISO sub-DAGs small enough for the per-partition ILP.
:func:`handle_bypass` detects the "bypass" pattern (an output combining a deep
main path with a shallow skip path) and splits it into a main and a bypass DAG so
each can be solved independently. These are two of Orbit's DAG-reduction steps
for partitioning and bypass handling.
"""
from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

import networkx as nx
from collections import deque

def _get_final_use_points(dag: Tdag, v_to_depth):
    final_use_depth = dict()
    for v in v_to_depth.keys():
        final = 0
        for w in dag.successors(v):
            final = max(final, v_to_depth[w])
        final_use_depth[v] = final
    return final_use_depth

def _do_partition(dag: Tdag, depth_to_vs: dict[int, list[str]], partition_depths: list[int]) -> list[Tdag]:
    dags_list = []
    partition_ranges = [(partition_depths[i], partition_depths[i+1]) for i in range(len(partition_depths)-1)]
    for i, (d_start, d_end) in enumerate(partition_ranges):
        new_dag = Tdag(dag.params, f"{dag.name}_{i}")
        # add input node
        assert len(depth_to_vs[d_start]) == 1, f"SISO partitioning requires single input node per partition, but found {len(depth_to_vs[d_start])}, depth {d_start}."
        p_root = depth_to_vs[d_start][0]
        pr_attr = dag.nodes[p_root]
        new_dag.add_node(p_root, level=pr_attr["level"], scale=pr_attr["scale"], 
                         op='input', op_descr={},
                         weight=pr_attr["weight"], comment=pr_attr['comment'])
        new_dag.inputs.add(p_root)
        # add middle nodes + output node 
        for d in range(d_start + 1, d_end + 1):
            for v in depth_to_vs[d]:
                new_dag.add_node(v, **dag.nodes[v])
                # add input edges
                for u in dag.predecessors(v):
                    if dag.nodes[u]['op'] == 'constant':
                        new_dag.add_node(u, **dag.nodes[u])
                    assert u in new_dag.nodes, f"Predecessor node {u} not found in partitioned DAG."
                    new_dag.add_edge(u, v, weight=dag.edges[u, v]['weight'])
        # update new dag outputs
        assert len(depth_to_vs[d_end]) == 1, "SISO partitioning requires single output node per partition"
        p_out = depth_to_vs[d_end][0]
        new_dag.outputs.add(p_out)
        # finalize
        dags_list.append(new_dag)
    return dags_list

def _filter_list(lst: list[int], depth_to_vs: dict[int, list[str]], delta: int) -> list[int]:
    if len(lst) <= 2:
        return lst
    prev_vs = [0]
    for i in range(1, len(depth_to_vs)):
        this_len = len(depth_to_vs[i-1]) if i-1 in depth_to_vs else 0
        prev_vs.append(prev_vs[-1] + this_len)
    result = [lst[0]]
    for i in range(1, len(lst)-1):
        if (prev_vs[lst[i]] - prev_vs[result[-1]] >= delta) or (prev_vs[lst[i+1]] - prev_vs[lst[i]] >= delta):
            result.append(lst[i])
    result.append(lst[-1])
    return result

def rdag_siso_partition(dag: Tdag, delta: int) -> list[Tdag]:
    """Split ``dag`` into SISO partitions at single-live-value depths.

    Walks the depth traversal tracking the live set; a depth where only one value
    is live is a valid cut point. ``delta`` filters cut points that are too close
    together (:func:`_filter_list`) so partitions stay a reasonable size. Returns
    the ordered list of partition sub-DAGs.
    """
    depth_to_vs_with_const = dag.get_depth_traversal()
    depth_to_vs = { d: [ v for v in vs if dag.nodes[v]['op'] != 'constant' ] for d, vs in depth_to_vs_with_const.items() }
    v_to_depth = { v: d for d, vs in depth_to_vs.items() for v in vs }
    final_use_depth = _get_final_use_points(dag, v_to_depth)
    
    partition_depths = []
    liveSet = set()
    for d, vs in sorted(depth_to_vs.items()):
        liveSet = { v for v in liveSet if final_use_depth[v] > d }
        for v in vs:
            liveSet.add(v)
        if len(liveSet) == 1:
            partition_depths.append(d)
    filtered_partition_depths = _filter_list(partition_depths, depth_to_vs, delta)
    rdags = _do_partition(dag, depth_to_vs, filtered_partition_depths)
    return rdags


def _get_all_ancestors(dag: Tdag, v: str) -> set[str]:
    anc = {v}
    queue = deque([v])
    while queue:
        u = queue.popleft()
        for pred in dag.predecessors(u):
            if pred not in anc:
                anc.add(pred)
                queue.append(pred)
    return anc

def _do_bypass_split(dag: Tdag, fork_inp: str, main_oup: str, bypass_oup: str, main_vs: set[str], bypass_vs: set[str]) -> tuple[Tdag, Tdag]:
    main_dag = Tdag(dag.params, f"{dag.name}_m")
    bypass_dag = Tdag(dag.params, f"{dag.name}_bypass")
    
    # adding fork input node to both dags
    fork_attr = dag.nodes[fork_inp]
    main_dag.add_node(fork_inp, level=fork_attr["level"], scale=fork_attr["scale"], 
                      op='input', op_descr={},
                      weight=fork_attr["weight"], comment=fork_attr['comment'])
    main_dag.inputs.add(fork_inp)
    bypass_dag.add_node(fork_inp, level=fork_attr["level"], scale=fork_attr["scale"], 
                        op='input', op_descr={},
                        weight=fork_attr["weight"], comment=fork_attr['comment'])
    bypass_dag.inputs.add(fork_inp)
    # adding main-oup node to bypass dag as "fake input"
    bypass_dag.add_node(main_oup, level=dag.nodes[main_oup]["level"], scale=dag.nodes[main_oup]["scale"],
                        op='input', op_descr={},
                        weight=dag.nodes[main_oup]["weight"], comment=dag.nodes[main_oup]['comment'])
    
    # add vertices
    for v in dag.nodes:
        if v == fork_inp:
            continue
        if v in main_vs:
            main_dag.add_node(v, **dag.nodes[v])
        else:
            # here we make dag.outputs part of bypass dag
            assert v in bypass_vs or v in dag.outputs, f"Node {v} not found in either main or bypass sets."
            bypass_dag.add_node(v, **dag.nodes[v])
    # set output nodes
    main_dag.outputs.add(main_oup)
    bypass_dag.outputs.add(list(dag.outputs)[0])  # original output node
    # add edges
    for u, v in dag.edges:
        if v in main_vs:
            assert u in main_vs, f"Edge ({u}->{v}) inconsistent: {v} in main set but {u} not."
            main_dag.add_edge(u, v, weight=dag.edges[u, v]['weight'])
        else:
            assert u in bypass_vs or u == main_oup, f"Edge ({u}->{v}) inconsistent: {v} in bypass set but {u} not."
            bypass_dag.add_edge(u, v, weight=dag.edges[u, v]['weight'])
    return main_dag, bypass_dag
        

def handle_bypass(dag: Tdag, dep_thres: int|None) -> tuple[Tdag, Tdag|None]:
    """Detect and split a bypass structure at the DAG's output.

    A bypass exists when the single output combines two non-constant inputs whose
    multiplicative-depth difference exceeds ``dep_thres``. When found, the DAG is
    split into ``(main, bypass)`` sub-DAGs; otherwise ``(dag, None)`` is returned.
    ``dep_thres=None`` disables bypass handling.
    """
    # bypass not enabled
    if dep_thres is None:
        return dag, None
    
    # check output structure: must have 2 inputs for the output node
    assert len(dag.inputs) == 1 and len(dag.outputs) == 1, "Bypass handling requires single input and single output nodes."
    out_v = list(dag.outputs)[0]
    if dag.in_degree(out_v) != 2:
        return dag, None
    out_inp_0 = list(dag.predecessors(out_v))[0]
    out_inp_1 = list(dag.predecessors(out_v))[1]
    # one constant input -> still no bypass
    if dag.nodes[out_inp_0]['op'] == 'constant' or dag.nodes[out_inp_1]['op'] == 'constant':
        return dag, None
    
    # check mult depth difference
    topo_vs = list(nx.topological_sort(dag))
    mul_depths = dict()
    for v in topo_vs:
        if dag.in_degree(v) == 0:
            mul_depths[v] = 0
        else:
            inp_mul = max([mul_depths[u] for u in dag.predecessors(v)])
            mul_depths[v] = inp_mul + (1 if dag.nodes[v]['op'] == 'mul' else 0)
    
    # skip bypass if depth difference is small
    if abs(mul_depths[out_inp_0] - mul_depths[out_inp_1]) < dep_thres:
        return dag, None
    
    main_oup, bypass_oup = out_inp_0, out_inp_1
    if mul_depths[out_inp_0] < mul_depths[out_inp_1]:
        main_oup, bypass_oup = out_inp_1, out_inp_0
    # ensure main_oup has higher mult depth

    main_vs = _get_all_ancestors(dag, main_oup)
    bypass_vs = _get_all_ancestors(dag, bypass_oup)
    forks = bypass_vs.intersection(main_vs)
    # input node should be exactly the only fork point
    if len(forks) != 1 or list(dag.inputs)[0] not in forks:
        return dag, None
    
    fork_inp = list(dag.inputs)[0]
    main_dag, bypass_dag = _do_bypass_split(dag, fork_inp, main_oup, bypass_oup, main_vs, bypass_vs)
    return main_dag, bypass_dag