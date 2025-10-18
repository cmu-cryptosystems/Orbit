from ...tdag import *
from collections import deque

def region_partition(tdag: Tdag) -> tuple[dict[int, Tdag], dict[str, int], dict[tuple[str, str], int]]:
    # Note: ReSBM region inputs means having input edges from previous regions
    #       So input nodes can have predecessors within the same region
    # Note: ReSBM region outputs are not really belong to the region, just for including all outgoing edges
    #       When calculating region latency, only consider internal nodes/edges
    in_degrees = dict(tdag.in_degree())
    queue = deque([v for v, d in in_degrees.items() if d == 0])
    fw_mul_depth = {v: 0 for v in queue}
    while queue:
        v = queue.popleft()
        for u in tdag.successors(v):
            if tdag.nodes[v]['op'] == 'mul':
                fw_mul_depth[u] = max(fw_mul_depth.get(u, 0), fw_mul_depth[v] + 1)
            else:
                fw_mul_depth[u] = max(fw_mul_depth.get(u, 0), fw_mul_depth[v])
            in_degrees[u] -= 1
            if in_degrees[u] == 0:
                queue.append(u)
    
    out_degrees = dict(tdag.out_degree())
    queue = deque([v for v, d in out_degrees.items() if d == 0])
    bw_mul_depth = {v: fw_mul_depth[v] for v in queue}
    while queue:
        v = queue.popleft()
        v_mul_depth = bw_mul_depth[v]
        if tdag.nodes[v]['op'] == 'mul':
            v_mul_depth -= 1
        for u in tdag.predecessors(v):
            bw_mul_depth[u] = min(bw_mul_depth.get(u, v_mul_depth), v_mul_depth)
            out_degrees[u] -= 1
            if out_degrees[u] == 0:
                queue.append(u)
    
    # special update for constants
    for v in tdag.nodes:
        if tdag.nodes[v]['op'] == 'constant':
            assert tdag.out_degree(v) == 1, f"Constant node {v} has out_degree {tdag.out_degree(v)}, expected 1"
            u = list(tdag.successors(v))[0]
            # constant should be in the same region as its only successor
            bw_mul_depth[v] = bw_mul_depth[u]
            
    regions = dict() # bw_mul_depth -> tdag
    node_to_region = dict()
    edge_to_region = dict()
    for v in tdag.nodes:
        r_id = bw_mul_depth[v]
        if r_id not in regions:
            regions[r_id] = Tdag(tdag.params, name=f"region_{r_id}")
        regions[r_id].add_node(v, **tdag.nodes[v])
        node_to_region[v] = r_id
    for u, v in tdag.edges:
        ru = node_to_region[u]
        rv = node_to_region[v]
        if v not in regions[ru].nodes:
            new_attr = tdag.nodes[v].copy()
            new_attr['op'] = 'output'
            new_attr['op_descr'] = dict()
            regions[ru].add_node(v, **new_attr)
            regions[ru].outputs.add(v)
            regions[rv].inputs.add(v)
            # only for adding output nodes, not really belong to this region
        regions[ru].add_edge(u, v, **tdag.get_edge_data(u, v))
        edge_to_region[(u, v)] = ru
        
    for inp in tdag.inputs:
        r_id = node_to_region[inp]
        regions[r_id].inputs.add(inp)
    
    return regions, node_to_region, edge_to_region