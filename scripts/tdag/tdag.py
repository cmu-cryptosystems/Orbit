import networkx as nx
from ..params.params import Params
from collections import deque, defaultdict

#################### Tdag Vertex ####################
# v: unique label name (str)
# v['op']: operation type (str)
# v['level']: operation target level (int)
# v['scale']: operation target scale (int)
# v['weight']: operation weight (due to compression) (int)
# v['op_descr']: operation description (dict)
# - 'input': {} (input node, no description)
# - 'constant': {'value': int, 'rms_var': float64} (constant value)
# - 'add': {'single': int, 'double': int} 
#    (indicates how many single/double additions are involved, 
#    'single'<=1, 'double' can be arbitrary)
# - 'mul': {'single': int, 'double': int} 
#    (indicates how many single/double multiplications are involved, 
#    ('single','double') can be either (0,1) or (1,0) )
# - 'rotate': {'offset': int, 'weight': int} 
#    (offset: rotation offset, weight: naf weight of the offset)
# - 'negate': {} (negation operation)
# - 'modswitch': {'downFactor': int} (downFactor: how many levels to drop)
# - 'rescale': {} (rescale operation)
# - 'upscale': {'upFactor': int} (upFactor: how many scales to raise)
# - 'bootstrap': {'targetLevel': int} 
#   (targetLevel: the level after bootstrapping)
# v['comment']: operation comment (str)
#################### Tdag Edge ####################
# e['weight']: edge weight (int)
#####################################################
class Tdag(nx.DiGraph):
    def __init__(self, params: Params, name:str):
        super().__init__()
        self.params = params
        self.name = name
        self.inputs = set() # input node labels
        self.outputs = set() # output node labels
        self.muldepth = None # multiplicative depth of the circuit
        
        # Assumption: all inputs' level/scale are the same
        #            all outputs' level/scale are the same
        #    Orbit: siso partitions, must be the same
        self.in_level = None # input level
        self.in_scale = None # input scale
        self.out_level = None # output level
        self.out_scale = None # output scale

    def get_v_weights(self, v: str) -> tuple[int, int]:
        op = self.nodes[v]['op']
        if op in ['input', 'constant', 'dummy', 'output']:
            single_cnt = 0
            double_cnt = 0
        elif op == 'add':
            single_cnt = self.nodes[v]['op_descr'].get('single', 0)
            double_cnt = self.nodes[v]['op_descr'].get('double', 0)
        elif op == 'mul':
            single_cnt = self.nodes[v]['op_descr'].get('single', 0) * self.nodes[v]['weight']
            double_cnt = self.nodes[v]['op_descr'].get('double', 0) * self.nodes[v]['weight']
        elif op == 'rotate':
            single_cnt = self.nodes[v]['op_descr'].get('weight', 0)
            double_cnt = 0
        else:
            single_cnt = self.nodes[v]['weight']
            double_cnt = 0
        return single_cnt, double_cnt
    
    def get_full_size(self) -> int:
        total_size = 0
        for v in self.nodes:
            total_size += self.nodes[v]['weight']
        return total_size
    
    def get_depth_traversal(self) -> dict[int, list[str]]:
        in_degrees = dict(self.in_degree())
        queue = deque([v for v, d in in_degrees.items() if d == 0])
        depths = {v: 0 for v in queue}
        
        processed_cnt = 0
        while queue:
            v = queue.popleft()
            processed_cnt += 1
            for u in self.successors(v):
                depths[u] = max(depths.get(u, 0), depths[v] + 1)
                in_degrees[u] -= 1
                if in_degrees[u] == 0:
                    queue.append(u)
        
        if processed_cnt != len(self.nodes):
            unreachable = set(self.nodes) - set(depths.keys())
            raise Exception(f"Graph has cycles or is not fully connected. Unreachable nodes: {unreachable}")
        
        results = defaultdict(list)
        for v, d in depths.items():
            results[d].append(v)
        return dict(results)
        

    def squash_ops(self, ops: list[str]):
        '''Squashes all vertices with the given operation'''
        print(f"\tRDAG: Squashing {ops}!")
        targets = [v for v in self.nodes if self.nodes[v]['op'] in ops]
        for v in targets:
            self.pop_vertex(v)
    
    def pop_vertex(self, v: str):
        assert self.in_degree(v) == 1, "Can only pop a vertex with in_degree=1"
        assert v not in self.inputs, "Cannot pop an input vertex"
        
        v_in = list(self.predecessors(v))[0]
        if v in self.outputs:
            self.outputs.remove(v)
            self.outputs.add(v_in)
        
        for u in self.successors(v):
            e_data = self.get_edge_data(v, u)
            self.add_edge(v_in, u, **e_data)
        self.remove_node(v)
        
    def insert_lsm_op(self, name: str, v_p: str, v_c:str, factor: int | None) -> str:
        '''Insert a level/scale modifying operation between v_p and v_c'''
        assert self.has_edge(v_p, v_c), f"Edge ({v_p}, {v_c}) does not exist"
        assert self.nodes[v_p].get('level', None) is not None, f"Node {v_p} has no level assigned"
        assert self.nodes[v_p].get('scale', None) is not None, f"Node {v_p} has no scale assigned"
        assert self.nodes[v_c].get('level', None) is not None, f"Node {v_c} has no level assigned"
        assert self.nodes[v_c].get('scale', None) is not None, f"Node {v_c} has no scale assigned"
        assert name in ['modswitch', 'rescale', 'upscale', 'bootstrap'], f"Unsupported level/scale management operation {name}"
        
        e_data = self.get_edge_data(v_p, v_c)
        assert e_data['weight'] == 1, "Edge weight must be 1"
        self.remove_edge(v_p, v_c)
        v_op = f"{v_p}_{name}_{factor}" if factor is not None else f"{v_p}_{name}"
        if v_op not in self.nodes:
            if name == 'modswitch':
                self.add_node(v_op, op=name, weight=1,
                                level=self.nodes[v_p]['level'] - factor,
                                scale=self.nodes[v_p]['scale'],
                                op_descr={'downFactor': factor},
                                comment=self.nodes[v_p]['comment']
                              )
            elif name == 'rescale':
                self.add_node(v_op, op=name, weight=1,
                                level=self.nodes[v_p]['level'] - 1,
                                scale=self.nodes[v_p]['scale'] - self.params.Sf,
                                op_descr={},
                                comment=self.nodes[v_p]['comment']
                              )
            elif name == 'upscale':
                self.add_node(v_op, op=name, weight=1,
                                level=self.nodes[v_p]['level'],
                                scale=self.nodes[v_p]['scale'] + factor,
                                op_descr={'upFactor': factor},
                                comment=self.nodes[v_p]['comment']
                              )
            elif name == 'bootstrap':
                self.add_node(v_op, op=name, weight=1,
                                level=factor,
                                scale=self.params.Sf,
                                op_descr={'targetLevel': factor},
                                comment=self.nodes[v_p]['comment']
                              )
        self.add_edge(v_p, v_op, weight=1)
        self.add_edge(v_op, v_c, **e_data)
        return v_op
    
    def copy_tdag(self) -> 'Tdag':
        new_tdag = Tdag(self.params, self.name+"_copy")
        for v in self.nodes:
            new_tdag.add_node(v, **self.nodes[v])
        for u, v in self.edges:
            new_tdag.add_edge(u, v, **self.get_edge_data(u, v))
        new_tdag.inputs = set(self.inputs)
        new_tdag.outputs = set(self.outputs)
        new_tdag.muldepth = self.muldepth
        new_tdag.in_level = self.in_level
        new_tdag.in_scale = self.in_scale
        new_tdag.out_level = self.out_level
        new_tdag.out_scale = self.out_scale
        return new_tdag