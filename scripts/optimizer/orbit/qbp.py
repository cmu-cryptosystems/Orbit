import joblib
from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params

class QBP:
    def __init__(self, dag: Tdag):
        self.dag = dag
        self.io_to_cost = dict()  # (in_lvl, in_scl) -> (out_lvl, out_scl) -> cost (float)
        self.io_to_assign = dict()  # (in_lvl, in_scl) -> (out_lvl, out_scl) -> Assign
    
    def add_io_result(self, in_key: tuple[int, int], out_key: tuple[int, int], cost: float, assign: Assign):
        if in_key not in self.io_to_cost:
            self.io_to_cost[in_key] = {out_key: cost}
            self.io_to_assign[in_key] = {out_key: assign}
        else:
            self.io_to_cost[in_key][out_key] = cost
            self.io_to_assign[in_key][out_key] = assign
    
    def append_io_results(self, new_io_to_cost: dict, new_io_to_assign: dict):
        for in_key, out_to_cost in new_io_to_cost.items():
            if in_key not in self.io_to_cost:
                self.io_to_cost[in_key] = out_to_cost.copy()
                self.io_to_assign[in_key] = new_io_to_assign[in_key].copy()
            else:
                self.io_to_cost[in_key].update(out_to_cost)
                self.io_to_assign[in_key].update(new_io_to_assign[in_key])
                
    def get_io_cost(self, in_key: tuple[int, int], out_key: tuple[int, int]) -> float|None:
        return self.io_to_cost[in_key].get(out_key, None) if in_key in self.io_to_cost else None
    def get_io_assign(self, in_key: tuple[int, int], out_key: tuple[int, int]) -> Assign|None:
        return self.io_to_assign[in_key].get(out_key, None) if in_key in self.io_to_assign else None
    
    def get_i_costs(self, in_key: tuple[int, int]) -> dict|None:
        return self.io_to_cost.get(in_key, None)
    def get_i_assigns(self, in_key: tuple[int, int]) -> dict|None:
        return self.io_to_assign.get(in_key, None)
    
    def get_all_costs(self) -> dict:
        return self.io_to_cost
    def get_all_assigns(self) -> dict:
        return self.io_to_assign
    
    def save_qbp(self, filepath: str):
        out_dict = {
            'name': self.dag.name,
            'v_ins': list(self.dag.inputs),
            'v_outs': list(self.dag.outputs),
            'nodes': {},
            'edges': {e: self.dag.edges[e]['weight'] for e in self.dag.edges},
            'io_to_cost': self.io_to_cost,
            'io_to_assign': {}
        }
        for v in self.dag.nodes:
            out_dict['nodes'][v] = {
                'op': self.dag.nodes[v]['op'],
                'weight': self.dag.nodes[v]['weight'],
                'op_descr': self.dag.nodes[v]['op_descr']
            }
        for in_key, out_to_assign in self.io_to_assign.items():
            for out_key, assign in out_to_assign.items():
                if in_key not in out_dict['io_to_assign']:
                    out_dict['io_to_assign'][in_key] = {out_key: assign.to_dict()}
                else:
                    out_dict['io_to_assign'][in_key][out_key] = assign.to_dict()
        joblib.dump(out_dict, f"{filepath}.joblib")
        
    @staticmethod
    def load_qbp(filepath: str, params: Params) -> 'QBP':
        out_dict = joblib.load(filepath)
        dag = Tdag(params, out_dict['name'])
        for v, v_data in out_dict['nodes'].items():
            dag.add_node(v, level=None, scale=None, comment=None, **v_data)
        for e, weight in out_dict['edges'].items():
            dag.add_edge(e[0], e[1], weight=weight)
        dag.inputs = set(out_dict['v_ins'])
        dag.outputs = set(out_dict['v_outs'])
        qbp = QBP(dag)
        qbp.io_to_cost = out_dict['io_to_cost']
        for in_key, out_to_assign in out_dict['io_to_assign'].items():
            for out_key, assign_dict in out_to_assign.items():
                assign = Assign.from_dict(dag, assign_dict)
                if in_key not in qbp.io_to_assign:
                    qbp.io_to_assign[in_key] = {out_key: assign}
                else:
                    qbp.io_to_assign[in_key][out_key] = assign
        return qbp
    
