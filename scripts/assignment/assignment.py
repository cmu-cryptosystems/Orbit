from ..tdag.tdag import Tdag
from ..params.params import Params

class Assign:
    def __init__(self, tdag: Tdag):
        self.tdag = tdag
        self.params = tdag.params
        self.v_lvl_out = dict()  # node -> output level, necessary for all nodes except constants
        self.v_scl_out = dict()  # node -> output scale, necessary for all nodes except constants
        self.v_lvl_in = dict()   # node -> input level, not necessary for all nodes
        self.v_scl_in = dict()   # node -> input scale, not necessary for all nodes
        self.e_lvl_out = dict()  # edge -> output level, necessary for all edges except from constants
        self.e_scl_out = dict()  # edge -> output scale, necessary for all edges except from constants
        self.v_err_in = dict()   # optional node -> estimated input error for resilience search
        self.v_err_out = dict()  # optional node -> estimated output error for resilience search
        self.e_err_out = dict()  # optional edge -> estimated output error for resilience search

    def to_dict(self) -> dict:
        return {
            'v_lvl_out': self.v_lvl_out,
            'v_scl_out': self.v_scl_out,
            'v_lvl_in': self.v_lvl_in,
            'v_scl_in': self.v_scl_in,
            'e_lvl_out': self.e_lvl_out,
            'e_scl_out': self.e_scl_out,
            'v_err_in': self.v_err_in,
            'v_err_out': self.v_err_out,
            'e_err_out': self.e_err_out,
        }
    
    @staticmethod
    def from_dict(tdag: Tdag, data: dict) -> 'Assign':
        assign = Assign(tdag)
        assign.v_lvl_out = data.get('v_lvl_out', dict())
        assign.v_scl_out = data.get('v_scl_out', dict())
        assign.v_lvl_in = data.get('v_lvl_in', dict())
        assign.v_scl_in = data.get('v_scl_in', dict())
        assign.e_lvl_out = data.get('e_lvl_out', dict())
        assign.e_scl_out = data.get('e_scl_out', dict())
        assign.v_err_in = data.get('v_err_in', dict())
        assign.v_err_out = data.get('v_err_out', dict())
        assign.e_err_out = data.get('e_err_out', dict())
        return assign
    
    def _deduce_in_lvl_scl(self, v: str) -> tuple[int|None, int|None]:
        v_deduced_il = None
        v_deduced_is = None
        preds = list(self.tdag.predecessors(v))
        p_v_il = []
        p_v_is = []
        for p in preds:
            if self.tdag.nodes[p]['op'] == 'constant':
                p_v_il.append(self.v_lvl_out[p])
                p_v_is.append(self.v_scl_out[p])
            else:
                p_v_il.append(self.e_lvl_out[(p, v)])
                p_v_is.append(self.e_scl_out[(p, v)])
        
        if len(p_v_il) == 0:
            return None, None
        if all(lvl == p_v_il[0] for lvl in p_v_il):
            v_deduced_il = p_v_il[0]
        else:
            raise ValueError(f"Node {v} has inconsistent input levels from predecessors: {p_v_il}")
        
        if self.tdag.nodes[v]['op'] == 'mul':
            if len(p_v_is) > 2 or len(p_v_is) == 0:
                raise ValueError(f"Node {v} with operation mul must have 1 or 2 predecessors.")
            if len(p_v_is) == 2:
                v_deduced_is = p_v_is[0] + p_v_is[1]
            else:
                v_deduced_is = p_v_is[0] * 2
        else:
            if all(scl == p_v_is[0] for scl in p_v_is):
                v_deduced_is = p_v_is[0]
            else:
                raise ValueError(f"Node {v} has inconsistent input scales from predecessors: {p_v_is}")
        
        return v_deduced_il, v_deduced_is

    def get_v_in_lvl_scl(self, v: str) -> tuple[int|None, int|None]:
        v_il = self.v_lvl_in.get(v, None)
        v_is = self.v_scl_in.get(v, None)
        deduced_il, deduced_is = self._deduce_in_lvl_scl(v)
        if v_il is not None and deduced_il is not None and v_il != deduced_il:
            raise ValueError(f"Node {v} with operation {self.tdag.nodes[v]['op']} has inconsistent input level assignment {v_il} vs deduced {deduced_il}.")
        if v_is is not None and deduced_is is not None and v_is != deduced_is:
            raise ValueError(f"Node {v} with operation {self.tdag.nodes[v]['op']} has inconsistent input scale assignment {v_is} vs deduced {deduced_is}.")
        if v_il is None and deduced_il is None:
            raise ValueError(f"Node {v} with operation {self.tdag.nodes[v]['op']} missing input level assignment and cannot be deduced.")
        if v_is is None and deduced_is is None:
            raise ValueError(f"Node {v} with operation {self.tdag.nodes[v]['op']} missing input scale assignment and cannot be deduced.")
        
        v_il = v_il if v_il is not None else deduced_il
        v_is = v_is if v_is is not None else deduced_is
        return v_il, v_is

    
    def check_assign(self):
        # Check node level/scale assignments
        for v in self.tdag.nodes:
            op = self.tdag.nodes[v]['op']
            v_ol = self.v_lvl_out.get(v, None)
            v_os = self.v_scl_out.get(v, None)
            if v_ol is None or v_os is None:
                raise ValueError(f"Node {v} with operation {op} missing output level/scale assignment.")

            if op == 'constant':
                continue

            scale_lb = self.params.scale_lower_bound(v, self.tdag.nodes[v], "out")
            if v_os < scale_lb:
                raise ValueError(
                    f"Node {v} output scale {v_os} below local lower bound {scale_lb}."
                )

            v_il, v_is = self.get_v_in_lvl_scl(v)
            input_scale_lb = self.params.scale_lower_bound(v, self.tdag.nodes[v], "in")
            if v_is < input_scale_lb:
                raise ValueError(
                    f"Node {v} input scale {v_is} below local lower bound {input_scale_lb}."
                )
            if not self.params.check_resbts(v_il, v_is, v_ol, v_os):
                raise ValueError(f"Node {v} with operation {op} has invalid level/scale transition: in({v_il}, {v_is}) -> out({v_ol}, {v_os}).")
        
        # Check edge level/scale assignments
        for v in self.tdag.nodes:
            for u in self.tdag.predecessors(v):
                if self.tdag.nodes[u]['op'] == 'constant':
                    continue
                e_ol = self.e_lvl_out.get((u, v), None)
                e_os = self.e_scl_out.get((u, v), None)
                if e_ol is None or e_os is None:
                    raise ValueError(f"Edge ({u} -> {v}) missing output level/scale assignment.")
                e_il = self.v_lvl_out.get(u, None)
                e_is = self.v_scl_out.get(u, None)
                edge_scale_lb = max(
                    self.params.scale_lower_bound(u, self.tdag.nodes[u], "out"),
                    self.params.scale_lower_bound(v, self.tdag.nodes[v], "in"),
                )
                if e_os < edge_scale_lb:
                    raise ValueError(
                        f"Edge ({u} -> {v}) output scale {e_os} below "
                        f"local lower bound {edge_scale_lb}."
                    )
                if not self.params.check_resbts(e_il, e_is, e_ol, e_os):
                    raise ValueError(f"Edge ({u} -> {v}) has invalid level/scale transition: in({e_il}, {e_is}) -> out({e_ol}, {e_os}).")
        
        return True
