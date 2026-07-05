from .assignment import Assign
from ..tdag.tdag import Tdag
from ..tdag.check_tdag import check_tdag
from ..visualize import visualize
from ..tdag.addition_squash import addition_desquash
import numpy as np

def _decode_edge(fdag: Tdag, u: str, v: str, scl: int|None):
    in_lvl = fdag.nodes[u]['level']
    in_scl = fdag.nodes[u]['scale']
    out_lvl = fdag.nodes[v]['level']
    out_scl = fdag.nodes[v]['scale'] if scl is None else scl
    
    Sf = fdag.params.Sf
    bts_input_level = int(getattr(fdag.params, 'bts_input_level', fdag.params.bts_lb))
    bts_input_scale = int(getattr(fdag.params, 'bts_input_scale', Sf))
    bts_output_scale = int(getattr(fdag.params, 'bts_output_scale', Sf))
    Smax = fdag.params.Sf + 2 * fdag.params.Sw
    
    def _insert_rescale(vc: str, vt: str, c_lvl: int, c_scl: int, t_lvl: int, t_scl: int):
        r = max(0, round(np.ceil((c_scl-t_scl)/Sf)))
        if t_lvl + r < c_lvl:
            # first do modswitch
            vc = fdag.insert_lsm_op('modswitch', vc, vt, c_lvl - t_lvl - r)
            c_lvl = t_lvl + r
        
        for i in range(r-1, -1, -1):
            this_target_scl = min(Smax-Sf, t_scl + i*Sf)
            if c_scl < this_target_scl + Sf:
                # need to upscale
                vc = fdag.insert_lsm_op('upscale', vc, vt, this_target_scl + Sf - c_scl)
                c_scl = this_target_scl + Sf
            # now do rescale
            vc = fdag.insert_lsm_op('rescale', vc, vt, None)
            c_lvl -= 1
            c_scl -= Sf

        # final upscale
        if c_scl < t_scl:
            vc = fdag.insert_lsm_op('upscale', vc, vt, t_scl - c_scl)
            c_scl = t_scl
        
        assert c_lvl == t_lvl and c_scl == t_scl, f"After rescaling, levels or scales do not match: current ({c_lvl}, {c_scl}), target ({t_lvl}, {t_scl})"
        return vc

    vc = u
    if in_lvl >= out_lvl and in_lvl * Sf - in_scl >= out_lvl * Sf - out_scl:
        # no bootstrap needed
        vc = _insert_rescale(vc, v, in_lvl, in_scl, out_lvl, out_scl)
    else:
        # need bootstrap
        vc = _insert_rescale(vc, v, in_lvl, in_scl, bts_input_level, bts_input_scale)
        r = max(0, round(np.ceil((bts_output_scale - out_scl)/Sf)))
        bts_target = out_lvl + r
        if not (fdag.params.bts_lb < bts_target <= fdag.params.bts_ub):
            raise ValueError(
                f"Bootstrap target level {bts_target} for edge ({u}, {v}) is out "
                f"of bounds [{fdag.params.bts_lb + 1}, {fdag.params.bts_ub}]."
            )
        vc = fdag.insert_lsm_op('bootstrap', vc, v, bts_target)
        vc = _insert_rescale(vc, v, bts_target, bts_output_scale, out_lvl, out_scl)

def decode_assign(assign: Assign, og_dag: Tdag, og_to_comp: dict[str, str]) -> Tdag:
    fdag = Tdag(og_dag.params, og_dag.name)
    resbts_edges = []
    
    for v in og_dag.nodes:
        v_attr = og_dag.nodes[v]
        if v_attr['op'] == 'constant':
            out_op = 'constant'
            out_op_descr = v_attr['op_descr']
        else:
            out_op = 'dummy'
            out_op_descr = dict()

        if v_attr['op'] == 'constant':
            fdag.add_node(f"{v}_out", op=out_op, weight=v_attr['weight'],
                          level=None, scale=None, 
                          op_descr=out_op_descr,
                          comment=v_attr['comment'])
        else:
            fdag.add_node(f"{v}_in", op=v_attr['op'], weight=v_attr['weight'],
                          level=None, scale=None,
                          op_descr=v_attr['op_descr'],
                          comment=v_attr['comment'])
            fdag.add_node(f"{v}_out", op=out_op, weight=v_attr['weight'],
                          level=None, scale=None,
                          op_descr=out_op_descr,
                          comment=v_attr['comment'])
            fdag.add_edge(f"{v}_in", f"{v}_out", weight=v_attr['weight'])
            resbts_edges.append((f"{v}_in", f"{v}_out", None))
    
    fdag.inputs = {f"{v}_in" for v in og_dag.inputs}
    fdag.outputs = {f"{v}_out" for v in og_dag.outputs}
    
    for v in og_dag.nodes:
        for u in og_dag.predecessors(v):
            fu = f"{u}_out"
            fv = f"{v}_in"
            edge_weight = og_dag.edges[u, v]['weight']
            fdag.add_edge(fu, fv, weight=edge_weight)
            if og_dag.nodes[u]['op'] != 'constant':
                comp_u = og_to_comp[u]
                comp_v = og_to_comp[v]
                resbts_edges.append((fu, fv, assign.e_scl_out[(comp_u, comp_v)]))
    
    for v in fdag.nodes:
        if v.endswith('_out'):
            comp_v = og_to_comp[v[:-4]]
            fdag.nodes[v]['level'] = assign.v_lvl_out[comp_v]
            fdag.nodes[v]['scale'] = assign.v_scl_out[comp_v]
        else:
            assert v.endswith('_in'), f"Unexpected node name {v}"
            comp_v = og_to_comp[v[:-3]]
            
            v_op = fdag.nodes[v]['op']
            # handle 'constant'
            assert v_op != 'constant', "Constant nodes should not have _in suffix"
            
            v_il, v_is = assign.get_v_in_lvl_scl(comp_v)
            assert v_il is not None and v_is is not None, f"Node {v} missing input level or scale"
            fdag.nodes[v]['level'] = v_il
            fdag.nodes[v]['scale'] = v_is
    
    for u, v, scl in resbts_edges:
        _decode_edge(fdag, u, v, scl)
    
    fdag.squash_ops(['dummy'])
    addition_desquash(fdag)
    # visualize(fdag, f"{fdag.name}_decoded_assign")
    
    assert check_tdag(fdag)
    return fdag
