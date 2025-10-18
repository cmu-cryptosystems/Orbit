from ...tdag import *
from ...params.params import Params
from ...latency_estimator import *
from ...assignment import *
from .rescale_assign import getRescaleRegions
from .region_resbts import tryRegionResBts, assignRegionResBts
from collections import defaultdict

def resbm_core(tdag: Tdag, regions: dict[int, Tdag], node_to_region: dict[str, int], edge_to_region: dict[tuple[str, str], int], le: LatencyEstimator):
    
    region_ids = sorted(regions.keys())
    params = regions[region_ids[0]].params
    
    region_to_id = {r: i for i, r in enumerate(region_ids)}
    edge_to_region_ids = dict()
    for (u, v), r in edge_to_region.items():
        edge_to_region_ids[(u, v)] = region_to_id[r]
    
    # Initialize minLat for each region
    minLat = {i: float('inf') for i in range(len(region_ids)+1)}
    minLat[0] = 0

    minLatRegionIO = defaultdict(dict) # region_id -> {prev region_id -> (in_lvl, in_scl, out_lvl, out_scl, is_rescale, is_bootstrap)}
    minLatRegionIO[0] = dict() # initial input level/scale
    
    # use region_id = len(region_ids) as aggregation of final result
    for src in range(len(region_ids)):
        for dst in range(src+1, len(region_ids)+1):
            last_out_scl = params.Sw if src == 0 else minLatRegionIO[src][src-1][3]
            rescale_regs = getRescaleRegions(regions, region_ids, src, dst, last_out_scl, params)
            l_bts = len(rescale_regs)-1 if src in rescale_regs else len(rescale_regs)
            l_bts += params.bts_lb
            if l_bts > params.bts_ub:
                break
            last_out_lvl = l_bts if src == 0 else minLatRegionIO[src][src-1][2]
            
            this_lat = 0
            this_regionIO = minLatRegionIO[src].copy()
            
            # special handling for the first region
            # src = 0
            # only input vertex, should not have mult -> not consuming level
            for reg_id in range(src, dst):
                reg = regions[region_ids[reg_id]]
                is_rescale = reg_id in rescale_regs
                is_bootstrap = (reg_id == src and reg_id > 0)
                reg_lat, out_lvl, out_scl = tryRegionResBts(reg, last_out_lvl, last_out_scl, l_bts,                 # region + input level/scale
                                                               tdag, edge_to_region_ids, this_regionIO,     # previous region info (for cross-multiple-region edges)
                                                               le,                                                  # latency estimator
                                                               is_rescale, is_bootstrap)                            # whether this region is for rescaling/bootstrapping
                if reg_lat == float('inf'):
                    this_lat = float('inf')
                    break
                this_lat += reg_lat
                this_regionIO[reg_id] = (last_out_lvl, last_out_scl, out_lvl, out_scl, l_bts, is_rescale, is_bootstrap)
                last_out_lvl, last_out_scl = out_lvl, out_scl
            
            new_lat = minLat[src] + this_lat
            if new_lat < minLat[dst]:
                minLat[dst] = new_lat
                minLatRegionIO[dst] = this_regionIO

    print(f"Final minimum latency: {minLat[len(region_ids)]}")
    # print("Final region input/output assignments:")
    # for k, v in minLatRegionIO[len(region_ids)].items():
    #     is_x2 = 0
    #     reg = regions[region_ids[k]]
    #     for inp in reg.inputs:
    #         if reg.nodes[inp]['op'] == 'mul':
    #             is_x2 = max(is_x2, 1)
    #             if reg.nodes[inp]['op_descr']['double']:
    #                 is_x2 = 2
    #                 break
    #     print(f"Region {k}: in_lvl={v[0]}, in_scl={v[1]}, out_lvl={v[2]}, out_scl={v[3]}, is_rescale={v[5]}, is_bootstrap={v[6]}, is_x2_reg={is_x2}")
    return get_final_assign(tdag, regions, edge_to_region_ids, region_ids, minLatRegionIO[len(region_ids)], le)


def get_final_assign(tdag: Tdag, regions: dict[int, Tdag], edge_to_region_ids, region_ids, regionIO, le) -> Assign:
    fassign = Assign(tdag)
    for inp in tdag.inputs:
        fassign.v_lvl_in[inp] = regionIO[0][0]
        fassign.v_scl_in[inp] = regionIO[0][1]
    
    for reg_id in region_ids:
        reg = regions[reg_id]
        # handling all real nodes in this region, and their input edges
        vl, vs, el, es = assignRegionResBts(reg, tdag, edge_to_region_ids, regionIO, regionIO[reg_id], le)
        for v, val in vl.items():
            if v not in reg.nodes or reg.nodes[v]['op'] == 'output':
                continue
            if v in fassign.v_lvl_out:
                assert fassign.v_lvl_out[v] == val, f"Node {v} has inconsistent output level assignment {fassign.v_lvl_out[v]} vs {val}."
            else:
                fassign.v_lvl_out[v] = val
        for v, val in vs.items():
            if v not in reg.nodes or reg.nodes[v]['op'] == 'output':
                continue
            if v in fassign.v_scl_out:
                assert fassign.v_scl_out[v] == val, f"Node {v} has inconsistent output scale assignment {fassign.v_scl_out[v]} vs {val}."
            else:
                fassign.v_scl_out[v] = val
        for e, val in el.items():
            if reg.nodes[e[1]]['op'] == 'output':
                continue
            if e in fassign.e_lvl_out:
                assert fassign.e_lvl_out[e] == val, f"Edge {e} has inconsistent output level assignment {fassign.e_lvl_out[e]} vs {val}."
            else:
                fassign.e_lvl_out[e] = val
        for e, val in es.items():
            if reg.nodes[e[1]]['op'] == 'output':
                continue
            if e in fassign.e_scl_out:
                assert fassign.e_scl_out[e] == val, f"Edge {e} has inconsistent output scale assignment {fassign.e_scl_out[e]} vs {val}."
            else:
                fassign.e_scl_out[e] = val
    return fassign