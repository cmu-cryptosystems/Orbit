from ...tdag import *
from ...params.params import Params


def _get_region_addscl(reg: Tdag, params: Params, inp_scl: int) -> int:
    add_scl = 0
    for inp in reg.inputs:
        if reg.nodes[inp]['op'] != 'mul':
            continue
        # assume that reg.nodes[inp]['op_descr'] = {'single':1, 'double':0} or {'single':0, 'double':1}
        assert 'single' in reg.nodes[inp]['op_descr'], f"Input node {inp} in region {reg.name} does not have 'single' in op_descr, got {reg.nodes[inp]['op_descr']}"
        if reg.nodes[inp]['op_descr']['single']:
            add_scl = max(add_scl, params.Sw)
        else:
            add_scl = max(add_scl, inp_scl)
    return add_scl

def _get_regions_outscl(regions, region_ids, src, dst, in_scl, rescale_regs, params:Params):
    last_scl = in_scl
    for reg_id in range(src, dst+1):
        if reg_id >= len(region_ids):
            continue
        reg = regions[region_ids[reg_id]]
        out_scl = last_scl + _get_region_addscl(reg, params, last_scl)
        if reg_id in rescale_regs:
            out_scl = max(params.Sw, out_scl - params.Sf)
        if reg_id == src:
            assert out_scl <= params.Sf, f"Bootstrap region (src) {reg.name} has output scale exactly before bootstrap {out_scl} > Sf={params.Sf}"
            out_scl = params.Sf
        last_scl = out_scl
    return last_scl

def getRescaleRegions(regions, region_ids, src, dst, in_scl, params:Params):
    
    rescale_regs = set()
    last_scl = in_scl
    
    for reg_id in range(src, dst+1):
        if reg_id >= len(region_ids):
            continue
        reg = regions[region_ids[reg_id]]
        assert isinstance(reg, Tdag)
        
        out_scl = last_scl + _get_region_addscl(reg, params, last_scl)
        if out_scl >= params.Sf + params.Sw:
            rescale_regs.add(reg_id)
            out_scl -= params.Sf
        if reg_id == src and reg_id > 0:
            assert out_scl <= 2 * params.Sf, f"Bootstrap region (src) {reg.name} has output scale {out_scl} > 2*Sf={2*params.Sf}"
            if out_scl > params.Sf:
                rescale_regs.add(reg_id)
            out_scl = params.Sf
        if reg_id == dst and out_scl > params.Sf and reg_id not in rescale_regs:
            rescale_regs.add(reg_id)
            out_scl = max(params.Sw, out_scl - params.Sf)
        last_scl = out_scl
    
    while last_scl > params.Sf:
        for reg_id in range(dst, src-1, -1):
            if reg_id not in rescale_regs:
                rescale_regs.add(reg_id)
                break
        last_scl = _get_regions_outscl(regions, region_ids, src, dst, in_scl, rescale_regs, params)
    
    # if dst == len(regions):
    #     print(f"Rescale regions between region {src} and {dst}: {rescale_regs}")
    #     print(f"Final output scale after rescaling: {last_scl}")
    return rescale_regs

def getRescaleRegions_ReSBMold(regions, region_ids, src, dst, in_scl, params:Params):
    rescale_regs = set()
    cur_reg = src
    
    if src > 0:
        src_scl_out = in_scl + _get_region_addscl(regions[region_ids[src]], params, in_scl)
        assert src_scl_out <= 2 * params.Sf, f"Bootstrap region (src) {regions[region_ids[src]].name} has output scale {src_scl_out} > 2*Sf={2*params.Sf}, in_scl = {in_scl}"
        if src_scl_out > params.Sf:
            rescale_regs.add(src)
            cur_reg = src + 1
    
    while cur_reg <= dst:
        best_scl = None
        smo_reg = None
        this_rescale_regs = rescale_regs.copy()
        for can_reg in range(cur_reg, dst+1):
            this_rescale_regs.add(can_reg)
            out_scl = _get_regions_outscl(regions, region_ids, src, dst, in_scl, this_rescale_regs, params)
            this_rescale_regs.remove(can_reg)
            if best_scl is not None and out_scl >= best_scl:
                break
            best_scl = out_scl
            smo_reg = can_reg
        rescale_regs.add(smo_reg)
        cur_reg = smo_reg + 1
        if best_scl <= params.Sf:
            break
    last_scl = _get_regions_outscl(regions, region_ids, src, dst, in_scl, rescale_regs, params)
    while last_scl > params.Sf:
        for reg_id in range(dst, src-1, -1):
            if reg_id not in rescale_regs:
                rescale_regs.add(reg_id)
                break
        last_scl = _get_regions_outscl(regions, region_ids, src, dst, in_scl, rescale_regs, params)
    return rescale_regs
        
