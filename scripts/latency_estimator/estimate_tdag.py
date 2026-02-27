from .latency_estimator import LatencyEstimator
from ..params.params import Params
from ..tdag.tdag import Tdag
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import numpy as np

def get_tdag_vertex_lat(tdag: Tdag, v: str, lvl: int, le: LatencyEstimator) -> float:
    op = tdag.nodes[v]['op']
    single_cnt, double_cnt = tdag.get_v_weights(v)
    lats = 0
    if single_cnt > 0 and f"{op}_single" in le.op_lmaps:
        lats += le.op_lmaps[f"{op}_single"][lvl] * single_cnt
    if double_cnt > 0 and f"{op}_double" in le.op_lmaps:
        lats += le.op_lmaps[f"{op}_double"][lvl] * double_cnt
    return lats

def estimate_tdag_latency_breakdown(tdag: Tdag, le: LatencyEstimator) -> tuple[dict[str, dict[int, int]], dict[str, dict[int, float]]]:
    op_stats = defaultdict(lambda : defaultdict(int))
    op_costs = defaultdict(lambda : defaultdict(float))
    
    for v in tdag.nodes:
        op = tdag.nodes[v]['op']
        single_cnt, double_cnt = tdag.get_v_weights(v)

        if single_cnt > 0:
            op_stats[f"{op}_single"][tdag.nodes[v]['level']] += single_cnt
        if double_cnt > 0:
            op_stats[f"{op}_double"][tdag.nodes[v]['level']] += double_cnt

    for key, stats in op_stats.items():
        if key not in le.op_lmaps:
            continue
        for level, weight in stats.items():
            op_costs[key][level] = le.op_lmaps[key][level] * weight
    
    op_stats_normal_dict = dict()
    op_costs_normal_dict = dict()
    for key, stats in op_stats.items():
        op_stats_normal_dict[key] = dict(stats)
    for key, stats in op_costs.items():
        op_costs_normal_dict[key] = dict(stats)
    
    return op_stats_normal_dict, op_costs_normal_dict

def estimate_tdag_latency(tdag: Tdag, le: LatencyEstimator) -> float:
    _, op_costs = estimate_tdag_latency_breakdown(tdag, le)
    total_cost = 0.0
    for stats in op_costs.values():
        total_cost += sum(stats.values())
    return total_cost

def estimate_tdag_latency_breakdown_to_file(tdag: Tdag, le: LatencyEstimator, filename: str):
    op_stats, op_costs = estimate_tdag_latency_breakdown(tdag, le)
    
    os.makedirs("log/stats/stats_txt", exist_ok=True)
    with open(f"log/stats/stats_txt/{filename}.txt", "w") as f:
        f.write("Operation Nums:\n")
        for key, stats in sorted(op_stats.items()):
            total_num = sum(stats.values())
            f.write(f"{key.ljust(20)}: total={total_num} : ")
            for level, num in sorted(stats.items()):
                f.write(f"{level}:{num},")
            f.write("\n")
        
        f.write("\n")
        f.write("Operation Costs:\n")
        
        total_total_cost = 0
        nomsus_total_cost = 0
        
        total_cost_dict = dict()
        for key, stats in sorted(op_costs.items()):
            total_cost = sum(stats.values())
            total_total_cost += total_cost
            if (not key.startswith("modswitch")) and (not key.startswith("upscale")):
                nomsus_total_cost += total_cost
            f.write(f"{key.ljust(20)}: total={total_cost} : ")
            total_cost_dict[key] = total_cost
            for level, num in sorted(stats.items()):
                f.write(f"{level}:{num},")
            f.write("\n")
        
        f.write("\n")
        f.write(f"Total Cost: {total_total_cost/1000000}\n")
        f.write(f"Total Cost (excluding modswitch and upscale): {nomsus_total_cost/1000000}\n")
        f.write("\n")
        
        f.write("Total Cost Portion:\n")
        for key, cost in sorted(total_cost_dict.items()):
            f.write(f"{key.ljust(20)}: {cost/total_total_cost:.4%}  {cost/1000000} sec\n")
        f.write("\n")

        print(f"Total Cost: {total_total_cost/1000000} sec")
        print(f"No Modswitch/Upscale Cost: {nomsus_total_cost/1000000} sec")
        print()
        print("Total Cost Portion:")
        for key, cost in sorted(total_cost_dict.items()):
            print(f"{key.ljust(20)}: {cost/total_total_cost:.4%}  {cost/1000000} sec")
        print()
