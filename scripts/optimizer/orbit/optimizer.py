from ...tdag import *
from ...assignment import *
from ...latency_estimator import *
from ...visualize import *
from ...params.params import Params
from ...utils.fix_random import set_global_seed

from .orbit_core import orbit_core

import argparse
import copy
import time
import os


def _optimize_and_emit_mlir(og_dag, output_file: str, params: Params, le: LatencyEstimator, timestamps: dict) -> None:
    """Shared Orbit path: compress, place, decode, and write MLIR."""
    print(f"Placement backend: {params.placement_backend}")
    if params.placement_backend == "ilp":
        print(f"ILP solver: {params.ilp_solver}")
    if params.resilience_profile is not None:
        print(f"Loaded resilience profile: {params.resilience_profile.describe()}")
        print(f"Resilience mode: {params.resilience_mode}")
        print(f"Resilience constraint policy: {params.resilience_constraint_policy}")
        print(f"Resilience decomposition: {params.resilience_decomposition}")
    if params.comp:
        start_time = time.time()
        addition_squash(og_dag)
        if params.bpsdepth is not None:
            addition_desquash_bypass(og_dag)
        print(f"After Squashing, DAG has {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
        comp_dag, og_to_comp = auto_compress(og_dag, is_ignore_weight=True)
        print(f"After Compression, DAG has {len(comp_dag.nodes)} nodes and {len(comp_dag.edges)} edges.")
        timestamps['DAG Compression Time'] = time.time() - start_time
    else:
        comp_dag = og_dag.copy_tdag()
        og_to_comp = {node: node for node in og_dag.nodes}

    if params.resilience_profile is not None:
        report = params.resilience_profile.match_report(comp_dag)
        params.resilience_match_report = report
        print(params.resilience_profile.format_match_report(report))
        if report["unmatched_targets"]:
            preview = ", ".join(str(target) for target in report["unmatched_targets"][:5])
            suffix = "..." if len(report["unmatched_targets"]) > 5 else ""
            print(f"Unmatched resilience targets: {preview}{suffix}")
        if report["matched_nodes"] == 0 and not params.allow_empty_resilience_match:
            raise RuntimeError(
                "Resilience profile matched zero compressed TDAG nodes. "
                "Use --allow-empty-resilience-match only to debug old profiles."
            )

    assign, ilp_times = orbit_core(comp_dag, le, params)
    for k, v in ilp_times.items():
        timestamps[k] = v
    if assign is None:
        raise RuntimeError("No placement solution found for the whole DAG")

    assign_lat = estimate_assign(assign, le)
    print(f"Final assignment latency: {assign_lat/1000000:.3f} sec.")

    start_time = time.time()
    fdag = decode_assign(assign, og_dag, og_to_comp)
    timestamps['DAG Decode Time'] = time.time() - start_time

    final_lat = estimate_tdag_latency(fdag, le)
    print(f"Final tdag latency: {final_lat/1000000:.3f} sec. (accurate estimation)")
    final_lat_bd_num, final_lat_bd_costs = estimate_tdag_latency_breakdown(fdag, le)
    print("Final tdag latency breakdown (number of operations):")
    for op, stats in sorted(final_lat_bd_num.items()):
        total_op_num = sum(stats.values())
        print(f"  {op}: {total_op_num} ops.")
    print("=" * 40)
    print("Final tdag latency breakdown (in sec):")
    for op, stats in sorted(final_lat_bd_costs.items()):
        total_op_cost = sum(stats.values())
        print(f"  {op}: {total_op_cost/1000000:.3f} sec.")
    print("=" * 40)

    start_time = time.time()
    tdag_to_mlir(fdag, output_file)
    timestamps['DAG Write Time'] = time.time() - start_time

    print(f"Orbit Compilation time: {sum(timestamps.values()):.3f} sec.")
    print("Timestamps breakdown:")
    for k, v in timestamps.items():
        print(f"  '{k}': {v:.3f} sec.")
    print("=" * 40)


def _optimize_and_emit_mlir_with_bypass_retry(
    og_dag,
    output_file: str,
    params: Params,
    le: LatencyEstimator,
    timestamps: dict,
) -> None:
    """Run Orbit and fail open to the no-bypass path for known QBP registration gaps."""

    base_dag = og_dag.copy_tdag()
    try:
        _optimize_and_emit_mlir(
            base_dag.copy_tdag(),
            output_file,
            params,
            le,
            dict(timestamps),
        )
        return
    except (AssertionError, RuntimeError) as exc:
        retryable = (
            "QBP not found" in str(exc)
            or "No placement solution found for the whole DAG" in str(exc)
        )
        if params.bpsdepth is None or not retryable:
            raise
        print(
            "Bypass placement failed; retrying compile with bypass disabled. "
            f"Original error: {exc}"
        )

    retry_params = copy.copy(params)
    retry_params.bpsdepth = None
    retry_timestamps = dict(timestamps)
    retry_timestamps["Bypass Retry Triggered"] = 0.0
    _optimize_and_emit_mlir(
        base_dag.copy_tdag(),
        output_file,
        retry_params,
        le,
        retry_timestamps,
    )


def run(input_file: str, output_file: str, params: Params):
    le = LatencyEstimator(params)

    timestamps = dict()

    start_time = time.time()
    og_dag = build_from_mlir(input_file, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    timestamps['DAG Load Time'] = time.time() - start_time

    _optimize_and_emit_mlir_with_bypass_retry(og_dag, output_file, params, le, timestamps)


def run_from_rotom(manifest_path: str, output_file: str, params: Params) -> None:
    """Same as :func:`run` but builds the initial Tdag from a Rotom circuit manifest (``build_from_rotom``)."""
    le = LatencyEstimator(params)

    timestamps = dict()

    start_time = time.time()
    og_dag = build_from_rotom(manifest_path, params)
    og_dag.squash_ops(['rescale', 'upscale', 'modswitch', 'bootstrap'])
    print(f"Built original DAG from Rotom with {len(og_dag.nodes)} nodes and {len(og_dag.edges)} edges.")
    timestamps['DAG Load Time'] = time.time() - start_time

    _optimize_and_emit_mlir_with_bypass_retry(og_dag, output_file, params, le, timestamps)

def main():
    set_global_seed(42)
    parser = argparse.ArgumentParser(description='Orbit Optimizer')
    parser.add_argument('--inputfile', type=str, required=True, help='Input MLIR file')
    parser.add_argument('--outputfile', type=str, required=True, help='Output MLIR file')
    parser.add_argument('--input-format', type=str, default='auto',
                        choices=['auto', 'mlir', 'rotom'],
                        help='Input format (default: auto-detect from file extension)')
    parser.add_argument('--costjson', type=str, required=True, help='Cost model JSON file')
    parser.add_argument('--maxlevel', type=int, default=None, help='Maximum level budget (overrides costjson if specified)')
    parser.add_argument('--btsupperbound', type=int, default=None, help='Maximum Bootstrapping target level (overrides costjson if specified)')
    parser.add_argument('--btslevel', type=int, default=None, help='Bootstrapping level (overrides costjson if specified)')
    parser.add_argument('-Sw', '--waterscale', type=int, default=40, help='Waterline scale')
    parser.add_argument('-Csw', '--constantscale', type=int, default=None, help='Constant scale (overrides waterline scale if specified)')
    parser.add_argument('-Sf', '--rescale', type=int, default=None, help='Rescaling factor (overrides costjson if specified)')
    parser.add_argument('--nobypass', action='store_true', help='Disable Bypass handling')
    parser.add_argument('--no-compress', action='store_true', help='Disable DAG compression')
    parser.add_argument('--no-partition', action='store_true', help='Disable SISO partitioning')
    parser.add_argument('--enable-reqbp', action='store_true', help='Enable QBP cross-bench reusing')
    parser.add_argument('--bypass-dep', type=int, default=15, help='Bypass dependency level (default: 15)')
    parser.add_argument('--threads', type=int, default=16, help='Number of threads (default: 16)')
    parser.add_argument('--placement-backend', type=str, default='openevolve', choices=['openevolve', 'ilp'],
                        help='Placement backend: OpenEvolve-guided search (default) or legacy ILP')
    parser.add_argument('--ilp-solver', type=str, default=None, choices=['gurobi', 'pulp'],
                        help='MILP backend used only with --placement-backend ilp')
    parser.add_argument('--openevolve-config', type=str, default=None,
                        help='Optional OpenEvolve YAML config for runtime evolution')
    parser.add_argument('--openevolve-output-dir', type=str, default=None,
                        help='Directory for generated OpenEvolve workspaces/checkpoints')
    parser.add_argument('--openevolve-iterations', type=int, default=0,
                        help='OpenEvolve iterations; 0 uses deterministic conservative placement')
    parser.add_argument('--openevolve-seed', type=int, default=42)
    parser.add_argument('--openevolve-keep-workdir', action='store_true')
    parser.add_argument('--openevolve-provider', type=str, default='gemini',
                        choices=['gemini', 'openai', 'custom'],
                        help='OpenEvolve LLM provider used when --openevolve-config is not supplied')
    parser.add_argument('--openevolve-model', type=str, default='gemini-3.1-flash-lite',
                        help='OpenEvolve LLM model used for generated config')
    parser.add_argument('--openevolve-api-base', type=str, default=None,
                        help='OpenAI-compatible API base for OpenEvolve generated config')
    parser.add_argument('--openevolve-api-key-env', type=str, default='OPENAI_API_KEY',
                        help='Environment variable containing the OpenEvolve API key')
    parser.add_argument('--openevolve-primary-weight', type=float, default=1.0,
                        help='Sampling weight for the primary OpenEvolve model')
    parser.add_argument('--openevolve-secondary-provider', type=str, default='none',
                        choices=['none', 'gemini', 'openai', 'custom'],
                        help='Optional secondary OpenEvolve provider for multi-model ensembles')
    parser.add_argument('--openevolve-secondary-model', type=str, default='gpt-5.5',
                        help='Secondary model used when --openevolve-secondary-provider is set')
    parser.add_argument('--openevolve-secondary-api-base', type=str, default=None,
                        help='OpenAI-compatible API base for the secondary OpenEvolve model')
    parser.add_argument('--openevolve-secondary-api-key-env', type=str, default='OPENAI_API_KEY',
                        help='Environment variable containing the secondary provider API key')
    parser.add_argument('--openevolve-secondary-weight', type=float, default=0.25,
                        help='Sampling weight for the secondary OpenEvolve model')
    parser.add_argument('--openevolve-harness', type=str, default='compile',
                        choices=['compile', 'partition'],
                        help='OpenEvolve harness scope for positive iterations')
    parser.add_argument('--openevolve-search-mode', type=str, default='bootstrap-mcts',
                        choices=['legacy', 'beam', 'bootstrap-mcts'],
                        help='OpenEvolve placement search mode for positive iterations')
    parser.add_argument('--openevolve-granularity', type=str, default='layer-nonlinear',
                        choices=['compile', 'layer-nonlinear'],
                        help='OpenEvolve placement context granularity for positive iterations')
    parser.add_argument('--openevolve-leniency', type=str, default='repair',
                        choices=['repair', 'strict'],
                        help='OpenEvolve candidate handling: repair invalid hints or reject strictly')
    parser.add_argument('--openevolve-max-unit-samples', type=int, default=64,
                        help='Maximum layer/nonlinear units to expose in sampled OpenEvolve contexts')
    parser.add_argument('--openevolve-eval-suite', type=str, default='polybert-sampled',
                        choices=['toy', 'polybert-sampled', 'polybert-full'],
                        help='Evaluation bundle used inside the OpenEvolve harness')
    parser.add_argument('--openevolve-reference-json', type=str, default=None,
                        help='Optional precomputed reference metrics; never triggers an ILP solve')
    parser.add_argument('--openevolve-finalists', type=int, default=3,
                        help='Number of top candidates to rerun on full OpenEvolve bundle')
    parser.add_argument(
        '--openevolve-budget-aggressive',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Bias OpenEvolve scoring toward candidates that directly solve more budget records',
    )
    parser.add_argument(
        '--openevolve-target-bootstrap-count',
        type=int,
        default=0,
        help='Soft target for final bootstrap count; 0 disables the target',
    )
    parser.add_argument('--openevolve-llm-timeout-sec', type=int, default=180)
    parser.add_argument('--openevolve-llm-retries', type=int, default=1)
    parser.add_argument('--openevolve-llm-retry-delay-sec', type=int, default=2)
    parser.add_argument('--openevolve-evaluator-timeout-sec', type=int, default=180)
    parser.add_argument('--openevolve-parallel-evaluations', type=int, default=1)
    parser.add_argument('--openevolve-checkpoint-interval', type=int, default=5)
    parser.add_argument(
        '--openevolve-fail-open',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Return the best recovered/initial candidate if OpenEvolve runtime fails',
    )
    parser.add_argument(
        '--openevolve-reuse-output',
        action='store_true',
        help='Reuse an existing OpenEvolve output directory and skip the LLM call',
    )
    parser.add_argument(
        '--noise-estimator',
        choices=['off', 'finalists'],
        default='finalists',
        help='Estimator-backed OpenEvolve finalist gate; finalists is used for positive OpenEvolve runs',
    )
    parser.add_argument(
        '--noise-estimator-binary',
        type=str,
        default=None,
        help='Optional JSON sidecar binary for CKKS noise estimation',
    )
    parser.add_argument('--noise-estimator-timeout-sec', type=int, default=30)
    parser.add_argument('--noise-estimator-min-output-margin-bits', type=float, default=2.0)
    parser.add_argument(
        '--noise-estimator-alpha',
        type=float,
        default=14.0,
        help='Gaussian tail multiplier for the CKKS noise bound',
    )
    parser.add_argument(
        '--noise-estimator-max-trace-message-bits',
        type=float,
        default=20.0,
        help='Trace message-bit warning threshold; output/bootstrap gates remain hard checks',
    )
    parser.add_argument(
        '--noise-estimator-require-trace-safe',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Treat whole-trace precision/message warnings as hard finalist rejections',
    )
    parser.add_argument(
        '--scale-floor-policy',
        choices=['waterline', 'estimator-relaxed'],
        default='waterline',
        help='Scale lower-bound policy for OpenEvolve candidate evaluation',
    )
    parser.add_argument(
        '--scale-floor-min-bits',
        type=int,
        default=None,
        help='Minimum relaxed scale floor; defaults to max(24, Sw - 12) under estimator-relaxed',
    )
    parser.add_argument(
        '--openevolve-scale-floor-candidates',
        type=str,
        default=None,
        help='Comma-separated OpenEvolve scale floor candidates, e.g. 40,36,32,28',
    )
    parser.add_argument(
        '--resilience-profile',
        type=str,
        default=None,
        help='ckks-robustness-profiler JSON or Orbit resilience constraints JSON',
    )
    parser.add_argument(
        '--resilience-mode',
        choices=['waterline', 'error-state'],
        default='waterline',
        help='How resilience constraints affect placement',
    )
    parser.add_argument(
        '--allow-empty-resilience-match',
        action='store_true',
        help='Allow a loaded resilience profile to match zero TDAG nodes',
    )
    parser.add_argument(
        '--resilience-decomposition',
        choices=['off', 'bounded-dp'],
        default='off',
        help='Optional decomposition strategy for resilience error-state search',
    )
    parser.add_argument('--resilience-decompose-threshold', type=int, default=32)
    parser.add_argument('--resilience-max-boundary-states', type=int, default=8)
    parser.add_argument('--resilience-error-buckets', type=int, default=8)
    parser.add_argument('--ilp-task-time-limit-sec', type=float, default=0.0)
    parser.add_argument(
        '--resilience-constraint-policy',
        choices=['relax-only', 'hard-tau'],
        default='relax-only',
        help='Use relax-only for speedup-guided placement or hard-tau for strict error caps',
    )
    parser.add_argument('--netname', type=str, default="", help='Network name for qbp reusing purposes (default: mlirs_input/<netname>.mlir)')
    
    args = parser.parse_args()
    if args.nobypass:
        bypass_dep = None
    else:
        bypass_dep = args.bypass_dep
        
    if args.netname != "":
        netname = args.netname
    else:
        netname = os.path.splitext(os.path.basename(args.inputfile))[0]
    params = Params(args.costjson, "Orbit", mode="compile", 
                    Sw=args.waterscale, CSw=args.constantscale, bpsdepth=bypass_dep, threads=args.threads, 
                    comp=not args.no_compress, part=not args.no_partition, reqbp=args.enable_reqbp, 
                    netname=netname, ilp_solver=args.ilp_solver,
                    placement_backend=args.placement_backend,
                    resilience_profile=args.resilience_profile,
                    resilience_mode=args.resilience_mode,
                    allow_empty_resilience_match=args.allow_empty_resilience_match,
                    resilience_decomposition=args.resilience_decomposition,
                    resilience_decompose_threshold=args.resilience_decompose_threshold,
                    resilience_max_boundary_states=args.resilience_max_boundary_states,
                    resilience_error_buckets=args.resilience_error_buckets,
                    ilp_task_time_limit_sec=args.ilp_task_time_limit_sec,
                    resilience_constraint_policy=args.resilience_constraint_policy,
                    openevolve_config=args.openevolve_config,
                    openevolve_output_dir=args.openevolve_output_dir,
                    openevolve_iterations=args.openevolve_iterations,
                    openevolve_seed=args.openevolve_seed,
                    openevolve_keep_workdir=args.openevolve_keep_workdir,
                    openevolve_provider=args.openevolve_provider,
                    openevolve_model=args.openevolve_model,
                    openevolve_api_base=args.openevolve_api_base,
                    openevolve_api_key_env=args.openevolve_api_key_env,
                    openevolve_primary_weight=args.openevolve_primary_weight,
                    openevolve_secondary_provider=args.openevolve_secondary_provider,
                    openevolve_secondary_model=args.openevolve_secondary_model,
                    openevolve_secondary_api_base=args.openevolve_secondary_api_base,
                    openevolve_secondary_api_key_env=args.openevolve_secondary_api_key_env,
                    openevolve_secondary_weight=args.openevolve_secondary_weight,
                    openevolve_harness=args.openevolve_harness,
                    openevolve_search_mode=args.openevolve_search_mode,
                    openevolve_granularity=args.openevolve_granularity,
                    openevolve_leniency=args.openevolve_leniency,
                    openevolve_max_unit_samples=args.openevolve_max_unit_samples,
                    openevolve_eval_suite=args.openevolve_eval_suite,
                    openevolve_reference_json=args.openevolve_reference_json,
                    openevolve_finalists=args.openevolve_finalists,
                    openevolve_budget_aggressive=args.openevolve_budget_aggressive,
                    openevolve_target_bootstrap_count=args.openevolve_target_bootstrap_count,
                    openevolve_llm_timeout_sec=args.openevolve_llm_timeout_sec,
                    openevolve_llm_retries=args.openevolve_llm_retries,
                    openevolve_llm_retry_delay_sec=args.openevolve_llm_retry_delay_sec,
                    openevolve_evaluator_timeout_sec=args.openevolve_evaluator_timeout_sec,
                    openevolve_parallel_evaluations=args.openevolve_parallel_evaluations,
                    openevolve_checkpoint_interval=args.openevolve_checkpoint_interval,
                    openevolve_fail_open=args.openevolve_fail_open,
                    openevolve_reuse_output=args.openevolve_reuse_output,
                    noise_estimator=args.noise_estimator,
                    noise_estimator_binary=args.noise_estimator_binary,
                    noise_estimator_timeout_sec=args.noise_estimator_timeout_sec,
                    noise_estimator_min_output_margin_bits=args.noise_estimator_min_output_margin_bits,
                    noise_estimator_alpha=args.noise_estimator_alpha,
                    noise_estimator_max_trace_message_bits=args.noise_estimator_max_trace_message_bits,
                    noise_estimator_require_trace_safe=args.noise_estimator_require_trace_safe,
                    scale_floor_policy=args.scale_floor_policy,
                    scale_floor_min_bits=args.scale_floor_min_bits,
                    openevolve_scale_floor_candidates=args.openevolve_scale_floor_candidates)
    if args.maxlevel is not None:
        params.lvl_ub = args.maxlevel
    if args.btsupperbound is not None:
        params.bts_ub = args.btsupperbound
    if args.btslevel is not None:
        params.bts_lb = args.btslevel
    if args.rescale is not None:
        params.Sf = args.rescale
    
    # make sure output_file directory exists
    output_dir = os.path.dirname(args.outputfile)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    input_format = args.input_format
    if input_format == 'auto':
        input_format = 'rotom' if args.inputfile.endswith('.json') else 'mlir'
    if input_format == 'rotom':
        run_from_rotom(args.inputfile, args.outputfile, params)
    else:
        run(args.inputfile, args.outputfile, params)

if __name__ == "__main__":
    main()
