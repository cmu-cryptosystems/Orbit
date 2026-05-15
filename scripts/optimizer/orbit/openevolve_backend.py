from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import hashlib
import shutil
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from ...assignment import Assign
from ...latency_estimator import LatencyEstimator, estimate_assign
from ...params.params import Params
from ...tdag import Tdag


class PlacementError(Exception):
    pass


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_API_BASE = "https://api.openai.com/v1"
CONTEXT_SCHEMA_VERSION = "orbit-openevolve-placement-context-v2"
COMPILE_CONTEXT_SCHEMA_VERSION = "orbit-openevolve-compile-harness-v1"
BANNED_CANDIDATE_TOKENS = (
    "gurobipy",
    "pulp",
    "PULP_CBC_CMD",
    "subprocess",
    "os.system",
    "__import__",
    "open(",
    "pathlib",
    "shutil",
    ".write_text",
    ".write_bytes",
    ".unlink",
    ".rename",
    "git ",
    "apply_patch",
)


@dataclass
class _BudgetAttempt:
    source: str
    assign: Assign
    cost: float
    in_key: tuple[int, int]
    out_key: tuple[int, int]


class OpenEvolvePlacementWorker:
    """OpenEvolve-backed QBP worker.

    With ``openevolve_iterations == 0`` this uses the intentionally conservative
    waterline seed directly. Positive iteration counts run OpenEvolve once for
    the whole budget batch, then validate the best candidate through the same
    deterministic repair path.
    """

    def __init__(self, params: Params, le: LatencyEstimator):
        self.params = params
        self.le = le
        self.last_diagnostics: dict[str, Any] = {}

    def get_qbp(
        self,
        pdag: Tdag,
        io_budgets_list: list[dict],
    ) -> tuple[
        dict[tuple[int, int], dict[tuple[int, int], Assign]],
        dict[tuple[int, int], dict[tuple[int, int], float]],
    ]:
        hints: dict[str, Any] = {}
        compile_hints = getattr(self.params, "openevolve_compile_hints", None)
        if compile_hints is not None:
            hints = compile_hints
        elif self.params.openevolve_iterations > 0:
            hints = self._run_openevolve(pdag, io_budgets_list)
        diagnostics = {} if getattr(self.params, "openevolve_evaluating_candidate", False) else None
        result = solve_budget_batch(pdag, io_budgets_list, self.le, self.params, hints, diagnostics)
        self.last_diagnostics = diagnostics or {}
        return result

    def _run_openevolve(self, pdag: Tdag, io_budgets_list: list[dict]) -> dict[str, Any]:
        try:
            from openevolve import run_evolution
        except ImportError as exc:
            raise ImportError(
                "openevolve>=0.2.27 is required when --openevolve-iterations is positive."
            ) from exc

        root = self._workspace_root(pdag)
        root.mkdir(parents=True, exist_ok=True)
        context_path = root / "context.json"
        initial_path = root / "initial_program.py"
        evaluator_path = root / "evaluator.py"
        context = build_context(pdag, io_budgets_list, self.params)
        context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        initial_path.write_text(_initial_program_source(), encoding="utf-8")
        evaluator_path.write_text(_evaluator_source(context_path), encoding="utf-8")
        initial_hints = _load_candidate_hints(initial_path, context)

        output_dir = root / "openevolve_output"
        try:
            with self._openevolve_runtime_env():
                result = run_evolution(
                    initial_program=initial_path,
                    evaluator=evaluator_path,
                    config=self._openevolve_config_arg(),
                    iterations=self.params.openevolve_iterations,
                    output_dir=str(output_dir),
                    cleanup=False,
                )
        except Exception as exc:
            if not self.params.openevolve_fail_open:
                raise
            _write_recovery_summary(root, exc)
            return _recover_candidate_hints(output_dir, context, initial_hints)
        best_program = root / "best_program.py"
        best_program.write_text(result.best_code, encoding="utf-8")
        try:
            return _load_candidate_hints(best_program, context)
        except Exception:
            if not self.params.openevolve_fail_open:
                return {}
            return _recover_candidate_hints(output_dir, context, initial_hints, result.best_code)
        finally:
            if not self.params.openevolve_keep_workdir and self.params.openevolve_output_dir is None:
                shutil.rmtree(root, ignore_errors=True)

    def _workspace_root(self, pdag: Tdag) -> Path:
        if self.params.openevolve_output_dir:
            return Path(self.params.openevolve_output_dir).resolve() / f"pdag_{pdag.name}"
        return Path(tempfile.mkdtemp(prefix=f"orbit_openevolve_{pdag.name}_"))

    def _openevolve_config_arg(self):
        try:
            from openevolve.config import Config, LLMModelConfig, load_config
        except ImportError as exc:
            if self.params.openevolve_config is not None:
                return self.params.openevolve_config
            raise ImportError(
                "openevolve>=0.2.27 with config support is required for generated OpenEvolve config."
            ) from exc
        if self.params.openevolve_config is not None:
            config = load_config(self.params.openevolve_config)
            self._normalize_openevolve_seed(config)
            return config
        config = Config()
        api_base = self._openevolve_api_base()
        model = self.params.openevolve_model
        config.random_seed = self.params.openevolve_seed
        config.llm.api_base = api_base
        config.llm.timeout = self.params.openevolve_llm_timeout_sec
        config.llm.retries = self.params.openevolve_llm_retries
        config.llm.retry_delay = self.params.openevolve_llm_retry_delay_sec
        config.llm.max_tokens = max(256, min(4096, int(self.params.openevolve_llm_max_tokens)))
        config.llm.temperature = 0.3
        config.llm.models = [
            LLMModelConfig(
                name=model,
                api_base=api_base,
                timeout=config.llm.timeout,
                retries=config.llm.retries,
                retry_delay=config.llm.retry_delay,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                random_seed=self.params.openevolve_seed,
            )
        ]
        config.llm.evaluator_models = [
            LLMModelConfig(
                name=model,
                api_base=api_base,
                timeout=config.llm.timeout,
                retries=config.llm.retries,
                retry_delay=config.llm.retry_delay,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                random_seed=self.params.openevolve_seed,
            )
        ]
        config.database.feature_dimensions = [
            "final_latency_usec",
            "boundary_quality",
            "bootstrap_count",
            "rescale_count",
            "fallback_selected_budgets",
            "profile_risk",
            "placement_runtime_sec",
        ]
        if hasattr(config, "checkpoint_interval"):
            config.checkpoint_interval = max(1, int(self.params.openevolve_checkpoint_interval))
        if hasattr(config, "evaluator"):
            config.evaluator.timeout = self.params.openevolve_evaluator_timeout_sec
            config.evaluator.parallel_evaluations = self.params.openevolve_parallel_evaluations
            config.evaluator.max_retries = 0
        if hasattr(config, "prompt"):
            config.prompt.max_artifact_bytes = min(
                int(getattr(config.prompt, "max_artifact_bytes", 20 * 1024)),
                12 * 1024,
            )
        if hasattr(config, "database"):
            config.database.log_prompts = True
            config.database.num_islands = min(int(getattr(config.database, "num_islands", 5)), 3)
        config.llm.update_model_params(
            {
                "api_base": api_base,
                "timeout": config.llm.timeout,
                "retries": config.llm.retries,
                "retry_delay": config.llm.retry_delay,
                "max_tokens": config.llm.max_tokens,
                "temperature": config.llm.temperature,
                "random_seed": self.params.openevolve_seed,
            },
            overwrite=True,
        )
        self._normalize_openevolve_seed(config)
        return config

    def _openevolve_api_base(self) -> str:
        if self.params.openevolve_api_base:
            return self.params.openevolve_api_base
        if self.params.openevolve_provider == "gemini":
            return GEMINI_API_BASE
        if self.params.openevolve_provider == "openai":
            return OPENAI_API_BASE
        raise ValueError(
            "--openevolve-api-base is required when --openevolve-provider custom is used."
        )

    def _normalize_openevolve_seed(self, config) -> None:
        config.random_seed = self.params.openevolve_seed
        if hasattr(config, "database"):
            config.database.random_seed = self.params.openevolve_seed
        if hasattr(config, "llm"):
            config.llm.update_model_params({"random_seed": self.params.openevolve_seed})

    def _openevolve_key_env_candidates(self) -> list[str]:
        candidates = [
            self.params.openevolve_api_key_env or "OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ]
        if self.params.openevolve_provider == "gemini":
            candidates.append("GEMINI_API_KEY")
        deduped = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    @contextmanager
    def _openevolve_runtime_env(self):
        candidates = self._openevolve_key_env_candidates()
        found_name = next((name for name in candidates if os.environ.get(name)), None)
        if found_name is None:
            checked = ", ".join(candidates)
            raise ValueError(
                "OpenEvolve runtime search needs an API key in one of these "
                f"environment variables: {checked}. "
                "Use --openevolve-iterations 0 for deterministic no-LLM placement."
            )
        old_openai_key = os.environ.get("OPENAI_API_KEY")
        bridged = found_name != "OPENAI_API_KEY"
        if bridged:
            os.environ["OPENAI_API_KEY"] = os.environ[found_name]
        try:
            yield
        finally:
            if bridged:
                if old_openai_key is None:
                    os.environ.pop("OPENAI_API_KEY", None)
                else:
                    os.environ["OPENAI_API_KEY"] = old_openai_key


def _initial_program_source() -> str:
    return '''"""Initial OpenEvolve placement policy for Orbit."""

# EVOLVE-BLOCK-START
def place(context):
    """Return optional placement preferences and scheduler knobs.

    Orbit owns deterministic repair and validation. OpenEvolve should mutate
    compact knobs or sparse hints, not emit full Orbit source changes.
    """
    return {
        # Positive-iteration runs start from the low-latency policy. Orbit
        # validates each budget record and fills unsupported records with the
        # deterministic smoke seed.
        "strategy": "level_preserving",
        "prefer_level_preservation": True,
        "allow_bootstrap": False,
        "allow_seed_fallback": True,
        # Effective with strategy="level_preserving".
        "refresh_fanout_at_level_floor": False,
        "max_scale_candidates": 32,
        "bootstrap_penalty": 1000000000.0,
        "rescale_penalty": 0.0,
        "level_drop_penalty": 20000000.0,
        "min_internal_level": None,
        "scale_penalty": 0.0,
        "beam_width": 4,
        "state_cap_per_node": 8,
        "scale_lattice": "default",
        "preferred_node_levels": {},
        "preferred_node_scales": {},
        "preferred_edge_scales": {},
    }
# EVOLVE-BLOCK-END
'''


def _initial_compile_program_source() -> str:
    return '''"""Initial compile-level OpenEvolve placement algorithm for Orbit."""

from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder


# EVOLVE-BLOCK-START
def place(context):
    """Return an Orbit placement algorithm description.

    The candidate may either return explicit placement records or a policy
    built from Orbit's helper API. Orbit validates every assignment and owns
    final repair into Assign objects.
    """
    builder = PlacementBuilder(context)
    return builder.low_scale_frontier(
        allow_seed_fallback=False,
        max_scale_candidates=10,
        bootstrap_penalty=1000000000.0,
        rescale_penalty=0.0,
        level_drop_penalty=20000000.0,
        scale_penalty=0.0,
    )
# EVOLVE-BLOCK-END
'''


def _evaluator_source(context_path: Path) -> str:
    return f'''from openevolve.evaluation_result import EvaluationResult
from scripts.optimizer.orbit.openevolve_backend import evaluate_candidate_program


def evaluate(program_path):
    result = evaluate_candidate_program({str(context_path)!r}, program_path)
    return EvaluationResult(metrics=result["metrics"], artifacts=result["artifacts"])
'''


def _compile_evaluator_source(context_path: Path) -> str:
    return f'''from openevolve.evaluation_result import EvaluationResult
from scripts.optimizer.orbit.openevolve_backend import evaluate_compile_candidate_program


def evaluate(program_path):
    result = evaluate_compile_candidate_program({str(context_path)!r}, program_path)
    return EvaluationResult(metrics=result["metrics"], artifacts=result["artifacts"])
'''


def build_context(pdag: Tdag, io_budgets_list: list[dict], params: Params) -> dict[str, Any]:
    graph_summary = _graph_summary(pdag, params)
    return {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "tdag": {
            "name": pdag.name,
            "inputs": sorted(pdag.inputs),
            "outputs": sorted(pdag.outputs),
            "topological_order": [str(node) for node in nx.topological_sort(pdag)],
            "nodes": {
                str(node): _jsonable_attrs(dict(attrs))
                for node, attrs in pdag.nodes(data=True)
            },
            "edges": [
                {
                    "u": str(u),
                    "v": str(v),
                    "attrs": _jsonable_attrs(dict(attrs)),
                }
                for u, v, attrs in pdag.edges(data=True)
            ],
        },
        "model": {
            "name": params.netname or pdag.name,
            "architecture": params.netname or pdag.name,
        },
        "graph_summary": graph_summary,
        "io_budgets": [_jsonable_io_budget(item) for item in io_budgets_list],
        "budget_summary": _budget_summary(io_budgets_list),
        "params": {
            "le_json": str(Path(params.le_json).resolve()),
            "Sw": params.Sw,
            "Csw": params.Csw,
            "Sf": params.Sf,
            "lvl_lb": params.lvl_lb,
            "lvl_ub": params.lvl_ub,
            "bts_lb": params.bts_lb,
            "bts_ub": params.bts_ub,
            "threads": params.threads,
            "netname": params.netname,
            "bpsdepth": params.bpsdepth,
            "part": params.part,
            "comp": params.comp,
            "resilience_profile_path": (
                str(Path(params.resilience_profile_path).resolve())
                if params.resilience_profile_path
                else None
            ),
            "resilience_constraint_policy": params.resilience_constraint_policy,
            "resilience_mode": params.resilience_mode,
            "openevolve_seed": params.openevolve_seed,
            "openevolve_eval_suite": params.openevolve_eval_suite,
            "openevolve_reference_json": params.openevolve_reference_json,
            "openevolve_finalists": params.openevolve_finalists,
        },
        "latency_model": {
            "backend": params.backend,
            "poly_deg": params.poly_deg,
            "available_ops": sorted(params.latency_table.keys()),
        },
        "constraints": {
            "ckks": {
                "level_lower_bound": params.lvl_lb,
                "level_upper_bound": params.lvl_ub,
                "bootstrap_level_lower_bound": params.bts_lb,
                "bootstrap_level_upper_bound": params.bts_ub,
                "rescaling_factor": params.Sf,
                "input_waterline": params.Sw,
                "constant_waterline": params.Csw,
                "max_scale": _max_scale(params),
            },
            "local_scale_lower_bounds": {
                str(node): {
                    "in": params.scale_lower_bound(str(node), attrs, "in"),
                    "out": params.scale_lower_bound(str(node), attrs, "out"),
                }
                for node, attrs in pdag.nodes(data=True)
                if attrs.get("op") != "constant"
            },
        },
        "resilience": _resilience_context(pdag, params),
    }


def tdag_from_context(context: dict[str, Any]) -> Tdag:
    pdata = context["params"]
    params = Params(
        pdata["le_json"],
        "Orbit",
        "compile",
        Sw=pdata["Sw"],
        CSw=pdata["Csw"],
        threads=pdata.get("threads", 1),
        bpsdepth=pdata.get("bpsdepth"),
        comp=pdata.get("comp", True),
        part=pdata.get("part", True),
        placement_backend="openevolve",
        resilience_profile=pdata.get("resilience_profile_path"),
        resilience_mode=pdata.get("resilience_mode", "waterline"),
        resilience_constraint_policy=pdata.get("resilience_constraint_policy", "relax-only"),
        openevolve_eval_suite=pdata.get("openevolve_eval_suite", "polybert-sampled"),
        openevolve_reference_json=pdata.get("openevolve_reference_json"),
        openevolve_finalists=pdata.get("openevolve_finalists", 3),
    )
    params.Sf = int(pdata["Sf"])
    params.lvl_lb = int(pdata["lvl_lb"])
    params.lvl_ub = int(pdata["lvl_ub"])
    params.bts_lb = int(pdata["bts_lb"])
    params.bts_ub = int(pdata["bts_ub"])
    params.netname = pdata.get("netname", "")
    tdata = context["tdag"]
    tdag = Tdag(params, tdata["name"])
    for node, attrs in tdata["nodes"].items():
        tdag.add_node(node, **attrs)
    for edge in tdata["edges"]:
        tdag.add_edge(edge["u"], edge["v"], **edge["attrs"])
    tdag.inputs = set(tdata["inputs"])
    tdag.outputs = set(tdata["outputs"])
    return tdag


def build_compile_context(dag: Tdag, params: Params) -> dict[str, Any]:
    context = build_context(dag, [{"in_lvl": -1, "in_scl": params.Sw}], params)
    context["schema_version"] = COMPILE_CONTEXT_SCHEMA_VERSION
    context["harness"] = {
        "scope": "compile",
        "eval_suite": params.openevolve_eval_suite,
        "initial_prev_cost": {-1: {params.Sw: 0}},
        "reference_json": _load_reference_json(params.openevolve_reference_json),
        "finalists": params.openevolve_finalists,
    }
    return context


def run_compile_openevolve(dag: Tdag, le: LatencyEstimator, params: Params) -> dict[str, Any]:
    try:
        from openevolve import run_evolution
    except ImportError as exc:
        raise ImportError(
            "openevolve>=0.2.27 is required when --openevolve-iterations is positive."
        ) from exc

    root = _compile_workspace_root(dag, params)
    root.mkdir(parents=True, exist_ok=True)
    context = build_compile_context(dag, params)
    reference = _evaluate_compile_hints(context, {}, suppress_output=True)
    context["reference"] = {
        "final_latency_usec": reference.get("final_latency_usec"),
        "bootstrap_count": reference.get("bootstrap_count"),
        "rescale_count": reference.get("rescale_count"),
        "valid": reference.get("valid", False),
    }
    context["placement_profile"] = reference.get("bottleneck_summary", [])
    context_path = root / "compile_context.json"
    initial_path = root / "initial_program.py"
    evaluator_path = root / "evaluator.py"
    output_dir = root / "openevolve_output"
    context.setdefault("harness", {})["trace_dir"] = str(output_dir / "trace_repository")
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initial_path.write_text(_initial_compile_program_source(), encoding="utf-8")
    evaluator_path.write_text(_compile_evaluator_source(context_path), encoding="utf-8")
    initial_hints = _load_candidate_hints(initial_path, context)

    worker = OpenEvolvePlacementWorker(params, le)
    try:
        with worker._openevolve_runtime_env():
            result = run_evolution(
                initial_program=initial_path,
                evaluator=evaluator_path,
                config=worker._openevolve_config_arg(),
                iterations=params.openevolve_iterations,
                output_dir=str(output_dir),
                cleanup=False,
            )
    except Exception as exc:
        if not params.openevolve_fail_open:
            raise
        _write_recovery_summary(root, exc)
        return _recover_compile_hints(output_dir, context, initial_hints, params)
    best_program = root / "best_program.py"
    best_program.write_text(result.best_code, encoding="utf-8")
    try:
        finalist_hints = _run_full_bundle_finalists(
            root,
            output_dir,
            context,
            result.best_code,
            params,
        )
        if finalist_hints is not None:
            return finalist_hints
        return _load_candidate_hints(best_program, context)
    except Exception as exc:
        if not params.openevolve_fail_open:
            raise
        _write_recovery_summary(root, exc)
        return _recover_compile_hints(output_dir, context, initial_hints, params, result.best_code)
    finally:
        if not params.openevolve_keep_workdir and params.openevolve_output_dir is None:
            shutil.rmtree(root, ignore_errors=True)


def evaluate_compile_candidate_program(
    context_path: str | Path, program_path: str | Path
) -> dict[str, Any]:
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    try:
        hints = _load_candidate_hints(Path(program_path), context)
        static = _static_validate_hints(context, hints)
        if not static["valid"]:
            result = _invalid_compile_result("compile_static_gate", static["reasons"])
            _record_compile_trace(context, Path(program_path), hints, result, "STATIC_ONLY")
            return result
        result = _evaluate_compile_hints(context, hints, suppress_output=True)
        result["static"] = static
        reference = context.get("reference", {})
        ref_latency = _finite_float(reference.get("final_latency_usec"), result["final_latency_usec"])
        latency_ratio = (
            ref_latency / result["final_latency_usec"]
            if result["valid"] and result["final_latency_usec"] > 0
            else 0.0
        )
        latency_score = min(4.0, latency_ratio) / 4.0
        bootstrap_score = _relative_reduction(
            int(reference.get("bootstrap_count") or result["bootstrap_count"]),
            int(result["bootstrap_count"]),
        )
        rescale_score = _relative_reduction(
            int(reference.get("rescale_count") or result["rescale_count"]),
            int(result["rescale_count"]),
        )
        risk_score = 1.0 / (1.0 + result["profile_risk"])
        runtime_score = 1.0 / (1.0 + result["placement_runtime_sec"])
        fallback_score = 1.0 / (1.0 + result["fallback_selected_budgets"])
        boundary_score = max(0.0, min(1.0, float(result.get("boundary_quality", 0.0))))
        quality_score = (
            0.55 * latency_score
            + 0.13 * bootstrap_score
            + 0.07 * rescale_score
            + 0.08 * boundary_score
            + 0.07 * risk_score
            + 0.05 * fallback_score
            + 0.05 * runtime_score
        )
        if not result["valid"] or result["fallback_selected_budgets"] > 0:
            combined_score = min(0.999, 0.40 * result["validity"] + 0.30 * quality_score)
        else:
            combined_score = 1.0 + quality_score
        evaluation = {
            "metrics": {
                "combined_score": float(combined_score),
                "validity": float(result["validity"]),
                "latency_score": float(latency_score),
                "boundary_score": float(boundary_score),
                "fallback_score": float(fallback_score),
                "bootstrap_score": float(bootstrap_score),
                "rescale_score": float(rescale_score),
                "final_latency_usec": float(result["final_latency_usec"] if result["valid"] else 0.0),
                "boundary_quality": float(boundary_score),
                "bootstrap_count": float(result["bootstrap_count"]),
                "rescale_count": float(result["rescale_count"]),
                "profile_risk": float(result["profile_risk"]),
                "placement_runtime_sec": float(result["placement_runtime_sec"]),
                "fallback_selected_budgets": float(result["fallback_selected_budgets"]),
                "graph_nodes": float(len(context["tdag"]["nodes"])),
                "graph_edges": float(len(context["tdag"]["edges"])),
            },
            "artifacts": {
                "reference_final_latency_usec": str(reference.get("final_latency_usec", "none")),
                "candidate_final_latency_usec": f"{result['final_latency_usec']:.3f}",
                "latency_delta_usec": (
                    f"{result['final_latency_usec'] - ref_latency:.3f}"
                    if math.isfinite(ref_latency) and result["valid"]
                    else "none"
                ),
                "bootstrap_count": str(result["bootstrap_count"]),
                "rescale_count": str(result["rescale_count"]),
                "boundary_quality": f"{boundary_score:.6f}",
                "score_breakdown": json.dumps(
                    {
                        "latency_score": latency_score,
                        "bootstrap_score": bootstrap_score,
                        "rescale_score": rescale_score,
                        "boundary_score": boundary_score,
                        "risk_score": risk_score,
                        "fallback_score": fallback_score,
                        "runtime_score": runtime_score,
                    },
                    sort_keys=True,
                ),
                "bootstrap_locations": json.dumps(result["bootstrap_locations"], sort_keys=True),
                "rescale_locations": json.dumps(result["rescale_locations"], sort_keys=True),
                "bottleneck_summary": json.dumps(result.get("bottleneck_summary", []), sort_keys=True),
                "selected_output_state": json.dumps(result["selected_output_state"], sort_keys=True),
                "invalid_reasons": _compact_invalid_reasons(result["diagnostics"]),
                "candidate_invalid_reasons": _compact_invalid_reasons(
                    {"invalid_reasons": result["diagnostics"].get("candidate_invalid_reasons", {})}
                ),
                "policy_summary": _compact_policy_summary(hints),
                "unmatched_resilience_targets": json.dumps(
                    _profile_unmatched_targets(context), sort_keys=True
                ),
                "reference_json": json.dumps(context.get("harness", {}).get("reference_json", {}), sort_keys=True)[:4000],
                "replay_log_tail": result["log_tail"],
                "patchgate": json.dumps(static, sort_keys=True),
            },
        }
        _record_compile_trace(context, Path(program_path), hints, evaluation, "CLEAR_ONLY")
        return evaluation
    except Exception as exc:
        return {
            "metrics": {
                "combined_score": 1e-6,
                "validity": 0.0,
                "latency_score": 0.0,
                "final_latency_usec": 0.0,
                "boundary_quality": 0.0,
                "bootstrap_count": 0.0,
                "rescale_count": 0.0,
                "profile_risk": 1.0,
            },
            "artifacts": {
                "failure_stage": "compile_harness",
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            },
        }


def _evaluate_compile_hints(
    context: dict[str, Any], hints: dict[str, Any], *, suppress_output: bool
) -> dict[str, Any]:
    from .iterative_partition import solve_partition
    from .qbp_manager import QBPManager

    tdag = tdag_from_context(context)
    params = tdag.params
    params.openevolve_iterations = 0
    params.openevolve_harness = "compile"
    params.openevolve_compile_hints = hints
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = context.get("harness", {}).get("eval_suite", "polybert-sampled")
    le = LatencyEstimator(params)
    qbp_manager = QBPManager(params, le)
    log_buffer = io.StringIO()
    start = time.time()
    try:
        stream = log_buffer if suppress_output else None
        if stream is None:
            io_to_assign, io_to_cost = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
        else:
            with redirect_stdout(stream), redirect_stderr(stream):
                io_to_assign, io_to_cost = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
        if io_to_assign is None or io_to_cost is None:
            raise PlacementError("compile replay produced no QBP solution")
        final_io_choice, final_cost = _select_final_choice(io_to_cost)
        if final_io_choice is None:
            raise PlacementError("compile replay produced no final IO choice")
        assign = io_to_assign[final_io_choice[:2]][final_io_choice[2:]]
        assign.check_assign()
        counts = _maintenance_counts(assign)
        locations = _maintenance_locations(assign)
        diagnostics = _collect_qbp_diagnostics(qbp_manager)
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": float(estimate_assign(assign, le)),
            "aggregated_partition_cost_usec": float(final_cost),
            "bootstrap_count": int(counts["bootstrap"]),
            "rescale_count": int(counts["rescale"]),
            "boundary_quality": float(_boundary_quality(params, final_io_choice)),
            "profile_risk": float(_profile_risk([assign], params)),
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": int(diagnostics.get("fallback_selected_budgets", 0)),
            "selected_output_state": {
                "in_lvl": final_io_choice[0],
                "in_scl": final_io_choice[1],
                "out_lvl": final_io_choice[2],
                "out_scl": final_io_choice[3],
            },
            "bootstrap_locations": locations["bootstrap"],
            "rescale_locations": locations["rescale"],
            "bottleneck_summary": _bottleneck_summary(locations),
            "diagnostics": diagnostics,
            "log_tail": log_buffer.getvalue()[-3000:],
        }
    except Exception as exc:
        return {
            "valid": False,
            "validity": 0.0,
            "final_latency_usec": float("inf"),
            "aggregated_partition_cost_usec": float("inf"),
            "bootstrap_count": 0,
            "rescale_count": 0,
            "boundary_quality": 0.0,
            "profile_risk": 1.0,
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "diagnostics": {"invalid_reasons": {f"{type(exc).__name__}: {str(exc)[:240]}": 1}},
            "log_tail": log_buffer.getvalue()[-3000:],
        }


def _compile_workspace_root(dag: Tdag, params: Params) -> Path:
    name = f"compile_{dag.name}"
    if params.openevolve_output_dir:
        return Path(params.openevolve_output_dir).resolve() / name
    return Path(tempfile.mkdtemp(prefix=f"orbit_openevolve_{name}_"))


def _run_full_bundle_finalists(
    root: Path,
    output_dir: Path,
    sampled_context: dict[str, Any],
    best_code: str,
    params: Params,
) -> dict[str, Any] | None:
    if (
        params.openevolve_finalists <= 0
        or sampled_context.get("harness", {}).get("eval_suite") == "polybert-full"
    ):
        return None
    finalist_dir = root / "finalists"
    finalist_dir.mkdir(parents=True, exist_ok=True)
    full_context = json.loads(json.dumps(sampled_context))
    full_context.setdefault("harness", {})["eval_suite"] = "polybert-full"
    reference = _evaluate_compile_hints(full_context, {}, suppress_output=True)
    full_context["reference"] = {
        "final_latency_usec": reference.get("final_latency_usec"),
        "bootstrap_count": reference.get("bootstrap_count"),
        "rescale_count": reference.get("rescale_count"),
        "valid": reference.get("valid", False),
    }
    candidates = _discover_finalist_codes(output_dir, best_code, params.openevolve_finalists)
    summaries = []
    best = None
    for idx, code in enumerate(candidates):
        program_path = finalist_dir / f"finalist_{idx}.py"
        program_path.write_text(code, encoding="utf-8")
        try:
            hints = _load_candidate_hints(program_path, full_context)
            static = _static_validate_hints(full_context, hints)
            if not static["valid"]:
                raise PlacementError("; ".join(static["reasons"][:4]))
            result = _evaluate_compile_hints(full_context, hints, suppress_output=True)
            summary = {
                "index": idx,
                "valid": result["valid"],
                "final_latency_usec": result["final_latency_usec"],
                "boundary_quality": result.get("boundary_quality", 0.0),
                "bootstrap_count": result["bootstrap_count"],
                "rescale_count": result["rescale_count"],
                "fallback_selected_budgets": result["fallback_selected_budgets"],
                "selected_output_state": result["selected_output_state"],
                "bootstrap_locations": result.get("bootstrap_locations", {}),
                "rescale_locations": result.get("rescale_locations", {}),
            }
            if result["valid"]:
                item = (
                    int(result["fallback_selected_budgets"] > 0),
                    float(result["final_latency_usec"]),
                    idx,
                    hints,
                    summary,
                )
                if best is None or item[:3] < best[:3]:
                    best = item
        except Exception as exc:
            summary = {
                "index": idx,
                "valid": False,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }
        summaries.append(summary)
    (finalist_dir / "full_bundle_summary.json").write_text(
        json.dumps(
            {
                "reference": full_context["reference"],
                "candidates": summaries,
                "selected_index": None if best is None else best[2],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return None if best is None else best[3]


def _discover_finalist_codes(output_dir: Path, best_code: str, limit: int) -> list[str]:
    if limit <= 0:
        return []
    scored: list[tuple[float, int, int, str]] = []
    order = 0

    def add_code(code: Any, score: float, iteration: int = -1) -> None:
        nonlocal order
        if not isinstance(code, str) or not code.strip():
            return
        scored.append((float(score), int(iteration), order, code))
        order += 1

    add_code(best_code, float("inf"))
    best_dir = output_dir / "best"
    best_score = _program_info_score(best_dir / "best_program_info.json")
    if best_dir.exists():
        for program in sorted(best_dir.glob("*.py")):
            try:
                add_code(program.read_text(encoding="utf-8"), best_score)
            except Exception:
                continue
    for program in (output_dir / "best_program.py", output_dir / "initial_program.py"):
        if program.exists():
            try:
                add_code(program.read_text(encoding="utf-8"), _program_info_score(program.with_suffix(".json")))
            except Exception:
                continue
    checkpoint_root = output_dir / "checkpoints"
    checkpoints = []
    if checkpoint_root.exists():
        checkpoints = sorted(
            [path for path in checkpoint_root.iterdir() if path.is_dir()],
            key=lambda path: int(path.name.rsplit("_", 1)[-1]) if "_" in path.name and path.name.rsplit("_", 1)[-1].isdigit() else -1,
            reverse=True,
        )
    for checkpoint in checkpoints[:8]:
        iteration = _checkpoint_iteration(checkpoint)
        checkpoint_score = max(
            _program_info_score(checkpoint / "best_program_info.json"),
            _program_info_score(checkpoint / "best" / "best_program_info.json"),
        )
        for program in (checkpoint / "best_program.py", checkpoint / "best" / "best_program.py"):
            if program.exists():
                try:
                    add_code(program.read_text(encoding="utf-8"), checkpoint_score, iteration)
                except Exception:
                    continue
        for program_json in list((checkpoint / "programs").glob("*.json")) + list(
            (checkpoint / "database" / "programs").glob("*.json")
        ):
            try:
                data = json.loads(program_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            code = data.get("code")
            add_code(code, _program_data_score(data), iteration)
    for program_json in (output_dir / "programs").glob("*.json"):
        try:
            data = json.loads(program_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        add_code(data.get("code"), _program_data_score(data))
    scored.sort(key=lambda item: (item[0], item[1], -item[2]), reverse=True)
    result = []
    seen = set()
    for _score, _iteration, _order, code in scored:
        digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        result.append(code)
        if len(result) >= limit:
            break
    return result


def _checkpoint_iteration(path: Path) -> int:
    tail = path.name.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def _program_info_score(path: Path) -> float:
    if not path.exists():
        return 0.0
    try:
        return _program_data_score(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return 0.0


def _program_data_score(data: Any) -> float:
    if not isinstance(data, dict):
        return 0.0
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        for key in ("combined_score", "score", "fitness"):
            try:
                return float(metrics[key])
            except (KeyError, TypeError, ValueError):
                continue
    for key in ("combined_score", "score", "fitness"):
        try:
            return float(data[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0


def _select_final_choice(
    io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]]
) -> tuple[tuple[int, int, int, int] | None, float | None]:
    final_io_choice = None
    final_cost = None
    for (in_lvl, in_scale), out_to_cost in io_to_cost.items():
        for (out_lvl, out_scale), cost in out_to_cost.items():
            if final_cost is None or final_cost > cost:
                final_cost = cost
                final_io_choice = (in_lvl, in_scale, out_lvl, out_scale)
    return final_io_choice, final_cost


def _collect_qbp_diagnostics(qbp_manager) -> dict[str, Any]:
    totals = {
        "requested_budgets": 0,
        "solved_budgets": 0,
        "candidate_solved_budgets": 0,
        "fallback_selected_budgets": 0,
        "candidate_invalid_reasons": {},
    }
    for item in getattr(qbp_manager, "openevolve_diagnostics", []):
        for key in ("requested_budgets", "solved_budgets", "candidate_solved_budgets", "fallback_selected_budgets"):
            totals[key] += int(item.get(key, 0))
        for reason, count in item.get("candidate_invalid_reasons", {}).items():
            totals["candidate_invalid_reasons"][reason] = totals["candidate_invalid_reasons"].get(reason, 0) + int(count)
    return totals


def _load_reference_json(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": f"{type(exc).__name__}: {str(exc)[:240]}", "path": path}


def _write_recovery_summary(root: Path, exc: Exception) -> None:
    try:
        (root / "openevolve_recovery.json").write_text(
            json.dumps(
                {
                    "failure_stage": "openevolve_runtime",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "fail_open": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _recover_candidate_hints(
    output_dir: Path,
    context: dict[str, Any],
    initial_hints: dict[str, Any],
    best_code: str = "",
) -> dict[str, Any]:
    for code in _discover_finalist_codes(output_dir, best_code, 8):
        hints = _hints_from_code(code, context)
        if hints is not None and _static_validate_hints(context, hints)["valid"]:
            return hints
    return initial_hints


def _recover_compile_hints(
    output_dir: Path,
    context: dict[str, Any],
    initial_hints: dict[str, Any],
    params: Params,
    best_code: str = "",
) -> dict[str, Any]:
    best = None
    for idx, code in enumerate(_discover_finalist_codes(output_dir, best_code, max(8, params.openevolve_finalists))):
        hints = _hints_from_code(code, context)
        if hints is None:
            continue
        static = _static_validate_hints(context, hints)
        if not static["valid"]:
            continue
        result = _evaluate_compile_hints(context, hints, suppress_output=True)
        if not result.get("valid"):
            continue
        item = (
            int(result.get("fallback_selected_budgets", 0) > 0),
            float(result.get("final_latency_usec", float("inf"))),
            idx,
            hints,
        )
        if best is None or item[:3] < best[:3]:
            best = item
    return initial_hints if best is None else best[3]


def _hints_from_code(code: str, context: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(code, str) or not code.strip():
        return None
    with tempfile.TemporaryDirectory(prefix="orbit_openevolve_recover_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(code, encoding="utf-8")
        try:
            return _load_candidate_hints(path, context)
        except Exception:
            return None


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _invalid_candidate_result(stage: str, reasons: list[str]) -> dict[str, Any]:
    reason_counts = {reason: 1 for reason in reasons[:8]}
    return {
        "metrics": {
            "combined_score": 1e-6,
            "validity": 0.0,
            "effective_validity": 0.0,
            "latency_score": 0.0,
            "avg_latency_usec": 0.0,
            "bootstrap_count": 0.0,
            "rescale_count": 0.0,
            "profile_risk": 1.0,
            "solved_budgets": 0.0,
            "candidate_solved_budgets": 0.0,
            "fallback_selected_budgets": 0.0,
        },
        "artifacts": {
            "failure_stage": stage,
            "invalid_reasons": json.dumps(reason_counts, sort_keys=True),
        },
    }


def _invalid_compile_result(stage: str, reasons: list[str]) -> dict[str, Any]:
    reason_counts = {reason: 1 for reason in reasons[:8]}
    return {
        "metrics": {
            "combined_score": 1e-6,
            "validity": 0.0,
            "latency_score": 0.0,
            "final_latency_usec": 0.0,
            "boundary_quality": 0.0,
            "bootstrap_count": 0.0,
            "rescale_count": 0.0,
            "profile_risk": 1.0,
            "fallback_selected_budgets": 0.0,
        },
        "artifacts": {
            "failure_stage": stage,
            "invalid_reasons": json.dumps(reason_counts, sort_keys=True),
        },
    }


def run_trial(
    context: dict[str, Any],
    program_path: str | Path,
    eval_mode: str = "CLEAR_ONLY",
) -> dict[str, Any]:
    hints = _load_candidate_hints(Path(program_path), context)
    static = _static_validate_hints(context, hints)
    if eval_mode == "STATIC_ONLY":
        return {
            "mode": "STATIC_ONLY",
            "valid": static["valid"],
            "hints": hints,
            "static": static,
        }
    if eval_mode != "CLEAR_ONLY":
        return {
            "mode": eval_mode,
            "valid": False,
            "hints": hints,
            "static": static,
            "error": "FHE_LIGHT/FHE_FULL are external verification modes in this harness",
        }
    if not static["valid"]:
        return {"mode": "CLEAR_ONLY", "valid": False, "hints": hints, "static": static}
    result = _evaluate_compile_hints(context, hints, suppress_output=True)
    result["mode"] = "CLEAR_ONLY"
    result["static"] = static
    result["hints"] = hints
    return result


def _static_validate_hints(context: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    nodes = context.get("tdag", {}).get("nodes", {})
    edges = {
        _edge_key(str(edge.get("u")), str(edge.get("v")))
        for edge in context.get("tdag", {}).get("edges", [])
        if isinstance(edge, dict)
    }
    ckks = _ckks_dict(context)
    local_bounds = context.get("constraints", {}).get("local_scale_lower_bounds", {})
    for node, level in _int_map(hints.get("preferred_node_levels", {})).items():
        if node not in nodes:
            reasons.append(f"unknown node level target {node}")
            continue
        if not int(ckks["lvl_lb"]) <= level <= int(ckks["lvl_ub"]):
            reasons.append(f"node {node} preferred level {level} outside CKKS bounds")
    for node, scale in _int_map(hints.get("preferred_node_scales", {})).items():
        if node not in nodes:
            reasons.append(f"unknown node scale target {node}")
            continue
        lb_info = local_bounds.get(node, {})
        lb = _safe_int(lb_info.get("out"), int(ckks["Sw"])) if isinstance(lb_info, dict) else int(ckks["Sw"])
        if not lb <= scale <= int(ckks["max_scale"]):
            reasons.append(f"node {node} preferred scale {scale} outside [{lb}, {ckks['max_scale']}]")
    for edge, scale in _int_map(hints.get("preferred_edge_scales", {})).items():
        if edge not in edges:
            reasons.append(f"unknown edge scale target {edge}")
            continue
        if not 0 <= scale <= int(ckks["max_scale"]):
            reasons.append(f"edge {edge} preferred scale {scale} outside CKKS max scale")
    for item in hints.get("portfolio", []):
        if isinstance(item, dict):
            child = _static_validate_hints({**context, "_portfolio_child": True}, item)
            reasons.extend(f"portfolio: {reason}" for reason in child["reasons"])
    return {"valid": not reasons, "reasons": reasons[:16]}


def _record_compile_trace(
    context: dict[str, Any],
    program_path: Path,
    hints: dict[str, Any],
    evaluation: dict[str, Any],
    eval_mode: str,
) -> None:
    trace_dir = context.get("harness", {}).get("trace_dir")
    if not trace_dir:
        return
    try:
        source = program_path.read_text(encoding="utf-8")
    except Exception:
        source = ""
    record = {
        "context_schema": context.get("schema_version"),
        "model": context.get("model", {}),
        "candidate_digest": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "policy_summary": _compact_policy_summary(hints),
        "eval_mode": eval_mode,
        "metrics": evaluation.get("metrics", {}),
        "artifacts": {
            key: value
            for key, value in evaluation.get("artifacts", {}).items()
            if key
            in {
                "invalid_reasons",
                "candidate_invalid_reasons",
                "selected_output_state",
                "bootstrap_locations",
                "rescale_locations",
                "bottleneck_summary",
                "boundary_quality",
            }
        },
    }
    try:
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "trials.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    except Exception:
        pass


def evaluate_candidate_program(context_path: str | Path, program_path: str | Path) -> dict[str, Any]:
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    tdag = tdag_from_context(context)
    le = LatencyEstimator(tdag.params)
    try:
        hints = _load_candidate_hints(Path(program_path), context)
        static = _static_validate_hints(context, hints)
        if not static["valid"]:
            return _invalid_candidate_result("placement_static_gate", static["reasons"])
        io_budgets = [_io_budget_from_json(item) for item in context["io_budgets"]]
        diagnostics: dict[str, Any] = {}
        io_to_assign, io_to_cost = solve_budget_batch(
            tdag, io_budgets, le, tdag.params, hints, diagnostics
        )
        assignments = list(diagnostics.get("assignments", []))
        costs = list(diagnostics.get("costs", []))
        solved = int(diagnostics.get("solved_budgets", len(costs)))
        candidate_solved = int(diagnostics.get("candidate_solved_budgets", solved))
        fallback_selected = int(diagnostics.get("fallback_selected_budgets", 0))
        candidate_improved = int(diagnostics.get("candidate_improved_budgets", 0))
        requested = max(1, len(io_budgets))
        effective_validity = solved / requested
        validity = candidate_solved / requested
        avg_cost = sum(costs) / solved if costs else float("inf")
        counts = _aggregate_counts(assignments)
        profile_risk = _profile_risk(assignments, tdag.params)
        reference = _reference_metrics(tdag, io_budgets, le)
        reference_avg = reference["avg_latency_usec"]
        latency_ratio = (
            reference_avg / avg_cost
            if costs and avg_cost > 0.0 and math.isfinite(reference_avg)
            else 0.0
        )
        latency_score = min(4.0, latency_ratio) / 4.0
        bootstrap_score = _relative_reduction(
            reference["bootstrap_count"], counts["bootstrap"]
        )
        rescale_score = _relative_reduction(reference["rescale_count"], counts["rescale"])
        risk_score = 1.0 / (1.0 + profile_risk)
        quality_score = (
            0.55 * latency_score
            + 0.20 * bootstrap_score
            + 0.10 * rescale_score
            + 0.15 * risk_score
        )
        if effective_validity < 1.0:
            combined_score = min(
                0.999,
                0.85 * effective_validity
                + 0.05 * validity
                + 0.099 * quality_score
            )
        elif candidate_solved == 0 or fallback_selected > 0:
            combined_score = min(0.999, 0.40 * validity + 0.30 * quality_score)
        else:
            improvement_fraction = candidate_improved / requested
            combined_score = (
                1.0
                + quality_score
                + 0.05 * improvement_fraction
                + 0.02 * validity
            )
        return {
            "metrics": {
                "combined_score": float(combined_score),
                "validity": float(validity),
                "effective_validity": float(effective_validity),
                "latency_score": float(latency_score),
                "avg_latency_usec": float(avg_cost if costs else 0.0),
                "bootstrap_count": float(counts["bootstrap"]),
                "rescale_count": float(counts["rescale"]),
                "profile_risk": float(profile_risk),
                "solved_budgets": float(solved),
                "candidate_solved_budgets": float(candidate_solved),
                "fallback_selected_budgets": float(fallback_selected),
                "candidate_improved_budgets": float(candidate_improved),
                "graph_nodes": float(len(tdag.nodes)),
                "graph_edges": float(len(tdag.edges)),
                "budget_count": float(len(io_budgets)),
            },
            "artifacts": {
                "solved_budgets": f"{solved}/{len(io_budgets)}",
                "candidate_solved_budgets": f"{candidate_solved}/{len(io_budgets)}",
                "fallback_selected_budgets": f"{fallback_selected}/{len(io_budgets)}",
                "candidate_improved_budgets": f"{candidate_improved}/{len(io_budgets)}",
                "reference_avg_latency_usec": f"{reference_avg:.3f}",
                "latency_delta_usec": (
                    f"{avg_cost - reference_avg:.3f}"
                    if costs and math.isfinite(reference_avg)
                    else "none"
                ),
                "best_latency_usec": str(min(costs) if costs else "none"),
                "avg_latency_usec": f"{avg_cost:.3f}" if costs else "none",
                "bootstrap_count": str(counts["bootstrap"]),
                "rescale_count": str(counts["rescale"]),
                "bootstrap_delta": str(counts["bootstrap"] - reference["bootstrap_count"]),
                "rescale_delta": str(counts["rescale"] - reference["rescale_count"]),
                "invalid_reasons": _compact_invalid_reasons(diagnostics),
                "candidate_invalid_reasons": _compact_invalid_reasons(
                    {"invalid_reasons": diagnostics.get("candidate_invalid_reasons", {})}
                ),
                "policy_summary": _compact_policy_summary(hints),
                "unmatched_resilience_targets": json.dumps(
                    _profile_unmatched_targets(context), sort_keys=True
                ),
                "profiler_plan_hints": _compact_profile_plan(context),
                "patchgate": json.dumps(static, sort_keys=True),
            },
        }
    except Exception as exc:
        return {
            "metrics": {
                "combined_score": 1e-6,
                "validity": 0.0,
                "latency_score": 0.0,
                "bootstrap_count": 0.0,
                "rescale_count": 0.0,
                "profile_risk": 1.0,
            },
            "artifacts": {
                "failure_stage": "placement_validation",
                "error": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            },
        }


def solve_budget_batch(
    pdag: Tdag,
    io_budgets_list: list[dict],
    le: LatencyEstimator,
    params: Params,
    hints: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], Assign]],
    dict[tuple[int, int], dict[tuple[int, int], float]],
]:
    io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]] = {}
    io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    hints = _with_default_policy(hints)
    if diagnostics is not None:
        diagnostics["requested_budgets"] = len(io_budgets_list)
        diagnostics["solved_budgets"] = 0
        diagnostics["candidate_solved_budgets"] = 0
        diagnostics["fallback_solved_budgets"] = 0
        diagnostics["fallback_selected_budgets"] = 0
        diagnostics["candidate_improved_budgets"] = 0
        diagnostics["costs"] = []
        diagnostics["assignments"] = []
    for io_budget in io_budgets_list:
        attempts: list[_BudgetAttempt] = []
        last_error = None
        record = _placement_record_for_budget(hints, pdag.name, io_budget)
        if record is not None:
            try:
                attempts.append(_solve_one_record_attempt(pdag, params, io_budget, le, record))
            except Exception as exc:
                last_error = exc
                if diagnostics is not None:
                    _record_invalid_reason(diagnostics, "candidate_invalid_reasons", exc)
        for source, policy_hints in _budget_policy_attempts(hints, params):
            try:
                attempts.append(_solve_one_budget_attempt(pdag, params, io_budget, le, source, policy_hints))
            except Exception as exc:
                last_error = exc
                if diagnostics is not None and source.startswith("candidate"):
                    _record_invalid_reason(diagnostics, "candidate_invalid_reasons", exc)

        if not attempts:
            if diagnostics is not None:
                _record_invalid_reason(
                    diagnostics,
                    "invalid_reasons",
                    last_error or PlacementError("no feasible placement policy"),
                )
            continue

        candidate_attempt = _best_attempt(
            attempt for attempt in attempts if attempt.source.startswith("candidate")
        )
        fallback_attempt = _best_attempt(
            attempt for attempt in attempts if attempt.source.startswith("seed_fallback")
        )
        best_attempt = _best_attempt(attempts)
        if best_attempt is None:
            continue

        if diagnostics is not None:
            diagnostics["solved_budgets"] += 1
            diagnostics["costs"].append(best_attempt.cost)
            diagnostics["assignments"].append(best_attempt.assign)
            if candidate_attempt is not None:
                diagnostics["candidate_solved_budgets"] += 1
            if fallback_attempt is not None:
                diagnostics["fallback_solved_budgets"] += 1
            if best_attempt.source.startswith("seed_fallback"):
                diagnostics["fallback_selected_budgets"] += 1
            if (
                candidate_attempt is not None
                and fallback_attempt is not None
                and candidate_attempt.cost + 1e-9 < fallback_attempt.cost
            ):
                diagnostics["candidate_improved_budgets"] += 1

        current = io_to_cost.get(best_attempt.in_key, {}).get(best_attempt.out_key)
        if current is None or best_attempt.cost < current:
            io_to_cost.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.cost
            io_to_assign.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.assign
    return io_to_assign, io_to_cost


def _budget_policy_attempts(
    hints: dict[str, Any], params: Params
) -> list[tuple[str, dict[str, Any]]]:
    attempts = []
    for idx, policy_hints in enumerate(_portfolio_policies(hints)):
        label = "candidate" if idx == 0 else f"candidate:{idx}"
        attempts.append((label, policy_hints))
        attempts.append((f"{label}:relaxed", _relaxed_scheduler_policy(policy_hints, params)))
    if _should_use_seed_fallback(hints):
        seed = _default_policy_hints()
        attempts.extend(
            [
                ("seed_fallback", seed),
                ("seed_fallback_relaxed", _relaxed_scheduler_policy(seed, params)),
            ]
        )
    return attempts


def _portfolio_policies(hints: dict[str, Any]) -> list[dict[str, Any]]:
    portfolio = hints.get("portfolio")
    if not isinstance(portfolio, list) or not portfolio:
        return [hints]
    policies = []
    seen = set()
    for item in [hints] + [policy for policy in portfolio if isinstance(policy, dict)]:
        policy = _with_default_policy(item)
        digest = json.dumps(
            {
                key: value
                for key, value in policy.items()
                if key not in {"portfolio", "placement_records", "api_version"}
            },
            sort_keys=True,
            default=str,
        )
        if digest in seen:
            continue
        seen.add(digest)
        policies.append(policy)
    return policies


def _should_use_seed_fallback(hints: dict[str, Any]) -> bool:
    explicit = hints.get("allow_seed_fallback")
    if explicit is not None:
        return _bool_hint(explicit, False)
    return str(hints.get("strategy", "waterline_seed")) != "waterline_seed"


def _solve_one_budget_attempt(
    pdag: Tdag,
    params: Params,
    io_budget: dict,
    le: LatencyEstimator,
    source: str,
    policy_hints: dict[str, Any],
) -> _BudgetAttempt:
    assign = build_conservative_assign(pdag, params, io_budget, policy_hints, le)
    assign.check_assign()
    pdag_vin = list(pdag.inputs)[0]
    pdag_vout = list(pdag.outputs)[0]
    in_key = (assign.v_lvl_in[pdag_vin], assign.v_scl_in[pdag_vin])
    out_key = (assign.v_lvl_out[pdag_vout], assign.v_scl_out[pdag_vout])
    if io_budget.get("in_lvl", -1) >= 0 and in_key[0] != io_budget["in_lvl"]:
        raise PlacementError("input level mismatch")
    if io_budget.get("in_scl", -1) >= 0 and in_key[1] != io_budget["in_scl"]:
        raise PlacementError("input scale mismatch")
    if io_budget.get("out_lvl", -1) >= 0 and out_key[0] != io_budget["out_lvl"]:
        raise PlacementError("output level mismatch")
    return _BudgetAttempt(source, assign, estimate_assign(assign, le), in_key, out_key)


def _solve_one_record_attempt(
    pdag: Tdag,
    params: Params,
    io_budget: dict,
    le: LatencyEstimator,
    record: dict[str, Any],
) -> _BudgetAttempt:
    assign = _assign_from_serialized(pdag, record.get("assignment", {}))
    assign.check_assign()
    pdag_vin = list(pdag.inputs)[0]
    pdag_vout = list(pdag.outputs)[0]
    in_key = (assign.v_lvl_in[pdag_vin], assign.v_scl_in[pdag_vin])
    out_key = (assign.v_lvl_out[pdag_vout], assign.v_scl_out[pdag_vout])
    if io_budget.get("in_lvl", -1) >= 0 and in_key[0] != io_budget["in_lvl"]:
        raise PlacementError("record input level mismatch")
    if io_budget.get("in_scl", -1) >= 0 and in_key[1] != io_budget["in_scl"]:
        raise PlacementError("record input scale mismatch")
    if io_budget.get("out_lvl", -1) >= 0 and out_key[0] != io_budget["out_lvl"]:
        raise PlacementError("record output level mismatch")
    return _BudgetAttempt("candidate_record", assign, estimate_assign(assign, le), in_key, out_key)


def _best_attempt(attempts) -> _BudgetAttempt | None:
    best = None
    for attempt in attempts:
        if best is None or attempt.cost < best.cost:
            best = attempt
    return best


def _placement_record_for_budget(
    hints: dict[str, Any], pdag_name: str, io_budget: dict[str, Any]
) -> dict[str, Any] | None:
    records = hints.get("placement_records")
    if not isinstance(records, list):
        return None
    target = (
        pdag_name,
        int(io_budget.get("in_lvl", -1)),
        int(io_budget.get("in_scl", -1)),
        int(io_budget.get("out_lvl", -1)),
    )
    for record in records:
        if not isinstance(record, dict):
            continue
        key = (
            str(record.get("pdag", "")),
            int(record.get("in_lvl", -1)),
            int(record.get("in_scl", -1)),
            int(record.get("out_lvl", -1)),
        )
        if key == target:
            return record
    return None


def _assign_from_serialized(tdag: Tdag, data: dict[str, Any]) -> Assign:
    if not isinstance(data, dict):
        raise PlacementError("placement record missing assignment object")
    assign = Assign(tdag)
    for name in ("v_lvl_out", "v_scl_out", "v_lvl_in", "v_scl_in"):
        value = data.get(name, {})
        if isinstance(value, dict):
            setattr(assign, name, {str(k): int(v) for k, v in value.items()})
    for name in ("e_lvl_out", "e_scl_out"):
        value = data.get(name, {})
        if not isinstance(value, dict):
            continue
        parsed = {}
        for key, item in value.items():
            if isinstance(key, tuple):
                edge = (str(key[0]), str(key[1]))
            else:
                parts = str(key).split("->", 1)
                if len(parts) != 2:
                    raise PlacementError(f"bad serialized edge key {key!r}")
                edge = (parts[0], parts[1])
            parsed[edge] = int(item)
        setattr(assign, name, parsed)
    return assign


def _record_invalid_reason(
    diagnostics: dict[str, Any], key: str, exc: Exception
) -> None:
    invalid_reasons = diagnostics.setdefault(key, {})
    reason = f"{type(exc).__name__}: {str(exc)[:240]}"
    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1


def build_conservative_assign(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    hints: dict[str, Any] | None = None,
    le: LatencyEstimator | None = None,
) -> Assign:
    hints = _with_default_policy(hints)
    if str(hints.get("strategy")) == "latency_beam":
        return _build_best_assign_from_variants(tdag, params, io_budget, hints, le)
    if "maino_v" in io_budget:
        best = None
        main_qbp_cost = io_budget.get("main_qbp_cost", {})
        for main_key, main_cost in sorted(main_qbp_cost.items(), key=lambda item: item[1]):
            fixed = {io_budget["maino_v"]: main_key}
            try:
                assign = _build_assign_once(tdag, params, io_budget, hints, fixed, le)
                cost = _assignment_score(assign, le) + float(main_cost)
            except Exception:
                continue
            if best is None or cost < best[0]:
                best = (cost, assign)
        if best is None:
            raise PlacementError("no feasible main-output choice for bypass placement")
        return best[1]
    return _build_assign_once(tdag, params, io_budget, hints, {}, le)


def _build_best_assign_from_variants(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    hints: dict[str, Any],
    le: LatencyEstimator | None,
) -> Assign:
    best = None
    for variant in _latency_beam_variants(hints, params):
        try:
            if "maino_v" in io_budget:
                main_qbp_cost = io_budget.get("main_qbp_cost", {})
                for main_key, main_cost in sorted(main_qbp_cost.items(), key=lambda item: item[1]):
                    assign = _build_assign_once(
                        tdag,
                        params,
                        io_budget,
                        variant,
                        {io_budget["maino_v"]: main_key},
                        le,
                    )
                    assign.check_assign()
                    cost = _assignment_score(assign, le) + float(main_cost)
                    item = (cost, assign)
                    if best is None or item[0] < best[0]:
                        best = item
            else:
                assign = _build_assign_once(tdag, params, io_budget, variant, {}, le)
                assign.check_assign()
                item = (_assignment_score(assign, le), assign)
                if best is None or item[0] < best[0]:
                    best = item
        except Exception:
            continue
    if best is None:
        raise PlacementError("latency_beam found no feasible policy variant")
    return best[1]


def _latency_beam_variants(hints: dict[str, Any], params: Params) -> list[dict[str, Any]]:
    width = max(1, min(8, _int_hint(hints.get("beam_width"), 4)))
    base = dict(hints)
    base["strategy"] = "level_preserving"
    variants = [
        base,
        {
            **base,
            "refresh_fanout_at_level_floor": True,
            "allow_bootstrap": False,
            "min_internal_level": max(params.lvl_lb, params.bts_lb + 1),
        },
        {
            **base,
            "allow_bootstrap": True,
            "bootstrap_penalty": max(1.0, _float_hint(base.get("bootstrap_penalty"), 1_000_000_000.0) * 0.25),
            "level_drop_penalty": max(0.0, _float_hint(base.get("level_drop_penalty"), 20_000_000.0) * 0.5),
        },
        {
            **base,
            "rescale_penalty": max(250_000.0, _float_hint(base.get("rescale_penalty"), 0.0)),
            "scale_penalty": max(100.0, _float_hint(base.get("scale_penalty"), 0.0)),
        },
        {
            **base,
            "min_internal_level": params.lvl_lb,
            "level_drop_penalty": 0.0,
        },
    ]
    return [_with_default_policy(variant) for variant in variants[:width]]


def _relaxed_scheduler_policy(hints: dict[str, Any], params: Params) -> dict[str, Any]:
    relaxed = dict(hints)
    relaxed.update(
        {
            "refresh_fanout_at_level_floor": False,
            "min_internal_level": params.lvl_lb,
            "level_drop_penalty": 0.0,
        }
    )
    return relaxed


def _build_assign_once(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    hints: dict[str, Any],
    fixed_node_inputs: dict[str, tuple[int, int]],
    le: LatencyEstimator | None,
) -> Assign:
    assign = Assign(tdag)
    node_level_hints = _int_map(hints.get("preferred_node_levels", {}))
    node_scale_hints = _int_map(hints.get("preferred_node_scales", {}))
    edge_scale_hints = _int_map(hints.get("preferred_edge_scales", {}))
    policy = _policy_options(hints, params)

    for v in nx.topological_sort(tdag):
        op = tdag.nodes[v]["op"]
        if op == "constant":
            continue
        if v in tdag.inputs or tdag.in_degree(v) == 0:
            in_level, in_scale = _input_level_scale(tdag, v, params, io_budget, fixed_node_inputs)
            assign.v_lvl_in[v] = in_level
            assign.v_scl_in[v] = in_scale
        else:
            in_level, in_scale = _incoming_level_scale(
                tdag,
                assign,
                v,
                params,
                fixed_node_inputs.get(v),
                node_level_hints.get(v),
                node_scale_hints.get(v),
                edge_scale_hints,
                policy,
                le,
            )
            assign.v_lvl_in[v] = in_level
            assign.v_scl_in[v] = in_scale

        output_level, output_scale = _choose_output_state(
            tdag,
            v,
            params,
            assign.v_lvl_in[v],
            assign.v_scl_in[v],
            io_budget.get("out_lvl") if v in tdag.outputs else None,
            node_level_hints.get(v),
            node_scale_hints.get(v),
            policy,
            le,
        )
        assign.v_lvl_out[v] = output_level
        assign.v_scl_out[v] = output_scale

    for v in tdag.nodes:
        if tdag.nodes[v]["op"] == "constant" and v not in assign.v_lvl_out:
            succ = list(tdag.successors(v))
            if not succ:
                assign.v_lvl_out[v] = params.lvl_lb
                assign.v_scl_out[v] = params.Csw
            else:
                target = succ[0]
                assign.v_lvl_out[v] = assign.v_lvl_in[target]
                assign.v_scl_out[v] = (
                    params.Csw
                    if tdag.nodes[target]["op"] == "mul"
                    else assign.v_scl_in[target]
                )

    return assign


def _input_level_scale(
    tdag: Tdag,
    v: str,
    params: Params,
    io_budget: dict,
    fixed_node_inputs: dict[str, tuple[int, int]],
) -> tuple[int, int]:
    if v in fixed_node_inputs:
        level, scale = fixed_node_inputs[v]
    else:
        scale = io_budget.get("in_scl", params.scale_lower_bound(v, tdag.nodes[v], "in"))
        if scale < 0:
            scale = params.scale_lower_bound(v, tdag.nodes[v], "in")
        scale = max(int(scale), params.scale_lower_bound(v, tdag.nodes[v], "in"))
        level = int(io_budget.get("in_lvl", -1))
        if level < 0:
            level = _highest_decryptable_level(params, scale)
    _assert_decryptable(params, level, scale)
    return level, scale


def _max_scale(params: Params) -> int:
    return params.max_scale()


def _default_policy_hints() -> dict[str, Any]:
    return _low_scale_frontier_policy()


def _waterline_seed_policy() -> dict[str, Any]:
    return {
        "strategy": "waterline_seed",
        "prefer_level_preservation": True,
        "allow_bootstrap": False,
        "allow_seed_fallback": None,
        "refresh_fanout_at_level_floor": False,
        "max_scale_candidates": 32,
        "bootstrap_penalty": 1_000_000_000.0,
        "rescale_penalty": 0.0,
        "level_drop_penalty": 20_000_000.0,
        "min_internal_level": None,
        "scale_penalty": 0.0,
        "preferred_node_levels": {},
        "preferred_node_scales": {},
        "preferred_edge_scales": {},
    }


def _low_scale_frontier_policy() -> dict[str, Any]:
    return {
        "strategy": "level_preserving",
        "prefer_level_preservation": True,
        "allow_bootstrap": False,
        "allow_seed_fallback": False,
        "refresh_fanout_at_level_floor": True,
        "max_scale_candidates": 10,
        "bootstrap_penalty": 1_000_000_000.0,
        "rescale_penalty": 0.0,
        "level_drop_penalty": 20_000_000.0,
        "min_internal_level": None,
        "scale_penalty": 0.0,
        "preferred_node_levels": {},
        "preferred_node_scales": {},
        "preferred_edge_scales": {},
    }


def _with_default_policy(hints: dict[str, Any] | None) -> dict[str, Any]:
    result = _default_policy_hints()
    if not isinstance(hints, dict):
        return result
    for key, value in hints.items():
        if key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
            merged = dict(result[key])
            if isinstance(value, dict):
                merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


def _policy_options(hints: dict[str, Any], params: Params) -> dict[str, Any]:
    return {
        "strategy": str(hints.get("strategy", "waterline_seed")),
        "prefer_level_preservation": _bool_hint(hints.get("prefer_level_preservation"), True),
        "allow_bootstrap": _bool_hint(hints.get("allow_bootstrap"), False),
        "refresh_fanout_at_level_floor": _bool_hint(
            hints.get("refresh_fanout_at_level_floor"), False
        ),
        "max_scale_candidates": max(3, min(32, _int_hint(hints.get("max_scale_candidates"), 32))),
        "bootstrap_penalty": max(0.0, _float_hint(hints.get("bootstrap_penalty"), 1_000_000_000.0)),
        "rescale_penalty": max(0.0, _float_hint(hints.get("rescale_penalty"), 0.0)),
        "level_drop_penalty": max(0.0, _float_hint(hints.get("level_drop_penalty"), 20_000_000.0)),
        "min_internal_level": max(
            params.lvl_lb,
            min(params.lvl_ub, _int_hint(hints.get("min_internal_level"), params.bts_lb + 1)),
        ),
        "scale_penalty": max(0.0, _float_hint(hints.get("scale_penalty"), 0.0)),
        "beam_width": max(1, min(8, _int_hint(hints.get("beam_width"), 4))),
        "state_cap_per_node": max(1, min(32, _int_hint(hints.get("state_cap_per_node"), 8))),
        "scale_lattice": str(hints.get("scale_lattice", "default")),
        "max_scale": _max_scale(params),
    }


def _bool_hint(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if value is None:
        return default
    return bool(value)


def _int_hint(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_hint(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _incoming_level_scale(
    tdag: Tdag,
    assign: Assign,
    v: str,
    params: Params,
    fixed_input: tuple[int, int] | None,
    preferred_level: int | None,
    preferred_scale: int | None,
    edge_scale_hints: dict[str, int],
    policy: dict[str, Any],
    le: LatencyEstimator | None,
) -> tuple[int, int]:
    preds = list(tdag.predecessors(v))
    if not preds:
        raise PlacementError(f"node {v} has no predecessor and is not an input")

    if fixed_input is not None:
        target_level, node_scale = fixed_input
        edge_scales = _edge_scales_for_fixed_input(tdag, v, preds, params, node_scale)
        if _incoming_level_ok(tdag, assign, v, preds, edge_scales, target_level, node_scale, params):
            _store_incoming(assign, tdag, v, preds, edge_scales, target_level, params)
            return target_level, node_scale
        raise PlacementError(f"fixed input for node {v} is infeasible")

    if policy["strategy"] == "waterline_seed":
        node_input_lb = max(params.scale_lower_bound(v, tdag.nodes[v], "in"), preferred_scale or 0)
        edge_scales = _default_edge_scales(tdag, v, preds, params, node_input_lb, edge_scale_hints)
        node_scale = _node_input_scale(tdag, v, edge_scales)
        level = _common_incoming_level(
            tdag, assign, v, preds, edge_scales, node_scale, params, preferred_level
        )
        _store_incoming(assign, tdag, v, preds, edge_scales, level, params)
        return level, node_scale

    best_no_bootstrap = None
    best_bootstrap = None
    best_any = None
    levels = [
        level
        for level in _level_candidates(params, preferred_level)
        if level >= int(policy["min_internal_level"])
    ]
    if not levels:
        levels = _level_candidates(params, preferred_level)
    for edge_scales, node_scale in _incoming_scale_options(
        tdag, assign, v, preds, params, preferred_scale, edge_scale_hints, policy
    ):
        for level in levels:
            if not _incoming_level_ok(tdag, assign, v, preds, edge_scales, level, node_scale, params):
                continue
            if (
                level <= int(policy["min_internal_level"])
                and not params.check_res(level, node_scale, params.bts_lb, params.Sf)
            ):
                continue
            score = policy["scale_penalty"] * node_scale
            has_bootstrap = False
            for u in preds:
                if tdag.nodes[u]["op"] == "constant":
                    continue
                this_score = _transition_score(
                    params,
                    le,
                    assign.v_lvl_out[u],
                    assign.v_scl_out[u],
                    level,
                    edge_scales[u],
                    policy,
                )
                if this_score is None:
                    score = None
                    break
                score += this_score
                has_bootstrap = has_bootstrap or not params.check_res(
                    assign.v_lvl_out[u], assign.v_scl_out[u], level, edge_scales[u]
                )
            if score is None:
                continue
            item = (score, level, edge_scales, node_scale)
            if best_any is None or item[0] < best_any[0]:
                best_any = item
            if not has_bootstrap and (best_no_bootstrap is None or item[0] < best_no_bootstrap[0]):
                best_no_bootstrap = item

    if best_no_bootstrap is not None and not policy["allow_bootstrap"]:
        _, level, edge_scales, node_scale = best_no_bootstrap
    elif best_any is not None:
        _, level, edge_scales, node_scale = best_any
    else:
        raise PlacementError(f"no feasible incoming level/scale for node {v}")
    _store_incoming(assign, tdag, v, preds, edge_scales, level, params)
    return level, node_scale


def _incoming_scale_options(
    tdag: Tdag,
    assign: Assign,
    v: str,
    preds: list[str],
    params: Params,
    preferred_scale: int | None,
    edge_scale_hints: dict[str, int],
    policy: dict[str, Any],
) -> list[tuple[dict[str, int], int]]:
    node_input_lb = max(params.scale_lower_bound(v, tdag.nodes[v], "in"), preferred_scale or 0)
    max_scale = int(policy["max_scale"])
    if tdag.nodes[v]["op"] != "mul":
        if any(tdag.nodes[u]["op"] == "constant" for u in preds):
            node_input_lb = max(node_input_lb, params.Csw)
        bases = [node_input_lb, preferred_scale, params.Sw, params.Csw, params.Sf]
        for u in preds:
            bases.append(edge_scale_hints.get(_edge_key(u, v)))
            if tdag.nodes[u]["op"] == "constant":
                bases.append(params.Csw)
            else:
                bases.append(assign.v_scl_out[u])
        return [
            ({u: scale for u in preds}, scale)
            for scale in _scale_candidates(bases, node_input_lb, max_scale, params, policy)
        ]

    per_pred: list[tuple[str, list[int]]] = []
    for u in preds:
        if tdag.nodes[u]["op"] == "constant":
            per_pred.append((u, [params.Csw]))
            continue
        edge_lb = _edge_scale_lb(tdag, u, v, params)
        edge_max = max_scale // 2 if len(preds) == 1 else max_scale
        bases = [
            edge_lb,
            edge_scale_hints.get(_edge_key(u, v)),
            assign.v_scl_out[u],
            math.ceil(assign.v_scl_out[u] / 2),
            params.Sw,
            params.Csw,
            params.Sf,
        ]
        per_pred.append((u, _scale_candidates(bases, edge_lb, edge_max, params, policy)))

    options: list[tuple[dict[str, int], int]] = []
    if len(per_pred) == 1:
        u, candidates = per_pred[0]
        for scale in candidates:
            node_scale = 2 * scale
            if node_input_lb <= node_scale <= max_scale:
                options.append(({u: scale}, node_scale))
        return _dedupe_scale_options(options)

    if len(per_pred) > 2:
        # Orbit mul nodes are expected to have one or two effective inputs. Keep a
        # deterministic fallback for malformed imports rather than exploding the
        # candidate product.
        edge_scales = _default_edge_scales(tdag, v, preds, params, node_input_lb, edge_scale_hints)
        return [(edge_scales, _node_input_scale(tdag, v, edge_scales))]

    (u0, c0), (u1, c1) = per_pred
    for s0 in c0:
        for s1 in c1:
            edge_scales = {u0: s0, u1: s1}
            node_scale = s0 + s1
            if node_input_lb <= node_scale <= max_scale:
                options.append((edge_scales, node_scale))
    if not options:
        edge_scales = _default_edge_scales(tdag, v, preds, params, node_input_lb, edge_scale_hints)
        return [(edge_scales, _node_input_scale(tdag, v, edge_scales))]
    return _dedupe_scale_options(options)


def _scale_candidates(
    values: list[Any],
    lower_bound: int,
    upper_bound: int,
    params: Params,
    policy: dict[str, Any],
) -> list[int]:
    candidates = {int(lower_bound), int(upper_bound), int(params.Sw), int(params.Csw), int(params.Sf)}
    for value in values:
        try:
            base = int(value)
        except (TypeError, ValueError):
            continue
        for delta in (0, -params.Sf, -2 * params.Sf, params.Sf):
            candidates.add(base + delta)
        candidates.add(math.ceil(base / 2))
        candidates.add(max(lower_bound, min(upper_bound, base)))
    filtered = sorted({v for v in candidates if lower_bound <= v <= upper_bound})
    if not filtered:
        return [int(lower_bound)]
    max_count = int(policy["max_scale_candidates"])
    if len(filtered) <= max_count:
        return filtered
    anchors = [v for v in values if isinstance(v, int)]
    if not anchors:
        anchors = [lower_bound]
    ranked = sorted(
        filtered,
        key=lambda value: (
            min(abs(value - anchor) for anchor in anchors),
            abs(value - params.Sw),
            value,
        ),
    )
    keep = sorted(set(ranked[:max_count] + [lower_bound]))
    return keep


def _dedupe_scale_options(
    options: list[tuple[dict[str, int], int]]
) -> list[tuple[dict[str, int], int]]:
    seen = set()
    result = []
    for edge_scales, node_scale in options:
        key = (tuple(sorted(edge_scales.items())), node_scale)
        if key in seen:
            continue
        seen.add(key)
        result.append((edge_scales, node_scale))
    return result


def _default_edge_scales(
    tdag: Tdag,
    v: str,
    preds: list[str],
    params: Params,
    node_input_lb: int,
    edge_scale_hints: dict[str, int],
) -> dict[str, int]:
    if tdag.nodes[v]["op"] != "mul":
        target = node_input_lb
        for u in preds:
            target = max(target, _edge_scale_lb(tdag, u, v, params), edge_scale_hints.get(_edge_key(u, v), 0))
        return {u: target for u in preds}

    scales = {}
    for u in preds:
        scales[u] = max(_edge_scale_lb(tdag, u, v, params), edge_scale_hints.get(_edge_key(u, v), 0))
        if tdag.nodes[u]["op"] == "constant":
            scales[u] = params.Csw
    current = _node_input_scale(tdag, v, scales)
    if current < node_input_lb:
        adjustable = next((u for u in preds if tdag.nodes[u]["op"] != "constant"), preds[0])
        if len(preds) == 1:
            scales[adjustable] = max(scales[adjustable], math.ceil(node_input_lb / 2))
        else:
            scales[adjustable] += node_input_lb - current
    return scales


def _edge_scales_for_fixed_input(
    tdag: Tdag,
    v: str,
    preds: list[str],
    params: Params,
    node_scale: int,
) -> dict[str, int]:
    if tdag.nodes[v]["op"] != "mul":
        return {u: node_scale for u in preds}
    if len(preds) == 1:
        if node_scale % 2:
            raise PlacementError("single-input mul fixed scale must be even")
        return {preds[0]: node_scale // 2}
    constants = [u for u in preds if tdag.nodes[u]["op"] == "constant"]
    if constants:
        edge_scales = {
            u: params.Csw if u in constants else _edge_scale_lb(tdag, u, v, params)
            for u in preds
        }
        current = _node_input_scale(tdag, v, edge_scales)
        if current > node_scale:
            raise PlacementError("fixed mul input scale below constant lower bound")
        adjustable = next((u for u in preds if u not in constants), None)
        if adjustable is None:
            raise PlacementError("fixed mul input scale has no adjustable nonconstant input")
        edge_scales[adjustable] += node_scale - current
        return edge_scales
    lbs = [_edge_scale_lb(tdag, u, v, params) for u in preds]
    first = max(lbs[0], node_scale - lbs[1])
    second = node_scale - first
    if second < lbs[1]:
        raise PlacementError("fixed mul input scale below edge lower bounds")
    return {preds[0]: first, preds[1]: second}


def _common_incoming_level(
    tdag: Tdag,
    assign: Assign,
    v: str,
    preds: list[str],
    edge_scales: dict[str, int],
    node_scale: int,
    params: Params,
    preferred_level: int | None,
) -> int:
    for level in _level_candidates(params, preferred_level):
        if _incoming_level_ok(tdag, assign, v, preds, edge_scales, level, node_scale, params):
            return level
    raise PlacementError(f"no feasible incoming level for node {v}")


def _incoming_level_ok(
    tdag: Tdag,
    assign: Assign,
    v: str,
    preds: list[str],
    edge_scales: dict[str, int],
    level: int,
    node_scale: int,
    params: Params,
) -> bool:
    if not _is_decryptable(params, level, node_scale):
        return False
    for u in preds:
        scale = edge_scales[u]
        if not _is_decryptable(params, level, scale):
            return False
        if tdag.nodes[u]["op"] == "constant":
            continue
        if not _transition_ok(params, assign.v_lvl_out[u], assign.v_scl_out[u], level, scale):
            return False
    return True


def _store_incoming(
    assign: Assign,
    tdag: Tdag,
    v: str,
    preds: list[str],
    edge_scales: dict[str, int],
    level: int,
    params: Params,
) -> None:
    for u in preds:
        if tdag.nodes[u]["op"] == "constant":
            assign.v_lvl_out[u] = level
            assign.v_scl_out[u] = params.Csw if tdag.nodes[v]["op"] == "mul" else edge_scales[u]
            continue
        assign.e_lvl_out[(u, v)] = level
        assign.e_scl_out[(u, v)] = edge_scales[u]


def _choose_output_state(
    tdag: Tdag,
    v: str,
    params: Params,
    in_level: int,
    in_scale: int,
    exact_level: int | None,
    preferred_level: int | None,
    preferred_scale: int | None,
    policy: dict[str, Any],
    le: LatencyEstimator | None,
) -> tuple[int, int]:
    if exact_level is not None and exact_level < 0:
        exact_level = None
    if policy["strategy"] == "waterline_seed":
        output_scale = max(
            params.scale_lower_bound(v, tdag.nodes[v], "out"),
            preferred_scale or 0,
        )
        output_level = _transition_level(
            params,
            in_level,
            in_scale,
            output_scale,
            exact=exact_level,
            preferred=preferred_level,
        )
        return output_level, output_scale
    out_lb = max(params.scale_lower_bound(v, tdag.nodes[v], "out"), preferred_scale or 0)
    max_scale = int(policy["max_scale"])
    bases = [
        out_lb,
        preferred_scale,
        in_scale,
        in_scale - params.Sf,
        math.ceil(in_scale / 2),
        params.Sw,
        params.Csw,
        params.Sf,
    ]
    scales = _scale_candidates(bases, out_lb, max_scale, params, policy)
    if exact_level is not None:
        levels = [int(exact_level)]
    else:
        min_level = int(policy["min_internal_level"])
        levels = [
            level
            for level in _level_candidates(params, preferred_level)
            if level >= min_level
        ]
    best_no_bootstrap = None
    best_bootstrap = None
    best_any = None
    for out_scale in scales:
        for out_level in levels:
            if not _is_decryptable(params, out_level, out_scale):
                continue
            if (
                exact_level is None
                and out_level <= int(policy["min_internal_level"])
                and not params.check_res(out_level, out_scale, params.bts_lb, params.Sf)
            ):
                continue
            score = _transition_score(
                params, le, in_level, in_scale, out_level, out_scale, policy
            )
            if score is None:
                continue
            score += policy["scale_penalty"] * out_scale
            has_bootstrap = not params.check_res(in_level, in_scale, out_level, out_scale)
            item = (score, out_level, out_scale)
            if best_any is None or item[0] < best_any[0]:
                best_any = item
            if not has_bootstrap and (best_no_bootstrap is None or item[0] < best_no_bootstrap[0]):
                best_no_bootstrap = item
            if has_bootstrap and (best_bootstrap is None or item[0] < best_bootstrap[0]):
                best_bootstrap = item

    if (
        exact_level is None
        and policy["refresh_fanout_at_level_floor"]
        and in_level <= int(policy["min_internal_level"]) + 1
        and tdag.out_degree(v) > 1
        and best_bootstrap is not None
    ):
        _, out_level, out_scale = best_bootstrap
        return out_level, out_scale
    if best_no_bootstrap is not None and not policy["allow_bootstrap"]:
        _, out_level, out_scale = best_no_bootstrap
        return out_level, out_scale
    if best_any is not None:
        _, out_level, out_scale = best_any
        return out_level, out_scale
    raise PlacementError(f"no feasible output state for node {v}")


def _transition_score(
    params: Params,
    le: LatencyEstimator | None,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
    policy: dict[str, Any],
) -> float | None:
    if not _transition_ok(params, in_lvl, in_scl, out_lvl, out_scl):
        return None
    try:
        cost = float(le.resbts_cost(in_lvl, in_scl, out_lvl, out_scl)) if le is not None else 0.0
    except Exception:
        return None
    counts = _transition_count_values(params, in_lvl, in_scl, out_lvl, out_scl)
    return (
        cost
        + policy["bootstrap_penalty"] * counts["bootstrap"]
        + policy["rescale_penalty"] * counts["rescale"]
        + policy["level_drop_penalty"] * max(0, in_lvl - out_lvl)
    )


def _transition_level(
    params: Params,
    in_level: int,
    in_scale: int,
    out_scale: int,
    *,
    exact: int | None = None,
    preferred: int | None = None,
) -> int:
    if exact is not None:
        level = int(exact)
        if (
            params.lvl_lb <= level <= params.lvl_ub
            and _is_decryptable(params, level, out_scale)
            and _transition_ok(params, in_level, in_scale, level, out_scale)
        ):
            return level
        raise PlacementError(f"cannot transition to exact output level {exact}")
    for level in _level_candidates(params, preferred):
        if _is_decryptable(params, level, out_scale) and _transition_ok(
            params, in_level, in_scale, level, out_scale
        ):
            return level
    raise PlacementError("no feasible output level")


def _level_candidates(params: Params, preferred: int | None = None) -> list[int]:
    top = params.lvl_ub if preferred is None else min(params.lvl_ub, max(params.lvl_lb, preferred))
    candidates = list(range(top, params.lvl_lb - 1, -1))
    candidates.extend(level for level in range(params.lvl_ub, top, -1))
    return candidates


def _highest_decryptable_level(params: Params, scale: int) -> int:
    for level in range(params.lvl_ub, params.lvl_lb - 1, -1):
        if _is_decryptable(params, level, scale):
            return level
    raise PlacementError(f"scale {scale} is not decryptable at any level")


def _is_decryptable(params: Params, level: int, scale: int) -> bool:
    return params.lvl_lb <= level <= params.lvl_ub and scale <= params.Sf * (level - params.lvl_lb + 2) - 7


def _assert_decryptable(params: Params, level: int, scale: int) -> None:
    if not _is_decryptable(params, level, scale):
        raise PlacementError(f"(level={level}, scale={scale}) is not decryptable")


def _transition_ok(params: Params, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
    if not params.check_resbts(in_lvl, in_scl, out_lvl, out_scl):
        return False
    if params.check_res(in_lvl, in_scl, out_lvl, out_scl):
        return True
    r = max(0, round(math.ceil((params.Sf - out_scl) / params.Sf)))
    return out_lvl + r <= params.lvl_ub


def _edge_scale_lb(tdag: Tdag, u: str, v: str, params: Params) -> int:
    if tdag.nodes[u]["op"] == "constant":
        return max(params.Csw, params.scale_lower_bound(v, tdag.nodes[v], "in"))
    return max(
        params.scale_lower_bound(u, tdag.nodes[u], "out"),
        params.scale_lower_bound(v, tdag.nodes[v], "in"),
    )


def _node_input_scale(tdag: Tdag, v: str, edge_scales: dict[str, int]) -> int:
    values = list(edge_scales.values())
    if tdag.nodes[v]["op"] == "mul":
        return values[0] * 2 if len(values) == 1 else sum(values)
    if not all(scale == values[0] for scale in values):
        raise PlacementError(f"non-mul node {v} has inconsistent input scales")
    return values[0]


def _edge_key(u: str, v: str) -> str:
    return f"{u}->{v}"


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        try:
            result[str(key)] = int(item)
        except (TypeError, ValueError):
            continue
    return result


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cheap_cost(assign: Assign) -> float:
    return float(sum(assign.v_lvl_out.values()) + sum(assign.v_scl_out.values()))


def _assignment_score(assign: Assign, le: LatencyEstimator | None) -> float:
    if le is None:
        return _cheap_cost(assign)
    return float(estimate_assign(assign, le))


def _reference_metrics(
    tdag: Tdag,
    io_budgets: list[dict[str, Any]],
    le: LatencyEstimator,
) -> dict[str, float | int]:
    diagnostics: dict[str, Any] = {}
    solve_budget_batch(tdag, io_budgets, le, tdag.params, _default_policy_hints(), diagnostics)
    assignments = list(diagnostics.get("assignments", []))
    costs = list(diagnostics.get("costs", []))
    counts = _aggregate_counts(assignments)
    solved = int(diagnostics.get("solved_budgets", len(costs)))
    return {
        "avg_latency_usec": float(sum(costs) / solved) if solved else float("inf"),
        "bootstrap_count": int(counts["bootstrap"]),
        "rescale_count": int(counts["rescale"]),
        "solved_budgets": solved,
    }


def _relative_reduction(reference: int, candidate: int) -> float:
    if reference <= 0:
        return 1.0 if candidate <= 0 else 0.0
    return max(0.0, min(1.0, (reference - candidate) / reference))


def _compact_policy_summary(hints: dict[str, Any]) -> str:
    keys = [
        "strategy",
        "prefer_level_preservation",
        "allow_bootstrap",
        "allow_seed_fallback",
        "refresh_fanout_at_level_floor",
        "max_scale_candidates",
        "bootstrap_penalty",
        "rescale_penalty",
        "level_drop_penalty",
        "min_internal_level",
        "scale_penalty",
        "beam_width",
        "state_cap_per_node",
        "scale_lattice",
    ]
    summary = {key: hints.get(key) for key in keys if key in hints}
    for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
        value = hints.get(key)
        if isinstance(value, dict) and value:
            summary[f"{key}_count"] = len(value)
    return json.dumps(summary, sort_keys=True)


def _compact_invalid_reasons(diagnostics: dict[str, Any]) -> str:
    invalid_reasons = diagnostics.get("invalid_reasons", {})
    if not invalid_reasons:
        return "{}"
    top = sorted(invalid_reasons.items(), key=lambda item: (-item[1], item[0]))[:5]
    return json.dumps(dict(top), sort_keys=True)


def _profile_unmatched_targets(context: dict[str, Any]) -> list[str]:
    report = context.get("resilience", {}).get("match_report", {})
    targets = report.get("unmatched_targets", [])
    if not isinstance(targets, list):
        return []
    return [str(target) for target in targets[:20]]


def _compact_profile_plan(context: dict[str, Any]) -> str:
    best_plan = context.get("resilience", {}).get("best_plan")
    if not isinstance(best_plan, dict):
        return "{}"
    keys = [
        "action_counts",
        "expected_latency",
        "latency",
        "risk",
        "objective",
        "final_state",
    ]
    compact = {key: best_plan[key] for key in keys if key in best_plan}
    actions = best_plan.get("actions")
    if isinstance(actions, list):
        compact["actions_preview"] = actions[:5]
        compact["actions_count"] = len(actions)
    return json.dumps(compact, default=str, sort_keys=True)[:4000]


def _jsonable_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(attrs, default=str))


def _graph_summary(pdag: Tdag, params: Params) -> dict[str, Any]:
    topo = [str(node) for node in nx.topological_sort(pdag)]
    depth_to_output = _depth_to_output(pdag, topo)
    op_histogram = Counter(str(attrs.get("op", "")) for _, attrs in pdag.nodes(data=True))
    return {
        "node_count": len(pdag.nodes),
        "edge_count": len(pdag.edges),
        "op_histogram": dict(sorted(op_histogram.items())),
        "max_scale": _max_scale(params),
        "nodes": {
            str(node): {
                "op": str(attrs.get("op", "")),
                "in_degree": int(pdag.in_degree(node)),
                "out_degree": int(pdag.out_degree(node)),
                "depth_to_output": int(depth_to_output.get(str(node), 0)),
                "scale_lb_in": (
                    None
                    if attrs.get("op") == "constant"
                    else params.scale_lower_bound(str(node), attrs, "in")
                ),
                "scale_lb_out": (
                    None
                    if attrs.get("op") == "constant"
                    else params.scale_lower_bound(str(node), attrs, "out")
                ),
            }
            for node, attrs in pdag.nodes(data=True)
        },
    }


def _depth_to_output(pdag: Tdag, topo: list[str]) -> dict[str, int]:
    depth = {str(node): 0 for node in pdag.nodes}
    for node in reversed(topo):
        succ = [str(v) for v in pdag.successors(node)]
        if succ:
            depth[str(node)] = 1 + max(depth[v] for v in succ)
    return depth


def _budget_summary(io_budgets_list: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "max": None, "values": []}
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "values": sorted(set(values))[:32],
        }

    in_levels = [int(item["in_lvl"]) for item in io_budgets_list if int(item.get("in_lvl", -1)) >= 0]
    in_scales = [int(item["in_scl"]) for item in io_budgets_list if int(item.get("in_scl", -1)) >= 0]
    out_levels = [int(item["out_lvl"]) for item in io_budgets_list if int(item.get("out_lvl", -1)) >= 0]
    return {
        "count": len(io_budgets_list),
        "in_levels": stats(in_levels),
        "in_scales": stats(in_scales),
        "out_levels": stats(out_levels),
        "has_bypass_records": any("maino_v" in item for item in io_budgets_list),
    }


def _jsonable_io_budget(io_budget: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in io_budget.items() if k != "main_qbp_cost"}
    if "main_qbp_cost" in io_budget:
        result["main_qbp_cost"] = [
            {"level": key[0], "scale": key[1], "cost": value}
            for key, value in io_budget["main_qbp_cost"].items()
        ]
    return result


def _io_budget_from_json(data: dict[str, Any]) -> dict[str, Any]:
    result = {k: v for k, v in data.items() if k != "main_qbp_cost"}
    if "main_qbp_cost" in data:
        result["main_qbp_cost"] = {
            (int(item["level"]), int(item["scale"])): float(item["cost"])
            for item in data["main_qbp_cost"]
        }
    return result


class PlacementBuilder:
    """Small helper API exposed to OpenEvolve candidate programs."""

    def __init__(self, context: dict[str, Any]):
        self.context = context
        self._patches: list[dict[str, Any]] = []

    def level_preserving(self, **overrides: Any) -> dict[str, Any]:
        policy = {
            "strategy": "level_preserving",
            "prefer_level_preservation": True,
            "allow_bootstrap": False,
            "allow_seed_fallback": True,
            "refresh_fanout_at_level_floor": False,
            "max_scale_candidates": 32,
            "bootstrap_penalty": 1_000_000_000.0,
            "rescale_penalty": 0.0,
            "level_drop_penalty": 20_000_000.0,
            "min_internal_level": None,
            "scale_penalty": 0.0,
            "preferred_node_levels": {},
            "preferred_node_scales": {},
            "preferred_edge_scales": {},
        }
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def waterline_seed(self, **overrides: Any) -> dict[str, Any]:
        policy = _waterline_seed_policy()
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def depth_fanout_aware(self, **overrides: Any) -> dict[str, Any]:
        """Return a level-preserving policy seeded by graph depth and fanout.

        Candidate programs can mutate the scalar penalties and the sparse
        preferred-node maps without emitting full assignments. Orbit still
        validates the final assignment through the normal repair path.
        """
        ckks = _ckks_dict(self.context)
        nodes = self.context.get("graph_summary", {}).get("nodes", {})
        depths = [
            _safe_int(info.get("depth_to_output"), 0)
            for info in nodes.values()
            if isinstance(info, dict)
        ]
        max_depth = max(depths or [0])
        depth_threshold = max(2, max_depth // 2)
        preferred_levels: dict[str, int] = {}
        preferred_scales: dict[str, int] = {}
        for node, info in nodes.items():
            if not isinstance(info, dict):
                continue
            op = str(info.get("op", ""))
            if op in {"constant", "input"}:
                continue
            fanout = _safe_int(info.get("out_degree"), 0)
            depth = _safe_int(info.get("depth_to_output"), 0)
            scale_lb = _safe_int(info.get("scale_lb_out"), int(ckks["Sw"]))
            if fanout >= 2 or depth >= depth_threshold:
                preferred_levels[str(node)] = int(ckks["lvl_ub"])
                preferred_scales[str(node)] = max(int(ckks["Sw"]), scale_lb)
        policy = {
            "strategy": "level_preserving",
            "prefer_level_preservation": True,
            "allow_bootstrap": False,
            "allow_seed_fallback": True,
            "refresh_fanout_at_level_floor": True,
            "max_scale_candidates": 40,
            "bootstrap_penalty": 900_000_000.0,
            "rescale_penalty": 250_000.0,
            "level_drop_penalty": 16_000_000.0,
            "min_internal_level": max(int(ckks["lvl_lb"]), int(ckks["bts_lb"]) + 1),
            "scale_penalty": 1_000.0,
            "preferred_node_levels": preferred_levels,
            "preferred_node_scales": preferred_scales,
            "preferred_edge_scales": {},
        }
        for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
            values = overrides.pop(key, None)
            if isinstance(values, dict):
                policy[key].update(values)
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def latency_beam(self, **overrides: Any) -> dict[str, Any]:
        policy = self.depth_fanout_aware()["policy"]
        policy.update(
            {
                "strategy": "latency_beam",
                "beam_width": 4,
                "state_cap_per_node": 8,
                "allow_seed_fallback": True,
                "refresh_fanout_at_level_floor": True,
                "bootstrap_penalty": 750_000_000.0,
                "rescale_penalty": 500_000.0,
                "level_drop_penalty": 12_000_000.0,
                "scale_lattice": "waterline_sf",
            }
        )
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def low_scale_frontier(self, **overrides: Any) -> dict[str, Any]:
        """Policy that keeps partition boundary scales low when CKKS permits it."""
        policy = _low_scale_frontier_policy()
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def portfolio(self, *policies: dict[str, Any], **overrides: Any) -> dict[str, Any]:
        normalized = [_with_default_policy(policy) for policy in policies if isinstance(policy, dict)]
        if not normalized:
            normalized = [self.depth_fanout_aware()["policy"]]
        base = dict(normalized[0])
        base.update(overrides)
        return {
            "api_version": "placement-builder-v1",
            "policy": base,
            "portfolio": normalized,
            "placement_records": [],
            "patches": list(self._patches),
        }

    def set_policy(self, key: str, value: Any) -> "PlacementBuilder":
        self._patches.append({"op": "set_policy", "key": str(key), "value": value})
        return self

    def prefer_node_level(self, node: str, level: int) -> "PlacementBuilder":
        self._patches.append({"op": "prefer_node_level", "node": str(node), "level": int(level)})
        return self

    def prefer_node_scale(self, node: str, scale: int) -> "PlacementBuilder":
        self._patches.append({"op": "prefer_node_scale", "node": str(node), "scale": int(scale)})
        return self

    def prefer_edge_scale(self, u: str, v: str, scale: int) -> "PlacementBuilder":
        self._patches.append(
            {"op": "prefer_edge_scale", "u": str(u), "v": str(v), "scale": int(scale)}
        )
        return self

    def target_layer(self, layer: str, operation: str = "prefer_high_level") -> "PlacementBuilder":
        self._patches.append({"op": "target_layer", "layer": str(layer), "operation": str(operation)})
        return self

    def disable_seed_fallback(self) -> "PlacementBuilder":
        self._patches.append({"op": "disable_seed_fallback"})
        return self

    def build(self, fallback_policy: dict[str, Any] | None = None) -> dict[str, Any]:
        policy = fallback_policy or self.depth_fanout_aware()["policy"]
        return {
            "api_version": "placement-builder-v1",
            "policy": policy,
            "placement_records": [],
            "patches": list(self._patches),
        }

    def placement_records(
        self, records: list[dict[str, Any]], fallback_policy: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return {
            "api_version": "placement-builder-v1",
            "policy": fallback_policy or self.level_preserving()["policy"],
            "placement_records": records,
            "patches": list(self._patches),
        }


def valid_transition(context_or_params: Any, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
    ckks = _ckks_dict(context_or_params)
    sf = int(ckks["Sf"])
    bts_lb = int(ckks["bts_lb"])
    bts_ub = int(ckks["bts_ub"])
    lvl_lb = int(ckks["lvl_lb"])
    lvl_ub = int(ckks["lvl_ub"])
    max_scale = int(ckks["max_scale"])

    def decryptable(level: int, scale: int) -> bool:
        return (
            lvl_lb <= level <= lvl_ub
            and 0 <= scale <= max_scale
            and scale <= sf * (level - lvl_lb + 2) - 7
        )

    def check_res(a_lvl: int, a_scl: int, b_lvl: int, b_scl: int) -> bool:
        return a_lvl >= b_lvl and sf * a_lvl - a_scl >= sf * b_lvl - b_scl

    if not decryptable(in_lvl, in_scl) or not decryptable(out_lvl, out_scl):
        return False
    if check_res(in_lvl, in_scl, out_lvl, out_scl):
        return True
    if not check_res(in_lvl, in_scl, bts_lb, sf):
        return False
    r = max(0, math.ceil((sf - out_scl) / sf))
    return bts_lb < out_lvl + r <= bts_ub and check_res(out_lvl + r, sf, out_lvl, out_scl)


def transition_cost(context_or_params: Any, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> float:
    if not valid_transition(context_or_params, in_lvl, in_scl, out_lvl, out_scl):
        return float("inf")
    ckks = _ckks_dict(context_or_params)
    sf = int(ckks["Sf"])
    bootstrap = 0 if (in_lvl >= out_lvl and sf * in_lvl - in_scl >= sf * out_lvl - out_scl) else 1
    rescale = max(0, int(math.ceil((in_scl - out_scl) / max(sf, 1))))
    return float(bootstrap * 1_000_000_000 + rescale * 1_000_000 + max(0, in_lvl - out_lvl))


def candidate_scales(
    context_or_params: Any, values: list[Any], lower_bound: int | None = None, max_count: int = 32
) -> list[int]:
    ckks = _ckks_dict(context_or_params)
    params = _ParamsProxy(ckks)
    policy = {"max_scale_candidates": max_count}
    lb = int(lower_bound if lower_bound is not None else ckks["Sw"])
    return _scale_candidates(values, lb, int(ckks["max_scale"]), params, policy)


def serialize_assignment_record(pdag_name: str, io_budget: dict[str, Any], assign: Assign) -> dict[str, Any]:
    return {
        "pdag": pdag_name,
        "in_lvl": io_budget.get("in_lvl", -1),
        "in_scl": io_budget.get("in_scl", -1),
        "out_lvl": io_budget.get("out_lvl", -1),
        "assignment": _serialize_assign(assign),
    }


def _ckks_dict(context_or_params: Any) -> dict[str, Any]:
    if isinstance(context_or_params, Params):
        return {
            "Sf": context_or_params.Sf,
            "Sw": context_or_params.Sw,
            "Csw": context_or_params.Csw,
            "lvl_lb": context_or_params.lvl_lb,
            "lvl_ub": context_or_params.lvl_ub,
            "bts_lb": context_or_params.bts_lb,
            "bts_ub": context_or_params.bts_ub,
            "max_scale": context_or_params.max_scale(),
        }
    if isinstance(context_or_params, dict):
        ckks = context_or_params.get("constraints", {}).get("ckks", {})
        return {
            "Sf": ckks.get("rescaling_factor", context_or_params.get("params", {}).get("Sf", 51)),
            "Sw": ckks.get("input_waterline", context_or_params.get("params", {}).get("Sw", 40)),
            "Csw": ckks.get("constant_waterline", context_or_params.get("params", {}).get("Csw", 40)),
            "lvl_lb": ckks.get("level_lower_bound", context_or_params.get("params", {}).get("lvl_lb", 1)),
            "lvl_ub": ckks.get("level_upper_bound", context_or_params.get("params", {}).get("lvl_ub", 16)),
            "bts_lb": ckks.get("bootstrap_level_lower_bound", context_or_params.get("params", {}).get("bts_lb", 3)),
            "bts_ub": ckks.get("bootstrap_level_upper_bound", context_or_params.get("params", {}).get("bts_ub", 16)),
            "max_scale": ckks.get("max_scale", 131),
        }
    raise TypeError("expected Params or OpenEvolve context")


class _ParamsProxy:
    def __init__(self, ckks: dict[str, Any]):
        self.Sf = int(ckks["Sf"])
        self.Sw = int(ckks["Sw"])
        self.Csw = int(ckks["Csw"])
        self.lvl_lb = int(ckks["lvl_lb"])
        self.lvl_ub = int(ckks["lvl_ub"])
        self.bts_lb = int(ckks["bts_lb"])
        self.bts_ub = int(ckks["bts_ub"])


def _load_candidate_hints(program_path: Path, context: dict[str, Any]) -> dict[str, Any]:
    _validate_candidate_source(program_path)
    spec = importlib.util.spec_from_file_location("orbit_openevolve_candidate", program_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "place"):
        return {}
    hints = module.place(context)
    return _normalize_candidate_hints(hints, context)


def _validate_candidate_source(program_path: Path) -> None:
    source = program_path.read_text(encoding="utf-8")
    for token in BANNED_CANDIDATE_TOKENS:
        if token in source:
            raise PlacementError(f"candidate program uses banned token {token!r}")


def _normalize_candidate_hints(value: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    if "policy" not in value and "placement_records" not in value and "portfolio" not in value and "patches" not in value:
        return _apply_patch_vocabulary(_with_default_policy(value), value.get("patches", []), context)
    policy = value.get("policy") if isinstance(value.get("policy"), dict) else {}
    result = _with_default_policy(policy)
    records = value.get("placement_records", [])
    if isinstance(records, list):
        result["placement_records"] = records
    portfolio = value.get("portfolio", [])
    if isinstance(portfolio, list):
        result["portfolio"] = [
            _apply_patch_vocabulary(_with_default_policy(item), value.get("patches", []), context)
            for item in portfolio
            if isinstance(item, dict)
        ]
    result = _apply_patch_vocabulary(result, value.get("patches", []), context)
    result["api_version"] = value.get("api_version", "placement-builder-v1")
    return result


def _apply_patch_vocabulary(
    hints: dict[str, Any],
    patches: Any,
    context: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(patches, list):
        return hints
    result = dict(hints)
    result.setdefault("preferred_node_levels", {})
    result.setdefault("preferred_node_scales", {})
    result.setdefault("preferred_edge_scales", {})
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        op = str(patch.get("op", ""))
        if op == "set_policy":
            key = str(patch.get("key", ""))
            if key in _PATCHABLE_POLICY_KEYS:
                result[key] = patch.get("value")
        elif op == "prefer_node_level":
            node = str(patch.get("node", ""))
            if node:
                result["preferred_node_levels"][node] = patch.get("level")
        elif op == "prefer_node_scale":
            node = str(patch.get("node", ""))
            if node:
                result["preferred_node_scales"][node] = patch.get("scale")
        elif op == "prefer_edge_scale":
            edge = str(patch.get("edge") or _edge_key(str(patch.get("u", "")), str(patch.get("v", ""))))
            if "->" in edge:
                result["preferred_edge_scales"][edge] = patch.get("scale")
        elif op == "disable_seed_fallback":
            result["allow_seed_fallback"] = False
        elif op == "target_layer":
            _apply_layer_patch(result, patch, context)
    return result


_PATCHABLE_POLICY_KEYS = {
    "strategy",
    "prefer_level_preservation",
    "allow_bootstrap",
    "allow_seed_fallback",
    "refresh_fanout_at_level_floor",
    "max_scale_candidates",
    "bootstrap_penalty",
    "rescale_penalty",
    "level_drop_penalty",
    "min_internal_level",
    "scale_penalty",
    "beam_width",
    "state_cap_per_node",
    "scale_lattice",
}


def _apply_layer_patch(
    hints: dict[str, Any],
    patch: dict[str, Any],
    context: dict[str, Any] | None,
) -> None:
    if not context:
        return
    layer = str(patch.get("layer", ""))
    operation = str(patch.get("operation", "prefer_high_level"))
    ckks = _ckks_dict(context)
    for node, attrs in context.get("tdag", {}).get("nodes", {}).items():
        if layer and layer not in _node_location(attrs):
            continue
        if str(attrs.get("op", "")) in {"input", "constant"}:
            continue
        if operation in {"prefer_high_level", "avoid_bootstrap", "protect"}:
            hints["preferred_node_levels"][str(node)] = int(ckks["lvl_ub"])
        if operation in {"prefer_waterline_scale", "protect"}:
            hints["preferred_node_scales"][str(node)] = int(ckks["Sw"])


def _serialize_assign(assign: Assign) -> dict[str, Any]:
    def edge_map(data: dict[tuple[str, str], Any]) -> dict[str, Any]:
        return {_edge_key(u, v): value for (u, v), value in data.items()}

    return {
        "v_lvl_out": dict(assign.v_lvl_out),
        "v_scl_out": dict(assign.v_scl_out),
        "v_lvl_in": dict(assign.v_lvl_in),
        "v_scl_in": dict(assign.v_scl_in),
        "e_lvl_out": edge_map(assign.e_lvl_out),
        "e_scl_out": edge_map(assign.e_scl_out),
    }


def _aggregate_counts(assignments: list[Assign]) -> dict[str, int]:
    counts = {"bootstrap": 0, "rescale": 0}
    for assign in assignments:
        counts["bootstrap"] += _maintenance_counts(assign)["bootstrap"]
        counts["rescale"] += _maintenance_counts(assign)["rescale"]
    return counts


def _maintenance_counts(assign: Assign) -> dict[str, int]:
    params = assign.params
    counts = {"bootstrap": 0, "rescale": 0}
    for v in assign.tdag.nodes:
        if assign.tdag.nodes[v]["op"] == "constant":
            continue
        in_lvl, in_scl = assign.get_v_in_lvl_scl(v)
        counts = _count_transition(params, in_lvl, in_scl, assign.v_lvl_out[v], assign.v_scl_out[v], counts)
    for u, v in assign.tdag.edges:
        if assign.tdag.nodes[u]["op"] == "constant":
            continue
        counts = _count_transition(
            params,
            assign.v_lvl_out[u],
            assign.v_scl_out[u],
            assign.e_lvl_out[(u, v)],
            assign.e_scl_out[(u, v)],
            counts,
        )
    return counts


def _boundary_quality(params: Params, final_io_choice: tuple[int, int, int, int] | None) -> float:
    if final_io_choice is None:
        return 0.0
    _in_lvl, _in_scl, out_lvl, out_scl = final_io_choice
    level_span = max(1, params.lvl_ub - params.lvl_lb)
    level_score = max(0.0, min(1.0, (out_lvl - params.lvl_lb) / level_span))
    target_scale = max(params.Sw, params.Sf)
    scale_score = 1.0 / (1.0 + abs(out_scl - target_scale) / max(1, params.Sf))
    decrypt_score = 1.0 if _is_decryptable(params, out_lvl, out_scl) else 0.0
    return 0.55 * level_score + 0.30 * scale_score + 0.15 * decrypt_score


def _maintenance_locations(assign: Assign) -> dict[str, dict[str, int]]:
    locations = {"bootstrap": Counter(), "rescale": Counter()}
    params = assign.params
    for v in assign.tdag.nodes:
        if assign.tdag.nodes[v]["op"] == "constant":
            continue
        in_lvl, in_scl = assign.get_v_in_lvl_scl(v)
        delta = _transition_count_values(params, in_lvl, in_scl, assign.v_lvl_out[v], assign.v_scl_out[v])
        location = _node_location(assign.tdag.nodes[v])
        locations["bootstrap"][location] += delta["bootstrap"]
        locations["rescale"][location] += delta["rescale"]
    for u, v in assign.tdag.edges:
        if assign.tdag.nodes[u]["op"] == "constant":
            continue
        delta = _transition_count_values(
            params,
            assign.v_lvl_out[u],
            assign.v_scl_out[u],
            assign.e_lvl_out[(u, v)],
            assign.e_scl_out[(u, v)],
        )
        location = _node_location(assign.tdag.nodes[v])
        locations["bootstrap"][location] += delta["bootstrap"]
        locations["rescale"][location] += delta["rescale"]
    return {
        "bootstrap": {key: value for key, value in locations["bootstrap"].items() if value},
        "rescale": {key: value for key, value in locations["rescale"].items() if value},
    }


def _bottleneck_summary(locations: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    combined: Counter[str] = Counter()
    for key, count in locations.get("bootstrap", {}).items():
        combined[key] += 4 * int(count)
    for key, count in locations.get("rescale", {}).items():
        combined[key] += int(count)
    return [
        {
            "location": key,
            "score": score,
            "bootstraps": int(locations.get("bootstrap", {}).get(key, 0)),
            "rescales": int(locations.get("rescale", {}).get(key, 0)),
        }
        for key, score in combined.most_common(10)
    ]


def _bootstrap_locations(assign: Assign) -> dict[str, int]:
    return _maintenance_locations(assign)["bootstrap"]


def _node_location(attrs: dict[str, Any]) -> str:
    comment = str(attrs.get("comment", ""))
    layer = "none"
    op = str(attrs.get("op", "unknown"))
    scope = ""
    for part in comment.split(";"):
        if part.startswith("layer="):
            layer = part.split("=", 1)[1]
        elif part.startswith("op="):
            op = part.split("=", 1)[1]
        elif part.startswith("scope="):
            scope = part.split("=", 1)[1]
    if layer == "none" and scope:
        pieces = [piece for piece in scope.split(".") if piece]
        if "layer" in pieces:
            idx = pieces.index("layer")
            if idx + 1 < len(pieces):
                layer = f"layer.{pieces[idx + 1]}"
        elif pieces:
            layer = ".".join(pieces[: min(3, len(pieces))])
    return f"layer={layer};op={op}"


def _count_transition(
    params: Params,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
    counts: dict[str, int],
) -> dict[str, int]:
    delta = _transition_count_values(params, in_lvl, in_scl, out_lvl, out_scl)
    counts["bootstrap"] += delta["bootstrap"]
    counts["rescale"] += delta["rescale"]
    return counts


def _transition_count_values(
    params: Params,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
) -> dict[str, int]:
    return {
        "bootstrap": 0 if params.check_res(in_lvl, in_scl, out_lvl, out_scl) else 1,
        "rescale": max(0, int(math.ceil((in_scl - out_scl) / max(params.Sf, 1)))),
    }


def _profile_risk(assignments: list[Assign], params: Params) -> float:
    if params.resilience_profile is None:
        return 0.0
    risk = 0.0
    for assign in assignments:
        for v in assign.tdag.nodes:
            if assign.tdag.nodes[v]["op"] == "constant":
                continue
            for port, scale in (("in", assign.get_v_in_lvl_scl(v)[1]), ("out", assign.v_scl_out[v])):
                scale_lb = params.scale_lower_bound(v, assign.tdag.nodes[v], port)
                risk += max(0.0, float(scale_lb - scale))
    return risk


def _resilience_context(pdag: Tdag, params: Params) -> dict[str, Any]:
    if params.resilience_profile is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "profile": params.resilience_profile.describe(),
        "match_report": params.resilience_profile.match_report(pdag),
        "best_plan": params.resilience_profile.best_plan,
        "artifacts": params.resilience_profile.artifacts,
    }
