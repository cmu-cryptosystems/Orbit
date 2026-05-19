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
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx

from ...assignment import Assign
from ...latency_estimator import LatencyEstimator, estimate_assign
from ...params.params import Params
from ...tdag import Tdag
from .noise_estimator import estimate_compile_result_noise, write_noise_summary


class PlacementError(Exception):
    pass


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_API_BASE = "https://api.openai.com/v1"
CONTEXT_SCHEMA_VERSION = "orbit-openevolve-placement-context-v2"
COMPILE_CONTEXT_SCHEMA_VERSION = "orbit-openevolve-compile-harness-v2"
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
    actual_cost: float | None = None


@dataclass(frozen=True)
class _OpenEvolveModelSpec:
    provider: str
    name: str
    api_base: str
    api_key_env: str
    weight: float


@dataclass(frozen=True)
class MCTSAction:
    """A bounded placement action evaluated by the bootstrap-MCTS scheduler."""

    name: str
    policy: dict[str, Any]
    prior: float = 0.0


@dataclass
class _MCTSNodeStats:
    visits: int = 0
    reward_sum: float = 0.0
    best_reward: float = float("-inf")
    best_assign: Assign | None = None
    best_cost: float = float("inf")
    invalid_reasons: Counter = field(default_factory=Counter)


class OpenEvolvePlacementWorker:
    """OpenEvolve-backed QBP worker.

    With ``openevolve_iterations == 0`` this uses a deterministic no-LLM policy.
    Positive iteration counts run OpenEvolve once for the whole budget batch,
    then validate the best candidate through the same deterministic repair path.
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
        elif getattr(self.params, "openevolve_search_mode", "bootstrap-mcts") == "bootstrap-mcts":
            hints = _bootstrap_mcts_seed_policy(self.params)
        else:
            hints = _zero_iteration_portfolio_hints()
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
        model_specs = self._openevolve_model_specs()
        api_base = model_specs[0].api_base
        multi_model = len(model_specs) > 1
        config.random_seed = self.params.openevolve_seed
        config.llm.api_base = api_base
        config.llm.timeout = self.params.openevolve_llm_timeout_sec
        config.llm.retries = self.params.openevolve_llm_retries
        config.llm.retry_delay = self.params.openevolve_llm_retry_delay_sec
        config.llm.max_tokens = max(256, min(4096, int(self.params.openevolve_llm_max_tokens)))
        config.llm.temperature = 0.3
        config.llm.models = [
            self._llm_model_config(LLMModelConfig, spec, config, multi_model)
            for spec in model_specs
        ]
        config.llm.evaluator_models = [
            self._llm_model_config(LLMModelConfig, model_specs[0], config, multi_model)
        ]
        config.database.feature_dimensions = [
            "final_latency_usec",
            "boundary_quality",
            "bootstrap_count",
            "rescale_count",
            "fallback_selected_budgets",
            "profile_risk",
            "placement_runtime_sec",
            "estimated_precision_bits",
            "output_margin_bits",
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

    def _llm_model_config(self, cls, spec: _OpenEvolveModelSpec, config, multi_model: bool):
        return cls(
            name=spec.name,
            api_base=spec.api_base,
            api_key=self._model_api_key(spec) if multi_model else None,
            weight=spec.weight,
            timeout=config.llm.timeout,
            retries=config.llm.retries,
            retry_delay=config.llm.retry_delay,
            max_tokens=config.llm.max_tokens,
            temperature=config.llm.temperature,
            random_seed=self.params.openevolve_seed,
        )

    def _openevolve_model_specs(self) -> list[_OpenEvolveModelSpec]:
        specs = [
            _OpenEvolveModelSpec(
                provider=self.params.openevolve_provider,
                name=self.params.openevolve_model,
                api_base=self._provider_api_base(
                    self.params.openevolve_provider,
                    self.params.openevolve_api_base,
                    "--openevolve-api-base",
                ),
                api_key_env=self.params.openevolve_api_key_env or "OPENAI_API_KEY",
                weight=max(0.0, float(getattr(self.params, "openevolve_primary_weight", 1.0))),
            )
        ]
        secondary_provider = getattr(self.params, "openevolve_secondary_provider", "none")
        if (
            secondary_provider != "none"
            and float(getattr(self.params, "openevolve_secondary_weight", 0.0)) > 0.0
        ):
            specs.append(
                _OpenEvolveModelSpec(
                    provider=secondary_provider,
                    name=getattr(self.params, "openevolve_secondary_model", "gpt-5.5"),
                    api_base=self._provider_api_base(
                        secondary_provider,
                        getattr(self.params, "openevolve_secondary_api_base", None),
                        "--openevolve-secondary-api-base",
                    ),
                    api_key_env=getattr(
                        self.params,
                        "openevolve_secondary_api_key_env",
                        "OPENAI_API_KEY",
                    )
                    or "OPENAI_API_KEY",
                    weight=max(0.0, float(getattr(self.params, "openevolve_secondary_weight", 0.25))),
                )
            )
        total = sum(spec.weight for spec in specs)
        if total <= 0.0:
            specs[0] = _OpenEvolveModelSpec(
                specs[0].provider,
                specs[0].name,
                specs[0].api_base,
                specs[0].api_key_env,
                1.0,
            )
            return specs
        return [
            _OpenEvolveModelSpec(spec.provider, spec.name, spec.api_base, spec.api_key_env, spec.weight / total)
            for spec in specs
        ]

    def _openevolve_api_base(self) -> str:
        return self._provider_api_base(
            self.params.openevolve_provider,
            self.params.openevolve_api_base,
            "--openevolve-api-base",
        )

    def _provider_api_base(self, provider: str, explicit: str | None, option_name: str) -> str:
        if explicit:
            return explicit
        if provider == "gemini":
            return GEMINI_API_BASE
        if provider == "openai":
            return OPENAI_API_BASE
        raise ValueError(
            f"{option_name} is required when the corresponding OpenEvolve provider is custom."
        )

    def _normalize_openevolve_seed(self, config) -> None:
        config.random_seed = self.params.openevolve_seed
        if hasattr(config, "database"):
            config.database.random_seed = self.params.openevolve_seed
        if hasattr(config, "llm"):
            config.llm.update_model_params({"random_seed": self.params.openevolve_seed})

    def _openevolve_key_env_candidates(self) -> list[str]:
        candidates = []
        for spec in self._openevolve_model_specs():
            candidates.extend(self._key_env_candidates_for_spec(spec))
        deduped = []
        for candidate in candidates:
            if candidate and candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _key_env_candidates_for_spec(self, spec: _OpenEvolveModelSpec) -> list[str]:
        candidates = []
        if spec.provider == "gemini":
            if spec.api_key_env and spec.api_key_env != "OPENAI_API_KEY":
                candidates.append(spec.api_key_env)
            candidates.extend(["GEMINI_API_KEY", "OPENAI_API_KEY"])
        elif spec.provider == "openai":
            if spec.api_key_env:
                candidates.append(spec.api_key_env)
            candidates.append("OPENAI_API_KEY")
        elif spec.api_key_env:
            candidates.append(spec.api_key_env)
        return candidates

    def _model_api_key(self, spec: _OpenEvolveModelSpec) -> str:
        for env_name in self._key_env_candidates_for_spec(spec):
            value = os.environ.get(env_name)
            if value:
                return value
        checked = ", ".join(self._key_env_candidates_for_spec(spec))
        raise ValueError(
            f"OpenEvolve model {spec.name!r} ({spec.provider}) needs an API key in one of: {checked}"
        )

    @contextmanager
    def _openevolve_runtime_env(self):
        specs = self._openevolve_model_specs()
        resolved: list[tuple[_OpenEvolveModelSpec, str]] = []
        missing: list[str] = []
        for spec in specs:
            found_name = next(
                (name for name in self._key_env_candidates_for_spec(spec) if os.environ.get(name)),
                None,
            )
            if found_name is None:
                missing.append(
                    f"{spec.name} checked {', '.join(self._key_env_candidates_for_spec(spec))}"
                )
            else:
                resolved.append((spec, found_name))
        if missing:
            raise ValueError(
                "OpenEvolve runtime search needs an API key in one of these "
                f"environment variables: {'; '.join(missing)}. "
                "Use --openevolve-iterations 0 for deterministic no-LLM placement."
            )
        old_openai_key = os.environ.get("OPENAI_API_KEY")
        bridged = len(specs) == 1 and resolved[0][1] != "OPENAI_API_KEY"
        if bridged:
            os.environ["OPENAI_API_KEY"] = os.environ[resolved[0][1]]
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
        "boundary_scale_penalty": 0.2,
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

from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder, PlacementMCTS


# EVOLVE-BLOCK-START
def place(context):
    """Return an Orbit placement algorithm description.

    The candidate may either return explicit placement records or a policy
    built from Orbit's helper API. Orbit validates every assignment and owns
    final repair into Assign objects.

    During sampled evolution, Orbit first rewards candidates that directly solve
    more requested budget records. Latency and bootstrap count only become
    competitive after a candidate covers enough budgets without seed fallback.
    """
    builder = PlacementBuilder(context)
    if context.get("harness", {}).get("search_mode") == "bootstrap-mcts":
        target_bootstraps = context.get("harness", {}).get("target_bootstrap_count", 9)
        return PlacementMCTS(context).low_bootstrap_seed(
            target_bootstraps=target_bootstraps,
            rollout_budget=48,
            exploration_weight=1.4,
            max_repair_bootstraps=max(4, int(target_bootstraps or 0)),
        )
    target_bootstraps = context.get("harness", {}).get("target_bootstrap_count", 0)
    portfolio = [
        builder.budget_fulfillment_beam()["policy"],
        builder.budget_fulfillment_beam(
            bootstrap_penalty=4_000_000_000.0,
            selection_bootstrap_penalty=500_000_000.0,
            beam_width=6,
            state_cap_per_node=16,
            max_scale_candidates=24,
        )["policy"],
        builder.low_scale_frontier()["policy"],
        builder.latency_beam(
            allow_bootstrap=True,
            budget_aggressive=True,
            selection_bootstrap_penalty=250_000_000.0,
        )["policy"],
        builder.profile_layer_refresh(include_attention=True)["policy"],
        builder.profile_layer_refresh(include_attention=False)["policy"],
        builder.bootstrap_safe_margin(min_internal_level=6)["policy"],
        builder.bootstrap_safe_margin(min_internal_level=8)["policy"],
        builder.noise_guarded_refresh()["policy"],
    ]
    global_policy = builder.low_scale_frontier()["policy"]
    unit_policies = []
    for unit in context.get("placement_units", [])[:16]:
        selector = unit.get("selector", {})
        if selector.get("kind") == "nonlinear":
            kind = selector.get("nonlinear_kind")
            if kind in {"attention_softmax", "reciprocal", "norm"}:
                policy = builder.bootstrap_safe_margin(min_internal_level=8)["policy"]
            else:
                policy = builder.low_scale_frontier(refresh_fanout_at_level_floor=True)["policy"]
            unit_policies.append(builder.unit_policy(selector, policy))
    if target_bootstraps:
        portfolio[0]["selection_bootstrap_penalty"] = 500_000_000.0
        portfolio[1]["selection_bootstrap_penalty"] = 750_000_000.0
    return builder.unit_portfolio(
        global_policy,
        *unit_policies,
        portfolio=portfolio,
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
    placement_units = _placement_units(pdag, params, params.openevolve_max_unit_samples)
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
        "placement_units": placement_units,
        "unit_op_histogram": _unit_op_histogram(placement_units),
        "unit_budget_summary": _unit_budget_summary(placement_units, io_budgets_list),
        "unit_hotspots": [],
        "unit_resilience_summary": _unit_resilience_summary(placement_units, params),
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
            "openevolve_search_mode": getattr(params, "openevolve_search_mode", "bootstrap-mcts"),
            "openevolve_granularity": params.openevolve_granularity,
            "openevolve_leniency": params.openevolve_leniency,
            "openevolve_max_unit_samples": params.openevolve_max_unit_samples,
            "openevolve_reference_json": params.openevolve_reference_json,
            "openevolve_finalists": params.openevolve_finalists,
            "openevolve_budget_aggressive": getattr(params, "openevolve_budget_aggressive", True),
            "openevolve_target_bootstrap_count": getattr(params, "openevolve_target_bootstrap_count", 0),
            "noise_estimator": params.noise_estimator,
            "noise_estimator_min_output_margin_bits": params.noise_estimator_min_output_margin_bits,
            "noise_estimator_alpha": params.noise_estimator_alpha,
            "noise_estimator_max_trace_message_bits": getattr(
                params, "noise_estimator_max_trace_message_bits", 20.0
            ),
            "noise_estimator_require_trace_safe": getattr(
                params, "noise_estimator_require_trace_safe", False
            ),
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
            "ilp_semantics": _ilp_semantics_summary(params),
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
        openevolve_search_mode=pdata.get("openevolve_search_mode", "bootstrap-mcts"),
        openevolve_granularity=pdata.get("openevolve_granularity", "layer-nonlinear"),
        openevolve_leniency=pdata.get("openevolve_leniency", "repair"),
        openevolve_max_unit_samples=pdata.get("openevolve_max_unit_samples", 64),
        openevolve_reference_json=pdata.get("openevolve_reference_json"),
        openevolve_finalists=pdata.get("openevolve_finalists", 3),
        openevolve_budget_aggressive=pdata.get("openevolve_budget_aggressive", True),
        openevolve_target_bootstrap_count=pdata.get("openevolve_target_bootstrap_count", 0),
        noise_estimator=pdata.get("noise_estimator", "finalists"),
        noise_estimator_min_output_margin_bits=pdata.get(
            "noise_estimator_min_output_margin_bits",
            2.0,
        ),
        noise_estimator_alpha=pdata.get("noise_estimator_alpha", 14.0),
        noise_estimator_max_trace_message_bits=pdata.get(
            "noise_estimator_max_trace_message_bits",
            20.0,
        ),
        noise_estimator_require_trace_safe=pdata.get(
            "noise_estimator_require_trace_safe",
            False,
        ),
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
        "search_mode": getattr(params, "openevolve_search_mode", "bootstrap-mcts"),
        "granularity": params.openevolve_granularity,
        "leniency": params.openevolve_leniency,
        "max_unit_samples": params.openevolve_max_unit_samples,
        "initial_prev_cost": {-1: {params.Sw: 0}},
        "reference_json": _load_reference_json(params.openevolve_reference_json),
        "finalists": params.openevolve_finalists,
        "budget_aggressive": getattr(params, "openevolve_budget_aggressive", True),
        "target_bootstrap_count": getattr(params, "openevolve_target_bootstrap_count", 0),
        "noise_estimator": params.noise_estimator,
        "noise_estimator_min_output_margin_bits": params.noise_estimator_min_output_margin_bits,
        "noise_estimator_alpha": params.noise_estimator_alpha,
        "noise_estimator_max_trace_message_bits": getattr(
            params, "noise_estimator_max_trace_message_bits", 20.0
        ),
        "noise_estimator_require_trace_safe": getattr(
            params, "noise_estimator_require_trace_safe", False
        ),
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
    if reference.get("sampled_budget_tasks"):
        context["sampled_budget_tasks"] = reference["sampled_budget_tasks"]
    context["placement_profile"] = reference.get("bottleneck_summary", [])
    context["unit_hotspots"] = _unit_hotspots_from_profile(
        context.get("placement_units", []),
        reference.get("bottleneck_summary", []),
    )
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
    if getattr(params, "openevolve_reuse_output", False):
        best_code = _load_reusable_best_code(output_dir)
        if not best_code:
            if not params.openevolve_fail_open:
                raise PlacementError(
                    f"--openevolve-reuse-output found no reusable programs under {output_dir}"
                )
            _write_recovery_summary(
                root,
                PlacementError(
                    f"--openevolve-reuse-output found no reusable programs under {output_dir}"
                ),
            )
            return _recover_compile_hints(output_dir, context, initial_hints, params)
    else:
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
        best_code = result.best_code
    best_program = root / "best_program.py"
    best_program.write_text(best_code, encoding="utf-8")
    try:
        finalist_hints = _run_full_bundle_finalists(
            root,
            output_dir,
            context,
            best_code,
            params,
            initial_hints,
        )
        if finalist_hints is not None:
            return finalist_hints
        return _load_candidate_hints(best_program, context)
    except Exception as exc:
        if not params.openevolve_fail_open:
            raise
        _write_recovery_summary(root, exc)
        return _recover_compile_hints(output_dir, context, initial_hints, params, best_code)
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
        sampled_reasons = _sampled_policy_static_reasons(context, hints)
        if sampled_reasons:
            result = _invalid_compile_result("compile_sampled_policy_gate", sampled_reasons)
            _record_compile_trace(context, Path(program_path), hints, result, "STATIC_ONLY")
            return result
        eval_suite = str(context.get("harness", {}).get("eval_suite", "polybert-sampled"))
        eval_hints = _compile_hints_for_eval_suite(hints, eval_suite)
        result = _evaluate_compile_hints(context, eval_hints, suppress_output=True)
        result["static"] = static
        diagnostics = result.get("diagnostics", {})
        requested_budgets = max(1, int(diagnostics.get("requested_budgets", 1)))
        candidate_solved = int(diagnostics.get("candidate_solved_budgets", 0))
        solved_budgets = int(diagnostics.get("solved_budgets", requested_budgets if result["valid"] else 0))
        candidate_validity = candidate_solved / requested_budgets
        effective_validity = solved_budgets / requested_budgets
        requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
        solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
        fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
        invalid_groups = int(diagnostics.get("invalid_boundary_groups", 0) or 0)
        boundary_group_validity = solved_groups / requested_groups
        candidate_qbp_coverage = candidate_groups / requested_groups
        repair_count = _repair_count(eval_hints)
        unit_coverage = _unit_coverage(context, eval_hints)
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
        fallback_score = 1.0 / (
            1.0 + result["fallback_selected_budgets"] + fallback_groups
        )
        repair_score = 1.0 / (1.0 + repair_count)
        boundary_score = max(0.0, min(1.0, float(result.get("boundary_quality", 0.0))))
        target_bootstrap_count = _context_target_bootstrap_count(context)
        target_bootstrap_score = _target_bootstrap_score(
            target_bootstrap_count,
            result["bootstrap_count"],
            reference.get("bootstrap_count"),
        )
        noise_estimate = _fast_compile_noise_estimate(context, result)
        noise_score = 1.0 if noise_estimate.get("valid", False) else 0.0
        reserve_score = _reserve_quality_score(result.get("reserve_summary", {}))
        quality_score = (
            0.36 * latency_score
            + 0.08 * bootstrap_score
            + 0.10 * target_bootstrap_score
            + 0.05 * rescale_score
            + 0.08 * boundary_score
            + 0.07 * risk_score
            + 0.08 * noise_score
            + 0.09 * reserve_score
            + 0.04 * fallback_score
            + 0.04 * runtime_score
        )
        if not result["valid"] or boundary_group_validity < 1.0:
            bootstrap_frontier = target_bootstrap_score if candidate_validity > 0.0 else 0.0
            combined_score = min(
                0.999,
                0.14 * effective_validity
                + 0.18 * boundary_group_validity
                + 0.14 * candidate_qbp_coverage
                + 0.44 * bootstrap_frontier
                + 0.10 * quality_score
            )
        elif result["fallback_selected_budgets"] > 0 or fallback_groups > 0 or repair_count > 0:
            if _context_budget_aggressive(context):
                combined_score = min(
                    0.999,
                    0.10 * effective_validity
                    + 0.22 * boundary_group_validity
                    + 0.34 * candidate_qbp_coverage
                    + 0.14 * quality_score
                    + 0.08 * unit_coverage
                    + 0.06 * repair_score,
                )
            else:
                combined_score = min(
                    0.999,
                    0.30 * effective_validity
                    + 0.25 * candidate_validity
                    + 0.25 * quality_score
                    + 0.12 * unit_coverage
                    + 0.08 * repair_score,
                )
        else:
            combined_score = 1.0 + quality_score + 0.05 * unit_coverage + 0.02 * candidate_qbp_coverage
        evaluation = {
            "metrics": {
                "combined_score": float(combined_score),
                "validity": float(result["validity"]),
                "effective_validity": float(effective_validity),
                "candidate_validity": float(candidate_validity),
                "boundary_group_validity": float(boundary_group_validity),
                "candidate_qbp_coverage": float(candidate_qbp_coverage),
                "repair_count": float(repair_count),
                "unit_coverage": float(unit_coverage),
                "latency_score": float(latency_score),
                "boundary_score": float(boundary_score),
                "fallback_score": float(fallback_score),
                "bootstrap_score": float(bootstrap_score),
                "target_bootstrap_score": float(target_bootstrap_score),
                "rescale_score": float(rescale_score),
                "final_latency_usec": float(result["final_latency_usec"] if result["valid"] else 0.0),
                "boundary_quality": float(boundary_score),
                "bootstrap_count": float(result["bootstrap_count"]),
                "target_bootstrap_count": float(target_bootstrap_count),
                "rescale_count": float(result["rescale_count"]),
                "profile_risk": float(result["profile_risk"]),
                "placement_runtime_sec": float(result["placement_runtime_sec"]),
                "fallback_selected_budgets": float(result["fallback_selected_budgets"]),
                "requested_boundary_groups": float(requested_groups),
                "solved_boundary_groups": float(solved_groups),
                "candidate_solved_boundary_groups": float(candidate_groups),
                "fallback_selected_groups": float(fallback_groups),
                "invalid_boundary_groups": float(invalid_groups),
                "estimated_precision_bits": float(noise_estimate.get("estimated_precision_bits", 0.0)),
                "output_margin_bits": float(noise_estimate.get("output_margin_bits", 0.0)),
                "reserve_score": float(reserve_score),
                "min_decryptability_reserve_bits": float(
                    _finite_float(
                        result.get("reserve_summary", {}).get("min_decryptability_reserve_bits"),
                        0.0,
                    )
                ),
                "min_transition_reserve_bits": float(
                    _finite_float(
                        result.get("reserve_summary", {}).get("min_transition_reserve_bits"),
                        0.0,
                    )
                ),
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
                        "noise_score": noise_score,
                        "reserve_score": reserve_score,
                        "fallback_score": fallback_score,
                        "runtime_score": runtime_score,
                    },
                    sort_keys=True,
                ),
                "noise_estimator": json.dumps(noise_estimate, sort_keys=True)[:4000],
                "reserve_summary": json.dumps(
                    result.get("reserve_summary", {}), sort_keys=True
                )[:4000],
                "bootstrap_locations": json.dumps(result["bootstrap_locations"], sort_keys=True),
                "rescale_locations": json.dumps(result["rescale_locations"], sort_keys=True),
                "bottleneck_summary": json.dumps(result.get("bottleneck_summary", []), sort_keys=True),
                "selected_output_state": json.dumps(result["selected_output_state"], sort_keys=True),
                "boundary_group_validity": f"{solved_groups}/{requested_groups}",
                "candidate_qbp_coverage": f"{candidate_groups}/{requested_groups}",
                "fallback_selected_groups": f"{fallback_groups}/{requested_groups}",
                "invalid_boundary_groups": f"{invalid_groups}/{requested_groups}",
                "invalid_reasons": _compact_invalid_reasons(result["diagnostics"]),
                "candidate_invalid_reasons": _compact_invalid_reasons(
                    {"invalid_reasons": result["diagnostics"].get("candidate_invalid_reasons", {})}
                ),
                "candidate_only_avg_latency_usec": (
                    f"{sum(result['diagnostics'].get('candidate_costs', [])) / len(result['diagnostics'].get('candidate_costs', [])):.3f}"
                    if result["diagnostics"].get("candidate_costs")
                    else "none"
                ),
                "selected_source_counts": json.dumps(
                    result["diagnostics"].get("selected_source_counts", {}), sort_keys=True
                ),
                "assignment_count_summary": json.dumps(
                    result["diagnostics"].get("assignment_count_summary", {}),
                    sort_keys=True,
                ),
                "boundary_group_count_summary": json.dumps(
                    result["diagnostics"].get("boundary_group_count_summary", {}),
                    sort_keys=True,
                ),
                "policy_summary": _compact_policy_summary(eval_hints),
                "unmatched_resilience_targets": json.dumps(
                    _profile_unmatched_targets(context), sort_keys=True
                ),
                "reference_json": json.dumps(context.get("harness", {}).get("reference_json", {}), sort_keys=True)[:4000],
                "replay_log_tail": result["log_tail"],
                "patchgate": json.dumps(static, sort_keys=True),
                "repair_summary": json.dumps(_unit_policy_summary(context, eval_hints), sort_keys=True),
                "per_unit_score_table": json.dumps(
                    _per_unit_score_table(context, result.get("bottleneck_summary", []), eval_hints),
                    sort_keys=True,
                )[:4000],
            },
        }
        _record_compile_trace(context, Path(program_path), eval_hints, evaluation, "CLEAR_ONLY")
        return evaluation
    except Exception as exc:
        tb = traceback.format_exc()
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
                "placement_runtime_sec": 0.0,
                "fallback_selected_budgets": 0.0,
                "estimated_precision_bits": 0.0,
                "output_margin_bits": 0.0,
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
    eval_suite = context.get("harness", {}).get("eval_suite", "polybert-sampled")
    if eval_suite != "polybert-full" and context.get("sampled_budget_tasks"):
        return _evaluate_sampled_budget_tasks(context, hints, suppress_output=suppress_output)

    from .iterative_partition import solve_partition
    from .qbp_manager import QBPManager

    tdag = tdag_from_context(context)
    params = tdag.params
    params.openevolve_iterations = 0
    params.openevolve_harness = "compile"
    params.openevolve_compile_hints = _compile_hints_for_eval_suite(hints, eval_suite)
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = eval_suite
    le = LatencyEstimator(params)
    qbp_manager = QBPManager(params, le)
    log_buffer = io.StringIO()
    start = time.time()
    try:
        stream = log_buffer if suppress_output else None
        try:
            if stream is None:
                partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            else:
                with redirect_stdout(stream), redirect_stderr(stream):
                    partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
        except (AssertionError, RuntimeError) as exc:
            if params.bpsdepth is None or not _is_retryable_bypass_failure(exc):
                raise
            if stream is None:
                print(
                    "OpenEvolve compile replay bypass failed; retrying with bypass disabled. "
                    f"Original error: {exc}"
                )
            else:
                stream.write(
                    "OpenEvolve compile replay bypass failed; retrying with bypass disabled. "
                    f"Original error: {exc}\n"
                )
            params.bpsdepth = None
            qbp_manager = QBPManager(params, le)
            if stream is None:
                partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            else:
                with redirect_stdout(stream), redirect_stderr(stream):
                    partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
        if partition_result is None:
            raise PlacementError("compile replay produced no valid final partitioning")
        io_to_assign, io_to_cost = partition_result
        if io_to_assign is None or io_to_cost is None:
            raise PlacementError("compile replay produced no QBP solution")
        final_io_choice, final_cost = _select_final_choice(io_to_cost)
        if final_io_choice is None:
            raise PlacementError("compile replay produced no final IO choice")
        assign = io_to_assign[final_io_choice[:2]][final_io_choice[2:]]
        assign.check_assign()
        counts = _maintenance_counts(assign)
        locations = _maintenance_locations(assign)
        reserve_summary = _assignment_reserve_summary(assign, tdag, params)
        diagnostics = _collect_qbp_diagnostics(qbp_manager)
        requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
        return {
            "valid": True,
            "validity": 1.0,
            "boundary_group_validity": float(
                int(diagnostics.get("solved_boundary_groups", 0) or 0) / requested_groups
            ),
            "candidate_qbp_coverage": float(
                int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
                / requested_groups
            ),
            "final_latency_usec": float(estimate_assign(assign, le)),
            "aggregated_partition_cost_usec": float(final_cost),
            "bootstrap_count": int(counts["bootstrap"]),
            "rescale_count": int(counts["rescale"]),
            "boundary_quality": float(_boundary_quality(params, final_io_choice)),
            "profile_risk": float(_profile_risk([assign], params)),
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": int(diagnostics.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(
                diagnostics.get("fallback_selected_boundary_groups", 0)
            ),
            "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0)),
            "selected_output_state": {
                "in_lvl": final_io_choice[0],
                "in_scl": final_io_choice[1],
                "out_lvl": final_io_choice[2],
                "out_scl": final_io_choice[3],
            },
            "reserve_summary": reserve_summary,
            "assignment": _serialize_assign(assign),
            "bootstrap_locations": locations["bootstrap"],
            "rescale_locations": locations["rescale"],
            "bottleneck_summary": _bottleneck_summary(locations),
            "diagnostics": diagnostics,
            "sampled_budget_tasks": _sampled_budget_tasks_from_qbp_manager(
                qbp_manager,
                params,
            ),
            "log_tail": log_buffer.getvalue()[-3000:],
        }
    except Exception as exc:
        tb = traceback.format_exc()
        diagnostics = _collect_qbp_diagnostics(qbp_manager)
        assignments = list(diagnostics.get("assignments", []))
        if (
            eval_suite != "polybert-full"
            and assignments
            and isinstance(exc, PlacementError)
            and "compile replay produced no valid final partitioning" in str(exc)
        ):
            diagnostics["sampled_progress_only"] = True
            diagnostics["invalid_reasons"] = {f"{type(exc).__name__}: {str(exc)[:240]}": 1}
            diagnostics["traceback"] = tb[-4000:]
            return _sampled_progress_compile_result(
                context,
                tdag,
                params,
                diagnostics,
                start,
                log_buffer,
                sampled_budget_tasks=_sampled_budget_tasks_from_qbp_manager(
                    qbp_manager,
                    params,
                ),
            )
        count_summary = _assignment_count_summary(assignments)
        counts = {
            "bootstrap": count_summary["avg_bootstrap"],
            "rescale": count_summary["avg_rescale"],
        }
        diagnostics["assignment_count_summary"] = count_summary
        return {
            "valid": False,
            "validity": 0.0,
            "boundary_group_validity": 0.0,
            "candidate_qbp_coverage": 0.0,
            "final_latency_usec": float("inf"),
            "aggregated_partition_cost_usec": float("inf"),
            "bootstrap_count": float(counts["bootstrap"]),
            "rescale_count": float(counts["rescale"]),
            "boundary_quality": 0.0,
            "profile_risk": 1.0,
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": int(diagnostics.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(
                diagnostics.get("fallback_selected_boundary_groups", 0)
            ),
            "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0)),
            "selected_output_state": {},
            "reserve_summary": {},
            "assignment": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "diagnostics": {
                **diagnostics,
                "invalid_reasons": {f"{type(exc).__name__}: {str(exc)[:240]}": 1},
                "traceback": tb[-4000:],
            },
            "sampled_budget_tasks": _sampled_budget_tasks_from_qbp_manager(
                qbp_manager,
                params,
            ),
            "log_tail": (log_buffer.getvalue() + "\n" + tb)[-4000:],
        }


def _sampled_progress_compile_result(
    context: dict[str, Any],
    tdag: Tdag,
    params: Params,
    diagnostics: dict[str, Any],
    start: float,
    log_buffer: io.StringIO,
    sampled_budget_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return sampled-budget progress when sampled replay lacks a full DP path.

    PolyBERT sampled mode intentionally evaluates a stratified subset of budget
    records. That subset can be useful for OpenEvolve feedback while still being
    insufficient for Orbit's final global DP path. Full-bundle finalist replay
    remains strict; this path only makes sampled evolution score direct coverage
    and repair/fallback pressure instead of flattening every candidate to an
    invalid compile.
    """

    assignments = list(diagnostics.get("assignments", []))
    costs = [float(item) for item in diagnostics.get("costs", [])]
    count_summary = _assignment_count_summary(assignments)
    group_summary = _boundary_group_count_summary(
        list(diagnostics.get("boundary_group_summaries", []))
    )
    sampled_bootstrap_count = (
        group_summary["avg_bootstrap"]
        if group_summary["boundary_group_count"]
        else count_summary["avg_bootstrap"]
    )
    sampled_rescale_count = (
        group_summary["avg_rescale"]
        if group_summary["boundary_group_count"]
        else count_summary["avg_rescale"]
    )
    best_assign = None
    if assignments:
        if costs and len(costs) == len(assignments):
            best_idx = min(range(len(costs)), key=lambda idx: costs[idx])
            best_assign = assignments[best_idx]
        else:
            best_assign = assignments[0]
    locations = _maintenance_locations(best_assign) if best_assign is not None else {"bootstrap": {}, "rescale": {}}
    reserve_summary = _aggregate_reserve_summary(assignments, tdag, params)
    requested = max(1, int(diagnostics.get("requested_budgets", len(assignments)) or 1))
    solved = int(diagnostics.get("solved_budgets", len(assignments)) or 0)
    requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
    solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
    candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
    return {
        "valid": bool(solved),
        "validity": float(solved / requested),
        "boundary_group_validity": float(solved_groups / requested_groups),
        "candidate_qbp_coverage": float(candidate_groups / requested_groups),
        "sampled_progress_only": True,
        "final_latency_usec": float(sum(costs) / len(costs)) if costs else 0.0,
        "aggregated_partition_cost_usec": float(sum(costs)) if costs else 0.0,
        "bootstrap_count": float(sampled_bootstrap_count),
        "rescale_count": float(sampled_rescale_count),
        "boundary_quality": 0.0,
        "profile_risk": float(_profile_risk(assignments, params)) if assignments else 1.0,
        "placement_runtime_sec": time.time() - start,
        "fallback_selected_budgets": int(diagnostics.get("fallback_selected_budgets", 0)),
        "fallback_selected_groups": int(
            diagnostics.get("fallback_selected_boundary_groups", 0)
        ),
        "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0)),
        "selected_output_state": {},
        "reserve_summary": reserve_summary,
        "assignment": _serialize_assign(best_assign) if best_assign is not None else {},
        "bootstrap_locations": locations["bootstrap"],
        "rescale_locations": locations["rescale"],
        "bottleneck_summary": _bottleneck_summary(locations),
        "diagnostics": diagnostics,
        "sampled_budget_tasks": sampled_budget_tasks or [],
        "log_tail": log_buffer.getvalue()[-3000:],
    }


def _sampled_budget_tasks_from_qbp_manager(
    qbp_manager,
    params: Params,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for index, item in enumerate(getattr(qbp_manager, "openevolve_budget_tasks", [])):
        pdag = item.get("pdag") if isinstance(item, dict) else None
        budgets = item.get("io_budgets", []) if isinstance(item, dict) else []
        if pdag is None or not budgets:
            continue
        tasks.append(
            {
                "index": index,
                "kind": str(item.get("kind", "normal")),
                "context": build_context(pdag, budgets, params),
            }
        )
    return tasks


def _evaluate_sampled_budget_tasks(
    context: dict[str, Any],
    hints: dict[str, Any],
    *,
    suppress_output: bool,
) -> dict[str, Any]:
    """Evaluate cached sampled QBP batches without rerunning partition replay."""

    start = time.time()
    eval_suite = str(context.get("harness", {}).get("eval_suite", "polybert-sampled"))
    eval_hints = _compile_hints_for_eval_suite(hints, eval_suite)
    total: dict[str, Any] = {
        "requested_budgets": 0,
        "solved_budgets": 0,
        "candidate_solved_budgets": 0,
        "fallback_solved_budgets": 0,
        "fallback_selected_budgets": 0,
        "requested_boundary_groups": 0,
        "solved_boundary_groups": 0,
        "partial_boundary_groups": 0,
        "candidate_solved_boundary_groups": 0,
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
        "candidate_improved_budgets": 0,
        "candidate_invalid_reasons": {},
        "invalid_reasons": {},
        "candidate_costs": [],
        "costs": [],
        "assignments": [],
        "selected_source_counts": {},
        "boundary_group_summaries": [],
        "sampled_task_count": 0,
        "sampled_direct_budget_eval": False,
        "sampled_qbp_group_eval": True,
    }
    reserve_summaries: list[dict[str, Any]] = []
    best_assign: Assign | None = None
    best_cost = float("inf")
    log_buffer = io.StringIO()
    stdout_context = redirect_stdout(log_buffer) if suppress_output else nullcontext()
    stderr_context = redirect_stderr(log_buffer) if suppress_output else nullcontext()

    def merge_counts(key: str, item: dict[str, Any]) -> None:
        total[key] += int(item.get(key, 0))

    try:
        with stdout_context, stderr_context:
            for task in context.get("sampled_budget_tasks", []):
                if not isinstance(task, dict) or not isinstance(task.get("context"), dict):
                    continue
                task_context = task["context"]
                task_tdag = tdag_from_context(task_context)
                task_params = task_tdag.params
                task_params.openevolve_iterations = 0
                task_params.openevolve_harness = "compile"
                task_params.openevolve_compile_hints = eval_hints
                task_params.openevolve_evaluating_candidate = True
                task_params.openevolve_eval_suite = eval_suite
                task_le = LatencyEstimator(task_params)
                budgets = [
                    _io_budget_from_json(item)
                    for item in task_context.get("io_budgets", [])
                    if isinstance(item, dict)
                ]
                if not budgets:
                    continue
                task_diag: dict[str, Any] = {}
                solve_budget_batch(
                    task_tdag,
                    budgets,
                    task_le,
                    task_params,
                    eval_hints,
                    task_diag,
                )
                total["sampled_task_count"] += 1
                for key in (
                    "requested_budgets",
                    "solved_budgets",
                    "candidate_solved_budgets",
                    "fallback_solved_budgets",
                    "fallback_selected_budgets",
                    "requested_boundary_groups",
                    "solved_boundary_groups",
                    "partial_boundary_groups",
                    "candidate_solved_boundary_groups",
                    "fallback_selected_boundary_groups",
                    "invalid_boundary_groups",
                    "candidate_improved_budgets",
                ):
                    merge_counts(key, task_diag)
                if len(total["boundary_group_summaries"]) < 512:
                    remaining = 512 - len(total["boundary_group_summaries"])
                    total["boundary_group_summaries"].extend(
                        list(task_diag.get("boundary_group_summaries", []))[:remaining]
                    )
                for key in ("costs", "candidate_costs", "assignments"):
                    total[key].extend(task_diag.get(key, []))
                for source, count in task_diag.get("selected_source_counts", {}).items():
                    total["selected_source_counts"][source] = (
                        total["selected_source_counts"].get(source, 0) + int(count)
                    )
                for reason_key in ("invalid_reasons", "candidate_invalid_reasons"):
                    for reason, count in task_diag.get(reason_key, {}).items():
                        total[reason_key][reason] = total[reason_key].get(reason, 0) + int(count)
                task_assignments = list(task_diag.get("assignments", []))
                if task_assignments:
                    reserve_summaries.append(
                        _aggregate_reserve_summary(task_assignments, task_tdag, task_params)
                    )
                task_costs = [float(item) for item in task_diag.get("costs", [])]
                if task_costs and len(task_costs) == len(task_assignments):
                    idx = min(range(len(task_costs)), key=lambda pos: task_costs[pos])
                    if task_costs[idx] < best_cost:
                        best_cost = task_costs[idx]
                        best_assign = task_assignments[idx]

        assignments = list(total["assignments"])
        costs = [float(item) for item in total["costs"]]
        count_summary = _assignment_count_summary(assignments)
        group_summary = _boundary_group_count_summary(
            list(total.get("boundary_group_summaries", []))
        )
        total["assignment_count_summary"] = count_summary
        total["boundary_group_count_summary"] = group_summary
        sampled_bootstrap_count = (
            group_summary["avg_bootstrap"]
            if group_summary["boundary_group_count"]
            else count_summary["avg_bootstrap"]
        )
        sampled_rescale_count = (
            group_summary["avg_rescale"]
            if group_summary["boundary_group_count"]
            else count_summary["avg_rescale"]
        )
        locations = (
            _maintenance_locations(best_assign)
            if best_assign is not None
            else {"bootstrap": {}, "rescale": {}}
        )
        params = tdag_from_context(context).params
        reserve_summary = _merge_reserve_summaries(reserve_summaries)
        requested = max(1, int(total.get("requested_budgets", 0) or 1))
        solved = int(total.get("solved_budgets", 0) or 0)
        requested_groups = max(1, int(total.get("requested_boundary_groups", 0) or 1))
        solved_groups = int(total.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(total.get("candidate_solved_boundary_groups", 0) or 0)
        return {
            "valid": bool(solved),
            "validity": float(solved / requested),
            "boundary_group_validity": float(solved_groups / requested_groups),
            "candidate_qbp_coverage": float(candidate_groups / requested_groups),
            "sampled_progress_only": True,
            "final_latency_usec": float(sum(costs) / len(costs)) if costs else 0.0,
            "aggregated_partition_cost_usec": float(sum(costs)) if costs else 0.0,
            "bootstrap_count": float(sampled_bootstrap_count),
            "rescale_count": float(sampled_rescale_count),
            "boundary_quality": 0.0,
            "profile_risk": float(_profile_risk(assignments, params)) if assignments else 1.0,
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": int(total.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(total.get("fallback_selected_boundary_groups", 0)),
            "invalid_boundary_groups": int(total.get("invalid_boundary_groups", 0)),
            "selected_output_state": {},
            "reserve_summary": reserve_summary,
            "assignment": _serialize_assign(best_assign) if best_assign is not None else {},
            "bootstrap_locations": locations["bootstrap"],
            "rescale_locations": locations["rescale"],
            "bottleneck_summary": _bottleneck_summary(locations),
            "diagnostics": total,
            "sampled_budget_tasks": [],
            "log_tail": log_buffer.getvalue()[-3000:],
        }
    except Exception as exc:
        total["invalid_reasons"] = {f"{type(exc).__name__}: {str(exc)[:240]}": 1}
        total["traceback"] = traceback.format_exc()[-4000:]
        return {
            "valid": False,
            "validity": 0.0,
            "boundary_group_validity": 0.0,
            "candidate_qbp_coverage": 0.0,
            "sampled_progress_only": True,
            "final_latency_usec": float("inf"),
            "aggregated_partition_cost_usec": float("inf"),
            "bootstrap_count": 0.0,
            "rescale_count": 0.0,
            "boundary_quality": 0.0,
            "profile_risk": 1.0,
            "placement_runtime_sec": time.time() - start,
            "fallback_selected_budgets": int(total.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(total.get("fallback_selected_boundary_groups", 0)),
            "invalid_boundary_groups": int(total.get("invalid_boundary_groups", 0)),
            "selected_output_state": {},
            "reserve_summary": {},
            "assignment": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": total,
            "sampled_budget_tasks": [],
            "log_tail": (log_buffer.getvalue() + "\n" + traceback.format_exc())[-4000:],
        }


def _merge_reserve_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {
            "min_decryptability_reserve_bits": None,
            "min_transition_reserve_bits": None,
            "min_bootstrap_input_reserve_bits": None,
        }

    def min_present(key: str) -> int | float | None:
        values = [item.get(key) for item in summaries if item.get(key) is not None]
        return min(values) if values else None

    return {
        "assignment_count": sum(int(item.get("assignment_count", 0) or 0) for item in summaries),
        "min_decryptability_reserve_bits": min_present("min_decryptability_reserve_bits"),
        "min_transition_reserve_bits": min_present("min_transition_reserve_bits"),
        "min_bootstrap_input_reserve_bits": min_present("min_bootstrap_input_reserve_bits"),
        "first_assignment": summaries[0].get("first_assignment", {}),
    }


def _is_retryable_bypass_failure(exc: BaseException) -> bool:
    text = str(exc)
    return "QBP not found" in text or "No placement solution found for the whole DAG" in text


def _compile_hints_for_eval_suite(hints: dict[str, Any], eval_suite: str) -> dict[str, Any]:
    normalized = _with_default_policy(hints)
    if eval_suite == "polybert-full":
        return normalized
    return _bounded_sampled_policy(normalized)


def _bounded_sampled_policy(hints: dict[str, Any]) -> dict[str, Any]:
    bounded = _with_default_policy(hints)
    portfolio = bounded.pop("portfolio", None)
    budget_aggressive = _bool_hint(bounded.get("budget_aggressive"), False)
    # Sampled evolution is intentionally lenient: failed direct candidate budgets
    # are repaired through the deterministic seed, then scored via
    # candidate_validity and fallback_selected_budgets. Finalist replay disables
    # this so selected MLIRs still come from a concrete policy.
    bounded["allow_seed_fallback"] = True
    bounded["allow_bootstrap"] = _bool_hint(bounded.get("allow_bootstrap"), budget_aggressive)
    bounded["refresh_fanout_at_level_floor"] = True
    bounded["max_scale_candidates"] = min(
        24 if budget_aggressive else 16,
        _int_hint(bounded.get("max_scale_candidates"), 20 if budget_aggressive else 16),
    )
    bounded["state_cap_per_node"] = min(
        16 if budget_aggressive else 8,
        _int_hint(bounded.get("state_cap_per_node"), 12 if budget_aggressive else 8),
    )
    bounded["beam_width"] = min(
        6 if budget_aggressive else 4,
        _int_hint(bounded.get("beam_width"), 5 if budget_aggressive else 4),
    )
    if str(bounded.get("strategy")) == "bootstrap_mcts":
        bounded["mcts_rollout_budget"] = min(
            4 if budget_aggressive else 2,
            _int_hint(bounded.get("mcts_rollout_budget"), 4 if budget_aggressive else 2),
        )
        bounded["mcts_action_cap"] = min(
            4 if budget_aggressive else 2,
            _int_hint(bounded.get("mcts_action_cap"), 4 if budget_aggressive else 2),
        )
        bounded["mcts_max_repair_bootstraps"] = min(
            32 if budget_aggressive else 4,
            _int_hint(
                bounded.get("mcts_max_repair_bootstraps"),
                32 if budget_aggressive else 4,
            ),
        )
        bounded["boundary_state_cap"] = min(
            1,
            _int_hint(bounded.get("boundary_state_cap"), 1),
        )
    if isinstance(portfolio, list) and portfolio:
        sampled_portfolio = [
            _bounded_sampled_policy(item)
            for item in portfolio
            if isinstance(item, dict)
        ]
        sampled_portfolio.sort(
            key=lambda policy: (
                not _bool_hint(policy.get("budget_aggressive"), False),
                str(policy.get("strategy")) != "latency_beam",
            )
        )
        bounded["portfolio"] = sampled_portfolio[:2]
    return bounded


def _finalist_hints_for_full_bundle(hints: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded single-policy finalist replay candidate.

    Full-bundle replay may examine several policies from a candidate portfolio,
    but each variant is evaluated as one concrete scheduler. That keeps final
    selection faithful to a real placement algorithm instead of hiding a large
    per-budget search behind a single evolved program.
    """

    bounded = _with_default_policy(hints)
    bounded.pop("portfolio", None)
    bounded["allow_seed_fallback"] = False
    budget_aggressive = _bool_hint(bounded.get("budget_aggressive"), False)
    bounded["max_scale_candidates"] = min(
        24 if budget_aggressive else 16,
        _int_hint(bounded.get("max_scale_candidates"), 20 if budget_aggressive else 16),
    )
    bounded["state_cap_per_node"] = min(
        16 if budget_aggressive else 8,
        _int_hint(bounded.get("state_cap_per_node"), 12 if budget_aggressive else 8),
    )
    if str(bounded.get("strategy")) == "latency_beam":
        bounded["beam_width"] = min(
            6 if budget_aggressive else 3,
            _int_hint(bounded.get("beam_width"), 5 if budget_aggressive else 3),
        )
    if str(bounded.get("strategy")) == "bootstrap_mcts":
        bounded["mcts_rollout_budget"] = min(
            64 if budget_aggressive else 32,
            _int_hint(bounded.get("mcts_rollout_budget"), 48 if budget_aggressive else 24),
        )
        bounded["mcts_action_cap"] = min(
            24 if budget_aggressive else 16,
            _int_hint(bounded.get("mcts_action_cap"), 16 if budget_aggressive else 12),
        )
        bounded["mcts_max_repair_bootstraps"] = min(
            16 if budget_aggressive else 8,
            _int_hint(bounded.get("mcts_max_repair_bootstraps"), 12 if budget_aggressive else 6),
        )
        bounded["boundary_state_cap"] = min(
            4 if budget_aggressive else 3,
            _int_hint(bounded.get("boundary_state_cap"), 4 if budget_aggressive else 3),
        )
    return bounded


def _finalist_hint_variants(hints: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    variants: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def add(label: str, item: dict[str, Any]) -> None:
        bounded = _finalist_hints_for_full_bundle(item)
        digest = _hint_digest(
            {
                key: value
                for key, value in bounded.items()
                if key not in {"api_version", "placement_records", "portfolio"}
            }
        )
        if digest in seen:
            return
        seen.add(digest)
        variants.append((label, bounded))

    add("primary", hints)
    portfolio = hints.get("portfolio")
    if isinstance(portfolio, list):
        for idx, item in enumerate(portfolio):
            if not isinstance(item, dict):
                continue
            add(f"portfolio_{idx}", item)
            if len(variants) >= 6:
                break
    return variants


def _hint_digest(hints: dict[str, Any]) -> str:
    try:
        payload = json.dumps(hints, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        payload = repr(hints)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sampled_policy_static_reasons(context: dict[str, Any], hints: dict[str, Any]) -> list[str]:
    if context.get("harness", {}).get("eval_suite") != "polybert-sampled":
        return []
    if _context_leniency(context) == "repair":
        return []
    sampled = _compile_hints_for_eval_suite(hints, "polybert-sampled")
    reasons = []
    for idx, policy in enumerate(_portfolio_policies(sampled)):
        prefix = "sampled policy" if idx == 0 else f"sampled portfolio[{idx - 1}]"
        strategy = str(policy.get("strategy", "level_preserving"))
        if strategy not in {"level_preserving", "latency_beam"}:
            reasons.append(f"{prefix} rejects strategy {strategy!r}")
        if _bool_hint(policy.get("allow_bootstrap"), False):
            reasons.append(f"{prefix} rejects explicit bootstrap search")
        if _bool_hint(policy.get("allow_seed_fallback"), False):
            reasons.append(f"{prefix} rejects seed fallback")
        if not _bool_hint(policy.get("refresh_fanout_at_level_floor"), False):
            reasons.append(f"{prefix} requires low-scale frontier refresh")
        if _int_hint(policy.get("max_scale_candidates"), 32) > 16:
            reasons.append(f"{prefix} caps max_scale_candidates at 16")
        if _int_hint(policy.get("state_cap_per_node"), 8) > 8:
            reasons.append(f"{prefix} caps state_cap_per_node at 8")
        if _int_hint(policy.get("beam_width"), 4) > 4:
            reasons.append(f"{prefix} caps beam_width at 4")
    return reasons


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
    initial_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if (
        params.openevolve_finalists <= 0
        or sampled_context.get("harness", {}).get("eval_suite") == "polybert-full"
    ):
        return None
    finalist_dir = root / "finalists"
    finalist_dir.mkdir(parents=True, exist_ok=True)
    sampled_invalid_reason = _sampled_best_invalid_reason(output_dir)
    if sampled_invalid_reason is not None:
        fail_open = _bounded_fail_open_hints(initial_hints)
        (finalist_dir / "full_bundle_summary.json").write_text(
            json.dumps(
                {
                    "reference": {"valid": False},
                    "candidates": [],
                    "selected_index": None,
                    "skipped_full_bundle": True,
                    "skip_reason": sampled_invalid_reason,
                    "fail_open": True,
                    "fail_open_policy": _compact_policy_summary(fail_open or {}),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return fail_open
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
    estimator_records = []
    result_cache: dict[str, dict[str, Any]] = {}

    def write_progress() -> None:
        try:
            (finalist_dir / "full_bundle_progress.json").write_text(
                json.dumps(
                    {
                        "reference": full_context["reference"],
                        "candidates": summaries,
                        "updated_at": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    finalist_items: list[tuple[str, str | None, dict[str, Any] | None]] = [
        (f"candidate_{idx}", code, None)
        for idx, code in enumerate(candidates)
    ]
    if initial_hints is not None:
        finalist_items.append(("initial_seed", None, initial_hints))
    for item_idx, (label, code, preset_hints) in enumerate(finalist_items):
        program_path = finalist_dir / f"finalist_{item_idx}.py"
        if code is not None:
            program_path.write_text(code, encoding="utf-8")
        else:
            program_path.write_text(
                "# Materialized initial OpenEvolve seed policy.\n"
                "def place(context):\n"
                f"    return {preset_hints!r}\n",
                encoding="utf-8",
            )
        try:
            hints = (
                preset_hints
                if preset_hints is not None
                else _load_candidate_hints(program_path, full_context)
            )
            hint_variants = _finalist_hint_variants(hints)
        except Exception as exc:
            summaries.append(
                {
                    "index": len(summaries),
                    "label": label,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            )
            continue

        for variant_label, finalist_hints in hint_variants:
            summary_index = len(summaries)
            summary_label = label if variant_label == "primary" else f"{label}:{variant_label}"
            try:
                hint_digest = _hint_digest(finalist_hints)
                cached = result_cache.get(hint_digest)
                if cached is not None:
                    result = cached["result"]
                    noise = cached.get("noise")
                    summary = {
                        **cached["summary"],
                        "index": summary_index,
                        "label": summary_label,
                        "cached_from_index": cached["index"],
                    }
                else:
                    static = _static_validate_hints(full_context, finalist_hints)
                    if not static["valid"]:
                        raise PlacementError("; ".join(static["reasons"][:4]))
                    result = _evaluate_compile_hints(full_context, finalist_hints, suppress_output=True)
                    summary = {
                        "index": summary_index,
                        "label": summary_label,
                        "valid": result["valid"],
                        "final_latency_usec": result["final_latency_usec"],
                        "boundary_quality": result.get("boundary_quality", 0.0),
                        "bootstrap_count": result["bootstrap_count"],
                        "rescale_count": result["rescale_count"],
                        "fallback_selected_budgets": result["fallback_selected_budgets"],
                        "fallback_selected_groups": result.get("fallback_selected_groups", 0),
                        "candidate_qbp_coverage": result.get("candidate_qbp_coverage", 0.0),
                        "selected_output_state": result["selected_output_state"],
                        "reserve_summary": result.get("reserve_summary", {}),
                        "bootstrap_locations": result.get("bootstrap_locations", {}),
                        "rescale_locations": result.get("rescale_locations", {}),
                    }
                    noise = _noise_estimator_for_finalist(full_context, result, params)
                    if noise is not None:
                        summary["noise_estimator"] = noise
                        estimator_records.append({"index": summary_index, "label": summary_label, **noise})
                    result_cache[hint_digest] = {
                        "index": summary_index,
                        "result": result,
                        "noise": noise,
                        "summary": dict(summary),
                    }
                if result["valid"]:
                    noise_reject = int(noise is not None and not noise.get("valid", False))
                    margin = (
                        _finite_float(noise.get("output_margin_bits"), 0.0)
                        if noise is not None
                        else 0.0
                    )
                    latency_target = _finalist_latency_target_usec(full_context)
                    margin_target = _finalist_output_margin_target_bits(full_context)
                    forced_bootstrap_floor = _finalist_forced_bootstrap_floor(full_context)
                    target_bootstrap_count = _context_target_bootstrap_count(full_context)
                    bootstrap_excess = (
                        max(0, int(result["bootstrap_count"]) - target_bootstrap_count)
                        if target_bootstrap_count > 0
                        else 0
                    )
                    plaintext_quality_reject, plaintext_quality_reason = _finalist_plaintext_quality_reject(
                        full_context
                    )
                    profile_noise_reject, profile_noise_reason = _profile_noise_metadata_reject(
                        full_context
                    )
                    noise_warning_reject, noise_warning_reason = _finalist_noise_warning_reject(
                        full_context,
                        noise,
                        params,
                    )
                    latency_reject = int(
                        latency_target is not None
                        and float(result["final_latency_usec"]) > latency_target
                    )
                    margin_reject = int(
                        noise is not None
                        and margin_target is not None
                        and margin < margin_target
                    )
                    bootstrap_reject = int(
                        forced_bootstrap_floor is not None
                        and int(result["bootstrap_count"]) < forced_bootstrap_floor
                    )
                    summary["finalist_gate"] = {
                        "latency_target_usec": latency_target,
                        "latency_reject": bool(latency_reject),
                        "output_margin_target_bits": margin_target,
                        "output_margin_reject": bool(margin_reject),
                        "forced_bootstrap_floor": forced_bootstrap_floor,
                        "target_bootstrap_count": target_bootstrap_count,
                        "target_bootstrap_excess": bootstrap_excess,
                        "bootstrap_reject": bool(bootstrap_reject),
                        "plaintext_quality_reject": plaintext_quality_reject,
                        "plaintext_quality_reason": plaintext_quality_reason,
                        "profile_noise_reject": profile_noise_reject,
                        "profile_noise_reason": profile_noise_reason,
                        "noise_warning_reject": noise_warning_reject,
                        "noise_warning_reason": noise_warning_reason,
                    }
                    item = (
                        int(plaintext_quality_reject),
                        int(profile_noise_reject),
                        int(noise_warning_reject),
                        noise_reject,
                        margin_reject,
                        int(
                            result["fallback_selected_budgets"] > 0
                            or int(result.get("fallback_selected_groups", 0) or 0) > 0
                        ),
                        bootstrap_reject,
                        latency_reject,
                        bootstrap_excess,
                        float(result["final_latency_usec"]),
                        -float(margin),
                        summary_index,
                        finalist_hints,
                        summary,
                    )
                    if best is None or item[:12] < best[:12]:
                        best = item
            except Exception as exc:
                summary = {
                    "index": summary_index,
                    "label": summary_label,
                    "valid": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }
            summaries.append(summary)
            write_progress()
    (finalist_dir / "full_bundle_summary.json").write_text(
        json.dumps(
            {
                "reference": full_context["reference"],
                "candidates": summaries,
                "selected_index": None if best is None else best[11],
                "finalist_gate": {
                    "latency_target_usec": _finalist_latency_target_usec(full_context),
                    "output_margin_target_bits": _finalist_output_margin_target_bits(full_context),
                    "forced_bootstrap_floor": _finalist_forced_bootstrap_floor(full_context),
                    "target_bootstrap_count": _context_target_bootstrap_count(full_context),
                },
                "noise_estimator": {
                    "mode": params.noise_estimator,
                    "min_output_margin_bits": params.noise_estimator_min_output_margin_bits,
                    "records": estimator_records,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if best is None:
        return _bounded_fail_open_hints(initial_hints)
    fallback_reject = bool(best[5])
    if (
        best[0]
        or best[1]
        or fallback_reject
        or ((best[2] or best[3]) and params.noise_estimator == "finalists")
    ):
        _write_finalist_rejection_summary(
            finalist_dir,
            best[13],
            {
                "plaintext_quality_reject": bool(best[0]),
                "profile_noise_reject": bool(best[1]),
                "fallback_selected_reject": fallback_reject,
                "noise_warning_reject": bool(best[2]),
                "noise_estimator_reject": bool(best[3]),
                "selected_index": best[11],
            },
        )
        return _bounded_fail_open_hints(initial_hints)
    write_noise_summary(
        finalist_dir / "noise_estimator_summary.json",
        {
            "selected_index": best[11],
            "selected": best[13],
            "records": estimator_records,
        },
    )
    return best[12]


def _bounded_fail_open_hints(initial_hints: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the deterministic smoke portfolio after finalist/runtime failure.

    The finalist replay path intentionally strips portfolios and seed fallback
    so that a successful evolved finalist represents one concrete placement
    algorithm. That bounded policy is not a good fail-open artifact: if every
    finalist is invalid, returning another bounded invalid policy can make the
    production compile fail after OpenEvolve has already recovered. The
    zero-iteration portfolio is the validated no-LLM smoke path, so use it as
    the final safety net.
    """

    fallback = _zero_iteration_portfolio_hints()
    fallback["fail_open_reason"] = "zero_iteration_seed_portfolio"
    if initial_hints is not None:
        fallback["recovered_from_initial_digest"] = _hint_digest(initial_hints)
    return fallback


def _sampled_best_invalid_reason(output_dir: Path) -> str | None:
    info_path = output_dir / "best" / "best_program_info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    metrics = info.get("metrics") if isinstance(info, dict) else None
    if not isinstance(metrics, dict):
        return None
    validity = _finite_float(metrics.get("validity"), 0.0)
    effective_validity = _finite_float(metrics.get("effective_validity"), validity)
    combined_score = _finite_float(metrics.get("combined_score"), 0.0)
    if validity <= 0.0 and effective_validity <= 0.0 and combined_score < 1.0:
        return "sampled_best_solved_no_budgets"
    return None


def _write_finalist_rejection_summary(
    finalist_dir: Path,
    selected_summary: dict[str, Any],
    reason: dict[str, Any],
) -> None:
    try:
        (finalist_dir / "finalist_rejection_summary.json").write_text(
            json.dumps(
                {
                    "reason": reason,
                    "selected_summary": selected_summary,
                    "fail_open": True,
                    "fallback": "bounded_initial_seed",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def _finalist_latency_target_usec(context: dict[str, Any]) -> float | None:
    reference = context.get("harness", {}).get("reference_json", {})
    if not isinstance(reference, dict):
        return None
    for key in (
        "final_latency_usec",
        "final_tdag_latency_usec",
        "tdag_latency_usec",
        "baseline_latency_usec",
    ):
        value = reference.get(key)
        if value is not None:
            return _finite_float(value, float("inf"))
    for key in (
        "final_latency_sec",
        "final_tdag_latency_sec",
        "tdag_latency_sec",
        "baseline_latency_sec",
    ):
        value = reference.get(key)
        if value is not None:
            return _finite_float(value, float("inf")) * 1_000_000.0
    return None


def _finalist_output_margin_target_bits(context: dict[str, Any]) -> float | None:
    reference = context.get("harness", {}).get("reference_json", {})
    if not isinstance(reference, dict):
        return None
    for key in (
        "required_output_margin_bits",
        "min_output_margin_bits",
        "precision_margin_bits",
    ):
        value = reference.get(key)
        if value is not None:
            target = _finite_float(value, float("nan"))
            if math.isfinite(target):
                return target
    return None


def _finalist_forced_bootstrap_floor(context: dict[str, Any]) -> int | None:
    reference = context.get("harness", {}).get("reference_json", {})
    if not isinstance(reference, dict):
        return None
    for key in (
        "force_min_bootstrap_count",
        "forced_min_bootstrap_count",
        "force_bootstrap_floor",
        "debug_min_bootstrap_count",
    ):
        value = reference.get(key)
        if value is None:
            continue
        try:
            floor = int(value)
        except (TypeError, ValueError):
            continue
        if floor > 0:
            return floor
    return None


def _finalist_plaintext_quality_reject(context: dict[str, Any]) -> tuple[bool, str | None]:
    reference = context.get("harness", {}).get("reference_json", {})
    if not isinstance(reference, dict):
        return False, None
    sanity = reference.get("plaintext_sanity") or reference.get("output_quality_sanity")
    if not isinstance(sanity, dict):
        return False, None
    if sanity.get("near_constant_first2_logits") is True:
        return True, "near_constant_first2_logits"
    zero_records = int(sanity.get("zero_output_collapse_records", 0) or 0)
    records = int(sanity.get("records", 0) or 0)
    if records > 0 and zero_records >= records:
        return True, "zero_output_collapse"
    min_margin = sanity.get("first2_margin_min")
    target = reference.get("min_plaintext_logit_margin")
    if target is not None and min_margin is not None:
        if _finite_float(min_margin, 0.0) < _finite_float(target, 0.0):
            return True, "plaintext_margin_below_target"
    return False, None


def _profile_noise_metadata_reject(context: dict[str, Any]) -> tuple[bool, str | None]:
    resilience = context.get("resilience", {})
    if not isinstance(resilience, dict) or not resilience.get("enabled"):
        return False, None
    noise_model = resilience.get("ckks_noise_model")
    if not isinstance(noise_model, dict):
        return True, "missing_ckks_noise_model"
    if noise_model.get("fallback") is True:
        return True, "ckks_noise_model_fallback"
    if noise_model.get("model") not in {None, "tuneinsight-lattigo-v6"}:
        return True, "unexpected_ckks_noise_model"
    return False, None


def _finalist_noise_warning_reject(
    context: dict[str, Any],
    noise: dict[str, Any] | None,
    params: Params,
) -> tuple[bool, str | None]:
    if noise is None:
        return False, None
    warnings = noise.get("warning_ops") or []
    if not isinstance(warnings, list):
        warnings = [str(warnings)]
    critical = {
        "trace_precision_below_margin",
        "trace_message_too_large",
    }.intersection(str(item) for item in warnings)
    if not critical:
        return False, None
    if getattr(params, "noise_estimator_require_trace_safe", False) or _has_estimator_backed_profile(context):
        return True, ",".join(sorted(critical))
    return False, None


def _has_estimator_backed_profile(context: dict[str, Any]) -> bool:
    resilience = context.get("resilience", {})
    if not isinstance(resilience, dict) or not resilience.get("enabled"):
        return False
    noise_model = resilience.get("ckks_noise_model")
    return isinstance(noise_model, dict) and noise_model.get("fallback") is not True


def _noise_estimator_for_finalist(
    context: dict[str, Any],
    result: dict[str, Any],
    params: Params,
) -> dict[str, Any] | None:
    if params.noise_estimator != "finalists":
        return None
    return estimate_compile_result_noise(context, result, params)


def _fast_compile_noise_estimate(
    context: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    ckks = context.get("ckks", {})
    params_data = context.get("params", {})
    params = SimpleNamespace(
        noise_estimator_binary=None,
        noise_estimator_timeout_sec=1,
        noise_estimator_min_output_margin_bits=context.get("harness", {}).get(
            "noise_estimator_min_output_margin_bits",
            params_data.get("noise_estimator_min_output_margin_bits", 2.0),
        ),
        noise_estimator_alpha=context.get("harness", {}).get(
            "noise_estimator_alpha",
            params_data.get("noise_estimator_alpha", 14.0),
        ),
        noise_estimator_max_trace_message_bits=context.get("harness", {}).get(
            "noise_estimator_max_trace_message_bits",
            params_data.get("noise_estimator_max_trace_message_bits", 20.0),
        ),
        noise_estimator_require_trace_safe=context.get("harness", {}).get(
            "noise_estimator_require_trace_safe",
            params_data.get("noise_estimator_require_trace_safe", False),
        ),
        poly_deg=ckks.get("poly_degree", 32768),
        max_slot=ckks.get("max_slots", 16384),
        Sf=ckks.get("Sf", params_data.get("Sf", 40)),
        Sw=ckks.get("Sw", params_data.get("Sw", 40)),
        lvl_lb=ckks.get("lvl_lb", params_data.get("lvl_lb", 1)),
        lvl_ub=ckks.get("lvl_ub", params_data.get("lvl_ub", 16)),
    )
    try:
        return estimate_compile_result_noise(context, result, params)
    except Exception as exc:
        return {
            "valid": False,
            "fallback": True,
            "estimated_precision_bits": 0.0,
            "output_margin_bits": float("-inf"),
            "unsupported_ops": [f"{type(exc).__name__}: {str(exc)[:160]}"],
        }


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


def _load_reusable_best_code(output_dir: Path) -> str | None:
    """Load the best available program from a completed OpenEvolve workspace."""

    candidates = _discover_finalist_codes(output_dir, "", 1)
    if candidates:
        return candidates[0]
    for program in (
        output_dir / "best" / "best_program.py",
        output_dir / "best_program.py",
        output_dir / "initial_program.py",
    ):
        if not program.exists():
            continue
        try:
            code = program.read_text(encoding="utf-8")
        except Exception:
            continue
        if code.strip():
            return code
    return None


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
        "requested_boundary_groups": 0,
        "solved_boundary_groups": 0,
        "partial_boundary_groups": 0,
        "candidate_solved_boundary_groups": 0,
        "fallback_selected_boundary_groups": 0,
        "invalid_boundary_groups": 0,
        "candidate_invalid_reasons": {},
        "candidate_costs": [],
        "costs": [],
        "assignments": [],
        "selected_source_counts": {},
        "boundary_group_summaries": [],
    }
    for item in getattr(qbp_manager, "openevolve_diagnostics", []):
        for key in (
            "requested_budgets",
            "solved_budgets",
            "candidate_solved_budgets",
            "fallback_selected_budgets",
            "requested_boundary_groups",
            "solved_boundary_groups",
            "partial_boundary_groups",
            "candidate_solved_boundary_groups",
            "fallback_selected_boundary_groups",
            "invalid_boundary_groups",
        ):
            totals[key] += int(item.get(key, 0))
        totals["candidate_costs"].extend(item.get("candidate_costs", []))
        for source, count in item.get("selected_source_counts", {}).items():
            totals["selected_source_counts"][source] = (
                totals["selected_source_counts"].get(source, 0) + int(count)
            )
        totals["costs"].extend(item.get("costs", []))
        totals["assignments"].extend(item.get("assignments", []))
        for reason, count in item.get("candidate_invalid_reasons", {}).items():
            totals["candidate_invalid_reasons"][reason] = totals["candidate_invalid_reasons"].get(reason, 0) + int(count)
        if len(totals["boundary_group_summaries"]) < 512:
            remaining = 512 - len(totals["boundary_group_summaries"])
            totals["boundary_group_summaries"].extend(
                list(item.get("boundary_group_summaries", []))[:remaining]
            )
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
    return _bounded_fail_open_hints(initial_hints) if best is None else best[3]


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
            "placement_runtime_sec": 0.0,
            "fallback_selected_budgets": 0.0,
            "boundary_group_validity": 0.0,
            "candidate_qbp_coverage": 0.0,
            "fallback_selected_groups": 0.0,
            "invalid_boundary_groups": 0.0,
            "estimated_precision_bits": 0.0,
            "output_margin_bits": 0.0,
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
    if _context_leniency(context) == "strict":
        for idx, item in enumerate(hints.get("unit_policies", []) or []):
            if not isinstance(item, dict):
                reasons.append(f"unit_policies[{idx}] is not a policy object")
                continue
            if not _selector_matches_any_unit(item.get("selector", {}), context):
                reasons.append(f"unit_policies[{idx}] targets no known placement unit")
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
                "repair_summary",
                "per_unit_score_table",
                "selected_source_counts",
                "candidate_qbp_coverage",
                "boundary_group_validity",
                "fallback_selected_groups",
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
        candidate_costs = list(diagnostics.get("candidate_costs", []))
        solved = int(diagnostics.get("solved_budgets", len(costs)))
        candidate_solved = int(diagnostics.get("candidate_solved_budgets", solved))
        fallback_selected = int(diagnostics.get("fallback_selected_budgets", 0))
        requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
        solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
        fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
        invalid_groups = int(diagnostics.get("invalid_boundary_groups", 0) or 0)
        candidate_improved = int(diagnostics.get("candidate_improved_budgets", 0))
        requested = max(1, len(io_budgets))
        effective_validity = solved / requested
        validity = candidate_solved / requested
        boundary_group_validity = solved_groups / requested_groups
        candidate_qbp_coverage = candidate_groups / requested_groups
        repair_count = _repair_count(hints)
        unit_coverage = _unit_coverage(context, hints)
        avg_cost = sum(costs) / solved if costs else float("inf")
        candidate_avg_cost = (
            sum(candidate_costs) / len(candidate_costs) if candidate_costs else float("inf")
        )
        counts = _aggregate_counts(assignments)
        profile_risk = _profile_risk(assignments, tdag.params)
        reserve_summary = _aggregate_reserve_summary(assignments, tdag, tdag.params)
        reserve_score = _reserve_quality_score(reserve_summary)
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
        target_bootstrap_count = _context_target_bootstrap_count(context)
        target_bootstrap_score = _target_bootstrap_score(
            target_bootstrap_count,
            counts["bootstrap"],
            reference["bootstrap_count"],
        )
        rescale_score = _relative_reduction(reference["rescale_count"], counts["rescale"])
        risk_score = 1.0 / (1.0 + profile_risk)
        quality_score = (
            0.42 * latency_score
            + 0.12 * bootstrap_score
            + 0.12 * target_bootstrap_score
            + 0.08 * rescale_score
            + 0.13 * risk_score
            + 0.12 * reserve_score
        )
        if effective_validity < 1.0 or boundary_group_validity < 1.0:
            bootstrap_frontier = target_bootstrap_score if validity > 0.0 else 0.0
            combined_score = min(
                0.999,
                0.28 * effective_validity
                + 0.20 * boundary_group_validity
                + 0.18 * candidate_qbp_coverage
                + 0.24 * bootstrap_frontier
                + 0.10 * quality_score
            )
        elif candidate_solved == 0 or fallback_selected > 0 or fallback_groups > 0:
            if _context_budget_aggressive(context):
                combined_score = min(
                    0.999,
                    0.48 * validity
                    + 0.22 * candidate_qbp_coverage
                    + 0.18 * quality_score
                    + 0.08 * unit_coverage
                )
            else:
                combined_score = min(0.999, 0.40 * validity + 0.30 * quality_score)
        else:
            improvement_fraction = candidate_improved / requested
            combined_score = (
                1.0
                + quality_score
                + 0.05 * improvement_fraction
                + 0.02 * candidate_qbp_coverage
            )
        return {
            "metrics": {
                "combined_score": float(combined_score),
                "validity": float(validity),
                "effective_validity": float(effective_validity),
                "candidate_validity": float(validity),
                "boundary_group_validity": float(boundary_group_validity),
                "candidate_qbp_coverage": float(candidate_qbp_coverage),
                "repair_count": float(repair_count),
                "unit_coverage": float(unit_coverage),
                "latency_score": float(latency_score),
                "avg_latency_usec": float(avg_cost if costs else 0.0),
                "candidate_only_avg_latency_usec": float(
                    candidate_avg_cost if candidate_costs else 0.0
                ),
                "bootstrap_count": float(counts["bootstrap"]),
                "target_bootstrap_count": float(target_bootstrap_count),
                "target_bootstrap_score": float(target_bootstrap_score),
                "rescale_count": float(counts["rescale"]),
                "profile_risk": float(profile_risk),
                "solved_budgets": float(solved),
                "candidate_solved_budgets": float(candidate_solved),
                "fallback_selected_budgets": float(fallback_selected),
                "requested_boundary_groups": float(requested_groups),
                "solved_boundary_groups": float(solved_groups),
                "candidate_solved_boundary_groups": float(candidate_groups),
                "fallback_selected_groups": float(fallback_groups),
                "invalid_boundary_groups": float(invalid_groups),
                "candidate_improved_budgets": float(candidate_improved),
                "reserve_score": float(reserve_score),
                "min_decryptability_reserve_bits": float(
                    _finite_float(reserve_summary.get("min_decryptability_reserve_bits"), 0.0)
                ),
                "min_transition_reserve_bits": float(
                    _finite_float(reserve_summary.get("min_transition_reserve_bits"), 0.0)
                ),
                "graph_nodes": float(len(tdag.nodes)),
                "graph_edges": float(len(tdag.edges)),
                "budget_count": float(len(io_budgets)),
            },
            "artifacts": {
                "solved_budgets": f"{solved}/{len(io_budgets)}",
                "candidate_solved_budgets": f"{candidate_solved}/{len(io_budgets)}",
                "fallback_selected_budgets": f"{fallback_selected}/{len(io_budgets)}",
                "boundary_group_validity": f"{solved_groups}/{requested_groups}",
                "candidate_qbp_coverage": f"{candidate_groups}/{requested_groups}",
                "fallback_selected_groups": f"{fallback_groups}/{requested_groups}",
                "invalid_boundary_groups": f"{invalid_groups}/{requested_groups}",
                "candidate_improved_budgets": f"{candidate_improved}/{len(io_budgets)}",
                "reference_avg_latency_usec": f"{reference_avg:.3f}",
                "latency_delta_usec": (
                    f"{avg_cost - reference_avg:.3f}"
                    if costs and math.isfinite(reference_avg)
                    else "none"
                ),
                "best_latency_usec": str(min(costs) if costs else "none"),
                "avg_latency_usec": f"{avg_cost:.3f}" if costs else "none",
                "candidate_only_avg_latency_usec": (
                    f"{candidate_avg_cost:.3f}" if candidate_costs else "none"
                ),
                "bootstrap_count": str(counts["bootstrap"]),
                "rescale_count": str(counts["rescale"]),
                "bootstrap_delta": str(counts["bootstrap"] - reference["bootstrap_count"]),
                "rescale_delta": str(counts["rescale"] - reference["rescale_count"]),
                "selected_source_counts": json.dumps(
                    diagnostics.get("selected_source_counts", {}), sort_keys=True
                ),
                "invalid_reasons": _compact_invalid_reasons(diagnostics),
                "candidate_invalid_reasons": _compact_invalid_reasons(
                    {"invalid_reasons": diagnostics.get("candidate_invalid_reasons", {})}
                ),
                "reserve_summary": json.dumps(reserve_summary, sort_keys=True)[:4000],
                "policy_summary": _compact_policy_summary(hints),
                "repair_summary": json.dumps(_unit_policy_summary(context, hints), sort_keys=True),
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
        _init_budget_diagnostics(diagnostics, len(io_budgets_list))
    if str(hints.get("strategy")) == "bootstrap_mcts":
        if (
            not getattr(params, "openevolve_evaluating_candidate", False)
            and int(getattr(params, "openevolve_iterations", 0) or 0) == 0
            and not getattr(params, "openevolve_compile_hints", None)
        ):
            return _solve_budget_batch_fast_seed(
                pdag,
                io_budgets_list,
                le,
                params,
                diagnostics,
            )
        result = _solve_budget_batch_boundary_mcts(
            pdag,
            io_budgets_list,
            le,
            params,
            hints,
            diagnostics,
        )
        if (
            not result[0]
            and not getattr(params, "openevolve_evaluating_candidate", False)
            and bool(getattr(params, "openevolve_fail_open", True))
        ):
            if diagnostics is not None:
                diagnostics["fail_open_seed_replay"] = True
            return _solve_budget_batch_fast_seed(
                pdag,
                io_budgets_list,
                le,
                params,
                diagnostics,
            )
        return result
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
            _record_selected_attempt(diagnostics, best_attempt, candidate_attempt, fallback_attempt)

        current = io_to_cost.get(best_attempt.in_key, {}).get(best_attempt.out_key)
        if current is None or best_attempt.cost < current:
            io_to_cost.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.cost
            io_to_assign.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.assign
    return io_to_assign, io_to_cost


def _solve_budget_batch_fast_seed(
    pdag: Tdag,
    io_budgets_list: list[dict],
    le: LatencyEstimator,
    params: Params,
    diagnostics: dict[str, Any] | None,
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], Assign]],
    dict[tuple[int, int], dict[tuple[int, int], float]],
]:
    io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]] = {}
    io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    fallback_source, fallback_policy = _seed_fallback_attempts(params)[0]
    for io_budget in io_budgets_list:
        try:
            best_attempt = _solve_one_budget_attempt(
                pdag,
                params,
                io_budget,
                le,
                fallback_source,
                fallback_policy,
            )
        except Exception as exc:
            if diagnostics is not None:
                _record_invalid_reason(diagnostics, "invalid_reasons", exc)
            continue
        if diagnostics is not None:
            _record_selected_attempt(diagnostics, best_attempt, None, best_attempt)
        current = io_to_cost.get(best_attempt.in_key, {}).get(best_attempt.out_key)
        if current is None or best_attempt.cost < current:
            io_to_cost.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.cost
            io_to_assign.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.assign
    return io_to_assign, io_to_cost


def _init_budget_diagnostics(diagnostics: dict[str, Any], requested_budgets: int) -> None:
    diagnostics["requested_budgets"] = requested_budgets
    diagnostics["solved_budgets"] = 0
    diagnostics["candidate_solved_budgets"] = 0
    diagnostics["fallback_solved_budgets"] = 0
    diagnostics["fallback_selected_budgets"] = 0
    diagnostics["requested_boundary_groups"] = 0
    diagnostics["solved_boundary_groups"] = 0
    diagnostics["partial_boundary_groups"] = 0
    diagnostics["candidate_solved_boundary_groups"] = 0
    diagnostics["fallback_selected_boundary_groups"] = 0
    diagnostics["invalid_boundary_groups"] = 0
    diagnostics["candidate_improved_budgets"] = 0
    diagnostics["costs"] = []
    diagnostics["candidate_costs"] = []
    diagnostics["assignments"] = []
    diagnostics["selected_source_counts"] = {}
    diagnostics["boundary_group_summaries"] = []


def _record_selected_attempt(
    diagnostics: dict[str, Any],
    best_attempt: _BudgetAttempt,
    candidate_attempt: _BudgetAttempt | None,
    fallback_attempt: _BudgetAttempt | None,
) -> None:
    diagnostics["solved_budgets"] += 1
    diagnostics["costs"].append(_attempt_actual_cost(best_attempt))
    diagnostics["assignments"].append(best_attempt.assign)
    if candidate_attempt is not None:
        diagnostics["candidate_solved_budgets"] += 1
        diagnostics["candidate_costs"].append(_attempt_actual_cost(candidate_attempt))
    if fallback_attempt is not None:
        diagnostics["fallback_solved_budgets"] += 1
    if best_attempt.source.startswith("seed_fallback"):
        diagnostics["fallback_selected_budgets"] += 1
    source_counts = diagnostics["selected_source_counts"]
    source_counts[best_attempt.source] = source_counts.get(best_attempt.source, 0) + 1
    if (
        candidate_attempt is not None
        and fallback_attempt is not None
        and candidate_attempt.cost + 1e-9 < fallback_attempt.cost
    ):
        diagnostics["candidate_improved_budgets"] += 1


def _attempt_actual_cost(attempt: _BudgetAttempt) -> float:
    return float(attempt.actual_cost if attempt.actual_cost is not None else attempt.cost)


def _record_boundary_group_result(
    diagnostics: dict[str, Any],
    group_key: tuple,
    budgets: list[dict],
    selected_attempts: list[_BudgetAttempt],
    candidate_solved_count: int,
    fallback_selected_count: int,
) -> None:
    requested = len(budgets)
    solved = len(selected_attempts)
    diagnostics["requested_boundary_groups"] += 1
    if solved == requested and requested > 0:
        diagnostics["solved_boundary_groups"] += 1
    elif solved > 0:
        diagnostics["partial_boundary_groups"] += 1
    else:
        diagnostics["invalid_boundary_groups"] += 1
    if requested > 0 and candidate_solved_count == requested:
        diagnostics["candidate_solved_boundary_groups"] += 1
    if fallback_selected_count > 0:
        diagnostics["fallback_selected_boundary_groups"] += 1

    selected_counts = _assignment_count_summary([attempt.assign for attempt in selected_attempts])
    summary = {
        "group_key": {
            "in_lvl": int(group_key[0]),
            "in_scl": int(group_key[1]),
            "maino_v": str(group_key[2]),
            "main_dag_size": int(group_key[3]),
        },
        "requested_output_levels": sorted(
            {
                int(budget.get("out_lvl", -1))
                for budget in budgets
                if int(budget.get("out_lvl", -1)) >= 0
            }
        ),
        "requested_budgets": requested,
        "solved_budgets": solved,
        "candidate_solved_budgets": int(candidate_solved_count),
        "fallback_selected_budgets": int(fallback_selected_count),
        "complete": bool(requested > 0 and solved == requested),
        "candidate_complete": bool(requested > 0 and candidate_solved_count == requested),
        "avg_bootstrap": float(selected_counts["avg_bootstrap"]),
        "min_bootstrap": float(selected_counts["min_bootstrap"]),
        "max_bootstrap": float(selected_counts["max_bootstrap"]),
        "avg_rescale": float(selected_counts["avg_rescale"]),
        "min_rescale": float(selected_counts["min_rescale"]),
        "max_rescale": float(selected_counts["max_rescale"]),
    }
    summaries = diagnostics.setdefault("boundary_group_summaries", [])
    if len(summaries) < 256:
        summaries.append(summary)


def _solve_budget_batch_boundary_mcts(
    pdag: Tdag,
    io_budgets_list: list[dict],
    le: LatencyEstimator,
    params: Params,
    hints: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], Assign]],
    dict[tuple[int, int], dict[tuple[int, int], float]],
]:
    io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]] = {}
    io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    for _group_key, budgets in _budget_boundary_groups(io_budgets_list).items():
        group_attempts = _boundary_mcts_group_attempts(pdag, params, budgets, le, hints, diagnostics)
        selected_attempts: list[_BudgetAttempt] = []
        group_candidate_solved = 0
        group_fallback_selected = 0
        for index, io_budget in enumerate(budgets):
            attempts = list(group_attempts.get(index, []))
            last_error = None
            record = _placement_record_for_budget(hints, pdag.name, io_budget)
            if record is not None:
                try:
                    attempts.append(_solve_one_record_attempt(pdag, params, io_budget, le, record))
                except Exception as exc:
                    last_error = exc
                    if diagnostics is not None:
                        _record_invalid_reason(diagnostics, "candidate_invalid_reasons", exc)
            if _should_use_seed_fallback(hints):
                for source, policy_hints in _seed_fallback_attempts(params):
                    try:
                        attempts.append(_solve_one_budget_attempt(pdag, params, io_budget, le, source, policy_hints))
                    except Exception as exc:
                        last_error = exc
            if not attempts:
                if diagnostics is not None:
                    _record_invalid_reason(
                        diagnostics,
                        "invalid_reasons",
                        last_error or PlacementError("no feasible boundary-mcts placement policy"),
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
            if candidate_attempt is not None:
                group_candidate_solved += 1
            if best_attempt.source.startswith("seed_fallback"):
                group_fallback_selected += 1
            selected_attempts.append(best_attempt)
            if diagnostics is not None:
                _record_selected_attempt(diagnostics, best_attempt, candidate_attempt, fallback_attempt)
            current = io_to_cost.get(best_attempt.in_key, {}).get(best_attempt.out_key)
            if current is None or best_attempt.cost < current:
                io_to_cost.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.cost
                io_to_assign.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.assign
        if diagnostics is not None:
            _record_boundary_group_result(
                diagnostics,
                _group_key,
                budgets,
                selected_attempts,
                group_candidate_solved,
                group_fallback_selected,
            )
    return io_to_assign, io_to_cost


def _budget_boundary_groups(io_budgets_list: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for budget in io_budgets_list:
        key = (
            int(budget.get("in_lvl", -1)),
            int(budget.get("in_scl", -1)),
            str(budget.get("maino_v", "")),
            int(budget.get("main_dag_size", 0) or 0),
        )
        groups.setdefault(key, []).append(budget)
    for budgets in groups.values():
        budgets.sort(key=lambda item: int(item.get("out_lvl", -1)))
    return groups


def _seed_fallback_attempts(params: Params) -> list[tuple[str, dict[str, Any]]]:
    seed = _default_policy_hints()
    beam = _budget_fulfillment_beam_policy()
    if not getattr(params, "openevolve_evaluating_candidate", False):
        return [("seed_fallback_latency_beam", beam)]
    return [
        ("seed_fallback", seed),
        ("seed_fallback_relaxed", _relaxed_scheduler_policy(seed, params)),
        ("seed_fallback_latency_beam", beam),
    ]


def _budget_fulfillment_beam_policy() -> dict[str, Any]:
    beam = _low_scale_frontier_policy()
    beam.update(
        {
            "strategy": "latency_beam",
            "allow_bootstrap": True,
            "budget_aggressive": True,
            "selection_bootstrap_penalty": 500_000_000.0,
            "beam_width": 3,
            "state_cap_per_node": 8,
            "max_scale_candidates": 16,
        }
    )
    return beam


def _boundary_mcts_group_attempts(
    pdag: Tdag,
    params: Params,
    budgets: list[dict],
    le: LatencyEstimator,
    hints: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> dict[int, list[_BudgetAttempt]]:
    policy = _policy_options(hints, params)
    actions = _mcts_actions_from_hints(hints, params)
    rollout_budget = max(len(actions), int(policy["mcts_rollout_budget"]))
    exploration = float(policy["mcts_exploration_weight"])
    max_repair_bootstraps = int(policy["mcts_max_repair_bootstraps"])
    if not getattr(params, "openevolve_evaluating_candidate", False):
        selected = next(
            (action for action in actions if _bool_hint(action.policy.get("allow_bootstrap"), False)),
            actions[0],
        )
        actions = [selected]
        rollout_budget = 1
    stats = [_MCTSNodeStats() for _ in actions]
    attempts_by_budget: dict[int, list[_BudgetAttempt]] = {idx: [] for idx in range(len(budgets))}
    seen_actions: set[str] = set()
    direct_attempts = _direct_budget_beam_group_attempts(
        pdag,
        params,
        budgets,
        le,
        hints,
        diagnostics,
    )
    for budget_idx, attempt in direct_attempts.items():
        attempts_by_budget[budget_idx].append(attempt)
    if _boundary_mcts_group_target_met(
        list(direct_attempts.values()),
        len(budgets),
        params,
        hints,
    ):
        return attempts_by_budget

    for step in range(rollout_budget):
        idx = _select_mcts_action(actions, stats, step, exploration)
        action = actions[idx]
        action_digest = json.dumps(action.policy, sort_keys=True, default=str)
        if action_digest in seen_actions and step >= len(actions):
            stats[idx].visits += 1
            stats[idx].reward_sum += _mcts_invalid_reward(action, step)
            continue
        seen_actions.add(action_digest)
        action_attempts: list[_BudgetAttempt] = []
        invalid_reasons: Counter = Counter()
        for budget_idx, budget in enumerate(budgets):
            best_for_budget = None
            variants = (
                [action.policy]
                if _bool_hint(action.policy.get("direct_budget_policy"), False)
                else _boundary_policy_variants(pdag, params, budget, hints, action.policy)
            )
            for variant_idx, variant in enumerate(variants):
                source = f"candidate:boundary_mcts:{action.name}:{variant_idx}"
                try:
                    attempt = _solve_one_budget_attempt(pdag, params, budget, le, source, variant)
                    counts = _maintenance_counts(attempt.assign)
                    if max_repair_bootstraps >= 0 and counts["bootstrap"] > max_repair_bootstraps:
                        raise PlacementError(
                            f"boundary-mcts rollout exceeded bootstrap repair cap "
                            f"({counts['bootstrap']} > {max_repair_bootstraps})"
                        )
                    if best_for_budget is None or attempt.cost < best_for_budget.cost:
                        best_for_budget = attempt
                except Exception as exc:
                    invalid_reasons[f"{type(exc).__name__}: {str(exc)[:160]}"] += 1
            if best_for_budget is not None:
                attempts_by_budget[budget_idx].append(best_for_budget)
                action_attempts.append(best_for_budget)

        reward = _boundary_mcts_group_reward(action_attempts, len(budgets), params, hints)
        stats[idx].visits += 1
        stats[idx].reward_sum += reward
        if action_attempts:
            best_cost = min(attempt.cost for attempt in action_attempts)
            best_assign = min(action_attempts, key=lambda attempt: attempt.cost).assign
            if reward > stats[idx].best_reward:
                stats[idx].best_reward = reward
                stats[idx].best_assign = best_assign
                stats[idx].best_cost = best_cost
        else:
            stats[idx].reward_sum += _mcts_invalid_reward(action, step)
            if diagnostics is not None:
                for reason, count in invalid_reasons.most_common(3):
                    diagnostics.setdefault("candidate_invalid_reasons", {})[
                        f"{action.name}: {reason}"
                    ] = diagnostics.setdefault("candidate_invalid_reasons", {}).get(
                        f"{action.name}: {reason}", 0
                    ) + count
        if _boundary_mcts_group_target_met(action_attempts, len(budgets), params, hints):
            break
    return attempts_by_budget


def _direct_budget_beam_group_attempts(
    pdag: Tdag,
    params: Params,
    budgets: list[dict],
    le: LatencyEstimator,
    hints: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> dict[int, _BudgetAttempt]:
    policy = _direct_budget_beam_policy_from_hints(hints)
    attempts: dict[int, _BudgetAttempt] = {}
    for budget_idx, budget in enumerate(budgets):
        try:
            attempts[budget_idx] = _solve_one_budget_attempt(
                pdag,
                params,
                budget,
                le,
                "candidate:direct_budget_beam",
                policy,
            )
        except Exception as exc:
            if diagnostics is not None:
                _record_invalid_reason(diagnostics, "candidate_invalid_reasons", exc)
    return attempts


def _direct_budget_beam_policy_from_hints(hints: dict[str, Any]) -> dict[str, Any]:
    policy = _budget_fulfillment_beam_policy()
    for key in (
        "bootstrap_penalty",
        "selection_bootstrap_penalty",
        "rescale_penalty",
        "level_drop_penalty",
        "scale_penalty",
        "boundary_scale_penalty",
        "beam_width",
        "state_cap_per_node",
        "max_scale_candidates",
        "scale_lattice",
        "min_internal_level",
    ):
        if key in hints:
            policy[key] = hints[key]
    policy["strategy"] = "latency_beam"
    policy["allow_bootstrap"] = True
    policy["budget_aggressive"] = True
    policy["allow_seed_fallback"] = False
    return _with_default_policy(policy)


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
    actual_cost = estimate_assign(assign, le)
    return _BudgetAttempt(
        source,
        assign,
        _policy_assignment_score(assign, le, policy_hints, params),
        in_key,
        out_key,
        actual_cost,
    )


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
    actual_cost = estimate_assign(assign, le)
    return _BudgetAttempt("candidate_record", assign, actual_cost, in_key, out_key, actual_cost)


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
    if str(hints.get("strategy")) == "bootstrap_mcts":
        return _build_bootstrap_mcts_assign(tdag, params, io_budget, hints, le)
    if str(hints.get("strategy")) == "latency_beam":
        return _build_best_assign_from_variants(tdag, params, io_budget, hints, le)
    if "maino_v" in io_budget:
        best = None
        main_qbp_cost = io_budget.get("main_qbp_cost", {})
        for main_key, main_cost in sorted(main_qbp_cost.items(), key=lambda item: item[1]):
            fixed = {io_budget["maino_v"]: main_key}
            try:
                assign = _build_assign_once(tdag, params, io_budget, hints, fixed, le)
                cost = _policy_assignment_score(assign, le, hints, params) + float(main_cost)
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
                    cost = _policy_assignment_score(assign, le, variant, params) + float(main_cost)
                    item = (cost, assign)
                    if best is None or item[0] < best[0]:
                        best = item
            else:
                assign = _build_assign_once(tdag, params, io_budget, variant, {}, le)
                assign.check_assign()
                item = (_policy_assignment_score(assign, le, variant, params), assign)
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


def _boundary_policy_variants(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    root_hints: dict[str, Any],
    action_policy: dict[str, Any],
) -> list[dict[str, Any]]:
    pdag_vout = list(tdag.outputs)[0]
    base = _with_default_policy(action_policy)
    if str(base.get("strategy")) == "bootstrap_mcts":
        base["strategy"] = "level_preserving"
    scales = _boundary_scale_candidates(tdag, params, io_budget, base)
    if not scales:
        scales = [params.scale_lower_bound(pdag_vout, tdag.nodes[pdag_vout], "out")]
    state_cap = max(
        1,
        min(
            8,
            _int_hint(base.get("state_cap_per_node"), 8),
            _int_hint(base.get("boundary_state_cap"), 3),
        ),
    )
    scales = scales[:state_cap]
    anchors = _bootstrap_anchor_nodes(tdag, params, root_hints, base)
    variants: list[dict[str, Any]] = []
    for scale in scales:
        variant = dict(base)
        node_levels = dict(_int_map(base.get("preferred_node_levels", {})))
        node_scales = dict(_int_map(base.get("preferred_node_scales", {})))
        node_scales[pdag_vout] = int(scale)
        if int(io_budget.get("out_lvl", -1)) >= 0:
            node_levels[pdag_vout] = int(io_budget["out_lvl"])
        for anchor in anchors:
            node_levels[anchor] = _anchor_output_level(params, base)
            node_scales[anchor] = max(
                params.scale_lower_bound(anchor, tdag.nodes[anchor], "out"),
                min(params.Sf, params.Sw),
            )
        variant["preferred_node_levels"] = node_levels
        variant["preferred_node_scales"] = node_scales
        variants.append(_with_default_policy(variant))
    return variants


def _boundary_scale_candidates(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    policy: dict[str, Any],
) -> list[int]:
    pdag_vout = list(tdag.outputs)[0]
    lower = params.scale_lower_bound(pdag_vout, tdag.nodes[pdag_vout], "out")
    out_lvl = int(io_budget.get("out_lvl", -1))
    upper = int(policy.get("max_scale", _max_scale(params)))
    if out_lvl >= 0:
        upper = min(upper, params.boundary_output_scale_bound(out_lvl))
    if upper < lower:
        return []
    values = [
        lower,
        params.Sw,
        params.Csw,
        params.Sf,
        policy.get("preferred_boundary_scale"),
        policy.get("boundary_scale"),
    ]
    mode = str(policy.get("boundary_scale_policy", "frontier"))
    if mode == "low":
        values.extend([lower + params.Sf, lower + 2 * params.Sf])
    elif mode == "waterline":
        values.extend([params.Sw, params.Sw + params.Sf, max(lower, params.Sw - params.Sf)])
    elif mode == "sf":
        values.extend([params.Sf, params.Sf + params.Sw, params.Sf * 2])
    else:
        values.extend([lower, params.Sw, params.Sf, upper, max(lower, upper - params.Sf)])
    return _scale_candidates(values, lower, upper, params, policy)


def _bootstrap_anchor_nodes(
    tdag: Tdag,
    params: Params,
    root_hints: dict[str, Any],
    policy: dict[str, Any],
) -> list[str]:
    explicit = policy.get("bootstrap_anchors") or root_hints.get("bootstrap_anchors")
    if isinstance(explicit, list):
        anchors = [
            str(node)
            for node in explicit
            if str(node) in tdag.nodes and tdag.nodes[str(node)].get("op") not in {"input", "constant"}
        ]
        if anchors:
            return anchors[: max(0, _int_hint(policy.get("bootstrap_anchor_count"), len(anchors)))]
    count = max(0, min(64, _int_hint(policy.get("bootstrap_anchor_count"), 0)))
    if count == 0:
        return []
    depths = _node_depths(tdag)

    def rank(node: str) -> tuple[float, int, int, str]:
        attrs = dict(tdag.nodes[node])
        nonlinear_bonus = 1 if _node_nonlinear_kind(attrs) else 0
        op_bonus = 1 if attrs.get("op") == "mul" else 0
        fanout = int(tdag.out_degree(node))
        return (-(2 * nonlinear_bonus + op_bonus), -fanout, -depths.get(node, 0), node)

    candidates = [
        str(node)
        for node in tdag.nodes
        if tdag.nodes[node].get("op") not in {"input", "constant"}
    ]
    return sorted(candidates, key=rank)[:count]


def _node_depths(tdag: Tdag) -> dict[str, int]:
    depths: dict[str, int] = {}
    for node in nx.topological_sort(tdag):
        preds = list(tdag.predecessors(node))
        depths[str(node)] = 0 if not preds else 1 + max(depths.get(str(pred), 0) for pred in preds)
    return depths


def _anchor_output_level(params: Params, policy: dict[str, Any]) -> int:
    preferred = _int_hint(policy.get("bootstrap_anchor_level"), params.bts_ub)
    return max(params.bts_lb + 1, min(params.lvl_ub, preferred))


def _boundary_mcts_group_reward(
    attempts: list[_BudgetAttempt],
    requested: int,
    params: Params,
    hints: dict[str, Any],
) -> float:
    if requested <= 0:
        return 0.0
    coverage = len(attempts) / requested
    if not attempts:
        return -0.1
    summary = _assignment_count_summary([attempt.assign for attempt in attempts])
    target = max(
        0,
        _int_hint(
            hints.get("target_bootstrap_count"),
            int(getattr(params, "openevolve_target_bootstrap_count", 0)),
        ),
    )
    bootstrap_score = _target_bootstrap_score(target, summary["avg_bootstrap"], None)
    rescale_score = 1.0 / (1.0 + summary["avg_rescale"] / 32.0)
    cost_score = 1.0 / (1.0 + (sum(attempt.cost for attempt in attempts) / len(attempts)) / 1_000_000_000.0)
    return 0.50 * coverage + 0.30 * bootstrap_score + 0.12 * rescale_score + 0.08 * cost_score


def _boundary_mcts_group_target_met(
    attempts: list[_BudgetAttempt],
    requested: int,
    params: Params,
    hints: dict[str, Any],
) -> bool:
    if requested <= 0 or len(attempts) < requested:
        return False
    target = max(
        0,
        _int_hint(
            hints.get("target_bootstrap_count"),
            int(getattr(params, "openevolve_target_bootstrap_count", 0)),
        ),
    )
    if target <= 0:
        return False
    summary = _assignment_count_summary([attempt.assign for attempt in attempts])
    return summary["avg_bootstrap"] <= target


def _build_bootstrap_mcts_assign(
    tdag: Tdag,
    params: Params,
    io_budget: dict,
    hints: dict[str, Any],
    le: LatencyEstimator | None,
) -> Assign:
    policy = _policy_options(hints, params)
    actions = _mcts_actions_from_hints(hints, params)
    rollout_budget = max(len(actions), int(policy["mcts_rollout_budget"]))
    exploration = float(policy["mcts_exploration_weight"])
    max_repair_bootstraps = int(policy["mcts_max_repair_bootstraps"])
    stats = [_MCTSNodeStats() for _ in actions]
    invalid_reasons: Counter = Counter()
    best: tuple[tuple[float, float, float, float], Assign, str] | None = None

    for step in range(rollout_budget):
        idx = _select_mcts_action(actions, stats, step, exploration)
        action = actions[idx]
        try:
            assign = build_conservative_assign(tdag, params, io_budget, action.policy, le)
            assign.check_assign()
            counts = _aggregate_counts([assign])
            if max_repair_bootstraps >= 0 and counts["bootstrap"] > max_repair_bootstraps:
                raise PlacementError(
                    f"mcts rollout exceeded bootstrap repair cap "
                    f"({counts['bootstrap']} > {max_repair_bootstraps})"
                )
            cost = _policy_assignment_score(assign, le, action.policy, params)
            reward = _mcts_rollout_reward(assign, cost, action.policy, params, hints)
            stats[idx].visits += 1
            stats[idx].reward_sum += reward
            if reward > stats[idx].best_reward:
                stats[idx].best_reward = reward
                stats[idx].best_assign = assign
                stats[idx].best_cost = cost
            item_key = (
                -float(counts["bootstrap"]),
                -float(counts["rescale"]),
                float(reward),
                -float(cost),
            )
            if best is None or item_key > best[0]:
                best = (item_key, assign, action.name)
            if _mcts_target_met(assign, params, hints):
                break
        except Exception as exc:
            reason = f"{action.name}: {type(exc).__name__}: {str(exc)[:160]}"
            invalid_reasons[reason] += 1
            stats[idx].visits += 1
            stats[idx].invalid_reasons[reason] += 1
            stats[idx].reward_sum += _mcts_invalid_reward(action, step)

    if best is not None:
        return best[1]
    common = "; ".join(f"{reason} x{count}" for reason, count in invalid_reasons.most_common(4))
    raise PlacementError(f"bootstrap_mcts found no feasible rollout ({common})")


def _mcts_actions_from_hints(hints: dict[str, Any], params: Params) -> list[MCTSAction]:
    actions: list[MCTSAction] = []
    raw_actions = hints.get("mcts_actions")
    if isinstance(raw_actions, list):
        for idx, item in enumerate(raw_actions):
            if not isinstance(item, dict):
                continue
            policy = _with_default_policy(item.get("policy", item))
            policy["strategy"] = (
                "latency_beam" if str(policy.get("strategy")) == "latency_beam" else "level_preserving"
            )
            actions.append(
                MCTSAction(
                    name=str(item.get("name", f"candidate_action_{idx}")),
                    policy=policy,
                    prior=_float_hint(item.get("prior"), 0.0),
                )
            )
    if not actions:
        actions = _default_bootstrap_mcts_actions(hints, params)
    return actions[: max(1, min(64, _int_hint(hints.get("mcts_action_cap"), 24)))]


def _default_bootstrap_mcts_actions(hints: dict[str, Any], params: Params) -> list[MCTSAction]:
    base = _with_default_policy(hints)
    base.update(
        {
            "strategy": "level_preserving",
            "allow_seed_fallback": False,
            "refresh_fanout_at_level_floor": False,
            "min_internal_level": params.lvl_lb,
            "max_scale_candidates": min(32, _int_hint(base.get("max_scale_candidates"), 24)),
            "bootstrap_penalty": max(
                1_000_000_000.0, _float_hint(base.get("bootstrap_penalty"), 1_000_000_000.0)
            ),
            "selection_bootstrap_penalty": max(
                250_000_000.0, _float_hint(base.get("selection_bootstrap_penalty"), 0.0)
            ),
            "rescale_penalty": max(0.0, _float_hint(base.get("rescale_penalty"), 0.0)),
            "level_drop_penalty": max(0.0, _float_hint(base.get("level_drop_penalty"), 0.0)),
            "scale_lattice": str(base.get("scale_lattice", "waterline_sf")),
            "boundary_scale_policy": str(base.get("boundary_scale_policy", "frontier")),
            "boundary_state_cap": max(1, min(8, _int_hint(base.get("boundary_state_cap"), 3))),
            "bootstrap_anchor_count": max(0, _int_hint(base.get("bootstrap_anchor_count"), 0)),
        }
    )
    target = max(0, _int_hint(base.get("target_bootstrap_count"), 0))
    variants = [
        (
            "strict_no_bootstrap",
            {
                **base,
                "forbid_bootstrap": True,
                "allow_bootstrap": False,
                "max_scale_candidates": min(32, max(20, _int_hint(base.get("max_scale_candidates"), 24))),
                "boundary_scale_policy": "low",
                "bootstrap_anchor_count": 0,
            },
            0.30,
        ),
        (
            "no_bootstrap_repair",
            {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": False,
                "max_scale_candidates": min(32, max(24, _int_hint(base.get("max_scale_candidates"), 24))),
                "boundary_scale_policy": "waterline",
                "bootstrap_anchor_count": 0,
            },
            0.20,
        ),
        (
            "minimal_bootstrap_repair",
            {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "bootstrap_penalty": max(2_500_000_000.0, _float_hint(base.get("bootstrap_penalty"), 0.0)),
                "selection_bootstrap_penalty": max(
                    500_000_000.0, _float_hint(base.get("selection_bootstrap_penalty"), 0.0)
                ),
                "level_drop_penalty": 0.0,
                "rescale_penalty": max(25_000.0, _float_hint(base.get("rescale_penalty"), 0.0)),
                "boundary_scale_policy": "frontier",
                "bootstrap_anchor_count": max(1, min(4, target or 2)),
            },
            0.10,
        ),
        (
            "budget_fulfillment_beam",
            {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "selection_bootstrap_penalty": max(
                    1_250_000_000.0,
                    _float_hint(base.get("selection_bootstrap_penalty"), 0.0),
                ),
            },
            0.05,
        ),
        (
            "latency_mcts_repair",
            {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "beam_width": min(8, max(4, _int_hint(base.get("beam_width"), 6))),
                "state_cap_per_node": min(32, max(12, _int_hint(base.get("state_cap_per_node"), 16))),
                "bootstrap_penalty": max(1_500_000_000.0, _float_hint(base.get("bootstrap_penalty"), 0.0)),
                "selection_bootstrap_penalty": max(
                    300_000_000.0, _float_hint(base.get("selection_bootstrap_penalty"), 0.0)
                ),
                "boundary_scale_policy": "sf",
                "bootstrap_anchor_count": max(1, min(6, target or 3)),
            },
            0.0,
        ),
        (
            "boundary_safe_repair",
            {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "refresh_fanout_at_level_floor": True,
                "min_internal_level": max(params.lvl_lb, params.bts_lb + 1),
                "bootstrap_penalty": max(3_000_000_000.0, _float_hint(base.get("bootstrap_penalty"), 0.0)),
                "reserve_penalty": max(100_000.0, _float_hint(base.get("reserve_penalty"), 0.0)),
                "min_transition_reserve": max(2, _int_hint(base.get("min_transition_reserve"), 0)),
                "boundary_scale_policy": "frontier",
                "bootstrap_anchor_count": max(1, min(8, target or 4)),
            },
            -0.10,
        ),
    ]
    return [MCTSAction(name, _with_default_policy(policy), prior) for name, policy, prior in variants]


def _select_mcts_action(
    actions: list[MCTSAction],
    stats: list[_MCTSNodeStats],
    step: int,
    exploration_weight: float,
) -> int:
    for idx, item in enumerate(stats):
        if item.visits == 0:
            return idx
    total = max(1, sum(item.visits for item in stats))
    best_idx = 0
    best_score = float("-inf")
    for idx, item in enumerate(stats):
        mean = item.reward_sum / max(1, item.visits)
        explore = exploration_weight * math.sqrt(math.log(total + 1.0) / max(1, item.visits))
        score = mean + explore + actions[idx].prior
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def _mcts_rollout_reward(
    assign: Assign,
    cost: float,
    action_policy: dict[str, Any],
    params: Params,
    root_hints: dict[str, Any],
) -> float:
    counts = _aggregate_counts([assign])
    target = max(0, _int_hint(root_hints.get("target_bootstrap_count"), 0))
    if target <= 0:
        target = max(0, int(getattr(params, "openevolve_target_bootstrap_count", 0)))
    bootstrap_score = _target_bootstrap_score(target, counts["bootstrap"], None)
    op_penalty = 0.0005 * counts["rescale"] + 0.02 * counts["bootstrap"]
    latency_penalty = min(0.25, max(0.0, cost) / 1_000_000_000_000.0)
    no_bootstrap_bonus = 0.08 if counts["bootstrap"] == 0 else 0.0
    strict_bonus = 0.04 if action_policy.get("forbid_bootstrap") else 0.0
    return bootstrap_score + no_bootstrap_bonus + strict_bonus - op_penalty - latency_penalty


def _mcts_invalid_reward(action: MCTSAction, step: int) -> float:
    return -0.08 + min(0.04, step * 0.001) + max(-0.05, min(0.05, action.prior))


def _mcts_target_met(assign: Assign, params: Params, hints: dict[str, Any]) -> bool:
    target = max(0, _int_hint(hints.get("target_bootstrap_count"), 0))
    if target <= 0:
        target = max(0, int(getattr(params, "openevolve_target_bootstrap_count", 0)))
    return target > 0 and _aggregate_counts([assign])["bootstrap"] <= target


def _policy_assignment_score(
    assign: Assign,
    le: LatencyEstimator | None,
    hints: dict[str, Any],
    params: Params,
) -> float:
    score = _assignment_score(assign, le)
    boundary_scale_penalty = _float_hint(hints.get("boundary_scale_penalty"), 0.2)
    if boundary_scale_penalty > 0.0 and le is not None and assign.tdag.outputs:
        pdag_vout = list(assign.tdag.outputs)[0]
        output_scale = int(assign.v_scl_out.get(pdag_vout, 0))
        rescale_cost = float(le.lin_op_lmaps["rescale_single"][1])
        score += boundary_scale_penalty * rescale_cost * assign.tdag.get_full_size() * output_scale
    selection_bootstrap_penalty = _float_hint(hints.get("selection_bootstrap_penalty"), 0.0)
    if selection_bootstrap_penalty > 0.0:
        counts = _aggregate_counts([assign])
        score += selection_bootstrap_penalty * counts["bootstrap"]
    return score


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
    unit_policies = list(hints.get("unit_policies", []) or [])

    for v in nx.topological_sort(tdag):
        op = tdag.nodes[v]["op"]
        node_hints = _policy_hints_for_node(hints, unit_policies, str(v), dict(tdag.nodes[v]))
        policy = _policy_options(node_hints, params)
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


def _zero_iteration_portfolio_hints() -> dict[str, Any]:
    hints = _low_scale_frontier_policy()
    safe6 = _bootstrap_safe_margin_policy()
    safe6["min_internal_level"] = 6
    safe8 = _bootstrap_safe_margin_policy()
    safe8["min_internal_level"] = 8
    hints["portfolio"] = [
        safe6,
        safe8,
        _noise_guarded_refresh_policy(),
    ]
    return hints


def _bootstrap_mcts_seed_policy(params: Params | None = None) -> dict[str, Any]:
    target = int(getattr(params, "openevolve_target_bootstrap_count", 0) or 0) if params else 0
    rollout_budget = 2 if params is None else max(2, min(16, int(getattr(params, "openevolve_mcts_rollout_budget", 2))))
    budget_aggressive = bool(getattr(params, "openevolve_budget_aggressive", False)) if params else False
    sampled_repair_cap = 128 if budget_aggressive else max(4, target)
    policy = _low_scale_frontier_policy()
    policy.update(
        {
            "strategy": "bootstrap_mcts",
            "budget_aggressive": budget_aggressive,
            "prefer_level_preservation": True,
            "allow_bootstrap": False,
            "forbid_bootstrap": False,
            "allow_seed_fallback": True,
            "refresh_fanout_at_level_floor": False,
            "max_scale_candidates": 24,
            "bootstrap_penalty": 4_000_000_000.0,
            "selection_bootstrap_penalty": 750_000_000.0,
            "rescale_penalty": 25_000.0,
            "level_drop_penalty": 0.0,
            "scale_penalty": 0.0,
            "reserve_penalty": 75_000.0,
            "min_transition_reserve": 1,
            "min_decryptability_reserve": 1,
            "beam_width": 6,
            "state_cap_per_node": 16,
            "scale_lattice": "waterline_sf",
            "mcts_rollout_budget": rollout_budget,
            "mcts_exploration_weight": 1.4,
            "mcts_max_repair_bootstraps": sampled_repair_cap,
            "target_bootstrap_count": target,
            "mcts_action_cap": 2,
            "boundary_state_cap": 1,
        }
    )
    return policy


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
        "reserve_penalty": 0.0,
        "min_transition_reserve": 0,
        "min_decryptability_reserve": 0,
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
        "reserve_penalty": 0.0,
        "min_transition_reserve": 0,
        "min_decryptability_reserve": 0,
        "preferred_node_levels": {},
        "preferred_node_scales": {},
        "preferred_edge_scales": {},
    }


def _balanced_noise_margin_policy() -> dict[str, Any]:
    policy = _low_scale_frontier_policy()
    policy.update(
        {
            "refresh_fanout_at_level_floor": False,
            "max_scale_candidates": 8,
            "allow_seed_fallback": False,
        }
    )
    return policy


def _bootstrap_safe_margin_policy() -> dict[str, Any]:
    policy = _low_scale_frontier_policy()
    policy.update(
        {
            "refresh_fanout_at_level_floor": True,
            "max_scale_candidates": 16,
            "allow_seed_fallback": False,
            "min_internal_level": 8,
            "level_drop_penalty": 50_000_000.0,
            "rescale_penalty": 250_000.0,
            "scale_penalty": 0.0,
            "reserve_penalty": 150_000.0,
            "min_transition_reserve": 6,
            "min_decryptability_reserve": 6,
        }
    )
    return policy


def _noise_guarded_refresh_policy() -> dict[str, Any]:
    policy = _low_scale_frontier_policy()
    policy.update(
        {
            "strategy": "level_preserving",
            "refresh_fanout_at_level_floor": True,
            "max_scale_candidates": 12,
            "allow_bootstrap": False,
            "allow_seed_fallback": False,
            "min_internal_level": 12,
            "bootstrap_penalty": 1_000_000_000.0,
            "rescale_penalty": 500_000.0,
            "level_drop_penalty": 80_000_000.0,
            "scale_penalty": 0.0,
            "reserve_penalty": 250_000.0,
            "min_transition_reserve": 10,
            "min_decryptability_reserve": 8,
        }
    )
    return policy


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
    min_internal_level = _normalize_min_internal_level(hints.get("min_internal_level"), params)
    return {
        "strategy": str(hints.get("strategy", "waterline_seed")),
        "prefer_level_preservation": _bool_hint(hints.get("prefer_level_preservation"), True),
        "allow_bootstrap": _bool_hint(hints.get("allow_bootstrap"), False),
        "forbid_bootstrap": _bool_hint(hints.get("forbid_bootstrap"), False),
        "refresh_fanout_at_level_floor": _bool_hint(
            hints.get("refresh_fanout_at_level_floor"), False
        ),
        "max_scale_candidates": max(3, min(32, _int_hint(hints.get("max_scale_candidates"), 32))),
        "bootstrap_penalty": max(0.0, _float_hint(hints.get("bootstrap_penalty"), 1_000_000_000.0)),
        "rescale_penalty": max(0.0, _float_hint(hints.get("rescale_penalty"), 0.0)),
        "level_drop_penalty": max(0.0, _float_hint(hints.get("level_drop_penalty"), 20_000_000.0)),
        "min_internal_level": min_internal_level,
        "scale_penalty": max(0.0, _float_hint(hints.get("scale_penalty"), 0.0)),
        "boundary_scale_penalty": max(0.0, _float_hint(hints.get("boundary_scale_penalty"), 0.2)),
        "reserve_penalty": max(0.0, _float_hint(hints.get("reserve_penalty"), 0.0)),
        "min_transition_reserve": max(
            0,
            min(params.Sf * max(1, params.lvl_ub), _int_hint(hints.get("min_transition_reserve"), 0)),
        ),
        "min_decryptability_reserve": max(
            0,
            min(params.Sf * max(1, params.lvl_ub), _int_hint(hints.get("min_decryptability_reserve"), 0)),
        ),
        "beam_width": max(1, min(8, _int_hint(hints.get("beam_width"), 4))),
        "state_cap_per_node": max(1, min(32, _int_hint(hints.get("state_cap_per_node"), 8))),
        "scale_lattice": str(hints.get("scale_lattice", "default")),
        "mcts_rollout_budget": max(1, min(256, _int_hint(hints.get("mcts_rollout_budget"), 48))),
        "mcts_exploration_weight": max(
            0.0, min(8.0, _float_hint(hints.get("mcts_exploration_weight"), 1.4))
        ),
        "mcts_max_repair_bootstraps": max(
            0, min(1024, _int_hint(hints.get("mcts_max_repair_bootstraps"), 4))
        ),
        "max_scale": _max_scale(params),
    }


def _policy_hints_for_node(
    base_hints: dict[str, Any],
    unit_policies: list[dict[str, Any]],
    node: str,
    attrs: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base_hints)
    for item in unit_policies:
        if not isinstance(item, dict):
            continue
        if _selector_matches_node(item.get("selector", {}), attrs, node):
            policy = item.get("policy", {})
            if isinstance(policy, dict):
                for key, value in policy.items():
                    if key in _PATCHABLE_POLICY_KEYS:
                        merged[key] = value
    return merged


def _selector_matches_node(selector: Any, attrs: dict[str, Any], node: str = "") -> bool:
    if not isinstance(selector, dict):
        return False
    unit_id = str(selector.get("unit_id") or selector.get("unit") or "")
    node_units = set(_node_unit_ids(attrs))
    if unit_id and unit_id not in node_units:
        return False
    layer = str(selector.get("layer", ""))
    if layer and layer != _node_layer(attrs):
        return False
    nonlinear_kind = str(selector.get("nonlinear_kind") or "")
    if nonlinear_kind and nonlinear_kind != str(_node_nonlinear_kind(attrs) or ""):
        return False
    kind = str(selector.get("kind", ""))
    if kind == "nonlinear" and _node_nonlinear_kind(attrs) is None:
        return False
    scope_contains = str(selector.get("scope_contains", ""))
    if scope_contains and scope_contains not in _node_scope(attrs):
        return False
    op_contains = str(selector.get("op_contains", ""))
    if op_contains and op_contains not in _node_op_tag(attrs):
        return False
    node_id = str(selector.get("node", ""))
    if node_id and node_id != str(node):
        return False
    return True


def _normalize_min_internal_level(value: Any, params: Params) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return max(params.lvl_lb, min(params.lvl_ub, _int_hint(value, params.lvl_lb)))


def _effective_min_internal_level(policy: dict[str, Any], params: Params) -> int:
    value = policy.get("min_internal_level")
    return params.lvl_lb if value is None else int(value)


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
    min_internal_level = _effective_min_internal_level(policy, params)
    levels = [
        level
        for level in _level_candidates(params, preferred_level)
        if level >= min_internal_level
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
                policy["min_internal_level"] is not None
                and level <= min_internal_level
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
    elif policy.get("forbid_bootstrap"):
        raise PlacementError(f"no no-bootstrap incoming level/scale for node {v}")
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
        min_level = _effective_min_internal_level(policy, params)
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
                and policy["min_internal_level"] is not None
                and out_level <= min_level
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
        and policy["min_internal_level"] is not None
        and in_level <= _effective_min_internal_level(policy, params) + 1
        and tdag.out_degree(v) > 1
        and best_bootstrap is not None
    ):
        _, out_level, out_scale = best_bootstrap
        return out_level, out_scale
    if best_no_bootstrap is not None and not policy["allow_bootstrap"]:
        _, out_level, out_scale = best_no_bootstrap
        return out_level, out_scale
    if policy.get("forbid_bootstrap"):
        raise PlacementError(f"no no-bootstrap output state for node {v}")
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
    cache = policy.setdefault("_transition_score_cache", {})
    cache_key = (int(in_lvl), int(in_scl), int(out_lvl), int(out_scl))
    if cache_key in cache:
        return cache[cache_key]
    if not _transition_ok(params, in_lvl, in_scl, out_lvl, out_scl):
        cache[cache_key] = None
        return None
    try:
        cost = float(le.resbts_cost(in_lvl, in_scl, out_lvl, out_scl)) if le is not None else 0.0
    except Exception:
        cache[cache_key] = None
        return None
    counts = _transition_count_values(params, in_lvl, in_scl, out_lvl, out_scl)
    reserve_penalty = float(policy.get("reserve_penalty", 0.0))
    reserve_deficit = 0.0
    if reserve_penalty > 0.0:
        reserve_deficit += max(
            0.0,
            float(policy.get("min_transition_reserve", 0))
            - float(_transition_reserve_bits(params, in_lvl, in_scl, out_lvl, out_scl)),
        )
        reserve_deficit += max(
            0.0,
            float(policy.get("min_decryptability_reserve", 0))
            - float(_decryptability_reserve_bits(params, out_lvl, out_scl)),
        )
    score = (
        cost
        + policy["bootstrap_penalty"] * counts["bootstrap"]
        + policy["rescale_penalty"] * counts["rescale"]
        + policy["level_drop_penalty"] * max(0, in_lvl - out_lvl)
        + reserve_penalty * reserve_deficit
    )
    cache[cache_key] = score
    return score


def _decryptability_reserve_bits(params: Params, level: int, scale: int) -> int:
    return int(params.Sf * (level - params.lvl_lb + 2) - 7 - scale)


def _linear_noise_budget_bits(params: Params, level: int, scale: int) -> int:
    return int(params.Sf * level - scale)


def _transition_reserve_bits(
    params: Params,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
) -> int:
    """ILP-style level/scale slack for one legal rescale/bootstrap transition."""

    if params.check_res(in_lvl, in_scl, out_lvl, out_scl):
        return _linear_noise_budget_bits(params, in_lvl, in_scl) - _linear_noise_budget_bits(
            params, out_lvl, out_scl
        )
    bootstrap_input = int(params.Sf * (in_lvl - params.bts_lb + 1) - in_scl)
    bootstrap_output = _decryptability_reserve_bits(params, out_lvl, out_scl)
    return min(bootstrap_input, bootstrap_output)


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
    return params.is_decryptable_state(level, scale)


def _assert_decryptable(params: Params, level: int, scale: int) -> None:
    if not _is_decryptable(params, level, scale):
        raise PlacementError(f"(level={level}, scale={scale}) is not decryptable")


def _transition_ok(params: Params, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
    return valid_transition(params, in_lvl, in_scl, out_lvl, out_scl)


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


def _assignment_reserve_summary(assign: Assign, tdag: Tdag, params: Params) -> dict[str, Any]:
    decryptability_margins: list[int] = []
    transition_reserves: list[int] = []
    bootstrap_input_reserves: list[int] = []

    def add_state(level: int | None, scale: int | None) -> None:
        if level is None or scale is None:
            return
        decryptability_margins.append(_decryptability_reserve_bits(params, int(level), int(scale)))

    for node in tdag.nodes:
        add_state(assign.v_lvl_in.get(node), assign.v_scl_in.get(node))
        add_state(assign.v_lvl_out.get(node), assign.v_scl_out.get(node))
    for edge, level in assign.e_lvl_out.items():
        add_state(level, assign.e_scl_out.get(edge))

    for node in tdag.nodes:
        if tdag.nodes[node]["op"] == "constant":
            continue
        if node in assign.v_lvl_in and node in assign.v_scl_in and node in assign.v_lvl_out and node in assign.v_scl_out:
            reserve = _transition_reserve_bits(
                params,
                assign.v_lvl_in[node],
                assign.v_scl_in[node],
                assign.v_lvl_out[node],
                assign.v_scl_out[node],
            )
            transition_reserves.append(reserve)
            if not params.check_res(
                assign.v_lvl_in[node],
                assign.v_scl_in[node],
                assign.v_lvl_out[node],
                assign.v_scl_out[node],
            ):
                bootstrap_input_reserves.append(
                    int(params.Sf * (assign.v_lvl_in[node] - params.bts_lb + 1) - assign.v_scl_in[node])
                )
        for pred in tdag.predecessors(node):
            if tdag.nodes[pred]["op"] == "constant":
                continue
            edge = (pred, node)
            if edge not in assign.e_lvl_out or edge not in assign.e_scl_out:
                continue
            reserve = _transition_reserve_bits(
                params,
                assign.v_lvl_out[pred],
                assign.v_scl_out[pred],
                assign.e_lvl_out[edge],
                assign.e_scl_out[edge],
            )
            transition_reserves.append(reserve)
            if not params.check_res(
                assign.v_lvl_out[pred],
                assign.v_scl_out[pred],
                assign.e_lvl_out[edge],
                assign.e_scl_out[edge],
            ):
                bootstrap_input_reserves.append(
                    int(params.Sf * (assign.v_lvl_out[pred] - params.bts_lb + 1) - assign.v_scl_out[pred])
                )

    def stats(values: list[int]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "min": None, "p10": None, "median": None}
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": int(ordered[0]),
            "p10": int(ordered[min(len(ordered) - 1, max(0, len(ordered) // 10))]),
            "median": int(ordered[len(ordered) // 2]),
        }

    return {
        "decryptability": stats(decryptability_margins),
        "transition": stats(transition_reserves),
        "bootstrap_input": stats(bootstrap_input_reserves),
        "min_decryptability_reserve_bits": (
            min(decryptability_margins) if decryptability_margins else None
        ),
        "min_transition_reserve_bits": min(transition_reserves) if transition_reserves else None,
        "min_bootstrap_input_reserve_bits": (
            min(bootstrap_input_reserves) if bootstrap_input_reserves else None
        ),
    }


def _aggregate_reserve_summary(assignments: list[Assign], tdag: Tdag, params: Params) -> dict[str, Any]:
    summaries = [_assignment_reserve_summary(assign, tdag, params) for assign in assignments]
    if not summaries:
        return {
            "min_decryptability_reserve_bits": None,
            "min_transition_reserve_bits": None,
            "min_bootstrap_input_reserve_bits": None,
        }

    def min_present(key: str) -> int | None:
        values = [item.get(key) for item in summaries if item.get(key) is not None]
        return min(values) if values else None

    return {
        "assignment_count": len(summaries),
        "min_decryptability_reserve_bits": min_present("min_decryptability_reserve_bits"),
        "min_transition_reserve_bits": min_present("min_transition_reserve_bits"),
        "min_bootstrap_input_reserve_bits": min_present("min_bootstrap_input_reserve_bits"),
        "first_assignment": summaries[0],
    }


def _reserve_quality_score(summary: dict[str, Any]) -> float:
    values = [
        summary.get("min_decryptability_reserve_bits"),
        summary.get("min_transition_reserve_bits"),
    ]
    if summary.get("min_bootstrap_input_reserve_bits") is not None:
        values.append(summary.get("min_bootstrap_input_reserve_bits"))
    numeric = [_finite_float(value, float("-inf")) for value in values if value is not None]
    if not numeric:
        return 0.0
    weakest = min(numeric)
    return max(0.0, min(1.0, weakest / 16.0))


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


def _target_bootstrap_score(
    target: int | float | None,
    candidate: int | float,
    reference: int | float | None = None,
) -> float:
    target_int = max(0, _safe_int(target, 0))
    candidate_int = max(0, _safe_int(candidate, 0))
    if target_int <= 0:
        return 1.0
    if candidate_int <= target_int:
        return 1.0
    reference_value = _safe_int(reference, 0)
    if reference_value <= target_int:
        reference_int = max(candidate_int, target_int * 128, target_int + 1)
    else:
        reference_int = max(candidate_int, reference_value, target_int + 1)
    try:
        numerator = math.log(float(candidate_int) / float(target_int))
        denominator = math.log(float(reference_int) / float(target_int))
    except (ValueError, ZeroDivisionError):
        progress_score = 0.0
    else:
        progress_score = max(0.0, min(1.0, 1.0 - numerator / denominator))
    absolute_score = math.sqrt(float(target_int) / float(candidate_int))
    return max(progress_score, max(0.0, min(1.0, 0.35 * absolute_score)))


def _context_target_bootstrap_count(context: dict[str, Any]) -> int:
    return max(
        0,
        _safe_int(
            context.get("harness", {}).get(
                "target_bootstrap_count",
                context.get("params", {}).get("openevolve_target_bootstrap_count", 0),
            ),
            0,
        ),
    )


def _context_budget_aggressive(context: dict[str, Any]) -> bool:
    value = context.get("harness", {}).get(
        "budget_aggressive",
        context.get("params", {}).get("openevolve_budget_aggressive", True),
    )
    return _bool_hint(value, True)


def _compact_policy_summary(hints: dict[str, Any]) -> str:
    keys = [
        "strategy",
        "prefer_level_preservation",
        "allow_bootstrap",
        "forbid_bootstrap",
        "allow_seed_fallback",
        "refresh_fanout_at_level_floor",
        "max_scale_candidates",
        "bootstrap_penalty",
        "rescale_penalty",
        "level_drop_penalty",
        "min_internal_level",
        "scale_penalty",
        "reserve_penalty",
        "min_transition_reserve",
        "min_decryptability_reserve",
        "beam_width",
        "state_cap_per_node",
        "scale_lattice",
        "budget_aggressive",
        "selection_bootstrap_penalty",
        "mcts_rollout_budget",
        "mcts_exploration_weight",
        "mcts_max_repair_bootstraps",
        "mcts_action_cap",
        "boundary_scale_policy",
        "boundary_state_cap",
        "preferred_boundary_scale",
        "boundary_scale",
        "bootstrap_anchor_count",
        "bootstrap_anchor_level",
        "bootstrap_anchors",
    ]
    summary = {key: hints.get(key) for key in keys if key in hints}
    for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
        value = hints.get(key)
        if isinstance(value, dict) and value:
            summary[f"{key}_count"] = len(value)
    if hints.get("unit_policies"):
        summary["unit_policy_count"] = len(hints.get("unit_policies") or [])
    if hints.get("__repair_reasons"):
        summary["repair_count"] = _repair_count(hints)
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
                "comment": str(attrs.get("comment", "")),
                "scope": _node_scope(attrs),
                "layer": _node_layer(attrs),
                "nonlinear_kind": _node_nonlinear_kind(attrs),
                "unit_ids": _node_unit_ids(attrs),
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


def _placement_units(pdag: Tdag, params: Params, max_units: int) -> list[dict[str, Any]]:
    unit_nodes: dict[str, dict[str, Any]] = {}
    depth_to_output = _depth_to_output(pdag, [str(node) for node in nx.topological_sort(pdag)])
    for node, attrs in pdag.nodes(data=True):
        if attrs.get("op") in {"input", "constant"}:
            continue
        for unit_id in _node_unit_ids(attrs):
            unit = unit_nodes.setdefault(
                unit_id,
                {
                    "id": unit_id,
                    "selector": _selector_from_unit_id(unit_id),
                    "node_ids": [],
                    "op_histogram": {},
                    "min_depth_to_output": None,
                    "max_depth_to_output": None,
                    "max_out_degree": 0,
                    "min_scale_lb_out": None,
                },
            )
            unit["node_ids"].append(str(node))
            op = str(attrs.get("op", ""))
            unit["op_histogram"][op] = unit["op_histogram"].get(op, 0) + 1
            depth = int(depth_to_output.get(str(node), 0))
            unit["min_depth_to_output"] = (
                depth
                if unit["min_depth_to_output"] is None
                else min(int(unit["min_depth_to_output"]), depth)
            )
            unit["max_depth_to_output"] = (
                depth
                if unit["max_depth_to_output"] is None
                else max(int(unit["max_depth_to_output"]), depth)
            )
            unit["max_out_degree"] = max(int(unit["max_out_degree"]), int(pdag.out_degree(node)))
            lb = params.scale_lower_bound(str(node), attrs, "out")
            unit["min_scale_lb_out"] = (
                lb if unit["min_scale_lb_out"] is None else min(int(unit["min_scale_lb_out"]), lb)
            )
    ranked = sorted(
        unit_nodes.values(),
        key=lambda unit: (
            0 if unit["selector"].get("kind") == "nonlinear" else 1,
            str(unit["selector"].get("layer", "")),
            str(unit["selector"].get("nonlinear_kind", "")),
            str(unit["id"]),
        ),
    )
    return ranked[: max(1, int(max_units))]


def _selector_from_unit_id(unit_id: str) -> dict[str, Any]:
    parts = unit_id.split(":")
    if len(parts) >= 3 and parts[0] == "nonlinear":
        return {"kind": "nonlinear", "layer": parts[1], "nonlinear_kind": parts[2], "unit_id": unit_id}
    if len(parts) >= 2 and parts[0] == "layer":
        return {"kind": "layer", "layer": parts[1], "unit_id": unit_id}
    return {"kind": "unknown", "unit_id": unit_id}


def _unit_op_histogram(units: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for unit in units:
        for op, count in unit.get("op_histogram", {}).items():
            counts[str(op)] += int(count)
    return dict(sorted(counts.items()))


def _unit_budget_summary(units: list[dict[str, Any]], io_budgets_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "unit_count": len(units),
        "budget_count": len(io_budgets_list),
        "has_bypass_budgets": any("maino_v" in budget for budget in io_budgets_list),
    }


def _unit_resilience_summary(units: list[dict[str, Any]], params: Params) -> dict[str, Any]:
    if params.resilience_profile is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "unit_count": len(units),
        "profile": params.resilience_profile.describe(),
    }


def _unit_hotspots_from_profile(
    units: list[dict[str, Any]], bottleneck_summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    hotspots = []
    for item in bottleneck_summary or []:
        location = str(item.get("location", ""))
        matched = [
            unit["id"]
            for unit in units
            if _selector_matches_location(unit.get("selector", {}), location)
        ][:5]
        if matched:
            hotspots.append({"location": location, "unit_ids": matched, "score": item.get("score", 0)})
    return hotspots[:20]


def _selector_matches_location(selector: dict[str, Any], location: str) -> bool:
    layer = str(selector.get("layer", ""))
    nonlinear_kind = str(selector.get("nonlinear_kind", ""))
    return bool((not layer or layer in location) and (not nonlinear_kind or nonlinear_kind in location))


def _node_scope(attrs: dict[str, Any]) -> str:
    comment = str(attrs.get("comment", ""))
    for part in comment.split(";"):
        if part.startswith("scope="):
            return part.split("=", 1)[1]
    return ""


def _node_op_tag(attrs: dict[str, Any]) -> str:
    comment = str(attrs.get("comment", ""))
    for part in comment.split(";"):
        if part.startswith("op="):
            return part.split("=", 1)[1]
    return str(attrs.get("op", ""))


def _node_layer(attrs: dict[str, Any]) -> str:
    comment = str(attrs.get("comment", ""))
    for part in comment.split(";"):
        if part.startswith("layer="):
            return part.split("=", 1)[1]
    scope = _node_scope(attrs)
    pieces = [piece for piece in scope.split(".") if piece]
    if "layer" in pieces:
        idx = pieces.index("layer")
        if idx + 1 < len(pieces):
            return f"layer.{pieces[idx + 1]}"
    if scope.startswith("bert.pooler"):
        return "pooler"
    if scope.startswith("bert.classifier"):
        return "classifier"
    return "global"


def _node_nonlinear_kind(attrs: dict[str, Any]) -> str | None:
    text = f"{_node_scope(attrs)};{_node_op_tag(attrs)};{attrs.get('comment', '')}".lower()
    if "reciprocal" in text or "newton" in text:
        return "reciprocal"
    if "qk_softmax" in text or "softmax" in text:
        return "attention_softmax"
    if "activation" in text or "gelu" in text or "silu" in text or "quadratic" in text:
        return "activation"
    if "layernorm" in text or "layer_norm" in text or "polynorm" in text or "norm" in text:
        return "norm"
    return None


def _node_unit_ids(attrs: dict[str, Any]) -> list[str]:
    layer = _node_layer(attrs)
    unit_ids = [f"layer:{layer}"]
    nonlinear = _node_nonlinear_kind(attrs)
    if nonlinear is not None:
        unit_ids.insert(0, f"nonlinear:{layer}:{nonlinear}")
    return unit_ids


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


def target_units(context: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    """Return graph-first placement units ranked by hotspot/depth/fanout signals."""

    units = [unit for unit in context.get("placement_units", []) or [] if isinstance(unit, dict)]
    hotspots = {
        str(item.get("unit_id")): _finite_float(item.get("hotspot_score"), 0.0)
        for item in context.get("unit_hotspots", []) or []
        if isinstance(item, dict)
    }
    graph_nodes = context.get("graph_summary", {}).get("nodes", {})

    def unit_rank(unit: dict[str, Any]) -> tuple[float, int, int, str]:
        node_ids = [str(node) for node in unit.get("node_ids", []) or []]
        fanout = 0
        depth = 0
        for node in node_ids[:64]:
            info = graph_nodes.get(node, {}) if isinstance(graph_nodes, dict) else {}
            if isinstance(info, dict):
                fanout += _safe_int(info.get("out_degree"), 0)
                depth = max(depth, _safe_int(info.get("depth_to_output"), 0))
        return (
            -hotspots.get(str(unit.get("id", "")), 0.0),
            -fanout,
            -depth,
            str(unit.get("id", "")),
        )

    ranked = sorted(units, key=unit_rank)
    return ranked[:limit] if limit is not None else ranked


def candidate_actions(
    context: dict[str, Any],
    *,
    target_bootstraps: int | None = None,
    action_cap: int = 16,
) -> list[dict[str, Any]]:
    """Build bounded MCTS actions without relying on BERT-specific labels."""

    ckks = _ckks_dict(context)
    target = (
        int(target_bootstraps)
        if target_bootstraps is not None
        else _context_target_bootstrap_count(context)
    )
    base = {
        "strategy": "level_preserving",
        "prefer_level_preservation": True,
        "allow_seed_fallback": False,
        "refresh_fanout_at_level_floor": False,
        "max_scale_candidates": 24,
        "bootstrap_penalty": 4_000_000_000.0,
        "selection_bootstrap_penalty": 750_000_000.0,
        "rescale_penalty": 25_000.0,
        "level_drop_penalty": 0.0,
        "scale_penalty": 0.0,
        "reserve_penalty": 75_000.0,
        "min_transition_reserve": 1,
        "min_decryptability_reserve": 1,
        "beam_width": 6,
        "state_cap_per_node": 16,
        "scale_lattice": "waterline_sf",
        "target_bootstrap_count": target,
        "boundary_scale_policy": "frontier",
        "boundary_state_cap": 2,
        "bootstrap_anchor_count": 0,
    }
    actions: list[dict[str, Any]] = [
        {
            "name": "strict_no_bootstrap",
            "prior": 0.30,
            "policy": {
                **base,
                "forbid_bootstrap": True,
                "allow_bootstrap": False,
                "boundary_scale_policy": "low",
            },
        },
        {
            "name": "repair_without_planned_bootstrap",
            "prior": 0.20,
            "policy": {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": False,
                "boundary_scale_policy": "waterline",
            },
        },
        {
            "name": "minimal_bootstrap_repair",
            "prior": 0.10,
            "policy": {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "frontier",
                "bootstrap_anchor_count": max(1, min(4, target or 2)),
            },
        },
        {
            "name": "budget_fulfillment_beam",
            "prior": 0.05,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "selection_bootstrap_penalty": 1_250_000_000.0,
            },
        },
        {
            "name": "latency_mcts_repair",
            "prior": 0.00,
            "policy": {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "sf",
                "bootstrap_anchor_count": max(1, min(6, target or 3)),
            },
        },
    ]
    for idx, unit in enumerate(target_units(context, max(0, action_cap - len(actions)))):
        preferred_levels = {}
        preferred_scales = {}
        for node in unit.get("node_ids", []) or []:
            preferred_levels[str(node)] = int(ckks["lvl_lb"])
            preferred_scales[str(node)] = int(ckks["Sw"])
        actions.append(
            {
                "name": f"unit_low_bootstrap_{idx}",
                "prior": -0.03,
                "policy": {
                    **base,
                    "forbid_bootstrap": False,
                    "allow_bootstrap": True,
                    "preferred_node_levels": preferred_levels,
                    "preferred_node_scales": preferred_scales,
                },
            }
        )
    return actions[: max(1, int(action_cap))]


def constraint_violations(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Return compact static violation/hotspot hints for evolved MCTS policies."""

    violations = []
    for item in context.get("placement_profile", [])[:32]:
        if isinstance(item, dict):
            violations.append(
                {
                    "location": str(item.get("location", "")),
                    "score": _safe_int(item.get("score"), 0),
                    "reason": str(item.get("reason", item.get("kind", "hotspot"))),
                }
            )
    return violations


def repair_partial(policy: dict[str, Any], *, allow_bootstrap: bool = True) -> dict[str, Any]:
    repaired = _with_default_policy(policy)
    repaired["strategy"] = "level_preserving"
    repaired["forbid_bootstrap"] = False
    repaired["allow_bootstrap"] = bool(allow_bootstrap)
    repaired["allow_seed_fallback"] = False
    repaired["refresh_fanout_at_level_floor"] = False
    return repaired


def score_rollout(metrics: dict[str, Any], *, target_bootstraps: int = 9) -> float:
    bootstrap_count = _finite_float(metrics.get("bootstrap_count"), float("inf"))
    validity = _finite_float(metrics.get("candidate_validity", metrics.get("validity")), 0.0)
    repairs = _finite_float(metrics.get("repair_count"), 0.0)
    return (
        0.55 * _target_bootstrap_score(target_bootstraps, bootstrap_count, None)
        + 0.35 * max(0.0, min(1.0, validity))
        + 0.10 / (1.0 + repairs)
    )


class PlacementMCTS:
    """Constrained API exposed to OpenEvolve for low-bootstrap placement search."""

    def __init__(self, context: dict[str, Any]):
        self.context = context

    def low_bootstrap_seed(
        self,
        *,
        target_bootstraps: int = 9,
        rollout_budget: int = 48,
        exploration_weight: float = 1.4,
        max_repair_bootstraps: int = 4,
        action_cap: int = 16,
    ) -> dict[str, Any]:
        if bool(self.context.get("harness", {}).get("budget_aggressive", False)):
            max_repair_bootstraps = max(int(max_repair_bootstraps), 128)
        else:
            max_repair_bootstraps = max(int(max_repair_bootstraps), int(target_bootstraps))
        policy = _bootstrap_mcts_seed_policy()
        policy.update(
            {
                "budget_aggressive": bool(
                    self.context.get("harness", {}).get("budget_aggressive", False)
                ),
                "target_bootstrap_count": int(target_bootstraps),
                "mcts_rollout_budget": int(rollout_budget),
                "mcts_exploration_weight": float(exploration_weight),
                "mcts_max_repair_bootstraps": int(max_repair_bootstraps),
                "mcts_action_cap": int(action_cap),
                "mcts_actions": candidate_actions(
                    self.context,
                    target_bootstraps=int(target_bootstraps),
                    action_cap=int(action_cap),
                ),
            }
        )
        return {
            "api_version": "placement-mcts-v1",
            **policy,
            "placement_records": [],
        }

    def candidate_actions(self, **kwargs: Any) -> list[dict[str, Any]]:
        return candidate_actions(self.context, **kwargs)

    def target_units(self, limit: int | None = None) -> list[dict[str, Any]]:
        return target_units(self.context, limit)

    def constraint_violations(self) -> list[dict[str, Any]]:
        return constraint_violations(self.context)

    def repair_partial(self, policy: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return repair_partial(policy, **kwargs)

    def score_rollout(self, metrics: dict[str, Any], **kwargs: Any) -> float:
        return score_rollout(metrics, **kwargs)


class PlacementBuilder:
    """Small helper API exposed to OpenEvolve candidate programs."""

    def __init__(self, context: dict[str, Any]):
        self.context = context
        self.constraints = PlacementConstraints(context)
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
            "reserve_penalty": 0.0,
            "min_transition_reserve": 0,
            "min_decryptability_reserve": 0,
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
            "reserve_penalty": 75_000.0,
            "min_transition_reserve": 4,
            "min_decryptability_reserve": 4,
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

    def budget_fulfillment_beam(self, **overrides: Any) -> dict[str, Any]:
        """Beam policy that prioritizes directly satisfying requested budget records."""
        ckks = _ckks_dict(self.context)
        policy = self.latency_beam()["policy"]
        policy.update(
            {
                "strategy": "latency_beam",
                "allow_bootstrap": True,
                "allow_seed_fallback": True,
                "budget_aggressive": True,
                "refresh_fanout_at_level_floor": False,
                "beam_width": 6,
                "state_cap_per_node": 16,
                "max_scale_candidates": 24,
                "bootstrap_penalty": 2_500_000_000.0,
                "selection_bootstrap_penalty": 250_000_000.0,
                "rescale_penalty": 150_000.0,
                "level_drop_penalty": 2_000_000.0,
                "scale_penalty": 200.0,
                "reserve_penalty": 75_000.0,
                "min_internal_level": int(ckks["lvl_lb"]),
                "min_transition_reserve": 2,
                "min_decryptability_reserve": 2,
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

    def balanced_noise_margin(self, **overrides: Any) -> dict[str, Any]:
        """Policy seed that trades some latency for a larger CKKS noise margin."""
        policy = _balanced_noise_margin_policy()
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def bootstrap_safe_margin(self, **overrides: Any) -> dict[str, Any]:
        """Policy seed that preserves extra level headroom before bootstraps."""
        policy = _bootstrap_safe_margin_policy()
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def noise_guarded_refresh(self, **overrides: Any) -> dict[str, Any]:
        """Safety-first seed for deep PolyBERT graphs with tight noise margins."""
        policy = _noise_guarded_refresh_policy()
        policy.update(overrides)
        return {"api_version": "placement-builder-v1", "policy": policy, "placement_records": []}

    def profile_layer_refresh(
        self,
        *,
        include_attention: bool = True,
        scale_bits: int | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Anchor refreshes at HE-native transformer block boundary nodes.

        The profiler reasons at layer/module boundaries, while an unconstrained
        scheduler may refresh inside residual or Goldschmidt subexpressions
        where decoded values are much larger. This seed prefers the final
        ``polynorm_proxy`` nodes for each layer, and optionally the attention
        sublayer norm proxies, so OpenEvolve starts from numerically meaningful
        refresh sites.
        """

        ckks = _ckks_dict(self.context)
        preferred_levels: dict[str, int] = {}
        preferred_scales: dict[str, int] = {}
        target_scale = int(scale_bits) if scale_bits is not None else int(ckks["Sf"])
        target_scale = max(int(ckks["Sw"]), min(int(ckks["max_scale"]), target_scale))
        nodes = (self.context.get("tdag") or {}).get("nodes", {})
        for node, info in nodes.items():
            if not isinstance(info, dict) or str(info.get("op")) in {"constant", "input"}:
                continue
            comment = str(info.get("comment", ""))
            if "bert.encoder.layer." not in comment:
                continue
            is_norm_boundary = (
                "op=polynorm_proxy" in comment
                or "op=layer_norm" in comment
                or "LayerNorm" in comment
            )
            if not is_norm_boundary:
                continue
            is_attention = ".attention.output.LayerNorm" in comment
            is_block_output = ".output.LayerNorm" in comment and ".attention.output.LayerNorm" not in comment
            if not is_block_output and not (include_attention and is_attention):
                continue
            preferred_levels[str(node)] = int(ckks["lvl_ub"])
            preferred_scales[str(node)] = target_scale

        policy = _bootstrap_safe_margin_policy()
        policy.update(
            {
                "strategy": "level_preserving",
                "allow_bootstrap": False,
                "allow_seed_fallback": False,
                "refresh_fanout_at_level_floor": False,
                "max_scale_candidates": 20,
                "bootstrap_penalty": 650_000_000.0,
                "rescale_penalty": 350_000.0,
                "level_drop_penalty": 60_000_000.0,
                "min_internal_level": max(int(ckks["lvl_lb"]), int(ckks["bts_lb"]) + 1),
                "scale_penalty": 0.0,
                "reserve_penalty": 150_000.0,
                "min_transition_reserve": 6,
                "min_decryptability_reserve": 6,
                "preferred_node_levels": preferred_levels,
                "preferred_node_scales": preferred_scales,
                "preferred_edge_scales": {},
            }
        )
        for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
            values = overrides.pop(key, None)
            if isinstance(values, dict):
                policy[key].update(values)
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

    def target_unit(self, unit_id: str, operation: str = "prefer_high_level") -> "PlacementBuilder":
        self._patches.append({"op": "target_unit", "unit_id": str(unit_id), "operation": str(operation)})
        return self

    def target_nonlinear(
        self, nonlinear_kind: str, layer: str = "", operation: str = "prefer_high_level"
    ) -> "PlacementBuilder":
        self._patches.append(
            {
                "op": "target_nonlinear",
                "nonlinear_kind": str(nonlinear_kind),
                "layer": str(layer),
                "operation": str(operation),
            }
        )
        return self

    def unit_policy(self, selector: dict[str, Any] | str, policy: dict[str, Any]) -> dict[str, Any]:
        if isinstance(selector, str):
            selector = {"unit_id": selector}
        return {"selector": dict(selector), "policy": dict(policy)}

    def unit_portfolio(
        self,
        global_policy: dict[str, Any] | None = None,
        *unit_policies: dict[str, Any],
        **overrides: Any,
    ) -> dict[str, Any]:
        portfolio = overrides.pop("portfolio", None)
        policy = dict(global_policy or self.low_scale_frontier()["policy"])
        policy.update(overrides)
        normalized_portfolio = (
            [_with_default_policy(item) for item in portfolio if isinstance(item, dict)]
            if isinstance(portfolio, list)
            else []
        )
        return {
            "api_version": "placement-builder-v1",
            "global_policy": policy,
            "portfolio": normalized_portfolio,
            "unit_policies": [item for item in unit_policies if isinstance(item, dict)],
            "placement_records": [],
            "patches": list(self._patches),
        }

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


class PlacementConstraints:
    """Solver-free CKKS legality helpers mirroring Orbit's ILP constraints."""

    def __init__(self, context_or_params: Any):
        self.context_or_params = context_or_params
        self.ckks = _ckks_dict(context_or_params)

    @property
    def sf(self) -> int:
        return int(self.ckks["Sf"])

    @property
    def lvl_lb(self) -> int:
        return int(self.ckks["lvl_lb"])

    @property
    def lvl_ub(self) -> int:
        return int(self.ckks["lvl_ub"])

    @property
    def sw(self) -> int:
        return int(self.ckks["Sw"])

    @property
    def csw(self) -> int:
        return int(self.ckks["Csw"])

    @property
    def max_scale(self) -> int:
        return int(self.ckks["max_scale"])

    def decryptability_bound(self, level: int) -> int:
        if isinstance(self.context_or_params, Params):
            return self.context_or_params.decryptable_scale_bound(level)
        return int(self.sf * (int(level) - self.lvl_lb + 2) - 7)

    def boundary_output_scale_bound(self, level: int) -> int:
        # This is the explicit terminal-boundary constraint in add_ilp_io_budgets.
        if isinstance(self.context_or_params, Params):
            return self.context_or_params.boundary_output_scale_bound(level)
        return int(self.sf * (int(level) + 1) - 7)

    def is_decryptable(self, level: int, scale: int) -> bool:
        if isinstance(self.context_or_params, Params):
            return self.context_or_params.is_decryptable_state(level, scale)
        return (
            self.lvl_lb <= int(level) <= self.lvl_ub
            and 0 <= int(scale) <= self.max_scale
            and int(scale) <= self.decryptability_bound(int(level))
        )

    def valid_transition(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        return valid_transition(self.context_or_params, in_lvl, in_scl, out_lvl, out_scl)

    def transition_slack(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> dict[str, Any]:
        return transition_slack(self.context_or_params, in_lvl, in_scl, out_lvl, out_scl)

    def transition_cost(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> float:
        return transition_cost(self.context_or_params, in_lvl, in_scl, out_lvl, out_scl)

    def node_scale_lower_bound(self, node: str, port: str = "out") -> int:
        node_info = self._node_info(node)
        key = "scale_lb_in" if port == "in" else "scale_lb_out"
        value = node_info.get(key)
        if value is None:
            return self.sw
        return int(value)

    def edge_scale_lower_bound(self, u: str, v: str) -> int:
        source = self._node_info(u)
        if str(source.get("op", "")) == "constant":
            return max(self.csw, self.node_scale_lower_bound(v, "in"))
        return max(
            self.node_scale_lower_bound(u, "out"),
            self.node_scale_lower_bound(v, "in"),
        )

    def mul_input_scale(self, predecessor_scales: list[int]) -> int:
        if not predecessor_scales:
            raise ValueError("mul_input_scale needs at least one predecessor scale")
        if len(predecessor_scales) == 1:
            return int(predecessor_scales[0]) * 2
        return int(sum(int(value) for value in predecessor_scales))

    def candidate_scales(
        self,
        values: list[Any],
        lower_bound: int | None = None,
        max_count: int = 32,
    ) -> list[int]:
        return candidate_scales(self.context_or_params, values, lower_bound, max_count)

    def legal_output_states(
        self,
        in_lvl: int,
        in_scl: int,
        *,
        exact_out_lvl: int | None = None,
        lower_bound: int | None = None,
        max_count: int = 32,
        include_bootstrap: bool = True,
    ) -> list[dict[str, Any]]:
        bases = [
            lower_bound,
            in_scl,
            int(in_scl) - self.sf,
            math.ceil(int(in_scl) / 2),
            self.sw,
            self.csw,
            self.sf,
        ]
        scales = self.candidate_scales(
            [value for value in bases if value is not None],
            lower_bound if lower_bound is not None else self.sw,
            max_count,
        )
        if exact_out_lvl is not None and int(exact_out_lvl) >= 0:
            levels = [int(exact_out_lvl)]
        else:
            levels = list(range(self.lvl_ub, self.lvl_lb - 1, -1))
        states = []
        for level in levels:
            for scale in scales:
                if not self.is_decryptable(level, scale):
                    continue
                slack = self.transition_slack(in_lvl, in_scl, level, scale)
                if not slack["valid"]:
                    continue
                if slack["uses_bootstrap"] and not include_bootstrap:
                    continue
                states.append(
                    {
                        "level": int(level),
                        "scale": int(scale),
                        "uses_bootstrap": bool(slack["uses_bootstrap"]),
                        "rescale_count": int(slack["rescale_count"]),
                        "transition_reserve_bits": int(slack["transition_reserve_bits"]),
                        "decryptability_reserve_out_bits": int(
                            slack["decryptability_reserve_out_bits"]
                        ),
                        "cost": float(self.transition_cost(in_lvl, in_scl, level, scale)),
                    }
                )
        states.sort(
            key=lambda item: (
                item["uses_bootstrap"],
                item["cost"],
                -item["transition_reserve_bits"],
                -item["level"],
                item["scale"],
            )
        )
        return states

    def boundary_penalty(self, out_lvl: int, out_scl: int, dag_size: int = 1) -> float:
        bound = self.boundary_output_scale_bound(out_lvl)
        hard_violation = max(0, int(out_scl) - bound)
        return float(1_000_000 * hard_violation + 0.2 * max(1, int(dag_size)) * int(out_scl))

    def _node_info(self, node: str) -> dict[str, Any]:
        if isinstance(self.context_or_params, dict):
            graph_nodes = self.context_or_params.get("graph_summary", {}).get("nodes", {})
            info = graph_nodes.get(str(node))
            if isinstance(info, dict):
                return info
            tdag_nodes = self.context_or_params.get("tdag", {}).get("nodes", {})
            info = tdag_nodes.get(str(node))
            if isinstance(info, dict):
                return info
        return {}


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


def transition_slack(
    context_or_params: Any,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
) -> dict[str, Any]:
    """Expose Orbit ILP level/scale feasibility slack to evolved candidates.

    These fields mirror the original Gurobi/PuLP constraints without importing
    or calling either solver. Candidate programs can use this for guarded
    heuristics; Orbit still validates final assignments with ``Assign.check_assign``.
    """

    ckks = _ckks_dict(context_or_params)
    params = _ParamsProxy(ckks)
    valid = valid_transition(context_or_params, in_lvl, in_scl, out_lvl, out_scl)
    no_bootstrap = params.check_res(in_lvl, in_scl, out_lvl, out_scl)
    rescale_count = max(0, int(math.ceil((int(in_scl) - int(out_scl)) / max(params.Sf, 1))))
    decrypt_in = int(params.Sf * (int(in_lvl) - params.lvl_lb + 2) - 7 - int(in_scl))
    decrypt_out = int(params.Sf * (int(out_lvl) - params.lvl_lb + 2) - 7 - int(out_scl))
    linear_in = int(params.Sf * int(in_lvl) - int(in_scl))
    linear_out = int(params.Sf * int(out_lvl) - int(out_scl))
    bootstrap_input = int(params.Sf * (int(in_lvl) - params.bts_lb + 1) - int(in_scl))
    if no_bootstrap:
        transition_reserve = linear_in - linear_out
    else:
        transition_reserve = min(bootstrap_input, decrypt_out)
    return {
        "valid": bool(valid),
        "uses_bootstrap": bool(valid and not no_bootstrap),
        "rescale_count": int(rescale_count),
        "decryptability_reserve_in_bits": decrypt_in,
        "decryptability_reserve_out_bits": decrypt_out,
        "linear_noise_budget_in_bits": linear_in,
        "linear_noise_budget_out_bits": linear_out,
        "transition_reserve_bits": int(transition_reserve),
        "bootstrap_input_reserve_bits": int(bootstrap_input),
        "violations": _transition_slack_violations(
            params, int(in_lvl), int(in_scl), int(out_lvl), int(out_scl)
        ),
    }


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
        self.max_scale = int(ckks["max_scale"])

    def check_res(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if in_lvl < out_lvl:
            return False
        return self.Sf * in_lvl - in_scl >= self.Sf * out_lvl - out_scl

    def check_resbts(self, in_lvl: int, in_scl: int, out_lvl: int, out_scl: int) -> bool:
        if self.check_res(in_lvl, in_scl, out_lvl, out_scl):
            return True
        if not self.check_res(in_lvl, in_scl, self.bts_lb, self.Sf):
            return False
        r = max(0, int(math.ceil((self.Sf - out_scl) / max(self.Sf, 1))))
        if not (self.bts_lb < out_lvl + r <= self.bts_ub):
            return False
        return self.check_res(out_lvl + r, self.Sf, out_lvl, out_scl)


def _ilp_semantics_summary(params: Params) -> dict[str, Any]:
    return {
        "source": "Orbit ILP VarPool/add_ilp_constraints semantics",
        "solver_free": True,
        "variables": {
            "node": ["v_lvl_in", "v_scl_in", "v_lvl_out", "v_scl_out", "v_use_r", "v_use_b"],
            "edge": ["e_lvl_in", "e_scl_in", "e_lvl_out", "e_scl_out", "e_use_r"],
        },
        "constraints": {
            "decryptable": "scale <= Sf * (level - lvl_lb + 2) - 7",
            "mul_input_scale": "mul input scale equals sum of predecessor output scales",
            "no_bootstrap": [
                "out_lvl <= in_lvl - r",
                "out_scl >= in_scl - Sf * r",
            ],
            "bootstrap": [
                "in_scl <= Sf * (in_lvl - bts_lb + 1)",
                "out_lvl >= bts_lb + 1",
                "out_scl >= Sf",
            ],
            "edge_rescale": [
                "edge_out_lvl <= edge_in_lvl - r",
                "edge_out_scl >= edge_in_scl - Sf * r",
            ],
            "boundary_output": "output_scale <= Sf * (output_level + 1) - 7",
            "local_bounds": "node/edge scales obey resilience scale_lower_bound",
        },
        "objective_terms": [
            "bootstrap latency * use_b",
            "rescale latency * use_r",
            "operator latency at input level",
            "small boundary output scale penalty",
        ],
        "ckks": {
            "Sf": params.Sf,
            "Sw": params.Sw,
            "Csw": params.Csw,
            "lvl_lb": params.lvl_lb,
            "lvl_ub": params.lvl_ub,
            "bts_lb": params.bts_lb,
            "bts_ub": params.bts_ub,
            "max_scale": params.max_scale(),
        },
    }


def _transition_slack_violations(
    params: _ParamsProxy,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
) -> list[str]:
    violations: list[str] = []

    def decryptable(level: int, scale: int, label: str) -> None:
        if not params.lvl_lb <= level <= params.lvl_ub:
            violations.append(f"{label}_level_out_of_bounds")
        if scale < 0:
            violations.append(f"{label}_scale_negative")
        if scale > params.max_scale:
            violations.append(f"{label}_scale_above_max")
        if scale > params.Sf * (level - params.lvl_lb + 2) - 7:
            violations.append(f"{label}_not_decryptable")

    decryptable(in_lvl, in_scl, "input")
    decryptable(out_lvl, out_scl, "output")
    if params.check_res(in_lvl, in_scl, out_lvl, out_scl):
        return violations
    if not params.check_res(in_lvl, in_scl, params.bts_lb, params.Sf):
        violations.append("bootstrap_input_above_refresh_bound")
    r = max(0, int(math.ceil((params.Sf - out_scl) / max(params.Sf, 1))))
    if not (params.bts_lb < out_lvl + r <= params.bts_ub):
        violations.append("bootstrap_output_level_window")
    if not params.check_res(out_lvl + r, params.Sf, out_lvl, out_scl):
        violations.append("bootstrap_output_rescale_invalid")
    return violations


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
    structured_keys = {
        "policy",
        "global_policy",
        "placement_records",
        "portfolio",
        "patches",
        "unit_policies",
        "mcts_actions",
    }
    if not any(key in value for key in structured_keys):
        return _sanitize_candidate_hints(
            _apply_patch_vocabulary(_with_default_policy(value), value.get("patches", []), context),
            context,
        )
    policy = (
        value.get("global_policy")
        if isinstance(value.get("global_policy"), dict)
        else value.get("policy")
        if isinstance(value.get("policy"), dict)
        else value
        if any(key in value for key in _PATCHABLE_POLICY_KEYS | {"strategy"})
        else {}
    )
    result = _with_default_policy(policy)
    if isinstance(value.get("mcts_actions"), list):
        result["mcts_actions"] = value["mcts_actions"]
    if "mcts_action_cap" in value:
        result["mcts_action_cap"] = value["mcts_action_cap"]
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
    result["unit_policies"] = _normalize_unit_policies(value.get("unit_policies"), context)
    result = _apply_patch_vocabulary(result, value.get("patches", []), context)
    result["api_version"] = value.get("api_version", "placement-builder-v1")
    return _sanitize_candidate_hints(result, context)


def _normalize_unit_policies(value: Any, context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            continue
        selector = item.get("selector", {})
        if isinstance(selector, str):
            selector = {"unit_id": selector}
        if not isinstance(selector, dict):
            selector = {}
        policy = item.get("policy", {})
        if not isinstance(policy, dict):
            policy = {}
        normalized.append({"selector": dict(selector), "policy": _with_default_policy(policy)})
    return normalized


def _sanitize_candidate_hints(
    hints: dict[str, Any], context: dict[str, Any] | None
) -> dict[str, Any]:
    if context is None or _context_leniency(context) != "repair":
        return hints
    repairs: list[str] = []
    result = dict(hints)
    _sanitize_policy_values(result, context, repairs, "global")
    result["preferred_node_levels"] = _sanitize_node_level_map(
        result.get("preferred_node_levels", {}), context, repairs
    )
    result["preferred_node_scales"] = _sanitize_node_scale_map(
        result.get("preferred_node_scales", {}), context, repairs
    )
    result["preferred_edge_scales"] = _sanitize_edge_scale_map(
        result.get("preferred_edge_scales", {}), context, repairs
    )
    sanitized_units = []
    for idx, item in enumerate(result.get("unit_policies", []) or []):
        if not isinstance(item, dict):
            repairs.append(f"unit_policies[{idx}] dropped non-dict item")
            continue
        selector = item.get("selector", {})
        if not _selector_matches_any_unit(selector, context):
            repairs.append(f"unit_policies[{idx}] dropped unmatched selector")
            continue
        policy = _with_default_policy(item.get("policy", {}))
        _sanitize_policy_values(policy, context, repairs, f"unit_policies[{idx}]")
        sanitized_units.append({"selector": selector, "policy": policy})
    result["unit_policies"] = sanitized_units
    sanitized_portfolio = []
    for idx, item in enumerate(result.get("portfolio", []) or []):
        if not isinstance(item, dict):
            repairs.append(f"portfolio[{idx}] dropped non-dict item")
            continue
        policy = _with_default_policy(item)
        _sanitize_policy_values(policy, context, repairs, f"portfolio[{idx}]")
        sanitized_portfolio.append(policy)
    if sanitized_portfolio:
        result["portfolio"] = sanitized_portfolio
    result["__repair_reasons"] = repairs[:64]
    return result


def _context_leniency(context: dict[str, Any]) -> str:
    return str(
        context.get("harness", {}).get(
            "leniency",
            context.get("params", {}).get("openevolve_leniency", "repair"),
        )
    )


def _sanitize_policy_values(
    policy: dict[str, Any], context: dict[str, Any], repairs: list[str], prefix: str
) -> None:
    ckks = _ckks_dict(context)

    def clamp_int(key: str, low: int, high: int) -> None:
        if key not in policy or policy[key] is None:
            return
        original = policy[key]
        value = max(low, min(high, _int_hint(original, low)))
        if value != original:
            repairs.append(f"{prefix}.{key} clamped to {value}")
        policy[key] = value

    if str(policy.get("strategy", "level_preserving")) not in {
        "level_preserving",
        "latency_beam",
        "waterline_seed",
        "bootstrap_mcts",
    }:
        repairs.append(f"{prefix}.strategy reset to level_preserving")
        policy["strategy"] = "level_preserving"
    clamp_int("max_scale_candidates", 3, 32)
    clamp_int("beam_width", 1, 8)
    clamp_int("state_cap_per_node", 1, 32)
    clamp_int("mcts_rollout_budget", 1, 256)
    clamp_int("mcts_max_repair_bootstraps", 0, 1024)
    clamp_int("mcts_action_cap", 1, 64)
    clamp_int("boundary_state_cap", 1, 8)
    clamp_int("bootstrap_anchor_count", 0, 64)
    clamp_int("bootstrap_anchor_level", int(ckks["lvl_lb"]), int(ckks["lvl_ub"]))
    if policy.get("min_internal_level") is not None:
        original = policy["min_internal_level"]
        value = max(int(ckks["lvl_lb"]), min(int(ckks["lvl_ub"]), _int_hint(original, int(ckks["lvl_lb"]))))
        if value != original:
            repairs.append(f"{prefix}.min_internal_level clamped to {value}")
        policy["min_internal_level"] = value
    for key in (
        "bootstrap_penalty",
        "rescale_penalty",
        "level_drop_penalty",
        "scale_penalty",
        "reserve_penalty",
        "selection_bootstrap_penalty",
        "mcts_exploration_weight",
    ):
        if key in policy:
            value = max(0.0, _float_hint(policy[key], 0.0))
            if value != policy[key]:
                repairs.append(f"{prefix}.{key} clamped to {value}")
            policy[key] = value


def _sanitize_node_level_map(
    values: Any, context: dict[str, Any], repairs: list[str]
) -> dict[str, int]:
    nodes = context.get("tdag", {}).get("nodes", {})
    ckks = _ckks_dict(context)
    result = {}
    for node, raw in _int_map(values).items():
        if node not in nodes:
            repairs.append(f"preferred_node_levels dropped unknown node {node}")
            continue
        value = max(int(ckks["lvl_lb"]), min(int(ckks["lvl_ub"]), int(raw)))
        if value != raw:
            repairs.append(f"preferred_node_levels[{node}] clamped to {value}")
        result[node] = value
    return result


def _sanitize_node_scale_map(
    values: Any, context: dict[str, Any], repairs: list[str]
) -> dict[str, int]:
    nodes = context.get("tdag", {}).get("nodes", {})
    ckks = _ckks_dict(context)
    local_bounds = context.get("constraints", {}).get("local_scale_lower_bounds", {})
    result = {}
    for node, raw in _int_map(values).items():
        if node not in nodes:
            repairs.append(f"preferred_node_scales dropped unknown node {node}")
            continue
        lb_info = local_bounds.get(node, {})
        lb = _safe_int(lb_info.get("out"), int(ckks["Sw"])) if isinstance(lb_info, dict) else int(ckks["Sw"])
        value = max(lb, min(int(ckks["max_scale"]), int(raw)))
        if value != raw:
            repairs.append(f"preferred_node_scales[{node}] clamped to {value}")
        result[node] = value
    return result


def _sanitize_edge_scale_map(
    values: Any, context: dict[str, Any], repairs: list[str]
) -> dict[str, int]:
    edges = {
        _edge_key(str(edge.get("u")), str(edge.get("v")))
        for edge in context.get("tdag", {}).get("edges", [])
        if isinstance(edge, dict)
    }
    ckks = _ckks_dict(context)
    result = {}
    for edge, raw in _int_map(values).items():
        if edge not in edges:
            repairs.append(f"preferred_edge_scales dropped unknown edge {edge}")
            continue
        value = max(0, min(int(ckks["max_scale"]), int(raw)))
        if value != raw:
            repairs.append(f"preferred_edge_scales[{edge}] clamped to {value}")
        result[edge] = value
    return result


def _selector_matches_any_unit(selector: Any, context: dict[str, Any]) -> bool:
    if not isinstance(selector, dict):
        return False
    units = context.get("placement_units", [])
    if not units:
        return True
    return any(_selector_matches_unit(selector, unit) for unit in units if isinstance(unit, dict))


def _selector_matches_unit(selector: dict[str, Any], unit: dict[str, Any]) -> bool:
    unit_selector = unit.get("selector", {})
    unit_id = str(unit.get("id", ""))
    if selector.get("unit_id") and str(selector.get("unit_id")) != unit_id:
        return False
    if selector.get("unit") and str(selector.get("unit")) != unit_id:
        return False
    for key in ("kind", "layer", "nonlinear_kind"):
        if selector.get(key) and str(selector.get(key)) != str(unit_selector.get(key)):
            return False
    return True


def _unit_coverage(context: dict[str, Any], hints: dict[str, Any]) -> float:
    units = [unit for unit in context.get("placement_units", []) if isinstance(unit, dict)]
    if not units:
        return 0.0
    matched = set()
    for item in hints.get("unit_policies", []) or []:
        if not isinstance(item, dict):
            continue
        selector = item.get("selector", {})
        for unit in units:
            if _selector_matches_unit(selector, unit):
                matched.add(str(unit.get("id", "")))
    return len(matched) / max(1, len(units))


def _repair_count(hints: dict[str, Any]) -> int:
    reasons = hints.get("__repair_reasons", [])
    return len(reasons) if isinstance(reasons, list) else 0


def _unit_policy_summary(context: dict[str, Any], hints: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_policy_count": len(hints.get("unit_policies", []) or []),
        "unit_coverage": _unit_coverage(context, hints),
        "repair_count": _repair_count(hints),
        "repair_reasons": list(hints.get("__repair_reasons", []) or [])[:12],
    }


def _per_unit_score_table(
    context: dict[str, Any],
    bottleneck_summary: list[dict[str, Any]],
    hints: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    covered = set()
    for item in hints.get("unit_policies", []) or []:
        if not isinstance(item, dict):
            continue
        for unit in context.get("placement_units", []) or []:
            if isinstance(unit, dict) and _selector_matches_unit(item.get("selector", {}), unit):
                covered.add(str(unit.get("id", "")))
    for unit in context.get("placement_units", [])[:32]:
        unit_id = str(unit.get("id", ""))
        selector = unit.get("selector", {})
        hotspot_score = 0
        for hotspot in bottleneck_summary or []:
            if _selector_matches_location(selector, str(hotspot.get("location", ""))):
                hotspot_score += int(hotspot.get("score", 0) or 0)
        rows.append(
            {
                "unit_id": unit_id,
                "layer": selector.get("layer"),
                "nonlinear_kind": selector.get("nonlinear_kind"),
                "node_count": len(unit.get("node_ids", []) or []),
                "covered_by_candidate": unit_id in covered,
                "hotspot_score": hotspot_score,
            }
        )
    return rows


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
            _apply_selector_patch(
                result,
                {"layer": str(patch.get("layer", ""))},
                str(patch.get("operation", "prefer_high_level")),
                context,
            )
        elif op == "target_unit":
            _apply_selector_patch(
                result,
                {"unit_id": str(patch.get("unit_id") or patch.get("unit") or "")},
                str(patch.get("operation", "prefer_high_level")),
                context,
            )
        elif op == "target_nonlinear":
            _apply_selector_patch(
                result,
                {
                    "layer": str(patch.get("layer", "")),
                    "nonlinear_kind": str(patch.get("nonlinear_kind") or patch.get("kind") or ""),
                    "kind": "nonlinear",
                },
                str(patch.get("operation", "prefer_high_level")),
                context,
            )
    return result


_PATCHABLE_POLICY_KEYS = {
    "strategy",
    "prefer_level_preservation",
    "allow_bootstrap",
    "forbid_bootstrap",
    "allow_seed_fallback",
    "refresh_fanout_at_level_floor",
    "max_scale_candidates",
    "bootstrap_penalty",
    "rescale_penalty",
    "level_drop_penalty",
    "min_internal_level",
    "scale_penalty",
    "boundary_scale_penalty",
    "reserve_penalty",
    "min_transition_reserve",
    "min_decryptability_reserve",
    "beam_width",
    "state_cap_per_node",
    "scale_lattice",
    "budget_aggressive",
    "selection_bootstrap_penalty",
    "mcts_rollout_budget",
    "mcts_exploration_weight",
    "mcts_max_repair_bootstraps",
    "mcts_action_cap",
    "boundary_scale_policy",
    "boundary_state_cap",
    "preferred_boundary_scale",
    "boundary_scale",
    "bootstrap_anchor_count",
    "bootstrap_anchor_level",
    "bootstrap_anchors",
}


def _apply_layer_patch(
    hints: dict[str, Any],
    patch: dict[str, Any],
    context: dict[str, Any] | None,
) -> None:
    _apply_selector_patch(
        hints,
        {"layer": str(patch.get("layer", ""))},
        str(patch.get("operation", "prefer_high_level")),
        context,
    )


def _apply_selector_patch(
    hints: dict[str, Any],
    selector: dict[str, Any],
    operation: str,
    context: dict[str, Any] | None,
) -> None:
    if not context:
        return
    ckks = _ckks_dict(context)
    for node, attrs in context.get("tdag", {}).get("nodes", {}).items():
        if not _selector_matches_node(selector, attrs, node):
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


def _assignment_count_summary(assignments: list[Assign]) -> dict[str, float]:
    """Summarize alternative QBP assignments without pretending they are one circuit.

    During sampled compile replay, diagnostics may include many candidate QBP
    assignments for different input/output budgets. Original Orbit's final
    bootstrap count is computed from one DP-selected assignment path, not from
    the sum of all budget alternatives. For failed/partial replay feedback, use
    per-budget averages/minima and keep the aggregate only as a diagnostic.
    """

    if not assignments:
        return {
            "assignment_count": 0.0,
            "aggregate_bootstrap": 0.0,
            "aggregate_rescale": 0.0,
            "avg_bootstrap": 0.0,
            "avg_rescale": 0.0,
            "min_bootstrap": 0.0,
            "min_rescale": 0.0,
            "max_bootstrap": 0.0,
            "max_rescale": 0.0,
        }
    bootstraps: list[int] = []
    rescales: list[int] = []
    for assign in assignments:
        counts = _maintenance_counts(assign)
        bootstraps.append(int(counts["bootstrap"]))
        rescales.append(int(counts["rescale"]))
    aggregate_bootstrap = sum(bootstraps)
    aggregate_rescale = sum(rescales)
    count = len(assignments)
    return {
        "assignment_count": float(count),
        "aggregate_bootstrap": float(aggregate_bootstrap),
        "aggregate_rescale": float(aggregate_rescale),
        "avg_bootstrap": float(aggregate_bootstrap / count),
        "avg_rescale": float(aggregate_rescale / count),
        "min_bootstrap": float(min(bootstraps)),
        "min_rescale": float(min(rescales)),
        "max_bootstrap": float(max(bootstraps)),
        "max_rescale": float(max(rescales)),
    }


def _boundary_group_count_summary(summaries: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize complete QBP boundary groups without summing alternatives.

    Each group represents the Orbit object that DP consumes: all output states
    for one partition input boundary. The representative bootstrap/rescale
    count is the average within each group, then averaged across groups.
    """

    usable = [
        item
        for item in summaries
        if isinstance(item, dict) and int(item.get("solved_budgets", 0) or 0) > 0
    ]
    if not usable:
        return {
            "boundary_group_count": 0.0,
            "avg_bootstrap": 0.0,
            "avg_rescale": 0.0,
            "min_bootstrap": 0.0,
            "min_rescale": 0.0,
            "max_bootstrap": 0.0,
            "max_rescale": 0.0,
        }
    avg_bootstraps = [float(item.get("avg_bootstrap", 0.0) or 0.0) for item in usable]
    avg_rescales = [float(item.get("avg_rescale", 0.0) or 0.0) for item in usable]
    min_bootstraps = [float(item.get("min_bootstrap", 0.0) or 0.0) for item in usable]
    min_rescales = [float(item.get("min_rescale", 0.0) or 0.0) for item in usable]
    max_bootstraps = [float(item.get("max_bootstrap", 0.0) or 0.0) for item in usable]
    max_rescales = [float(item.get("max_rescale", 0.0) or 0.0) for item in usable]
    count = len(usable)
    return {
        "boundary_group_count": float(count),
        "avg_bootstrap": float(sum(avg_bootstraps) / count),
        "avg_rescale": float(sum(avg_rescales) / count),
        "min_bootstrap": float(min(min_bootstraps)),
        "min_rescale": float(min(min_rescales)),
        "max_bootstrap": float(max(max_bootstraps)),
        "max_rescale": float(max(max_rescales)),
    }


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
        "ckks_noise_model": params.resilience_profile.ckks_noise_model,
        "artifacts": params.resilience_profile.artifacts,
    }
