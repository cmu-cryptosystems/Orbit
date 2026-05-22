from __future__ import annotations

import importlib.util
import concurrent.futures
import io
import itertools
import json
import math
import multiprocessing
import os
import hashlib
import queue as queue_module
import shutil
import signal
import tempfile
import time
import traceback
from collections import Counter
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import networkx as nx

from ...assignment import Assign
from ...assignment.decode_assign import decode_assign
from ...latency_estimator import LatencyEstimator, estimate_assign
from ...params.params import Params
from ...tdag import Tdag
from ...tdag.tdag2mlir import tdag_to_mlir
from .noise_estimator import estimate_compile_result_noise, write_noise_summary


class PlacementError(Exception):
    pass


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENAI_API_BASE = "https://api.openai.com/v1"
CONTEXT_SCHEMA_VERSION = "orbit-openevolve-placement-context-v4"
COMPILE_CONTEXT_SCHEMA_VERSION = "orbit-openevolve-compile-harness-v4"
SAMPLED_QBP_TASK_CACHE_SCHEMA_VERSION = "orbit-openevolve-sampled-qbp-task-v1"
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
TRACE_PREVIEW_CHARS = 5000
TRACE_FEEDBACK_CHARS = 20000


@dataclass
class _BudgetAttempt:
    source: str
    assign: Assign
    cost: float
    in_key: tuple[int, int]
    out_key: tuple[int, int]
    actual_cost: float | None = None
    policy: dict[str, Any] | None = None


@dataclass
class _QBPDPState:
    level: int
    scale: int
    cost: float
    bootstrap_count: float = 0.0
    rescale_count: float = 0.0
    assign_data: dict[str, Any] | None = None


@dataclass
class _QBPDPBudgetResult:
    valid: bool
    in_key: tuple[int, int]
    out_key: tuple[int, int] | None
    cost: float
    bootstrap_count: float
    rescale_count: float
    assign: Assign | None = None
    reason: str = ""


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
        _apply_active_scale_floor_from_hints(self.params, hints)
        collect_diagnostics = bool(
            getattr(self.params, "openevolve_evaluating_candidate", False)
            or getattr(self.params, "openevolve_collect_diagnostics", False)
        )
        diagnostics = {} if collect_diagnostics else None
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
        if hasattr(config, "max_code_length"):
            config.max_code_length = 50_000
        if hasattr(config, "diff_based_evolution"):
            config.diff_based_evolution = False
        config.llm.api_base = api_base
        config.llm.timeout = self.params.openevolve_llm_timeout_sec
        config.llm.retries = self.params.openevolve_llm_retries
        config.llm.retry_delay = self.params.openevolve_llm_retry_delay_sec
        config.llm.max_tokens = max(256, min(50_000, int(self.params.openevolve_llm_max_tokens)))
        config.llm.temperature = 0.3
        config.llm.models = [
            self._llm_model_config(LLMModelConfig, spec, config, multi_model)
            for spec in model_specs
        ]
        config.llm.evaluator_models = [
            self._llm_model_config(LLMModelConfig, model_specs[0], config, multi_model)
        ]
        config.database.feature_dimensions = [
            "objective_cost_usec",
            "objective_cost_ratio_vs_seed",
            "boundary_group_validity",
            "candidate_direct_group_coverage",
            "unsolved_reachable_boundary_groups",
            "total_frontier_cost_usec",
            "final_latency_usec",
            "bootstrap_count",
            "component_bootstrap_score",
            "placement_effect_score",
            "policy_effect_score",
            "action_effect_score",
            "seed_equivalent_policy",
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
            config.prompt.max_artifact_bytes = max(
                int(getattr(config.prompt, "max_artifact_bytes", 20 * 1024)),
                48 * 1024,
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


def _initial_compile_program_source(search_mode: str | None = None) -> str:
    if search_mode == "bootstrap-mcts":
        return '''"""Initial compile-level OpenEvolve bootstrap-MCTS placement algorithm."""

from scripts.optimizer.orbit.openevolve_backend import PlacementMCTS


# EVOLVE-BLOCK-START
def place(context):
    """Return an Orbit bootstrap-MCTS placement policy.

    Prefer mutating mcts_action_presets first: it is a compact declarative
    table for action priors, enabled/disabled actions, beam settings, scale
    candidates, repair aggressiveness, and graph maintenance pressure. Use
    mcts_action_cap and mcts_action_allowlist/blocklist to force small
    ablations when selected_source_counts are unchanged. Candidate dictionaries
    are sparse overlays on the current best seed: omitted preset fields inherit
    the seed values, and replace_mcts_action_presets=True should be used only
    when intentionally replacing the full table. Orbit validates and repairs
    every assignment.

    Follow context["evolution_guidance"], context["harness"]["candidate_examples"],
    and context["harness"]["top_costly_boundary_groups"]: use evaluator artifacts
    as execution trace feedback. Keep path-diverse candidates for exploration,
    but the winning program must reduce objective_cost_usec versus the seed. If
    candidate_qbp_coverage is low, change boundary-state coverage; if candidate
    invalid reasons point to forced anchors, soften or retarget anchors. For
    expensive QBP groups, prefer boundary_group_policies with selectors copied
    from top_costly_boundary_groups and bounded policy overrides for
    boundary_state_cap, boundary_scale_policy, scale_lattice, max_scale_candidates,
    bootstrap_penalty, and MCTS action presets. Bootstrap counts are diagnostics
    only; do not optimize toward a fixed count. Candidate examples include
    candidate_mlir_preview and trace_features values. Treat those as execution
    traces: compare orbit.trace.bootstrap, orbit.trace.rescale,
    orbit.trace.boundary_group, and selected-source operations against
    objective_cost_usec, then mutate the active policy below. Good mutations
    change the selected path digest and lower latency; trace-only or
    seed-equivalent changes are exploration examples, not winners.
    """
    mcts = PlacementMCTS(context)
    actions = mcts.candidate_actions(action_cap=16)
    available = {str(action.get("name", "")) for action in actions}
    reference_action_names = sorted(
        name for name in available if name == "reference_boundary_cost_beam"
    )
    relaxed_floor_action_names = sorted(
        name for name in available if name.startswith("estimator_relaxed_floor_")
    )
    active_action_names = [
        name
        for name in [
            "budget_fulfillment_beam",
            "wide_boundary_cost_beam",
            "waterline_cost_beam",
            "dense_boundary_cost_beam",
            "nonlinear_phase_boundary_beam",
            *reference_action_names,
            "profile_waterline_repair",
            "tuneinsight_avgcase_cost_beam",
            "tuneinsight_deferred_bootstrap_beam",
            *relaxed_floor_action_names,
            "latency_mcts_repair",
            "component_budget_repair",
        ]
        if name in available
    ]
    if not bool(context.get("harness", {}).get("enable_trace_patch_initial", False)):
        compact_names = [
            name
            for name in [
                "budget_fulfillment_beam",
                "wide_boundary_cost_beam",
                *reference_action_names,
            ]
            if name in available
        ]
        # Keep trace-group switches explicit but disabled in the seed. This
        # gives OpenEvolve a small, concrete boolean surface to mutate from
        # prior MLIR/path traces without making the initial program broad or
        # slow.
        trace_group_overrides = [
            {
                "enabled": False,
                "group_index": 0,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "bootstrap_penalty": 35_000_000.0,
                "beam_width": 10,
                "state_cap_per_node": 48,
            },
            {
                "enabled": False,
                "group_index": 0,
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "bootstrap_penalty": 65_000_000.0,
                "beam_width": 8,
                "state_cap_per_node": 40,
            },
            {
                "enabled": False,
                "group_index": 1,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "dense",
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "bootstrap_penalty": 25_000_000.0,
                "beam_width": 12,
                "state_cap_per_node": 48,
            },
        ]
        costly_groups = mcts.top_costly_boundary_groups(limit=4)
        boundary_group_policies = []
        for item in trace_group_overrides:
            if not item.get("enabled"):
                continue
            group_index = int(item.get("group_index", 0))
            if group_index < 0 or group_index >= len(costly_groups):
                continue
            overrides = {key: value for key, value in item.items() if key not in {"enabled", "group_index"}}
            boundary_group_policies.append(mcts.boundary_group_policy(costly_groups[group_index], **overrides))
        return {
            "inherit_initial_policy": False,
            "strategy": "bootstrap_mcts",
            "budget_aggressive": True,
            "allow_seed_fallback": True,
            "max_scale_candidates": 24,
            "bootstrap_penalty": 25_000_000.0,
            "reserve_penalty": 75_000.0,
            "min_transition_reserve": 1,
            "min_decryptability_reserve": 1,
            "boundary_scale_policy": "waterline",
            "trace_group_overrides": trace_group_overrides,
            "boundary_group_policies": boundary_group_policies,
            "mcts_action_allowlist": compact_names,
            "mcts_action_cap": max(1, min(2, len(compact_names))),
            "mcts_rollout_budget": 4,
            "mcts_exploration_weight": 1.25,
            "mcts_max_repair_bootstraps": 16,
            "mcts_prior_order": True,
            "include_seed_repair_actions": False,
            "selection_objective": "cost",
            "mcts_action_presets": mcts.action_presets(
                budget_fulfillment_beam={
                    "prior": 0.60,
                    "policy": {
                        "strategy": "latency_beam",
                        "beam_width": 10,
                        "state_cap_per_node": 32,
                        "boundary_state_cap": 8,
                        "max_scale_candidates": 48,
                        "bootstrap_penalty": 25_000_000.0,
                        "rescale_penalty": 0.0,
                        "level_drop_penalty": 20_000_000.0,
                        "selection_objective": "cost",
                        "direct_budget_policy": True,
                        "boundary_scale_policy": "waterline",
                    },
                },
                wide_boundary_cost_beam={
                    "prior": 0.40,
                    "policy": {
                        "strategy": "latency_beam",
                        "beam_width": 10,
                        "state_cap_per_node": 32,
                        "boundary_state_cap": 8,
                        "max_scale_candidates": 48,
                        "boundary_scale_policy": "frontier",
                        "bootstrap_penalty": 25_000_000.0,
                        "rescale_penalty": 0.0,
                        "level_drop_penalty": 20_000_000.0,
                        "selection_objective": "cost",
                        "direct_budget_policy": True,
                    },
                },
                dense_boundary_cost_beam={
                    "prior": 0.50,
                    "policy": {
                        "strategy": "latency_beam",
                        "beam_width": 10,
                        "state_cap_per_node": 48,
                        "boundary_state_cap": 16,
                        "max_scale_candidates": 96,
                        "scale_lattice": "dense",
                        "boundary_scale_policy": "frontier",
                        "bootstrap_penalty": 25_000_000.0,
                        "rescale_penalty": 0.0,
                        "level_drop_penalty": 20_000_000.0,
                        "selection_objective": "cost",
                        "direct_budget_policy": True,
                    },
                },
                **{
                    name: {
                        "prior": 0.50,
                        "policy": {
                            "strategy": "latency_beam",
                            "beam_width": 10,
                            "state_cap_per_node": 48,
                            "boundary_state_cap": 12,
                            "max_scale_candidates": 80,
                            "boundary_scale_policy": "frontier",
                            "scale_lattice": "waterline_sf",
                            "bootstrap_anchor_selector": "reference_bootstrap_locations",
                            "force_bootstrap_anchors": False,
                            "bootstrap_penalty": 30_000_000.0,
                            "selection_bootstrap_penalty": 0.0,
                            "selection_objective": "cost",
                            "direct_budget_policy": True,
                        },
                    }
                    for name in reference_action_names
                },
            ),
        }
    boundary_group_policies = []
    for idx, group in enumerate(mcts.top_costly_boundary_groups(limit=6)):
        if idx % 3 == 0:
            boundary_group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=12,
                max_scale_candidates=80,
                boundary_scale_policy="frontier",
                scale_lattice="waterline_sf",
                bootstrap_penalty=35_000_000.0,
                beam_width=10,
                state_cap_per_node=48,
            ))
        elif idx % 3 == 1:
            boundary_group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=8,
                max_scale_candidates=64,
                boundary_scale_policy="waterline",
                scale_lattice="waterline_sf",
                bootstrap_penalty=65_000_000.0,
                beam_width=8,
                state_cap_per_node=40,
            ))
        else:
            boundary_group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=16,
                max_scale_candidates=96,
                boundary_scale_policy="frontier",
                scale_lattice="dense",
                bootstrap_penalty=25_000_000.0,
                beam_width=12,
                state_cap_per_node=48,
            ))
    policy = {
        "inherit_initial_policy": True,
        "strategy": "bootstrap_mcts",
        "mcts_action_allowlist": active_action_names,
        "mcts_action_cap": 12,
        "mcts_rollout_budget": 24,
        "mcts_exploration_weight": 1.25,
        "mcts_max_repair_bootstraps": 128,
        "mcts_prior_order": True,
        "include_seed_repair_actions": False,
        "selection_objective": "cost",
        "boundary_group_policies": boundary_group_policies,
        "mcts_action_presets": mcts.action_presets(
            budget_fulfillment_beam={
                "prior": 0.62,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 40,
                    "boundary_state_cap": 10,
                    "max_scale_candidates": 64,
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                    "direct_budget_policy": True,
                    "boundary_scale_policy": "waterline",
                },
            },
            wide_boundary_cost_beam={
                "prior": 0.46,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 40,
                    "boundary_state_cap": 10,
                    "max_scale_candidates": 64,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                    "direct_budget_policy": True,
                },
            },
            dense_boundary_cost_beam={
                "prior": 0.52,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 12,
                    "state_cap_per_node": 56,
                    "boundary_state_cap": 18,
                    "max_scale_candidates": 112,
                    "scale_lattice": "dense",
                    "boundary_scale_policy": "frontier",
                    "bootstrap_penalty": 25_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                    "direct_budget_policy": True,
                },
            },
            nonlinear_phase_boundary_beam={
                "prior": 0.40,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            tuneinsight_deferred_bootstrap_beam={
                "prior": 0.30,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "boundary_state_cap": 10,
                    "max_scale_candidates": 80,
                    "state_cap_per_node": 40,
                    "beam_width": 10,
                    "noise_slack_model": "tuneinsight_avgcase",
                    "reserve_penalty": 0.0,
                    "min_transition_reserve": 0,
                    "min_decryptability_reserve": 0,
                    "bootstrap_penalty": 350_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            **{
                name: {
                    "prior": 0.44,
                    "policy": {
                        "strategy": "latency_beam",
                        "beam_width": 10,
                        "state_cap_per_node": 48,
                        "boundary_state_cap": 12,
                        "max_scale_candidates": 80,
                        "boundary_scale_policy": "frontier",
                        "scale_lattice": "waterline_sf",
                        "bootstrap_anchor_selector": "reference_bootstrap_locations",
                        "force_bootstrap_anchors": False,
                        "bootstrap_penalty": 30_000_000.0,
                        "selection_bootstrap_penalty": 0.0,
                        "selection_objective": "cost",
                        "direct_budget_policy": True,
                    },
                }
                for name in reference_action_names
            },
            **{
                name: {
                    "prior": 0.26,
                    "policy": {
                        "strategy": "latency_beam",
                        "boundary_scale_policy": "frontier",
                        "scale_lattice": "estimator_relaxed",
                        "scale_floor_bits": int(name.rsplit("_", 1)[-1]),
                        "boundary_state_cap": 8,
                        "max_scale_candidates": 64,
                        "state_cap_per_node": 32,
                        "beam_width": 8,
                        "noise_slack_model": "tuneinsight_avgcase",
                        "reserve_penalty": 0.0,
                        "min_transition_reserve": 0,
                        "min_decryptability_reserve": 0,
                        "bootstrap_penalty": 35_000_000.0,
                        "selection_bootstrap_penalty": 0.0,
                        "selection_objective": "cost",
                    },
                }
                for name in relaxed_floor_action_names
            },
        ),
    }
    return policy

    # Legacy reference surface kept below for OpenEvolve context only. The
    # active policy above returns before this code runs.
    for action in actions:
        policy = action.get("policy", {})
        name = action.get("name", "")
        if name == "strict_no_bootstrap":
            action["prior"] = 0.05
            policy["boundary_scale_policy"] = "low"
            policy["max_scale_candidates"] = 24
        elif name == "budget_fulfillment_beam":
            action["prior"] = 0.54
            policy["beam_width"] = 8
            policy["state_cap_per_node"] = 32
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 40
            policy["bootstrap_penalty"] = 250_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "wide_boundary_cost_beam":
            action["prior"] = 0.36
            policy["beam_width"] = 8
            policy["state_cap_per_node"] = 32
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["boundary_scale_policy"] = "frontier"
            policy["bootstrap_penalty"] = 50_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "dense_boundary_cost_beam":
            action["prior"] = 0.40
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 16
            policy["max_scale_candidates"] = 96
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "dense"
            policy["bootstrap_penalty"] = 25_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "nonlinear_phase_boundary_beam":
            action["prior"] = 0.44
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 12
            policy["max_scale_candidates"] = 80
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["bootstrap_anchor_selector"] = "nonlinear_phase_boundaries"
            policy["bootstrap_anchor_count"] = 64
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_penalty"] = 35_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "reference_boundary_cost_beam":
            action["prior"] = 0.50
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 12
            policy["max_scale_candidates"] = 80
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["bootstrap_anchor_selector"] = "reference_bootstrap_locations"
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_penalty"] = 30_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "profile_waterline_repair":
            action["prior"] = 0.24
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "waterline"
            policy["scale_lattice"] = "waterline_sf"
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 48
            policy["state_cap_per_node"] = 24
            policy["beam_width"] = 6
            policy["bootstrap_penalty"] = 125_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
        elif name == "tuneinsight_avgcase_cost_beam":
            action["prior"] = 0.22
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "sf"
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["state_cap_per_node"] = 32
            policy["beam_width"] = 8
            policy["noise_slack_model"] = "tuneinsight_avgcase"
            policy["reserve_penalty"] = 0.0
            policy["min_transition_reserve"] = 0
            policy["min_decryptability_reserve"] = 0
            policy["bootstrap_penalty"] = 25_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "tuneinsight_deferred_bootstrap_beam":
            action["prior"] = 0.30
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["state_cap_per_node"] = 32
            policy["beam_width"] = 8
            policy["noise_slack_model"] = "tuneinsight_avgcase"
            policy["reserve_penalty"] = 0.0
            policy["min_transition_reserve"] = 0
            policy["min_decryptability_reserve"] = 0
            policy["bootstrap_penalty"] = 650_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "minimal_bootstrap_repair":
            action["prior"] = 0.18
            policy["bootstrap_anchor_count"] = 0
            policy["boundary_state_cap"] = 3
            policy["bootstrap_penalty"] = 2_500_000_000.0
            policy["selection_bootstrap_penalty"] = 250_000_000.0
            policy["selection_objective"] = "min_bootstrap"
        elif name == "component_budget_repair":
            action["prior"] = 0.28
            policy["selection_objective"] = "cost"
            policy["prefer_component_budget_fit"] = False
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_anchor_count"] = 0
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 48
            policy["bootstrap_penalty"] = 50_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
        elif name == "waterline_budget_repair":
            action["prior"] = 0.20
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "latency_mcts_repair":
            action["prior"] = 0.19
        action["policy"] = policy
    policy = mcts.low_bootstrap_seed(
        rollout_budget=20,
        exploration_weight=1.15,
        max_repair_bootstraps=16,
        action_cap=len(actions),
    )
    policy["mcts_actions"] = actions
    policy["mcts_action_cap"] = 8
    policy["mcts_rollout_budget"] = 24
    relaxed_floor_action_names = [
        str(action.get("name"))
        for action in actions
        if str(action.get("name", "")).startswith("estimator_relaxed_floor_")
    ]
    policy["mcts_action_allowlist"] = [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "waterline_cost_beam",
        "dense_boundary_cost_beam",
        "nonlinear_phase_boundary_beam",
        *reference_action_names,
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        *relaxed_floor_action_names,
        "latency_mcts_repair",
        "component_budget_repair",
        "minimal_bootstrap_repair",
        "waterline_budget_repair",
    ]
    policy["mcts_exploration_weight"] = 1.15
    group_policies = []
    for idx, group in enumerate(mcts.top_costly_boundary_groups(limit=6)):
        if idx % 3 == 0:
            group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=12,
                max_scale_candidates=80,
                boundary_scale_policy="frontier",
                scale_lattice="waterline_sf",
                bootstrap_penalty=35_000_000.0,
                beam_width=10,
                state_cap_per_node=48,
            ))
        elif idx % 3 == 1:
            group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=8,
                max_scale_candidates=64,
                boundary_scale_policy="waterline",
                scale_lattice="waterline_sf",
                bootstrap_penalty=65_000_000.0,
                beam_width=8,
                state_cap_per_node=40,
            ))
        else:
            group_policies.append(mcts.boundary_group_policy(
                group,
                boundary_state_cap=16,
                max_scale_candidates=96,
                boundary_scale_policy="frontier",
                scale_lattice="dense",
                bootstrap_penalty=25_000_000.0,
                beam_width=12,
                state_cap_per_node=48,
            ))
    policy["boundary_group_policies"] = group_policies
    policy["include_seed_repair_actions"] = False
    policy["mcts_prior_order"] = True
    policy["mcts_action_presets"] = mcts.action_presets(
        budget_fulfillment_beam={
            "prior": 0.54,
            "policy": {
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 6,
                "max_scale_candidates": 40,
                "bootstrap_penalty": 250_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        wide_boundary_cost_beam={
            "prior": 0.36,
            "policy": {
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        dense_boundary_cost_beam={
            "prior": 0.40,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "dense",
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        nonlinear_phase_boundary_beam={
            "prior": 0.44,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                "bootstrap_anchor_count": 64,
                "force_bootstrap_anchors": False,
                "bootstrap_penalty": 35_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        **{
            name: {
                "prior": 0.50,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "reference_bootstrap_locations",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 30_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            }
            for name in reference_action_names
        },
        profile_waterline_repair={
            "prior": 0.24,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "state_cap_per_node": 24,
                "beam_width": 6,
                "bootstrap_penalty": 125_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
        },
        tuneinsight_avgcase_cost_beam={
            "prior": 0.22,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "sf",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "state_cap_per_node": 32,
                "beam_width": 8,
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        tuneinsight_deferred_bootstrap_beam={
            "prior": 0.30,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "state_cap_per_node": 32,
                "beam_width": 8,
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 650_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        **{
            name: {
                "prior": 0.34,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "estimator_relaxed",
                    "scale_floor_bits": (
                        int(name.rsplit("_", 1)[-1])
                        if name.rsplit("_", 1)[-1].isdigit()
                        else int(context.get("params", {}).get("Sw", 40))
                    ),
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "state_cap_per_node": 32,
                    "beam_width": 8,
                    "noise_slack_model": "tuneinsight_avgcase",
                    "reserve_penalty": 0.0,
                    "min_transition_reserve": 0,
                    "min_decryptability_reserve": 0,
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            }
            for name in relaxed_floor_action_names
        },
        component_budget_repair={
            "prior": 0.28,
            "policy": {
                "selection_objective": "cost",
                "prefer_component_budget_fit": False,
                "force_bootstrap_anchors": False,
                "bootstrap_anchor_count": 0,
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
        },
    )
    # Promote the best deterministic policy-bank seed into the default program.
    # This avoids spending every short OpenEvolve smoke rediscovering the same
    # compact low-latency prior before mutations can explore around it.
    policy.update(
        {
            "allow_seed_fallback": True,
            "allow_bootstrap": False,
            "refresh_fanout_at_level_floor": True,
            "max_scale_candidates": 24,
            "bootstrap_penalty": 25_000_000.0,
            "rescale_penalty": 0.0,
            "level_drop_penalty": 20_000_000.0,
            "reserve_penalty": 75_000.0,
            "min_transition_reserve": 1,
            "min_decryptability_reserve": 1,
            "boundary_scale_policy": "waterline",
            "beam_width": 10,
            "state_cap_per_node": 32,
            "boundary_state_cap": 12,
        }
    )
    for key in (
        "component_bootstrap_budgets",
        "enable_direct_budget_beam",
        "forbid_bootstrap",
        "scale_lattice",
        "selection_bootstrap_penalty",
        "target_bootstrap_count",
    ):
        policy.pop(key, None)
    policy["boundary_group_policies"] = []
    policy["mcts_action_cap"] = 10
    policy["mcts_rollout_budget"] = 16
    policy["mcts_exploration_weight"] = 1.25
    policy["mcts_max_repair_bootstraps"] = 128
    policy["mcts_prior_order"] = True
    policy["include_seed_repair_actions"] = True
    policy["selection_objective"] = "cost"
    active_action_names = [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        *reference_action_names,
        "latency_mcts_repair",
    ]
    active_actions = [
        action for action in actions if str(action.get("name", "")) in set(active_action_names)
    ]
    for action in active_actions:
        name = str(action.get("name", ""))
        if name == "budget_fulfillment_beam":
            action["prior"] = 0.60
        elif name == "wide_boundary_cost_beam":
            action["prior"] = 0.40
        elif name == "dense_boundary_cost_beam":
            action["prior"] = 0.50
        elif name == "latency_mcts_repair":
            action["prior"] = 0.19
        elif name in reference_action_names:
            action["prior"] = 0.50
    policy["mcts_actions"] = active_actions
    policy["mcts_action_allowlist"] = active_action_names
    policy["mcts_action_presets"] = mcts.action_presets(
        budget_fulfillment_beam={
            "prior": 0.60,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 48,
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "boundary_scale_policy": "waterline",
                "target_bootstrap_count": 0,
            },
        },
        wide_boundary_cost_beam={
            "prior": 0.40,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 48,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "target_bootstrap_count": 0,
            },
        },
        dense_boundary_cost_beam={
            "prior": 0.50,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "scale_lattice": "dense",
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "target_bootstrap_count": 0,
            },
        },
        **{
            name: {
                "prior": 0.50,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "reference_bootstrap_locations",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 30_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                    "direct_budget_policy": True,
                    "target_bootstrap_count": 0,
                },
            }
            for name in reference_action_names
        },
    )
    return policy
# EVOLVE-BLOCK-END
'''
    return '''"""Initial compile-level OpenEvolve placement algorithm for Orbit."""

from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder, PlacementMCTS


# EVOLVE-BLOCK-START
def place(context):
    """Return an Orbit placement algorithm description.

    The candidate may either return explicit placement records or a policy
    built from Orbit's helper API. Orbit validates every assignment and owns
    final repair into Assign objects.

    Follow context["evolution_guidance"], context["harness"]["candidate_examples"],
    and context["harness"]["top_costly_boundary_groups"]: use previous evaluator
    artifacts as execution-trace feedback, keep CKKS legality intact, and
    mutate only the high-impact policy surface rather than Orbit source code.
    Candidate examples carry candidate_mlir_preview and trace_features fields;
    use them to compare maintenance placement, selected sources, boundary
    groups, and objective_cost_usec before changing policy priors.
    Use boundary_group_policies when one expensive QBP group needs a different
    boundary lattice, scale policy, or action preset than the global seed.

    During sampled evolution, Orbit ranks policies lexicographically: first
    solve every reachable QBP boundary group directly, then minimize total
    objective_cost_usec. Path-diverse but slower candidates are exploration
    examples only and cannot become final MLIR selections. Bootstrap/rescale
    counts are secondary diagnostics, not targets. Sampled
    averages are diagnostics only: finalists must replay the full QBP bundle,
    avoid seed fallback, and reduce final_latency_usec.
    """
    builder = PlacementBuilder(context)
    if context.get("harness", {}).get("search_mode") == "bootstrap-mcts":
        mcts = PlacementMCTS(context)
        # Mutate mcts_action_presets first for bootstrap-mcts runs. The legacy
        # portfolio below is ignored in this mode, so useful candidates should
        # change action priors/enabled flags, mcts_action_cap, optional
        # allow/block lists, beam settings, scale candidates, and repair
        # aggressiveness in this compact action table.
        actions = mcts.candidate_actions(action_cap=12)
        reference_action_names = [
            str(action.get("name"))
            for action in actions
            if str(action.get("name", "")) == "reference_boundary_cost_beam"
        ]
        for action in actions:
            policy = action.get("policy", {})
            name = action.get("name", "")
            if name == "strict_no_bootstrap":
                action["prior"] = 0.05
                policy["boundary_scale_policy"] = "low"
                policy["max_scale_candidates"] = 24
            elif name == "budget_fulfillment_beam":
                action["prior"] = 0.54
                policy["beam_width"] = 8
                policy["state_cap_per_node"] = 32
                policy["boundary_state_cap"] = 6
                policy["max_scale_candidates"] = 40
                policy["bootstrap_penalty"] = 250_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "wide_boundary_cost_beam":
                action["prior"] = 0.36
                policy["beam_width"] = 8
                policy["state_cap_per_node"] = 32
                policy["boundary_state_cap"] = 8
                policy["max_scale_candidates"] = 64
                policy["boundary_scale_policy"] = "frontier"
                policy["bootstrap_penalty"] = 50_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "dense_boundary_cost_beam":
                action["prior"] = 0.40
                policy["strategy"] = "latency_beam"
                policy["beam_width"] = 10
                policy["state_cap_per_node"] = 48
                policy["boundary_state_cap"] = 16
                policy["max_scale_candidates"] = 96
                policy["boundary_scale_policy"] = "frontier"
                policy["scale_lattice"] = "dense"
                policy["bootstrap_penalty"] = 25_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "nonlinear_phase_boundary_beam":
                action["prior"] = 0.44
                policy["strategy"] = "latency_beam"
                policy["beam_width"] = 10
                policy["state_cap_per_node"] = 48
                policy["boundary_state_cap"] = 12
                policy["max_scale_candidates"] = 80
                policy["boundary_scale_policy"] = "frontier"
                policy["scale_lattice"] = "waterline_sf"
                policy["bootstrap_anchor_selector"] = "nonlinear_phase_boundaries"
                policy["bootstrap_anchor_count"] = 64
                policy["force_bootstrap_anchors"] = False
                policy["bootstrap_penalty"] = 35_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "reference_boundary_cost_beam":
                action["prior"] = 0.50
                policy["strategy"] = "latency_beam"
                policy["beam_width"] = 10
                policy["state_cap_per_node"] = 48
                policy["boundary_state_cap"] = 12
                policy["max_scale_candidates"] = 80
                policy["boundary_scale_policy"] = "frontier"
                policy["scale_lattice"] = "waterline_sf"
                policy["bootstrap_anchor_selector"] = "reference_bootstrap_locations"
                policy["force_bootstrap_anchors"] = False
                policy["bootstrap_penalty"] = 30_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "profile_waterline_repair":
                action["prior"] = 0.24
                policy["strategy"] = "latency_beam"
                policy["boundary_scale_policy"] = "waterline"
                policy["scale_lattice"] = "waterline_sf"
                policy["boundary_state_cap"] = 6
                policy["max_scale_candidates"] = 48
                policy["state_cap_per_node"] = 24
                policy["beam_width"] = 6
                policy["bootstrap_penalty"] = 125_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
            elif name == "tuneinsight_avgcase_cost_beam":
                action["prior"] = 0.22
                policy["strategy"] = "latency_beam"
                policy["boundary_scale_policy"] = "sf"
                policy["boundary_state_cap"] = 8
                policy["max_scale_candidates"] = 64
                policy["state_cap_per_node"] = 32
                policy["beam_width"] = 8
                policy["noise_slack_model"] = "tuneinsight_avgcase"
                policy["reserve_penalty"] = 0.0
                policy["min_transition_reserve"] = 0
                policy["min_decryptability_reserve"] = 0
                policy["bootstrap_penalty"] = 25_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "tuneinsight_deferred_bootstrap_beam":
                action["prior"] = 0.30
                policy["strategy"] = "latency_beam"
                policy["boundary_scale_policy"] = "frontier"
                policy["scale_lattice"] = "waterline_sf"
                policy["boundary_state_cap"] = 8
                policy["max_scale_candidates"] = 64
                policy["state_cap_per_node"] = 32
                policy["beam_width"] = 8
                policy["noise_slack_model"] = "tuneinsight_avgcase"
                policy["reserve_penalty"] = 0.0
                policy["min_transition_reserve"] = 0
                policy["min_decryptability_reserve"] = 0
                policy["bootstrap_penalty"] = 650_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "minimal_bootstrap_repair":
                action["prior"] = 0.18
                policy["bootstrap_anchor_count"] = 0
                policy["boundary_state_cap"] = 3
                policy["bootstrap_penalty"] = 2_500_000_000.0
                policy["selection_bootstrap_penalty"] = 250_000_000.0
                policy["selection_objective"] = "min_bootstrap"
            elif name == "component_budget_repair":
                action["prior"] = 0.28
                policy["selection_objective"] = "cost"
                policy["prefer_component_budget_fit"] = False
                policy["force_bootstrap_anchors"] = False
                policy["bootstrap_anchor_count"] = 0
                policy["boundary_state_cap"] = 6
                policy["max_scale_candidates"] = 48
                policy["bootstrap_penalty"] = 50_000_000.0
                policy["selection_bootstrap_penalty"] = 0.0
            elif name == "waterline_budget_repair":
                action["prior"] = 0.20
                policy["selection_bootstrap_penalty"] = 0.0
                policy["selection_objective"] = "cost"
            elif name == "latency_mcts_repair":
                action["prior"] = 0.19
            action["policy"] = policy
        policy = mcts.low_bootstrap_seed(
            rollout_budget=20,
            exploration_weight=1.15,
            max_repair_bootstraps=16,
            action_cap=len(actions),
        )
        policy["mcts_actions"] = actions
        policy["mcts_action_cap"] = 8
        policy["mcts_rollout_budget"] = 24
        policy["mcts_action_allowlist"] = [
            "budget_fulfillment_beam",
            "wide_boundary_cost_beam",
            "waterline_cost_beam",
            "dense_boundary_cost_beam",
            "nonlinear_phase_boundary_beam",
            *reference_action_names,
            "profile_waterline_repair",
            "tuneinsight_avgcase_cost_beam",
            "tuneinsight_deferred_bootstrap_beam",
            "latency_mcts_repair",
            "component_budget_repair",
            "minimal_bootstrap_repair",
            "waterline_budget_repair",
        ]
        policy["mcts_exploration_weight"] = 1.15
        group_policies = []
        for idx, group in enumerate(mcts.top_costly_boundary_groups(limit=6)):
            if idx % 3 == 0:
                group_policies.append(mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=12,
                    max_scale_candidates=80,
                    boundary_scale_policy="frontier",
                    scale_lattice="waterline_sf",
                    bootstrap_penalty=35_000_000.0,
                    beam_width=10,
                    state_cap_per_node=48,
                ))
            elif idx % 3 == 1:
                group_policies.append(mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=8,
                    max_scale_candidates=64,
                    boundary_scale_policy="waterline",
                    scale_lattice="waterline_sf",
                    bootstrap_penalty=65_000_000.0,
                    beam_width=8,
                    state_cap_per_node=40,
                ))
            else:
                group_policies.append(mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=16,
                    max_scale_candidates=96,
                    boundary_scale_policy="frontier",
                    scale_lattice="dense",
                    bootstrap_penalty=25_000_000.0,
                    beam_width=12,
                    state_cap_per_node=48,
                ))
        policy["boundary_group_policies"] = group_policies
        policy["include_seed_repair_actions"] = False
        policy["mcts_prior_order"] = True
        policy["mcts_action_presets"] = mcts.action_presets(
            budget_fulfillment_beam={
                "prior": 0.54,
                "policy": {
                    "beam_width": 8,
                    "state_cap_per_node": 32,
                    "boundary_state_cap": 6,
                    "max_scale_candidates": 40,
                    "bootstrap_penalty": 250_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            wide_boundary_cost_beam={
                "prior": 0.36,
                "policy": {
                    "beam_width": 8,
                    "state_cap_per_node": 32,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "boundary_scale_policy": "frontier",
                    "bootstrap_penalty": 50_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            dense_boundary_cost_beam={
                "prior": 0.40,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 16,
                    "max_scale_candidates": 96,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "dense",
                    "bootstrap_penalty": 25_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            nonlinear_phase_boundary_beam={
                "prior": 0.44,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                    "bootstrap_anchor_count": 64,
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            **{
                name: {
                    "prior": 0.50,
                    "policy": {
                        "strategy": "latency_beam",
                        "beam_width": 10,
                        "state_cap_per_node": 48,
                        "boundary_state_cap": 12,
                        "max_scale_candidates": 80,
                        "boundary_scale_policy": "frontier",
                        "scale_lattice": "waterline_sf",
                        "bootstrap_anchor_selector": "reference_bootstrap_locations",
                        "force_bootstrap_anchors": False,
                        "bootstrap_penalty": 30_000_000.0,
                        "selection_bootstrap_penalty": 0.0,
                        "selection_objective": "cost",
                    },
                }
                for name in reference_action_names
            },
            profile_waterline_repair={
                "prior": 0.24,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "waterline",
                    "scale_lattice": "waterline_sf",
                    "boundary_state_cap": 6,
                    "max_scale_candidates": 48,
                    "state_cap_per_node": 24,
                    "beam_width": 6,
                    "bootstrap_penalty": 125_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                },
            },
            tuneinsight_avgcase_cost_beam={
                "prior": 0.22,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "sf",
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "state_cap_per_node": 32,
                    "beam_width": 8,
                    "noise_slack_model": "tuneinsight_avgcase",
                    "reserve_penalty": 0.0,
                    "min_transition_reserve": 0,
                    "min_decryptability_reserve": 0,
                    "bootstrap_penalty": 25_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            tuneinsight_deferred_bootstrap_beam={
                "prior": 0.30,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "state_cap_per_node": 32,
                    "beam_width": 8,
                    "noise_slack_model": "tuneinsight_avgcase",
                    "reserve_penalty": 0.0,
                    "min_transition_reserve": 0,
                    "min_decryptability_reserve": 0,
                    "bootstrap_penalty": 650_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            },
            component_budget_repair={
                "prior": 0.28,
                "policy": {
                    "selection_objective": "cost",
                    "prefer_component_budget_fit": False,
                    "force_bootstrap_anchors": False,
                    "bootstrap_anchor_count": 0,
                    "boundary_state_cap": 6,
                    "max_scale_candidates": 48,
                    "bootstrap_penalty": 50_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                },
            },
        )
        return policy
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
    return f'''import json
import multiprocessing as _multiprocessing
import os as _os
import queue as _queue
import signal as _signal
import traceback
from pathlib import Path

from openevolve.evaluation_result import EvaluationResult
from scripts.optimizer.orbit.openevolve_backend import (
    _invalid_compile_result,
    evaluate_compile_candidate_program,
)


def _evaluate_child(context_path, program_path, result_queue):
    try:
        _os.setsid()
    except Exception:
        pass
    try:
        result = evaluate_compile_candidate_program(context_path, program_path)
    except BaseException as exc:
        result = _invalid_compile_result(
            "compile_evaluator_exception",
            [f"{{type(exc).__name__}}: {{str(exc)[:240]}}"],
        )
        result.setdefault("artifacts", {{}})["traceback"] = traceback.format_exc()[-4000:]
    result_queue.put(result)


def evaluate(program_path):
    timeout = 180
    try:
        context = json.loads(Path({str(context_path)!r}).read_text(encoding="utf-8"))
        timeout = int(context.get("params", {{}}).get("openevolve_evaluator_timeout_sec", timeout))
    except Exception:
        timeout = 180
    ctx = _multiprocessing.get_context("fork")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_evaluate_child,
        args=({str(context_path)!r}, str(program_path), result_queue),
    )
    proc.start()
    proc.join(max(1, timeout))
    if proc.is_alive():
        try:
            _os.killpg(proc.pid, _signal.SIGTERM)
        except Exception:
            proc.terminate()
        proc.join(5)
        if proc.is_alive():
            try:
                _os.killpg(proc.pid, _signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.join(5)
        result = _invalid_compile_result(
            "compile_evaluator_timeout",
            [f"candidate exceeded evaluator timeout {{timeout}}s"],
        )
    else:
        try:
            result = result_queue.get_nowait()
        except _queue.Empty:
            result = _invalid_compile_result(
                "compile_evaluator_no_result",
                [f"candidate evaluator exited with code {{proc.exitcode}} and produced no result"],
            )
    return EvaluationResult(metrics=result["metrics"], artifacts=result["artifacts"])
'''


def build_context(pdag: Tdag, io_budgets_list: list[dict], params: Params) -> dict[str, Any]:
    graph_summary = _graph_summary(pdag, params)
    placement_units = _placement_units(pdag, params, params.openevolve_max_unit_samples)
    waterline_profile = _waterline_profile_summary(pdag, params, placement_units)
    unit_bootstrap_budget = _unit_bootstrap_budget_summary(
        placement_units,
        params,
        getattr(params, "openevolve_target_bootstrap_count", 0),
    )
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
        "waterline_profile": waterline_profile,
        "evolution_guidance": _alphaevolve_guidance(params, unit_bootstrap_budget),
        "placement_units": placement_units,
        "unit_op_histogram": _unit_op_histogram(placement_units),
        "unit_budget_summary": _unit_budget_summary(placement_units, io_budgets_list),
        "unit_hotspots": [],
        "unit_resilience_summary": _unit_resilience_summary(placement_units, params),
        "unit_bootstrap_budget": unit_bootstrap_budget,
        "nonlinear_maintenance_priors": unit_bootstrap_budget.get("kind_priors", {}),
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
            "openevolve_qbp_engine": getattr(params, "openevolve_qbp_engine", "mcts"),
            "openevolve_granularity": params.openevolve_granularity,
            "openevolve_leniency": params.openevolve_leniency,
            "openevolve_max_unit_samples": params.openevolve_max_unit_samples,
            "openevolve_reference_json": params.openevolve_reference_json,
            "openevolve_finalists": params.openevolve_finalists,
            "openevolve_parallel_evaluations": params.openevolve_parallel_evaluations,
            "openevolve_evaluator_timeout_sec": getattr(
                params, "openevolve_evaluator_timeout_sec", 180
            ),
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
            "scale_floor_policy": getattr(params, "scale_floor_policy", "waterline"),
            "scale_floor_min_bits": getattr(params, "scale_floor_min_bits", params.Sw),
            "openevolve_scale_floor_candidates": getattr(
                params, "openevolve_scale_floor_candidates", [params.Sw]
            ),
            "active_scale_floor_bits": (
                params.active_scale_floor_bits()
                if hasattr(params, "active_scale_floor_bits")
                else params.Sw
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
                "scale_floor_policy": getattr(params, "scale_floor_policy", "waterline"),
                "scale_floor_min_bits": getattr(params, "scale_floor_min_bits", params.Sw),
                "scale_floor_candidates": getattr(
                    params, "openevolve_scale_floor_candidates", [params.Sw]
                ),
                "active_scale_floor_bits": (
                    params.active_scale_floor_bits()
                    if hasattr(params, "active_scale_floor_bits")
                    else params.Sw
                ),
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
        openevolve_qbp_engine=pdata.get("openevolve_qbp_engine", "mcts"),
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
        scale_floor_policy=pdata.get("scale_floor_policy", "waterline"),
        scale_floor_min_bits=pdata.get("scale_floor_min_bits"),
        openevolve_scale_floor_candidates=pdata.get("openevolve_scale_floor_candidates"),
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
        "qbp_engine": getattr(params, "openevolve_qbp_engine", "mcts"),
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
        "scale_floor_policy": getattr(params, "scale_floor_policy", "waterline"),
        "scale_floor_min_bits": getattr(params, "scale_floor_min_bits", params.Sw),
        "openevolve_scale_floor_candidates": getattr(
            params, "openevolve_scale_floor_candidates", [params.Sw]
        ),
    }
    return context


def _load_cached_compile_context(path: Path, dag: Tdag, params: Params) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get("schema_version") != COMPILE_CONTEXT_SCHEMA_VERSION:
        return None
    tdata = cached.get("tdag", {})
    if tdata.get("name") != dag.name:
        return None
    if len(tdata.get("nodes", {})) != len(dag.nodes) or len(tdata.get("edges", [])) != len(dag.edges):
        return None
    harness = cached.get("harness", {})
    if harness.get("eval_suite") != params.openevolve_eval_suite:
        return None
    if not isinstance(cached.get("reference"), dict):
        return None
    if params.openevolve_eval_suite != "polybert-full" and not cached.get("sampled_budget_tasks"):
        return None
    return cached


def _stable_context_cache_dir(params: Params) -> Path | None:
    if not getattr(params, "openevolve_context_cache", True):
        return None
    explicit = getattr(params, "openevolve_context_cache_dir", None)
    if explicit:
        return Path(explicit).resolve()
    env_dir = os.environ.get("ORBIT_OPENEVOLVE_CONTEXT_CACHE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).resolve()
    if getattr(params, "openevolve_output_dir", None):
        base = Path(params.openevolve_output_dir).resolve()
        if base.name == "workdir" and base.parent.parent != base.parent:
            return base.parent.parent / ".openevolve_context_cache"
        return base.parent / ".openevolve_context_cache"
    return Path.cwd() / ".openevolve_context_cache"


def _file_digest(path: str | Path | None) -> str | None:
    if not path:
        return None
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _tdag_cache_digest(dag: Tdag) -> str:
    payload = {
        "name": dag.name,
        "inputs": sorted(str(item) for item in dag.inputs),
        "outputs": sorted(str(item) for item in dag.outputs),
        "nodes": [
            [str(node), _jsonable_attrs(dict(attrs))]
            for node, attrs in sorted(dag.nodes(data=True), key=lambda item: str(item[0]))
        ],
        "edges": [
            [str(u), str(v), _jsonable_attrs(dict(attrs))]
            for u, v, attrs in sorted(
                dag.edges(data=True),
                key=lambda item: (str(item[0]), str(item[1])),
            )
        ],
    }
    return _hint_digest(payload)


def _compile_context_cache_key(dag: Tdag, params: Params) -> str:
    profile = getattr(params, "resilience_profile", None)
    payload = {
        "schema_version": COMPILE_CONTEXT_SCHEMA_VERSION,
        "tdag": _tdag_cache_digest(dag),
        "cost_model_digest": _file_digest(getattr(params, "le_json", None)),
        "params": {
            "Sw": int(params.Sw),
            "Csw": int(params.Csw),
            "Sf": int(params.Sf),
            "lvl_lb": int(params.lvl_lb),
            "lvl_ub": int(params.lvl_ub),
            "bts_lb": int(params.bts_lb),
            "bts_ub": int(params.bts_ub),
            "bpsdepth": params.bpsdepth,
            "part": bool(params.part),
            "comp": bool(params.comp),
            "netname": str(params.netname),
            "eval_suite": str(params.openevolve_eval_suite),
            "search_mode": str(getattr(params, "openevolve_search_mode", "bootstrap-mcts")),
            "granularity": str(params.openevolve_granularity),
            "leniency": str(params.openevolve_leniency),
            "max_unit_samples": int(params.openevolve_max_unit_samples),
            "reference_json_digest": _file_digest(getattr(params, "openevolve_reference_json", None)),
            "scale_floor_policy": str(getattr(params, "scale_floor_policy", "waterline")),
            "scale_floor_min_bits": int(getattr(params, "scale_floor_min_bits", params.Sw)),
            "scale_floor_candidates": list(
                getattr(params, "openevolve_scale_floor_candidates", [params.Sw])
            ),
            "resilience_profile_fingerprint": getattr(profile, "fingerprint", None),
            "resilience_constraint_policy": str(params.resilience_constraint_policy),
            "resilience_mode": str(params.resilience_mode),
        },
    }
    return _hint_digest(payload)


def _stable_context_cache_path(dag: Tdag, params: Params) -> Path | None:
    cache_dir = _stable_context_cache_dir(params)
    if cache_dir is None:
        return None
    return cache_dir / f"{_compile_context_cache_key(dag, params)}.json"


def _historical_context_root(stable_context_path: Path | None) -> Path | None:
    if stable_context_path is None:
        return None
    cache_dir = stable_context_path.parent
    if cache_dir.name == ".openevolve_context_cache":
        return cache_dir.parent
    return cache_dir.parent if cache_dir.parent != cache_dir else cache_dir


def _historical_context_enabled(params: Params) -> bool:
    raw = os.environ.get("ORBIT_OPENEVOLVE_CONTEXT_HISTORY", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    if raw in {"1", "true", "yes", "on"}:
        return True
    return bool(getattr(params, "openevolve_sampled_only", False))


def _historical_context_limit() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_CONTEXT_HISTORY_LIMIT", "").strip()
    try:
        return max(1, int(raw)) if raw else 128
    except ValueError:
        return 128


def _historical_context_min_tasks() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_CONTEXT_HISTORY_MIN_TASKS", "").strip()
    try:
        return max(1, int(raw)) if raw else 8
    except ValueError:
        return 8


def _cached_context_reference_latency(cached: dict[str, Any] | None) -> float:
    if not isinstance(cached, dict):
        return float("inf")
    reference = cached.get("reference", {})
    if not isinstance(reference, dict):
        return float("inf")
    latency = _finite_float(
        reference.get(
            "sampled_dp_latency_usec",
            reference.get("objective_cost_usec", reference.get("final_latency_usec")),
        ),
        float("inf"),
    )
    return latency if math.isfinite(latency) and latency > 0 else float("inf")


def _cached_context_sampled_task_count(cached: dict[str, Any] | None) -> int:
    if not isinstance(cached, dict):
        return 0
    tasks = cached.get("sampled_budget_tasks")
    return len(tasks) if isinstance(tasks, list) else 0


def _should_use_historical_context(
    cached_context: dict[str, Any] | None,
    historical_context: dict[str, Any],
) -> bool:
    if cached_context is None:
        return True
    min_tasks = _historical_context_min_tasks()
    cached_tasks = _cached_context_sampled_task_count(cached_context)
    historical_tasks = _cached_context_sampled_task_count(historical_context)
    if historical_tasks >= min_tasks and cached_tasks < min_tasks:
        return True
    if cached_tasks >= min_tasks and historical_tasks < min_tasks:
        return False
    return _cached_context_reference_latency(historical_context) < _cached_context_reference_latency(
        cached_context
    )


def _load_historical_compile_context(
    stable_context_path: Path | None,
    dag: Tdag,
    params: Params,
) -> tuple[dict[str, Any], Path] | None:
    if not _historical_context_enabled(params):
        return None
    root = _historical_context_root(stable_context_path)
    if root is None or not root.exists():
        return None
    candidates: list[Path] = []
    candidates.extend(root.glob("*/workdir/*/compile_context.json"))
    candidates.extend(root.glob("*/compile_context.json"))
    candidates = [
        path
        for path in candidates
        if path.is_file()
        and (stable_context_path is None or path.resolve() != stable_context_path.resolve())
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    best: tuple[float, int, float, dict[str, Any], Path] | None = None
    fallback_best: tuple[float, int, float, dict[str, Any], Path] | None = None
    min_tasks = _historical_context_min_tasks()
    for path in candidates[: _historical_context_limit()]:
        cached = _load_cached_compile_context(path, dag, params)
        if cached is not None:
            latency = _cached_context_reference_latency(cached)
            task_count = len(cached.get("sampled_budget_tasks") or [])
            item = (latency, -task_count, -path.stat().st_mtime, cached, path)
            if fallback_best is None or item[:3] < fallback_best[:3]:
                fallback_best = item
            if task_count >= min_tasks and (best is None or item[:3] < best[:3]):
                best = item
    if best is None:
        best = fallback_best
    if best is None:
        return None
    return best[3], best[4]


def _store_stable_compile_context(path: Path | None, context: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _set_context_cache_paths(
    context: dict[str, Any],
    params: Params,
    stable_context_path: Path | None,
) -> None:
    if stable_context_path is None:
        return
    qbp_cache_dir = str(
        stable_context_path.parent / f"{stable_context_path.stem}_qbp_tasks"
    )
    context.setdefault("harness", {})["stable_context_cache_path"] = str(
        stable_context_path
    )
    context.setdefault("harness", {})["sampled_qbp_cache_dir"] = qbp_cache_dir
    setattr(params, "openevolve_sampled_qbp_cache_dir", qbp_cache_dir)
    for task in context.get("sampled_budget_tasks", []) or []:
        if isinstance(task, dict) and isinstance(task.get("context"), dict):
            task["context"].setdefault("harness", {})[
                "sampled_qbp_cache_dir"
            ] = qbp_cache_dir


def _clean_task_context_for_cache(task_context: dict[str, Any]) -> dict[str, Any]:
    cleaned = _jsonable_policy_hints(task_context)
    harness = cleaned.get("harness")
    if isinstance(harness, dict):
        for key in (
            "stable_context_cache_path",
            "sampled_qbp_cache_dir",
            "trace_dir",
            "timing_log_enabled",
        ):
            harness.pop(key, None)
    return cleaned


def _sampled_task_cache_dir(task_context: dict[str, Any]) -> Path | None:
    harness = task_context.get("harness", {})
    if not isinstance(harness, dict):
        return None
    raw = harness.get("sampled_qbp_cache_dir")
    if not raw:
        return None
    return Path(str(raw)).resolve()


def _sampled_task_cache_key(
    task_context: dict[str, Any],
    eval_hints: dict[str, Any],
    eval_suite: str,
) -> str:
    payload = {
        "schema_version": SAMPLED_QBP_TASK_CACHE_SCHEMA_VERSION,
        "task_context": _hint_digest(_clean_task_context_for_cache(task_context)),
        "eval_hints": _hint_digest(_jsonable_policy_hints(eval_hints)),
        "policy_effect": _hint_digest(_policy_effect_payload(eval_hints)),
        "eval_suite": str(eval_suite),
    }
    return _hint_digest(payload)


def _sampled_task_cache_path(
    task_context: dict[str, Any],
    eval_hints: dict[str, Any],
    eval_suite: str,
) -> Path | None:
    cache_dir = _sampled_task_cache_dir(task_context)
    if cache_dir is None:
        return None
    key = _sampled_task_cache_key(task_context, eval_hints, eval_suite)
    return cache_dir / key[:2] / f"{key}.json"


def _read_sampled_task_cache(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if cached.get("schema_version") != SAMPLED_QBP_TASK_CACHE_SCHEMA_VERSION:
        return None
    if not bool(cached.get("validated_assignments", False)):
        return None
    result = cached.get("result")
    if not isinstance(result, dict):
        return None
    diagnostics = result.setdefault("diagnostics", {})
    if isinstance(diagnostics, dict):
        timings = diagnostics.setdefault("sampled_task_timing_sec", {})
        if isinstance(timings, dict):
            timings["cache_read"] = 0.0
            timings["total"] = 0.0
        diagnostics["sampled_task_cache_hit"] = True
    return result


def _write_sampled_task_cache(
    path: Path | None,
    result: dict[str, Any],
    *,
    validated_assignments: bool,
) -> None:
    if path is None or not validated_assignments:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = result.setdefault("diagnostics", {})
        if isinstance(diagnostics, dict):
            diagnostics["sampled_task_cache_write"] = True
        payload = {
            "schema_version": SAMPLED_QBP_TASK_CACHE_SCHEMA_VERSION,
            "validated_assignments": True,
            "created_at": time.time(),
            "result": result,
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def _merge_cached_compile_context(
    context: dict[str, Any],
    cached: dict[str, Any],
    params: Params,
) -> None:
    reference = dict(cached.get("reference", {}))
    context["reference"] = reference
    cached_harness = cached.get("harness", {}) if isinstance(cached.get("harness"), dict) else {}
    harness = context.setdefault("harness", {})
    harness["seed_baseline"] = dict(cached_harness.get("seed_baseline") or reference)
    sampled_seed = cached_harness.get("sampled_seed_baseline")
    if isinstance(sampled_seed, dict):
        harness["sampled_seed_baseline"] = dict(sampled_seed)
    if cached.get("sampled_budget_tasks") and params.openevolve_eval_suite != "polybert-full":
        context["sampled_budget_tasks"] = cached["sampled_budget_tasks"]
        metrics = None
        if isinstance(cached_harness.get("sampled_task_seed_metrics"), list):
            metrics = cached_harness.get("sampled_task_seed_metrics")
        elif isinstance(sampled_seed, dict) and isinstance(
            sampled_seed.get("sampled_task_seed_metrics"), list
        ):
            metrics = sampled_seed.get("sampled_task_seed_metrics")
        elif isinstance(reference.get("sampled_task_seed_metrics"), list):
            metrics = reference.get("sampled_task_seed_metrics")
        if isinstance(metrics, list):
            harness["sampled_task_seed_metrics"] = deepcopy(metrics)
            _annotate_sampled_budget_tasks_with_seed_metrics(
                context["sampled_budget_tasks"],
                harness["sampled_task_seed_metrics"],
            )
    if isinstance(cached.get("placement_profile"), list):
        context["placement_profile"] = cached["placement_profile"]
    if isinstance(cached.get("unit_hotspots"), list):
        context["unit_hotspots"] = cached["unit_hotspots"]
    for key in ("top_costly_boundary_groups", "unsolved_boundary_groups"):
        if key in cached_harness:
            harness[key] = deepcopy(cached_harness[key])
        elif isinstance(sampled_seed, dict) and key in sampled_seed:
            harness[key] = deepcopy(sampled_seed[key])
    if isinstance(harness.get("sampled_task_seed_metrics"), list):
        harness["top_costly_boundary_groups"] = _top_costly_boundary_groups_from_task_metrics(
            harness["sampled_task_seed_metrics"]
        )


def _ensure_sampled_seed_metadata(
    context: dict[str, Any],
    initial_hints: dict[str, Any],
    params: Params,
) -> bool:
    if params.openevolve_eval_suite == "polybert-full":
        return False
    if not context.get("sampled_budget_tasks"):
        return False
    harness = context.setdefault("harness", {})
    metrics = harness.get("sampled_task_seed_metrics")
    if isinstance(metrics, list) and metrics:
        _annotate_sampled_budget_tasks_with_seed_metrics(
            context["sampled_budget_tasks"], metrics
        )
        if "top_costly_boundary_groups" not in harness:
            harness["top_costly_boundary_groups"] = _top_costly_boundary_groups_from_task_metrics(
                metrics
            )
        return False
    sampled_reference = _evaluate_sampled_budget_tasks(
        context,
        _compile_hints_for_eval_suite(initial_hints, params.openevolve_eval_suite),
        suppress_output=True,
    )
    sampled_baseline = _placement_baseline_from_result(sampled_reference)
    sampled_baseline["source"] = "sampled_seed_metadata"
    sampled_baseline["policy_summary"] = _compact_policy_summary(initial_hints)
    harness["sampled_seed_baseline"] = sampled_baseline
    harness["seed_baseline"] = dict(sampled_baseline)
    metrics = list(sampled_baseline.get("sampled_task_seed_metrics", []) or [])
    if metrics:
        harness["sampled_task_seed_metrics"] = metrics
        _annotate_sampled_budget_tasks_with_seed_metrics(
            context["sampled_budget_tasks"], metrics
        )
        harness["top_costly_boundary_groups"] = _top_costly_boundary_groups_from_task_metrics(
            metrics
        )
    harness["unsolved_boundary_groups"] = dict(
        sampled_baseline.get("unsolved_boundary_groups", {}) or {}
    )
    context["reference"] = dict(sampled_baseline)
    return True


def run_compile_openevolve(dag: Tdag, le: LatencyEstimator, params: Params) -> dict[str, Any]:
    try:
        from openevolve import run_evolution
    except ImportError as exc:
        raise ImportError(
            "openevolve>=0.2.27 is required when --openevolve-iterations is positive."
        ) from exc

    root = _compile_workspace_root(dag, params)
    root.mkdir(parents=True, exist_ok=True)
    context_path = root / "compile_context.json"
    initial_path = root / "initial_program.py"
    evaluator_path = root / "evaluator.py"
    output_dir = root / "openevolve_output"
    print(f"OpenEvolve compile workspace: {root}", flush=True)
    harness_start = time.perf_counter()
    harness_last = harness_start
    harness_timings: dict[str, float] = {}

    def record_harness_timing(stage: str) -> None:
        nonlocal harness_last
        now = time.perf_counter()
        elapsed = float(now - harness_last)
        harness_timings[stage] = elapsed
        harness_last = now
        total = float(now - harness_start)
        print(
            f"OpenEvolve timing: {stage}={elapsed:.3f}s total={total:.3f}s",
            flush=True,
        )
        try:
            (root / "compile_harness_timing.json").write_text(
                json.dumps(
                    {
                        "stages_sec": harness_timings,
                        "total_sec": total,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    stable_context_path = _stable_context_cache_path(dag, params)
    cached_context = _load_cached_compile_context(context_path, dag, params)
    cache_source = "workspace"
    if cached_context is None and stable_context_path is not None:
        cached_context = _load_cached_compile_context(stable_context_path, dag, params)
        cache_source = "stable"
    record_harness_timing("load_cached_context")
    context = build_compile_context(dag, params)
    context.setdefault("harness", {})["trace_dir"] = str(output_dir / "trace_repository")
    context.setdefault("harness", {})["timing_log_enabled"] = True
    _set_context_cache_paths(context, params, stable_context_path)
    record_harness_timing("build_compile_context")
    if stable_context_path is not None:
        historical = _load_historical_compile_context(stable_context_path, dag, params)
        if historical is not None:
            historical_context, historical_path = historical
            historical_latency = _cached_context_reference_latency(historical_context)
            cached_latency = _cached_context_reference_latency(cached_context)
            historical_tasks = _cached_context_sampled_task_count(historical_context)
            cached_tasks = _cached_context_sampled_task_count(cached_context)
            if _should_use_historical_context(cached_context, historical_context):
                cached_context = historical_context
                cache_source = f"history:{historical_path}"
                print(
                    "OpenEvolve compile harness: selected historical sampled context "
                    f"{historical_path} latency={historical_latency} "
                    f"tasks={historical_tasks}.",
                    flush=True,
                )
            else:
                print(
                    "OpenEvolve compile harness: keeping cached context over historical "
                    f"latency={cached_latency} tasks={cached_tasks} "
                    f"historical_latency={historical_latency} "
                    f"historical_tasks={historical_tasks}.",
                    flush=True,
                )
    initial_source = _initial_compile_program_source(
        getattr(params, "openevolve_search_mode", None)
    )
    initial_hints = _hints_from_code(initial_source, context) or _bootstrap_mcts_seed_policy(params)
    context.setdefault("harness", {})["initial_program_digest"] = _source_digest(initial_source)
    initial_path.write_text(initial_source, encoding="utf-8")
    record_harness_timing("initial_program")
    seed_trace_examples: list[dict[str, Any]] = []
    if cached_context is not None:
        print(
            "OpenEvolve compile harness: reusing cached seed replay/context "
            f"from {cache_source}.",
            flush=True,
        )
        _merge_cached_compile_context(context, cached_context, params)
        _set_context_cache_paths(context, params, stable_context_path)
        record_harness_timing("merge_cached_context")
        if _ensure_sampled_seed_metadata(context, initial_hints, params):
            record_harness_timing("sampled_seed_metadata")
    else:
        print("OpenEvolve compile harness: evaluating initial seed baseline.", flush=True)
        context_collection_hints = _context_collection_seed_hints(initial_hints, params)
        reference = _evaluate_compile_hints(context, context_collection_hints, suppress_output=True)
        record_harness_timing("seed_context_collection_replay")
        context["reference"] = {
            "final_latency_usec": reference.get("final_latency_usec"),
            "objective_cost_usec": reference.get(
                "objective_cost_usec", reference.get("final_latency_usec")
            ),
            "total_frontier_cost_usec": reference.get("total_frontier_cost_usec"),
            "sampled_dp_latency_usec": reference.get("sampled_dp_latency_usec"),
            "bootstrap_count": reference.get("bootstrap_count"),
            "rescale_count": reference.get("rescale_count"),
            "valid": reference.get("valid", False),
            "effective_qbp_digest": _effective_qbp_digest(reference.get("diagnostics", {})),
            "selected_path_digest": _selected_path_digest(reference, reference.get("diagnostics", {})),
            "source": "context_collection_seed"
            if context_collection_hints is not initial_hints
            else "initial_seed",
            "policy_summary": _compact_policy_summary(context_collection_hints),
        }
        context.setdefault("harness", {})["seed_baseline"] = dict(context["reference"])
        if reference.get("sampled_budget_tasks"):
            context["sampled_budget_tasks"] = reference["sampled_budget_tasks"]
            if params.openevolve_eval_suite != "polybert-full":
                sampled_reference = _evaluate_compile_hints(
                    context,
                    _compile_hints_for_eval_suite(initial_hints, params.openevolve_eval_suite),
                    suppress_output=True,
                )
                record_harness_timing("sampled_seed_replay")
                sampled_baseline = _placement_baseline_from_result(sampled_reference)
                sampled_baseline["source"] = "sampled_initial_seed"
                sampled_baseline["policy_summary"] = _compact_policy_summary(initial_hints)
                harness = context.setdefault("harness", {})
                harness["sampled_seed_baseline"] = sampled_baseline
                if sampled_baseline.get("sampled_task_seed_metrics"):
                    harness["sampled_task_seed_metrics"] = list(
                        sampled_baseline.get("sampled_task_seed_metrics", [])
                    )
                    _annotate_sampled_budget_tasks_with_seed_metrics(
                        context["sampled_budget_tasks"],
                        harness["sampled_task_seed_metrics"],
                    )
                harness["top_costly_boundary_groups"] = list(
                    sampled_baseline.get("top_costly_boundary_groups", []) or []
                )[:8]
                harness["unsolved_boundary_groups"] = dict(
                    sampled_baseline.get("unsolved_boundary_groups", {}) or {}
                )
                context["reference"] = dict(sampled_baseline)
                seed_trace_examples = _seed_mlir_trace_examples(
                    context,
                    initial_path,
                    initial_hints,
                    sampled_reference,
                )
        context["placement_profile"] = reference.get("bottleneck_summary", [])
        if not seed_trace_examples:
            seed_trace_examples = _seed_mlir_trace_examples(
                context,
                initial_path,
                initial_hints,
                reference,
            )
        print(
            "OpenEvolve compile harness: seed baseline "
            f"valid={context['reference'].get('valid')} "
            f"latency={context['reference'].get('final_latency_usec')} "
            f"bootstraps={context['reference'].get('bootstrap_count')}.",
            flush=True,
        )
        record_harness_timing("seed_trace_examples")
    context["unit_hotspots"] = _unit_hotspots_from_profile(
        context.get("placement_units", []),
        context.get("placement_profile", []),
    )
    if seed_trace_examples:
        context.setdefault("harness", {})["candidate_examples"] = _merge_candidate_examples(
            seed_trace_examples
        )
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _store_stable_compile_context(stable_context_path, context)
    record_harness_timing("write_context_before_policy_bank")
    policy_bank = _run_compile_policy_bank_prepass(
        context_path,
        context,
        initial_hints,
        output_dir,
        params,
    )
    record_harness_timing("policy_bank_prepass")
    if policy_bank.get("initial_hints") is not None:
        # A sampled policy-bank winner is a better latency reference and prompt
        # example, but it can be too broad for OpenEvolve's required initial
        # program evaluation. Keep the compact generated program as the actual
        # OpenEvolve seed; candidates must beat the policy-bank reference to
        # win, and the bank policy remains available in the context for trace
        # learning and optional full-bundle validation.
        policy_bank_initial_hints = policy_bank["initial_hints"]
        harness = context.setdefault("harness", {})
        harness["policy_bank_initial_hints"] = _jsonable_policy_hints(
            policy_bank_initial_hints
        )
        selected_record = policy_bank.get("selected_record")
        if isinstance(selected_record, dict):
            active_seed = _policy_bank_active_seed_baseline(selected_record)
            active_seed["source"] = "policy_bank"
            active_seed["policy_summary"] = selected_record.get("policy_summary", {})
            harness.setdefault("pre_policy_bank_reference", dict(context.get("reference", {})))
            if isinstance(harness.get("sampled_seed_baseline"), dict):
                harness.setdefault(
                    "pre_policy_bank_sampled_seed_baseline",
                    dict(harness["sampled_seed_baseline"]),
                )
                harness["sampled_seed_baseline"] = dict(active_seed)
            harness["active_seed_baseline"] = dict(active_seed)
            harness["seed_baseline"] = dict(active_seed)
            harness["top_costly_boundary_groups"] = list(
                selected_record.get("top_costly_boundary_groups", []) or []
            )[:8]
            harness["unsolved_boundary_groups"] = dict(
                selected_record.get("unsolved_boundary_groups", {}) or {}
            )
            context["reference"] = dict(active_seed)
        harness["initial_policy_source"] = "compact_seed_with_policy_bank_reference"
    elif getattr(params, "openevolve_reuse_output", False):
        reusable_initial = _load_reusable_policy_bank_initial_hints(output_dir, context)
        if reusable_initial is not None:
            initial_hints = reusable_initial
            initial_source = _program_source_from_hints(
                initial_hints,
                "Reused policy-bank latency-improved OpenEvolve initial program.",
            )
            harness = context.setdefault("harness", {})
            harness["initial_policy_source"] = "reused_policy_bank"
            label = str(initial_hints.get("policy_bank_selected_label", ""))
            if label:
                harness["reused_policy_bank_selected_label"] = label
    context.setdefault("harness", {})["initial_policy_hints"] = _jsonable_policy_hints(
        initial_hints
    )
    context.setdefault("harness", {})["candidate_examples"] = _merge_candidate_examples(
        seed_trace_examples,
        policy_bank.get("examples", []),
    )
    context.setdefault("harness", {})["policy_bank_summary"] = policy_bank.get("summary", {})
    context_path.write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    initial_path.write_text(initial_source, encoding="utf-8")
    evaluator_path.write_text(_compile_evaluator_source(context_path), encoding="utf-8")
    record_harness_timing("write_evaluator")
    print(
        "OpenEvolve compile harness: starting evolution "
        f"iterations={params.openevolve_iterations} "
        f"parallel_evaluations={params.openevolve_parallel_evaluations}.",
        flush=True,
    )

    worker = OpenEvolvePlacementWorker(params, le)
    if getattr(params, "openevolve_reuse_output", False):
        print(
            f"OpenEvolve compile harness: loading reusable program from {output_dir}.",
            flush=True,
        )
        best_code = _load_reusable_best_code(output_dir)
        print(
            "OpenEvolve compile harness: reusable program "
            f"{'loaded' if best_code else 'missing'}.",
            flush=True,
        )
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
            qbp_worker_override = _openevolve_evaluator_qbp_worker_count(params)
            with worker._openevolve_runtime_env(), _temporary_qbp_worker_env(qbp_worker_override):
                result = run_evolution(
                    initial_program=initial_path,
                    evaluator=evaluator_path,
                    config=worker._openevolve_config_arg(),
                    iterations=params.openevolve_iterations,
                    output_dir=str(output_dir),
                    cleanup=False,
                )
                record_harness_timing("run_evolution")
        except Exception as exc:
            if not params.openevolve_fail_open:
                raise
            _write_recovery_summary(root, exc)
            record_harness_timing("recover_after_evolution_failure")
            return _recover_compile_hints(output_dir, context, initial_hints, params)
        best_code = result.best_code
    best_program = root / "best_program.py"
    best_program.write_text(best_code, encoding="utf-8")
    record_harness_timing("write_best_program")
    try:
        promoted_hints = _run_sampled_promotion_pass(
            root,
            output_dir,
            context,
            best_code,
            params,
            initial_hints,
        )
        record_harness_timing("sampled_promotion")
        if promoted_hints is not None:
            _write_sampled_selection_summary(
                root,
                output_dir,
                "sampled_promotion_selected",
                fail_open=False,
                selected_sampled_best=True,
            )
            return promoted_hints
        sampled_reject_reason = _sampled_best_invalid_reason(output_dir)
        if (
            params.openevolve_finalists <= 0
            and params.openevolve_eval_suite != "polybert-full"
        ):
            _write_sampled_selection_summary(
                root,
                output_dir,
                sampled_reject_reason or "sampled_smoke_no_full_replay",
                fail_open=True,
                selected_sampled_best=sampled_reject_reason is None,
            )
            record_harness_timing("sampled_selection")
            return _bounded_fail_open_hints(initial_hints)
        if sampled_reject_reason is not None:
            _write_sampled_selection_summary(
                root,
                output_dir,
                sampled_reject_reason,
                fail_open=False,
                selected_sampled_best=False,
            )
            validated_policy_bank = _full_validate_policy_bank_seed(
                root,
                context,
                params,
                initial_hints,
                sampled_reject_reason,
            )
            if validated_policy_bank is not None:
                record_harness_timing("validated_policy_bank")
                return validated_policy_bank
            record_harness_timing("sampled_reject_fail_open")
            return _bounded_fail_open_hints(initial_hints)
        finalist_hints = _run_full_bundle_finalists(
            root,
            output_dir,
            context,
            best_code,
            params,
            initial_hints,
        )
        if finalist_hints is not None:
            record_harness_timing("full_bundle_finalists")
            return finalist_hints
        record_harness_timing("load_best_candidate")
        return _load_candidate_hints(best_program, context)
    except Exception as exc:
        if not params.openevolve_fail_open:
            raise
        _write_recovery_summary(root, exc)
        record_harness_timing("recover_after_selection_failure")
        return _recover_compile_hints(output_dir, context, initial_hints, params, best_code)
    finally:
        if not params.openevolve_keep_workdir and params.openevolve_output_dir is None:
            shutil.rmtree(root, ignore_errors=True)


def _evaluate_policy_bank_program(context_path: str, program_path: str) -> dict[str, Any]:
    return evaluate_compile_candidate_program(Path(context_path), Path(program_path))


def _policy_bank_full_validation_timeout_sec(params: Params) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_FULL_VALIDATION_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            return max(30, int(float(raw)))
        except ValueError:
            pass
    evaluator_timeout = int(getattr(params, "openevolve_evaluator_timeout_sec", 180) or 180)
    return max(60, min(1800, evaluator_timeout))


def _policy_bank_prepass_timeout_sec(params: Params, variant_count: int) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            return max(30, int(float(raw)))
        except ValueError:
            pass
    evaluator_timeout = int(getattr(params, "openevolve_evaluator_timeout_sec", 180) or 180)
    waves = max(
        1,
        math.ceil(
            max(1, int(variant_count or 1))
            / max(1, _policy_bank_worker_count(params, max(1, int(variant_count or 1))))
        ),
    )
    # The policy bank is a seeding aid. A wedged variant should produce a
    # timeout record and let OpenEvolve start, not block the whole compile.
    return max(120, min(900, evaluator_timeout * waves + 30))


def _mp_context():
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _policy_bank_full_validation_record(
    idx: int,
    label: str,
    full_context: dict[str, Any],
    replay_hints: dict[str, Any],
    sampled_reject_reason: str,
) -> dict[str, Any]:
    result = _evaluate_compile_hints(
        full_context,
        replay_hints,
        suppress_output=True,
        evaluating_candidate=False,
    )
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    valid = bool(result.get("valid", False))
    fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
    fallback_budgets = int(result.get("fallback_selected_budgets", 0) or 0)
    invalid_reasons = diagnostics.get("invalid_reasons", {})
    diagnostic_summary = {
        key: diagnostics.get(key)
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
            "unreachable_boundary_groups",
            "selected_source_counts",
            "invalid_reasons",
            "mcts_action_attempt_counts",
            "mcts_action_success_counts",
            "mcts_action_invalid_counts",
            "candidate_invalid_reasons",
        )
        if key in diagnostics
    }
    diagnostic_summary["boundary_group_summaries"] = list(
        diagnostics.get("boundary_group_summaries", []) or []
    )[:32]
    return {
        "index": idx,
        "label": label,
        "valid": valid,
        "sampled_reject_reason": sampled_reject_reason,
        "final_latency_usec": result.get("final_latency_usec"),
        "objective_cost_usec": result.get(
            "objective_cost_usec", result.get("final_latency_usec")
        ),
        "bootstrap_count": result.get("bootstrap_count"),
        "rescale_count": result.get("rescale_count"),
        "fallback_selected_groups": fallback_groups,
        "fallback_selected_budgets": fallback_budgets,
        "candidate_qbp_coverage": result.get("candidate_qbp_coverage"),
        "effective_qbp_digest": _effective_qbp_digest(diagnostics),
        "selected_path_digest": _selected_path_digest(result, diagnostics),
        "invalid_reasons": invalid_reasons if isinstance(invalid_reasons, dict) else {},
        "diagnostic_summary": diagnostic_summary,
        "log_tail": str(result.get("log_tail", ""))[-2000:],
        "policy_summary": _compact_policy_summary(replay_hints),
    }


def _policy_bank_full_validation_worker(
    result_queue,
    idx: int,
    label: str,
    full_context: dict[str, Any],
    replay_hints: dict[str, Any],
    sampled_reject_reason: str,
) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    try:
        record = _policy_bank_full_validation_record(
            idx,
            label,
            full_context,
            replay_hints,
            sampled_reject_reason,
        )
        result_queue.put({"ok": True, "record": record})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "record": {
                    "index": idx,
                    "label": label,
                    "valid": False,
                    "sampled_reject_reason": sampled_reject_reason,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "traceback": traceback.format_exc()[-4000:],
                    "policy_summary": _compact_policy_summary(replay_hints),
                },
            }
        )


def _run_policy_bank_full_validation_variant(
    idx: int,
    label: str,
    full_context: dict[str, Any],
    replay_hints: dict[str, Any],
    sampled_reject_reason: str,
    timeout_sec: int,
) -> dict[str, Any]:
    ctx = _mp_context()
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_policy_bank_full_validation_worker,
        args=(
            result_queue,
            idx,
            label,
            full_context,
            replay_hints,
            sampled_reject_reason,
        ),
    )
    proc.start()
    deadline = time.time() + float(timeout_sec)
    payload: dict[str, Any] | None = None
    while time.time() < deadline:
        try:
            payload = result_queue.get(timeout=min(0.5, max(0.0, deadline - time.time())))
            break
        except queue_module.Empty:
            if not proc.is_alive():
                break
    if payload is None:
        proc.join(0)
        try:
            payload = result_queue.get_nowait()
        except queue_module.Empty:
            payload = None
    if payload is None and proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        else:
            proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.join(5)
        return {
            "index": idx,
            "label": label,
            "valid": False,
            "sampled_reject_reason": sampled_reject_reason,
            "error": f"full validation timed out after {timeout_sec}s",
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "policy_summary": _compact_policy_summary(replay_hints),
        }
    if payload is None:
        return {
            "index": idx,
            "label": label,
            "valid": False,
            "sampled_reject_reason": sampled_reject_reason,
            "error": f"full validation worker exited with code {proc.exitcode}",
            "worker_exitcode": proc.exitcode,
            "policy_summary": _compact_policy_summary(replay_hints),
        }
    proc.join(5)
    if proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        else:
            proc.terminate()
        proc.join(5)
    record = payload.get("record") if isinstance(payload, dict) else None
    if isinstance(record, dict):
        return record
    return {
        "index": idx,
        "label": label,
        "valid": False,
        "sampled_reject_reason": sampled_reject_reason,
        "error": "full validation worker returned no record",
        "policy_summary": _compact_policy_summary(replay_hints),
    }


def _run_compile_policy_bank_prepass(
    context_path: Path,
    context: dict[str, Any],
    initial_hints: dict[str, Any],
    output_dir: Path,
    params: Params,
) -> dict[str, Any]:
    policy_bank_toggle = os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK", "").strip().lower()
    if policy_bank_toggle in {"0", "false", "no", "off"}:
        return {
            "examples": [],
            "summary": {
                "enabled": False,
                "reason": "disabled_by_environment",
            },
            "initial_hints": None,
        }
    if (
        params.openevolve_iterations <= 0
        or getattr(params, "openevolve_reuse_output", False)
        or getattr(params, "openevolve_search_mode", "bootstrap-mcts") != "bootstrap-mcts"
    ):
        return {"examples": [], "summary": {"enabled": False}, "initial_hints": None}
    output_dir.mkdir(parents=True, exist_ok=True)
    bank_dir = output_dir / "policy_bank"
    bank_dir.mkdir(parents=True, exist_ok=True)
    variants = _compile_policy_bank_variants(initial_hints, context, params)
    max_variants_raw = os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK_MAX", "").strip()
    if max_variants_raw:
        try:
            max_variants = max(1, int(max_variants_raw))
            variants = variants[:max_variants]
        except ValueError:
            pass
    records: list[dict[str, Any]] = []
    best_improved: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    workers = _policy_bank_worker_count(params, len(variants))
    print(
        "OpenEvolve compile harness: evaluating policy bank "
        f"variants={len(variants)} workers={workers}.",
        flush=True,
    )
    jobs: list[tuple[str, dict[str, Any], Path]] = []
    for idx, (label, hints) in enumerate(variants):
        eval_hints = dict(hints)
        if os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK_LIGHTWEIGHT", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            eval_hints["policy_bank_lightweight"] = True
        program_path = bank_dir / f"{idx:02d}_{_safe_filename(label)}.py"
        program_path.write_text(
            _program_source_from_hints(eval_hints, f"Policy-bank candidate {label}."),
            encoding="utf-8",
        )
        jobs.append((label, hints, program_path))

    def consume_record(label: str, hints: dict[str, Any], evaluation: dict[str, Any]) -> None:
        nonlocal best_improved
        record = _policy_bank_record(label, hints, evaluation)
        records.append(record)
        if record.get("latency_improved") and record.get("correct"):
            objective = _finite_float(record.get("objective_cost_usec"), float("inf"))
            if best_improved is None or objective < best_improved[0]:
                best_improved = (objective, hints, record)

    if workers <= 1:
        for label, hints, program_path in jobs:
            consume_record(
                label,
                hints,
                evaluate_compile_candidate_program(context_path, program_path),
            )
    else:
        qbp_worker_override = _policy_bank_qbp_worker_count(params, workers)
        with _temporary_qbp_worker_env(qbp_worker_override):
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers)
            future_to_job = {
                executor.submit(
                    _evaluate_policy_bank_program,
                    str(context_path),
                    str(program_path),
                ): (label, hints)
                for label, hints, program_path in jobs
            }
            pending = set(future_to_job)
            deadline = time.monotonic() + _policy_bank_prepass_timeout_sec(
                params, len(future_to_job)
            )
            try:
                while pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        for future in list(pending):
                            label, hints = future_to_job[future]
                            future.cancel()
                            evaluation = _invalid_compile_result(
                                "policy_bank_prepass_timeout",
                                [f"policy-bank variant timed out: {label}"],
                            )
                            consume_record(label, hints, evaluation)
                        pending.clear()
                        break
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=min(5.0, max(0.1, remaining)),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for future in done:
                        label, hints = future_to_job[future]
                        try:
                            evaluation = future.result()
                        except BaseException as exc:
                            evaluation = _invalid_compile_result(
                                "policy_bank_evaluator_exception",
                                [f"{type(exc).__name__}: {str(exc)[:240]}"],
                            )
                            evaluation.setdefault("artifacts", {})[
                                "traceback"
                            ] = traceback.format_exc()[-4000:]
                        consume_record(label, hints, evaluation)
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    summary = {
        "enabled": True,
        "variant_count": len(variants),
        "records": records,
        "selected_initial_label": (
            best_improved[2].get("label") if best_improved is not None else None
        ),
        "selected_initial_objective_cost_usec": (
            best_improved[0] if best_improved is not None else None
        ),
    }
    (bank_dir / "policy_bank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    examples = _policy_bank_candidate_examples(context, records)
    print(
        "OpenEvolve compile harness: policy bank "
        f"improved={best_improved is not None} "
        f"best={summary['selected_initial_label']}.",
        flush=True,
    )
    selected_hints = None
    if best_improved is not None:
        selected_hints = deepcopy(best_improved[1])
        selected_hints["policy_bank_validated_initial"] = True
        selected_hints["policy_bank_selected_label"] = str(best_improved[2].get("label", ""))
        selected_hints["policy_bank_selected_objective_cost_usec"] = float(best_improved[0])
        selected_hints["policy_bank_selected_top_costly_boundary_groups"] = list(
            best_improved[2].get("top_costly_boundary_groups", []) or []
        )[:8]
    return {
        "examples": examples,
        "summary": summary,
        "initial_hints": selected_hints,
        "selected_record": best_improved[2] if best_improved is not None else None,
    }


def _policy_bank_worker_count(params: Params, variant_count: int) -> int:
    workers_raw = os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK_WORKERS", "").strip()
    try:
        if workers_raw:
            workers = max(1, int(workers_raw))
        else:
            workers = max(
                1,
                min(
                    4,
                    int(getattr(params, "openevolve_parallel_evaluations", 1) or 1),
                ),
            )
    except ValueError:
        workers = max(
            1,
            min(
                4,
                int(getattr(params, "openevolve_parallel_evaluations", 1) or 1),
            ),
        )
    return min(workers, max(1, int(variant_count or 0)))


def _policy_bank_qbp_worker_count(params: Params, policy_bank_workers: int) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_POLICY_BANK_QBP_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(32, int(raw)))
        except ValueError:
            pass
    threads = max(1, int(getattr(params, "threads", 1) or 1))
    bank_workers = max(1, int(policy_bank_workers or 1))
    return max(1, min(8, threads // bank_workers))


def _openevolve_evaluator_qbp_worker_count(params: Params) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_EVALUATOR_QBP_WORKERS", "").strip()
    if raw:
        try:
            return max(1, min(32, int(raw)))
        except ValueError:
            pass
    parallel = max(1, int(getattr(params, "openevolve_parallel_evaluations", 1) or 1))
    threads = max(1, int(getattr(params, "threads", 1) or 1))
    return max(1, min(12, threads // parallel))


@contextmanager
def _temporary_qbp_worker_env(worker_count: int):
    old_qbp_workers = os.environ.get("ORBIT_OPENEVOLVE_QBP_WORKERS")
    old_qbp_cap = os.environ.get("ORBIT_OPENEVOLVE_MAX_QBP_WORKERS")
    worker_count = max(1, int(worker_count or 1))
    os.environ["ORBIT_OPENEVOLVE_QBP_WORKERS"] = str(worker_count)
    os.environ["ORBIT_OPENEVOLVE_MAX_QBP_WORKERS"] = str(worker_count)
    try:
        yield
    finally:
        if old_qbp_workers is None:
            os.environ.pop("ORBIT_OPENEVOLVE_QBP_WORKERS", None)
        else:
            os.environ["ORBIT_OPENEVOLVE_QBP_WORKERS"] = old_qbp_workers
        if old_qbp_cap is None:
            os.environ.pop("ORBIT_OPENEVOLVE_MAX_QBP_WORKERS", None)
        else:
            os.environ["ORBIT_OPENEVOLVE_MAX_QBP_WORKERS"] = old_qbp_cap


def _graph_maintenance_anchor_hint(context: dict[str, Any]) -> int:
    """Return a graph-derived exploration width, not a bootstrap target."""

    budget_info = context.get("unit_bootstrap_budget", {})
    pressures = (
        budget_info.get("unit_maintenance_pressure", {})
        if isinstance(budget_info, dict)
        else {}
    )
    if isinstance(pressures, dict) and pressures:
        positive = [
            _finite_float(item.get("pressure"), 0.0)
            for item in pressures.values()
            if isinstance(item, dict) and _finite_float(item.get("pressure"), 0.0) > 0
        ]
        if positive:
            return max(1, min(12, int(math.ceil(math.sqrt(len(positive))))))
    units = context.get("placement_units", []) or []
    nonlinear_units = [
        unit
        for unit in units
        if isinstance(unit, dict)
        and (unit.get("selector", {}) or {}).get("nonlinear_kind")
    ]
    if nonlinear_units:
        return max(1, min(12, int(math.ceil(math.sqrt(len(nonlinear_units))))))
    return 0


def _compile_policy_bank_variants(
    initial_hints: dict[str, Any],
    context: dict[str, Any],
    params: Params,
) -> list[tuple[str, dict[str, Any]]]:
    scale_candidates = _scale_floor_candidates_from_context(context)
    ckks = _ckks_dict(context)
    waterline = int(ckks["Sw"])
    anchor_hint = _graph_maintenance_anchor_hint(context)
    variants: list[tuple[str, dict[str, Any]]] = []

    def add(
        label: str,
        updates: dict[str, Any],
        preset_updates: dict[str, dict[str, Any]] | None = None,
        *,
        replace_presets: bool = False,
    ) -> None:
        hints = deepcopy(initial_hints)
        hints.update(updates)
        if preset_updates:
            presets = {} if replace_presets else deepcopy(hints.get("mcts_action_presets", {}))
            for name, patch in preset_updates.items():
                base = deepcopy(presets.get(name, {}))
                base_policy = deepcopy(base.get("policy", {}))
                base_policy.update(patch.get("policy", {}))
                base["policy"] = base_policy
                if "prior" in patch:
                    base["prior"] = patch["prior"]
                presets[name] = base
            hints["mcts_action_presets"] = presets
        variants.append((label, hints))

    top_groups: list[dict[str, Any]] = []
    harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
    for item in harness.get("top_costly_boundary_groups", []) or []:
        if isinstance(item, dict):
            top_groups.append(item)

    waterline_direct_policy = {
        "selection_objective": "cost",
        "bootstrap_penalty": 25_000_000.0,
        "max_scale_candidates": 32,
        "boundary_scale_policy": "waterline",
    }
    waterline_direct_actions = [
        action
        for action in candidate_actions(context, action_cap=12)
        if str(action.get("name", "")) != "reference_boundary_cost_beam"
    ][:10]
    add(
        "waterline_direct_cost_compact",
        {
            **waterline_direct_policy,
            "boundary_scale_policy": "waterline",
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 8,
            "max_scale_candidates": 32,
            "state_cap_per_node": 40,
            "beam_width": 12,
            "selection_bootstrap_penalty": 0.0,
            "mcts_action_cap": 10,
            "mcts_rollout_budget": 16,
            "mcts_exploration_weight": 1.25,
            "mcts_max_repair_bootstraps": 128,
            "mcts_actions": waterline_direct_actions,
            "mcts_action_allowlist": [
                "budget_fulfillment_beam",
                "dense_boundary_cost_beam",
                "nonlinear_phase_boundary_beam",
                "wide_boundary_cost_beam",
                "latency_mcts_repair",
            ],
        },
        {
            "budget_fulfillment_beam": {
                "prior": 0.60,
                "policy": {**waterline_direct_policy, "beam_width": 12},
            },
            "dense_boundary_cost_beam": {
                "prior": 0.50,
                "policy": {**waterline_direct_policy, "strategy": "latency_beam", "beam_width": 12},
            },
            "nonlinear_phase_boundary_beam": {
                "prior": 0.50,
                "policy": {**waterline_direct_policy, "strategy": "latency_beam", "beam_width": 12},
            },
            "wide_boundary_cost_beam": {
                "prior": 0.40,
                "policy": {**waterline_direct_policy, "beam_width": 10},
            },
        },
        replace_presets=True,
    )
    latency_beam_base = {
        "strategy": "latency_beam",
        "prefer_level_preservation": True,
        "allow_bootstrap": True,
        "allow_seed_fallback": False,
        "refresh_fanout_at_level_floor": True,
        "max_scale_candidates": 48,
        "bootstrap_penalty": 25_000_000.0,
        "rescale_penalty": 0.0,
        "level_drop_penalty": 20_000_000.0,
        "selection_objective": "cost",
        "beam_width": 10,
        "state_cap_per_node": 32,
        "boundary_state_cap": 8,
        "target_bootstrap_count": 0,
        "direct_budget_policy": True,
    }
    latency_beam_actions = [
        {
            "name": "budget_fulfillment_beam",
            "prior": 0.60,
            "policy": {**latency_beam_base, "boundary_scale_policy": "waterline"},
        },
        {
            "name": "wide_boundary_cost_beam",
            "prior": 0.40,
            "policy": {**latency_beam_base, "boundary_scale_policy": "frontier"},
        },
        {
            "name": "dense_boundary_cost_beam",
            "prior": 0.50,
            "policy": {
                **latency_beam_base,
                "scale_lattice": "dense",
                "boundary_state_cap": 16,
            },
        },
    ]
    add(
        "latency_beam_waterline48_compact",
        {
            "strategy": "bootstrap_mcts",
            "budget_aggressive": True,
            "mcts_rollout_budget": 16,
            "mcts_exploration_weight": 1.25,
            "mcts_max_repair_bootstraps": 128,
            "mcts_action_cap": 10,
            "mcts_prior_order": True,
            "mcts_actions": latency_beam_actions,
            "mcts_action_allowlist": [
                "budget_fulfillment_beam",
                "wide_boundary_cost_beam",
                "dense_boundary_cost_beam",
                "latency_mcts_repair",
            ],
            "boundary_scale_policy": "waterline",
            "scale_lattice": "waterline_sf",
            "bootstrap_penalty": 25_000_000.0,
            "rescale_penalty": 25_000.0,
        },
        {
            action["name"]: {
                "prior": action["prior"],
                "policy": action["policy"],
            }
            for action in latency_beam_actions
        },
        replace_presets=True,
    )
    if top_groups:
        # Keep these before the broad policy-bank sweep so capped short runs
        # still test whether trace-derived boundary overlays can move the DP
        # path. Each probe touches one costly boundary group only.
        trace_probe_specs = [
            ("frontier", "waterline_sf", 12, 80, 10, 48, 35_000_000.0),
            ("waterline", "waterline_sf", 8, 64, 8, 40, 65_000_000.0),
            ("frontier", "dense", 16, 96, 12, 48, 25_000_000.0),
        ]
        for group_idx, group in enumerate(top_groups[:2]):
            selector = group.get("group_key", group)
            if not isinstance(selector, dict):
                continue
            for (
                boundary_policy,
                scale_lattice,
                boundary_cap,
                max_scales,
                beam_width,
                state_cap,
                maintenance_penalty,
            ) in trace_probe_specs:
                overlay_policy = {
                    "boundary_state_cap": int(boundary_cap),
                    "max_scale_candidates": int(max_scales),
                    "boundary_scale_policy": boundary_policy,
                    "scale_lattice": scale_lattice,
                    "bootstrap_penalty": float(maintenance_penalty),
                    "selection_bootstrap_penalty": 0.0,
                    "beam_width": int(beam_width),
                    "state_cap_per_node": int(state_cap),
                    "selection_objective": "cost",
                }
                add(
                    (
                        f"trace_group_{group_idx}_{boundary_policy}_"
                        f"{scale_lattice}_cost_probe"
                    ),
                    {
                        "boundary_group_policies": [
                            {"selector": dict(selector), "policy": overlay_policy}
                        ],
                        "mcts_action_cap": 10,
                        "mcts_rollout_budget": 16,
                        "mcts_exploration_weight": 1.15,
                        "mcts_max_repair_bootstraps": 128,
                        "selection_objective": "cost",
                        "include_seed_repair_actions": False,
                        "mcts_action_allowlist": [
                            "budget_fulfillment_beam",
                            "wide_boundary_cost_beam",
                            "dense_boundary_cost_beam",
                            "latency_mcts_repair",
                        ],
                    },
                    {
                        "budget_fulfillment_beam": {
                            "prior": 0.70,
                            "policy": overlay_policy,
                        },
                        "wide_boundary_cost_beam": {
                            "prior": 0.55,
                            "policy": overlay_policy,
                        },
                        "dense_boundary_cost_beam": {
                            "prior": 0.58,
                            "policy": {
                                **overlay_policy,
                                "strategy": "latency_beam",
                                "boundary_state_cap": max(12, int(boundary_cap)),
                                "max_scale_candidates": max(80, int(max_scales)),
                            },
                        },
                    },
                )
    for label, max_scale, boundary_cap, beam_width, bootstrap_penalty in [
        ("waterline_direct_cost_wide48", 48, 8, 12, 25_000_000.0),
        ("waterline_direct_cost_wide64", 64, 8, 12, 25_000_000.0),
        ("waterline_direct_cost_deeper_boundary", 48, 12, 12, 25_000_000.0),
        ("waterline_direct_cost_beam16", 48, 10, 16, 25_000_000.0),
        ("waterline_direct_cost_lower_maintenance_penalty", 48, 8, 12, 10_000_000.0),
        ("waterline_direct_cost_higher_maintenance_penalty", 48, 8, 12, 75_000_000.0),
    ]:
        local_policy = {
            **waterline_direct_policy,
            "bootstrap_penalty": float(bootstrap_penalty),
            "max_scale_candidates": int(max_scale),
            "boundary_scale_policy": "waterline",
        }
        add(
            label,
            {
                **local_policy,
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": int(boundary_cap),
                "state_cap_per_node": 48 if boundary_cap >= 10 else 40,
                "beam_width": int(beam_width),
                "selection_bootstrap_penalty": 0.0,
                "mcts_action_cap": 10,
                "mcts_rollout_budget": 20 if beam_width >= 16 else 16,
                "mcts_exploration_weight": 1.25,
                "mcts_max_repair_bootstraps": 128,
                "mcts_actions": waterline_direct_actions,
                "mcts_action_allowlist": [
                    "budget_fulfillment_beam",
                    "dense_boundary_cost_beam",
                    "nonlinear_phase_boundary_beam",
                    "wide_boundary_cost_beam",
                    "latency_mcts_repair",
                ],
            },
            {
                "budget_fulfillment_beam": {
                    "prior": 0.60,
                    "policy": {**local_policy, "beam_width": int(beam_width)},
                },
                "dense_boundary_cost_beam": {
                    "prior": 0.50,
                    "policy": {
                        **local_policy,
                        "strategy": "latency_beam",
                        "beam_width": int(beam_width),
                    },
                },
                "nonlinear_phase_boundary_beam": {
                    "prior": 0.50,
                    "policy": {
                        **local_policy,
                        "strategy": "latency_beam",
                        "beam_width": int(beam_width),
                    },
                },
                "wide_boundary_cost_beam": {
                    "prior": 0.40,
                    "policy": {**local_policy, "beam_width": max(10, int(beam_width) - 2)},
                },
            },
            replace_presets=True,
        )

    add(
        "frontier_wide_low_bootstrap_penalty",
        {
            "boundary_scale_policy": "frontier",
            "boundary_state_cap": 8,
            "max_scale_candidates": 64,
            "state_cap_per_node": 32,
            "beam_width": 8,
            "mcts_action_cap": 10,
            "mcts_rollout_budget": 16,
            "selection_objective": "cost",
            "bootstrap_penalty": 50_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "wide_boundary_cost_beam": {
                "prior": 0.7,
                "policy": {
                    "boundary_scale_policy": "frontier",
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "bootstrap_penalty": 50_000_000.0,
                },
            },
            "budget_fulfillment_beam": {
                "prior": 0.55,
                "policy": {
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "selection_objective": "cost",
                },
            },
        },
    )
    add(
        "waterline_sf_coverage",
        {
            "boundary_scale_policy": "waterline",
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 6,
            "max_scale_candidates": 48,
            "state_cap_per_node": 24,
            "beam_width": 8,
            "mcts_action_cap": 10,
            "mcts_rollout_budget": 16,
            "selection_objective": "cost",
            "bootstrap_penalty": 125_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "profile_waterline_repair": {
                "prior": 0.65,
                "policy": {
                    "boundary_state_cap": 6,
                    "max_scale_candidates": 48,
                    "bootstrap_penalty": 125_000_000.0,
                },
            },
        },
    )
    add(
        "compact_prior_order_cost",
        {
            "mcts_action_cap": 6,
            "mcts_rollout_budget": 24,
            "mcts_exploration_weight": 1.25,
            "mcts_max_repair_bootstraps": 16,
            "mcts_prior_order": True,
            "include_seed_repair_actions": True,
            "selection_objective": "cost",
            "mcts_action_allowlist": [
                "budget_fulfillment_beam",
                "wide_boundary_cost_beam",
                "dense_boundary_cost_beam",
                "minimal_bootstrap_repair",
            ],
        },
        {
            "budget_fulfillment_beam": {
                "prior": 0.65,
                "policy": {
                    "beam_width": 12,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 48,
                    "bootstrap_penalty": 150_000_000.0,
                    "selection_objective": "cost",
                },
            },
            "wide_boundary_cost_beam": {
                "prior": 0.45,
                "policy": {
                    "beam_width": 12,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 10,
                    "max_scale_candidates": 64,
                    "boundary_scale_policy": "frontier",
                    "bootstrap_penalty": 30_000_000.0,
                    "selection_objective": "cost",
                },
            },
            "dense_boundary_cost_beam": {
                "prior": 0.50,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 16,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 20,
                    "max_scale_candidates": 96,
                    "scale_lattice": "dense",
                    "boundary_scale_policy": "frontier",
                    "bootstrap_penalty": 15_000_000.0,
                    "selection_objective": "cost",
                },
            },
            "minimal_bootstrap_repair": {
                "prior": 0.10,
                "policy": {
                    "bootstrap_anchor_count": 0,
                    "boundary_state_cap": 2,
                    "bootstrap_penalty": 5_000_000_000.0,
                    "selection_objective": "min_bootstrap",
                },
            },
        },
    )
    add(
        "maintenance_pressure_cost",
        {
            "boundary_state_cap": 6,
            "max_scale_candidates": 48,
            "mcts_action_cap": 10,
            "mcts_rollout_budget": 18,
            "selection_objective": "cost",
            "prefer_component_budget_fit": False,
            "bootstrap_anchor_count": anchor_hint,
            "force_bootstrap_anchors": False,
            "bootstrap_penalty": 80_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "component_budget_repair": {
                "prior": 0.75,
                "policy": {
                    "boundary_state_cap": 6,
                    "max_scale_candidates": 48,
                    "bootstrap_anchor_count": anchor_hint,
                    "selection_objective": "cost",
                    "force_bootstrap_anchors": False,
                },
            },
        },
    )
    add(
        "waterline_boundary_cost",
        {
            "boundary_scale_policy": "waterline",
            "preferred_boundary_scale": waterline,
            "boundary_scale": waterline,
            "max_scale": waterline,
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 8,
            "max_scale_candidates": 64,
            "state_cap_per_node": 32,
            "beam_width": 8,
            "mcts_action_cap": 12,
            "mcts_rollout_budget": 24,
            "selection_objective": "cost",
            "bootstrap_penalty": 65_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "budget_fulfillment_beam": {
                "prior": 0.74,
                "policy": {
                    "boundary_scale_policy": "waterline",
                    "preferred_boundary_scale": waterline,
                    "boundary_scale": waterline,
                    "max_scale": waterline,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "selection_objective": "cost",
                    "bootstrap_penalty": 65_000_000.0,
                },
            },
            "wide_boundary_cost_beam": {
                "prior": 0.62,
                "policy": {
                    "boundary_scale_policy": "waterline",
                    "preferred_boundary_scale": waterline,
                    "boundary_scale": waterline,
                    "max_scale": waterline,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "selection_objective": "cost",
                    "bootstrap_penalty": 65_000_000.0,
                },
            },
        },
    )
    add(
        "maintenance_pressure_boundary_cost",
        {
            "boundary_scale_policy": "waterline",
            "preferred_boundary_scale": waterline,
            "boundary_scale": waterline,
            "max_scale": waterline,
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 8,
            "max_scale_candidates": 64,
            "state_cap_per_node": 32,
            "beam_width": 8,
            "mcts_action_cap": 12,
            "mcts_rollout_budget": 24,
            "selection_objective": "cost",
            "prefer_component_budget_fit": False,
            "bootstrap_anchor_count": anchor_hint,
            "force_bootstrap_anchors": False,
            "bootstrap_penalty": 45_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "component_budget_repair": {
                "prior": 0.80,
                "policy": {
                    "boundary_scale_policy": "waterline",
                    "preferred_boundary_scale": waterline,
                    "boundary_scale": waterline,
                    "max_scale": waterline,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "bootstrap_anchor_count": anchor_hint,
                    "selection_objective": "cost",
                    "force_bootstrap_anchors": False,
                },
            },
        },
    )
    add(
        "nonlinear_phase_boundary_cost",
        {
            "boundary_scale_policy": "frontier",
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 12,
            "max_scale_candidates": 80,
            "state_cap_per_node": 48,
            "beam_width": 10,
            "mcts_action_cap": 12,
            "mcts_rollout_budget": 24,
            "selection_objective": "cost",
            "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
            "bootstrap_anchor_count": 64,
            "force_bootstrap_anchors": False,
            "bootstrap_penalty": 35_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        },
        {
            "nonlinear_phase_boundary_beam": {
                "prior": 0.90,
                "policy": {
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                    "bootstrap_anchor_count": 64,
                    "selection_objective": "cost",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 35_000_000.0,
                },
            },
            "budget_fulfillment_beam": {
                "prior": 0.62,
                "policy": {
                    "boundary_scale_policy": "frontier",
                    "boundary_state_cap": 10,
                    "max_scale_candidates": 72,
                    "selection_objective": "cost",
                    "bootstrap_penalty": 45_000_000.0,
                },
            },
        },
    )
    reference_patterns = _reference_bootstrap_patterns_from_context(context)
    if reference_patterns:
        reference_anchor_count = max(8, min(64, len(reference_patterns) * 2))
        reference_action_base = {
            **_budget_fulfillment_beam_policy(),
            "target_bootstrap_count": 0,
            "direct_budget_policy": True,
            "beam_width": 12,
            "state_cap_per_node": 48,
            "boundary_state_cap": 12,
            "max_scale_candidates": 80,
            "boundary_scale_policy": "frontier",
            "scale_lattice": "waterline_sf",
            "bootstrap_anchor_selector": "reference_bootstrap_locations",
            "bootstrap_anchor_include_patterns": reference_patterns,
            "bootstrap_anchor_count": reference_anchor_count,
            "force_bootstrap_anchors": False,
            "selection_objective": "cost",
            "selection_bootstrap_penalty": 0.0,
        }
        for label, boundary_policy, scale_lattice, boundary_cap, max_scales in (
            ("reference_location_frontier_cost", "frontier", "waterline_sf", 12, 80),
            ("reference_location_dense_cost", "frontier", "dense", 16, 96),
            ("reference_location_waterline_cost", "waterline", "waterline_sf", 10, 64),
        ):
            action_policy = {
                **reference_action_base,
                "boundary_scale_policy": boundary_policy,
                "scale_lattice": scale_lattice,
                "boundary_state_cap": int(boundary_cap),
                "max_scale_candidates": int(max_scales),
            }
            add(
                label,
                {
                    **action_policy,
                    "strategy": "bootstrap_mcts",
                    "budget_aggressive": True,
                    "mcts_action_cap": 12,
                    "mcts_rollout_budget": 24,
                    "mcts_exploration_weight": 1.2,
                    "mcts_max_repair_bootstraps": 128,
                    "mcts_prior_order": True,
                    "mcts_actions": [
                        {
                            "name": "reference_boundary_cost_beam",
                            "prior": 0.92,
                            "policy": action_policy,
                        },
                        {
                            "name": "budget_fulfillment_beam",
                            "prior": 0.62,
                            "policy": {
                                **action_policy,
                                "bootstrap_anchor_count": 0,
                                "bootstrap_anchor_selector": "",
                            },
                        },
                        {
                            "name": "dense_boundary_cost_beam",
                            "prior": 0.52,
                            "policy": {
                                **action_policy,
                                "scale_lattice": "dense",
                                "boundary_state_cap": max(12, int(boundary_cap)),
                                "max_scale_candidates": max(80, int(max_scales)),
                            },
                        },
                    ],
                    "mcts_action_allowlist": [
                        "reference_boundary_cost_beam",
                        "budget_fulfillment_beam",
                        "dense_boundary_cost_beam",
                    ],
                },
                {
                    "reference_boundary_cost_beam": {
                        "prior": 0.92,
                        "policy": action_policy,
                    },
                    "budget_fulfillment_beam": {
                        "prior": 0.62,
                        "policy": {
                            **action_policy,
                            "bootstrap_anchor_count": 0,
                            "bootstrap_anchor_selector": "",
                        },
                    },
                    "dense_boundary_cost_beam": {
                        "prior": 0.52,
                        "policy": {
                            **action_policy,
                            "scale_lattice": "dense",
                            "boundary_state_cap": max(12, int(boundary_cap)),
                            "max_scale_candidates": max(80, int(max_scales)),
                        },
                    },
                },
                replace_presets=True,
            )
        forced_reference_count = max(4, min(32, len(reference_patterns)))
        forced_reference_policy = {
            **reference_action_base,
            "bootstrap_anchor_count": forced_reference_count,
            "force_bootstrap_anchors": True,
            "boundary_scale_policy": "frontier",
            "scale_lattice": "waterline_sf",
            "boundary_state_cap": 12,
            "max_scale_candidates": 80,
            "bootstrap_penalty": 50_000_000.0,
            "selection_bootstrap_penalty": 0.0,
        }
        add(
            "reference_location_forced_sparse_cost",
            {
                **forced_reference_policy,
                "strategy": "bootstrap_mcts",
                "budget_aggressive": True,
                "mcts_action_cap": 10,
                "mcts_rollout_budget": 20,
                "mcts_exploration_weight": 1.1,
                "mcts_max_repair_bootstraps": 128,
                "mcts_prior_order": True,
                "mcts_actions": [
                    {
                        "name": "reference_forced_boundary_cost_beam",
                        "prior": 0.95,
                        "policy": forced_reference_policy,
                    },
                    {
                        "name": "budget_fulfillment_beam",
                        "prior": 0.55,
                        "policy": {
                            **forced_reference_policy,
                            "force_bootstrap_anchors": False,
                            "bootstrap_anchor_count": 0,
                            "bootstrap_anchor_selector": "",
                        },
                    },
                ],
                "mcts_action_allowlist": [
                    "reference_forced_boundary_cost_beam",
                    "budget_fulfillment_beam",
                ],
            },
            {
                "reference_forced_boundary_cost_beam": {
                    "prior": 0.95,
                    "policy": forced_reference_policy,
                },
                "budget_fulfillment_beam": {
                    "prior": 0.55,
                    "policy": {
                        **forced_reference_policy,
                        "force_bootstrap_anchors": False,
                        "bootstrap_anchor_count": 0,
                        "bootstrap_anchor_selector": "",
                    },
                },
            },
            replace_presets=True,
        )
    for floor in scale_candidates:
        if int(floor) >= waterline:
            continue
        add(
            f"relaxed_floor_{int(floor)}_frontier",
            {
                "scale_floor_bits": int(floor),
                "boundary_scale_policy": "frontier",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "state_cap_per_node": 32,
                "beam_width": 8,
                "mcts_action_cap": 10,
                "mcts_rollout_budget": 16,
                "selection_objective": "cost",
                "bootstrap_penalty": 75_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
            {
                f"estimator_relaxed_floor_{int(floor)}": {
                    "prior": 0.8,
                    "policy": {
                        "scale_floor_bits": int(floor),
                        "boundary_scale_policy": "frontier",
                        "boundary_state_cap": 8,
                        "max_scale_candidates": 64,
                        "selection_objective": "cost",
                    },
                },
            },
        )
        if int(floor) == max([int(item) for item in scale_candidates if int(item) < waterline] or [floor]):
            # Also test a waterline output boundary with the least aggressive
            # relaxed floor that can reduce rescales. This is a graph-general
            # scale policy, not a fixed bootstrap-count target.
            add(
                f"relaxed_floor_{int(floor)}_waterline_boundary",
                {
                    "scale_floor_bits": int(floor),
                    "boundary_scale_policy": "waterline",
                    "preferred_boundary_scale": waterline,
                    "boundary_scale": waterline,
                    "max_scale": waterline,
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "state_cap_per_node": 32,
                    "beam_width": 8,
                    "mcts_action_cap": 12,
                    "mcts_rollout_budget": 24,
                    "selection_objective": "cost",
                    "bootstrap_penalty": 65_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                },
                {
                    f"estimator_relaxed_floor_{int(floor)}": {
                        "prior": 0.86,
                        "policy": {
                            "scale_floor_bits": int(floor),
                            "boundary_scale_policy": "waterline",
                            "preferred_boundary_scale": waterline,
                            "boundary_scale": waterline,
                            "max_scale": waterline,
                            "boundary_state_cap": 8,
                            "max_scale_candidates": 64,
                            "selection_objective": "cost",
                            "bootstrap_penalty": 65_000_000.0,
                        },
                    },
                    "budget_fulfillment_beam": {
                        "prior": 0.68,
                        "policy": {
                            "boundary_scale_policy": "waterline",
                            "preferred_boundary_scale": waterline,
                            "boundary_scale": waterline,
                            "max_scale": waterline,
                            "boundary_state_cap": 8,
                            "max_scale_candidates": 64,
                            "selection_objective": "cost",
                            "bootstrap_penalty": 65_000_000.0,
                        },
                    },
                },
            )
    if top_groups:
        boundary_overlay_specs = [
            (
                "trace_cost_frontier_overlays",
                "frontier",
                "waterline_sf",
                12,
                80,
                10,
                48,
                35_000_000.0,
            ),
            (
                "trace_cost_waterline_overlays",
                "waterline",
                "waterline_sf",
                8,
                64,
                8,
                40,
                65_000_000.0,
            ),
            (
                "trace_cost_dense_overlays",
                "frontier",
                "dense",
                16,
                96,
                12,
                48,
                25_000_000.0,
            ),
        ]
        for (
            label,
            boundary_policy,
            scale_lattice,
            boundary_cap,
            max_scales,
            beam_width,
            state_cap,
            maintenance_penalty,
        ) in boundary_overlay_specs:
            overlays = []
            for group in top_groups[:6]:
                selector = group.get("group_key", group)
                if not isinstance(selector, dict):
                    continue
                overlays.append(
                    {
                        "selector": dict(selector),
                        "policy": {
                            "boundary_state_cap": int(boundary_cap),
                            "max_scale_candidates": int(max_scales),
                            "boundary_scale_policy": boundary_policy,
                            "scale_lattice": scale_lattice,
                            "bootstrap_penalty": float(maintenance_penalty),
                            "selection_bootstrap_penalty": 0.0,
                            "beam_width": int(beam_width),
                            "state_cap_per_node": int(state_cap),
                            "selection_objective": "cost",
                        },
                    }
                )
            if overlays:
                add(
                    label,
                    {
                        "boundary_group_policies": overlays,
                        "mcts_action_cap": 12,
                        "mcts_rollout_budget": 24,
                        "mcts_exploration_weight": 1.2,
                        "mcts_max_repair_bootstraps": 128,
                        "selection_objective": "cost",
                        "include_seed_repair_actions": False,
                    },
                    {
                        "budget_fulfillment_beam": {
                            "prior": 0.66,
                            "policy": {
                                "boundary_state_cap": int(boundary_cap),
                                "max_scale_candidates": int(max_scales),
                                "boundary_scale_policy": boundary_policy,
                                "scale_lattice": scale_lattice,
                                "bootstrap_penalty": float(maintenance_penalty),
                                "selection_objective": "cost",
                            },
                        },
                        "dense_boundary_cost_beam": {
                            "prior": 0.58,
                            "policy": {
                                "strategy": "latency_beam",
                                "boundary_state_cap": int(max(boundary_cap, 12)),
                                "max_scale_candidates": int(max(max_scales, 80)),
                                "boundary_scale_policy": boundary_policy,
                                "scale_lattice": scale_lattice,
                                "bootstrap_penalty": float(maintenance_penalty),
                                "selection_objective": "cost",
                            },
                        },
                    },
                )
    return variants


def _policy_bank_record(
    label: str,
    hints: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    artifacts = evaluation.get("artifacts", {}) if isinstance(evaluation, dict) else {}
    gate = _dict_from_jsonish(artifacts.get("correctness_gate", {}))
    trace = _dict_from_jsonish(artifacts.get("execution_trace", {}))
    objective = _finite_float(metrics.get("objective_cost_usec"), float("inf"))
    reference = _finite_float(metrics.get("reference_objective_cost_usec"), float("inf"))
    mlir_preview = str(artifacts.get("candidate_mlir_preview", ""))
    improved = bool(
        metrics.get("objective_improved_vs_seed", 0.0)
        or (math.isfinite(objective) and math.isfinite(reference) and objective < reference)
    )
    return {
        "label": label,
        "valid": bool(metrics.get("latency_only_correct", 0.0)),
        "correct": bool(metrics.get("latency_only_correct", 0.0)),
        "latency_improved": improved,
        "combined_score": _finite_float(metrics.get("combined_score"), 0.0),
        "objective_cost_usec": objective,
        "sampled_dp_latency_usec": _finite_float(
            metrics.get("sampled_dp_latency_usec"), objective
        ),
        "total_frontier_cost_usec": _finite_float(
            metrics.get("total_frontier_cost_usec"), objective
        ),
        "final_latency_usec": _finite_float(metrics.get("final_latency_usec"), objective),
        "reference_objective_cost_usec": reference,
        "objective_delta_usec": (
            objective - reference
            if math.isfinite(objective) and math.isfinite(reference)
            else float("inf")
        ),
        "seed_equivalent_path": bool(metrics.get("seed_equivalent_path", 0.0)),
        "candidate_qbp_coverage": _finite_float(metrics.get("candidate_qbp_coverage"), 0.0),
        "boundary_group_validity": _finite_float(metrics.get("boundary_group_validity"), 0.0),
        "fallback_selected_budgets": _finite_float(metrics.get("fallback_selected_budgets"), 0.0),
        "sampled_selected_path_bootstraps": _finite_float(
            metrics.get("sampled_selected_path_bootstraps"), 0.0
        ),
        "bootstrap_count": _finite_float(metrics.get("bootstrap_count"), 0.0),
        "sampled_frontier_total_bootstrap_count": _finite_float(
            metrics.get("sampled_frontier_total_bootstrap_count"), 0.0
        ),
        "rescale_count": _finite_float(metrics.get("rescale_count"), 0.0),
        "sampled_selected_path_rescales": _finite_float(
            metrics.get("sampled_selected_path_rescales"), 0.0
        ),
        "sampled_frontier_total_rescale_count": _finite_float(
            metrics.get("sampled_frontier_total_rescale_count"), 0.0
        ),
        "scale_floor_bits": _finite_float(metrics.get("scale_floor_bits"), 0.0),
        "selected_source_counts": {
            str(source): int(count or 0)
            for source, count in dict(
                trace.get("path", {}).get("selected_source_counts", {})
                if isinstance(trace.get("path"), dict)
                else {}
            ).items()
        },
        "top_costly_boundary_groups": list(
            trace.get("top_costly_boundary_groups", []) or []
        )[:8],
        "unsolved_boundary_groups": dict(trace.get("unsolved_boundary_groups", {}) or {}),
        "effective_qbp_digest": str(artifacts.get("effective_qbp_digest", ""))[:24],
        "selected_path_digest": str(
            artifacts.get("selected_path_digest")
            or artifacts.get("sampled_selected_path_digest")
            or ""
        )[:24],
        "path_digest": str(artifacts.get("sampled_selected_path_digest", ""))[:24],
        "candidate_mlir_digest": str(artifacts.get("candidate_mlir_digest", "")),
        "candidate_mlir_path": str(artifacts.get("candidate_mlir_path", "")),
        "candidate_mlir_preview": mlir_preview[:TRACE_PREVIEW_CHARS],
        "trace_features": _dict_from_jsonish(
            artifacts.get("candidate_mlir_trace_features", {})
        )
        or _mlir_trace_features(mlir_preview, result=evaluation),
        "gate_reasons": list(gate.get("reasons", []) or [])[:8],
        "policy_summary": json.loads(_compact_policy_summary(hints)),
    }


def _policy_bank_active_seed_baseline(record: dict[str, Any]) -> dict[str, Any]:
    objective = _finite_float(record.get("objective_cost_usec"), float("inf"))
    final_latency = _finite_float(record.get("final_latency_usec"), objective)
    sampled_latency = _finite_float(record.get("sampled_dp_latency_usec"), objective)
    total_frontier = _finite_float(record.get("total_frontier_cost_usec"), objective)
    bootstrap_count = _finite_float(
        record.get("bootstrap_count"),
        _finite_float(record.get("sampled_selected_path_bootstraps"), 0.0),
    )
    rescale_count = _finite_float(
        record.get("rescale_count"),
        _finite_float(record.get("sampled_selected_path_rescales"), 0.0),
    )
    return {
        "label": record.get("label"),
        "valid": bool(record.get("correct", record.get("valid", False))),
        "objective_cost_usec": objective,
        "sampled_dp_latency_usec": sampled_latency,
        "total_frontier_cost_usec": total_frontier,
        "final_latency_usec": final_latency,
        "bootstrap_count": bootstrap_count,
        "rescale_count": rescale_count,
        "effective_qbp_digest": str(record.get("effective_qbp_digest", ""))[:24],
        "selected_path_digest": str(
            record.get("selected_path_digest") or record.get("path_digest") or ""
        )[:24],
        "candidate_qbp_coverage": _finite_float(
            record.get("candidate_qbp_coverage"), 0.0
        ),
        "boundary_group_validity": _finite_float(
            record.get("boundary_group_validity"), 0.0
        ),
        "fallback_selected_budgets": _finite_float(
            record.get("fallback_selected_budgets"), 0.0
        ),
        "scale_floor_bits": _finite_float(record.get("scale_floor_bits"), 0.0),
        "top_costly_boundary_groups": list(
            record.get("top_costly_boundary_groups", []) or []
        )[:8],
        "unsolved_boundary_groups": dict(record.get("unsolved_boundary_groups", {}) or {}),
    }


def _policy_bank_candidate_examples(
    context: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seed = (
        context.get("harness", {}).get("sampled_seed_baseline")
        if isinstance(context.get("harness", {}), dict)
        else None
    )
    if not isinstance(seed, dict):
        seed = context.get("reference", {})
    if isinstance(seed, dict) and seed:
        examples.append(
            {
                "kind": "seed_reference",
                "objective_cost_usec": _finite_float(
                    seed.get("objective_cost_usec", seed.get("sampled_dp_latency_usec")),
                    0.0,
                ),
                "selected_path_digest": str(seed.get("selected_path_digest", ""))[:24],
                "bootstrap_count": _finite_float(seed.get("bootstrap_count"), 0.0),
                "rescale_count": _finite_float(seed.get("rescale_count"), 0.0),
                "top_costly_boundary_groups": list(
                    seed.get("top_costly_boundary_groups", []) or []
                )[:4],
            }
        )
    ranked = sorted(
        records,
        key=lambda item: (
            not bool(item.get("correct")),
            not bool(item.get("latency_improved")),
            _finite_float(item.get("objective_cost_usec"), float("inf")),
        ),
    )
    for item in ranked[:6]:
        examples.append(
            {
                "kind": (
                    "policy_bank_latency_improved"
                    if item.get("latency_improved")
                    else "policy_bank_loser"
                ),
                "label": item.get("label"),
                "correct": bool(item.get("correct")),
                "latency_improved": bool(item.get("latency_improved")),
                "objective_cost_usec": _finite_float(item.get("objective_cost_usec"), 0.0),
                "objective_delta_usec": _finite_float(item.get("objective_delta_usec"), 0.0),
                "selected_path_digest": str(item.get("path_digest", "")),
                "bootstrap_count": _finite_float(item.get("sampled_selected_path_bootstraps"), 0.0),
                "rescale_count": _finite_float(item.get("rescale_count"), 0.0),
                "scale_floor_bits": _finite_float(item.get("scale_floor_bits"), 0.0),
                "fallback_selected_budgets": _finite_float(
                    item.get("fallback_selected_budgets"), 0.0
                ),
                "top_costly_boundary_groups": list(
                    item.get("top_costly_boundary_groups", []) or []
                )[:4],
                "candidate_mlir_digest": item.get("candidate_mlir_digest", ""),
                "candidate_mlir_path": item.get("candidate_mlir_path", ""),
                "candidate_mlir_preview": str(item.get("candidate_mlir_preview", ""))[
                    :TRACE_PREVIEW_CHARS
                ],
                "trace_features": item.get("trace_features")
                or _mlir_trace_features(str(item.get("candidate_mlir_preview", ""))),
                "why_it_lost": _policy_bank_loss_reason(item),
            }
        )
    return examples[:8]


def _merge_candidate_examples(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            key = (
                str(item.get("kind", "")),
                str(item.get("candidate_mlir_digest", "")),
                str(item.get("selected_path_digest", "")),
            )
            if key in seen:
                continue
            seen.add(key)
            compact = dict(item)
            if "candidate_mlir_preview" in compact:
                compact["candidate_mlir_preview"] = str(compact["candidate_mlir_preview"])[
                    :TRACE_PREVIEW_CHARS
                ]
                compact.setdefault(
                    "trace_features",
                    _mlir_trace_features(str(compact.get("candidate_mlir_preview", ""))),
                )
            examples.append(compact)
    return examples[:8]


def _seed_mlir_trace_examples(
    context: dict[str, Any],
    program_path: Path,
    hints: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(result.get("valid", False)):
        return []
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    objective_cost = _finite_float(
        result.get("objective_cost_usec", result.get("final_latency_usec")),
        float("inf"),
    )
    if not math.isfinite(objective_cost):
        return []
    objective = {
        "objective_cost_usec": objective_cost,
        "base_objective_cost_usec": objective_cost,
        "reference_objective_cost_usec": objective_cost,
        "total_frontier_cost_usec": _finite_float(
            result.get("total_frontier_cost_usec", result.get("aggregated_partition_cost_usec")),
            objective_cost,
        ),
    }
    selected_path_digest = _selected_path_digest(result, diagnostics)
    correctness_gate = {
        "correct": True,
        "reasons": [],
        "seed_equivalent_path": True,
        "objective_improved_vs_seed": False,
    }
    execution_trace = {
        "objective": "seed_mlir_trace_reference",
        "correctness_gate": correctness_gate,
        "policy_summary": json.loads(_compact_policy_summary(hints)),
        "latency": {
            "objective_cost_usec": objective_cost,
            "final_latency_usec": _finite_float(result.get("final_latency_usec"), objective_cost),
            "total_frontier_cost_usec": objective["total_frontier_cost_usec"],
        },
        "path": {
            "selected_path_digest": selected_path_digest,
            "effective_qbp_digest": _effective_qbp_digest(diagnostics),
            "selected_source_counts": diagnostics.get("selected_source_counts", {}),
        },
        "top_costly_boundary_groups": _boundary_group_top_cost_summary(
            diagnostics.get("boundary_group_summaries", [])
        )[:8],
        "unsolved_boundary_groups": _boundary_group_unsolved_summary(
            diagnostics.get("boundary_group_summaries", [])
        ),
    }
    artifacts = _maybe_write_candidate_mlir_artifacts(
        context,
        program_path,
        hints,
        result,
        objective,
        correctness_gate,
        execution_trace,
    )
    if bool(artifacts.get("written", False)):
        mlir_digest = str(artifacts.get("mlir_digest", ""))
        mlir_path = str(artifacts.get("mlir_path", ""))
        mlir_preview = str(artifacts.get("mlir_preview", ""))[:TRACE_PREVIEW_CHARS]
        execution_trace_path = str(artifacts.get("execution_trace_path", ""))
        artifact_reason = ""
    else:
        mlir_preview = _mlir_trace_preview_from_result(result, diagnostics)
        mlir_digest = hashlib.sha256(mlir_preview.encode("utf-8")).hexdigest()
        mlir_path = ""
        execution_trace_path = ""
        artifact_reason = str(artifacts.get("reason", "mlir_artifact_not_written"))
    context.setdefault("harness", {})["seed_mlir_trace_summary"] = {
        "written": bool(artifacts.get("written", False)),
        "reason": artifact_reason,
        "candidate_mlir_digest": mlir_digest,
        "candidate_mlir_path": mlir_path,
        "mlir_preview_bytes": len(mlir_preview.encode("utf-8")),
    }
    return [
        {
            "kind": "seed_mlir_trace_reference",
            "label": "initial_seed",
            "correct": True,
            "latency_improved": False,
            "objective_cost_usec": objective_cost,
            "final_latency_usec": _finite_float(result.get("final_latency_usec"), objective_cost),
            "selected_path_digest": selected_path_digest[:24],
            "bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0),
            "rescale_count": _finite_float(result.get("rescale_count"), 0.0),
            "candidate_mlir_digest": mlir_digest,
            "candidate_mlir_path": mlir_path,
            "candidate_mlir_preview": mlir_preview[:TRACE_PREVIEW_CHARS],
            "candidate_mlir_artifact_reason": artifact_reason,
            "execution_trace_path": execution_trace_path,
            "trace_features": _mlir_trace_features(
                mlir_preview,
                result=result,
                diagnostics=diagnostics,
            ),
            "why_it_lost": "seed_reference_for_latency_and_trace_comparison",
            "top_costly_boundary_groups": execution_trace["top_costly_boundary_groups"][:4],
        }
    ]


def _mlir_trace_preview_from_result(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    limit: int = 80,
) -> str:
    """Build a compact MLIR-like placement trace when full MLIR emission fails."""

    lines = [
        'module attributes {orbit.trace = "placement"} {',
        f'  // objective_cost_usec = {_finite_float(result.get("objective_cost_usec"), 0.0):.3f}',
        f'  // selected_path_digest = {str(result.get("sampled_selected_path_digest", ""))[:32]}',
        f'  // bootstrap_count = {_finite_float(result.get("bootstrap_count"), 0.0):.3f}',
        f'  // rescale_count = {_finite_float(result.get("rescale_count"), 0.0):.3f}',
    ]

    def add_locations(kind: str, locations: Any) -> None:
        if not isinstance(locations, dict):
            return
        count = 0
        for node, attrs in sorted(locations.items(), key=lambda item: str(item[0])):
            if count >= limit:
                lines.append(f"  // {kind}: truncated")
                return
            if isinstance(attrs, dict):
                level = attrs.get("level", attrs.get("lvl", ""))
                scale = attrs.get("scale", attrs.get("scl", ""))
            else:
                level = ""
                scale = ""
            node_text = str(node).replace('"', '\\"')[:160]
            lines.append(
                f'  "orbit.trace.{kind}"() '
                f'{{node = "{node_text}", level = "{level}", scale = "{scale}"}} : () -> ()'
            )
            count += 1

    add_locations("bootstrap", result.get("bootstrap_locations", {}))
    add_locations("rescale", result.get("rescale_locations", {}))
    for idx, group in enumerate(
        _boundary_group_top_cost_summary(
            list(diagnostics.get("boundary_group_summaries", []) or []), limit=16
        )
    ):
        key = str(group.get("key", "")).replace('"', '\\"')[:160]
        source_counts = group.get("selected_source_counts", {})
        source = ""
        if isinstance(source_counts, dict) and source_counts:
            source = max(source_counts.items(), key=lambda item: int(item[1] or 0))[0]
        source_text = str(source).replace('"', '\\"')[:160]
        levels = ",".join(str(v) for v in list(group.get("requested_output_levels", []) or [])[:8])
        lines.append(
            f'  "orbit.trace.boundary_group"() '
            f'{{index = {idx}, key = "{key}", output_levels = "{levels}", '
            f'min_cost_usec = {_finite_float(group.get("min_cost_usec"), 0.0):.3f}, '
            f'min_bootstrap = {_finite_float(group.get("min_bootstrap"), 0.0):.3f}, '
            f'min_rescale = {_finite_float(group.get("min_rescale"), 0.0):.3f}, '
            f'dominant_source = "{source_text}"}} : () -> ()'
        )
    source_counts = diagnostics.get("selected_source_counts", {})
    if isinstance(source_counts, dict) and source_counts:
        for source, count in sorted(source_counts.items(), key=lambda item: str(item[0]))[:24]:
            source_text = str(source).replace('"', '\\"')[:160]
            lines.append(
                f'  "orbit.trace.selected_source"() '
                f'{{source = "{source_text}", count = {int(count or 0)}}} : () -> ()'
            )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _mlir_trace_features(
    mlir_preview: str,
    *,
    result: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract compact structured features from MLIR or MLIR-like trace text."""

    preview = str(mlir_preview or "")
    kind_counts: Counter = Counter()
    node_samples: dict[str, list[str]] = {"bootstrap": [], "rescale": []}
    selected_sources: Counter = Counter()
    objective_cost = None
    path_digest = ""
    for line in preview.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") and "=" in stripped:
            key, value = stripped[2:].split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "objective_cost_usec":
                objective_cost = _finite_float(value, 0.0)
            elif key == "selected_path_digest":
                path_digest = value[:32]
        if '"orbit.trace.' not in line:
            continue
        kind = line.split('"orbit.trace.', 1)[1].split('"', 1)[0]
        kind_counts[kind] += 1
        if kind in node_samples:
            node = _trace_attr(line, "node")
            if node and len(node_samples[kind]) < 12:
                node_samples[kind].append(node)
        if kind == "selected_source":
            source = _trace_attr(line, "source")
            count = _trace_int_attr(line, "count")
            if source:
                selected_sources[source] += max(1, count)
    if isinstance(result, dict):
        metrics = result.get("metrics", result)
        artifacts = result.get("artifacts", {})
        if objective_cost is None:
            objective_cost = _finite_float(
                metrics.get("objective_cost_usec", artifacts.get("objective_cost_usec")),
                0.0,
            )
        if not path_digest:
            path_digest = str(
                artifacts.get("sampled_selected_path_digest")
                or artifacts.get("selected_path_digest")
                or metrics.get("sampled_selected_path_digest")
                or ""
            )[:32]
        for kind, key in (("bootstrap", "bootstrap_locations"), ("rescale", "rescale_locations")):
            locations = result.get(key)
            if not isinstance(locations, dict):
                locations = artifacts.get(key) if isinstance(artifacts, dict) else None
            if isinstance(locations, dict):
                kind_counts[kind] = max(kind_counts[kind], len(locations))
                for node in list(locations)[:12]:
                    node_text = str(node)
                    if node_text not in node_samples[kind]:
                        node_samples[kind].append(node_text)
        artifact_sources = _dict_from_jsonish(
            artifacts.get("selected_source_counts", {})
            if isinstance(artifacts, dict)
            else {}
        )
        for source, count in artifact_sources.items():
            selected_sources[str(source)] += int(count or 0)
    diag = diagnostics if isinstance(diagnostics, dict) else {}
    boundary_groups = []
    if isinstance(diag.get("boundary_group_summaries"), list):
        boundary_groups = _boundary_group_top_cost_summary(diag["boundary_group_summaries"], limit=6)
    return {
        "trace_kind_counts": dict(sorted(kind_counts.items())),
        "objective_cost_usec": objective_cost if objective_cost is not None else 0.0,
        "selected_path_digest": path_digest,
        "bootstrap_nodes": node_samples["bootstrap"][:12],
        "rescale_nodes": node_samples["rescale"][:12],
        "selected_source_counts": dict(selected_sources.most_common(8)),
        "top_boundary_groups": boundary_groups,
        "preview_bytes": len(preview.encode("utf-8")),
    }


def _trace_learning_feedback(
    context: dict[str, Any],
    objective: dict[str, Any],
    correctness_gate: dict[str, Any],
    effective_path: dict[str, Any],
    trace_features: dict[str, Any],
) -> dict[str, Any]:
    """Turn MLIR-like trace features into concrete next mutation guidance."""

    cost = _finite_float(objective.get("objective_cost_usec"), float("inf"))
    reference = _finite_float(objective.get("reference_objective_cost_usec"), float("inf"))
    if not math.isfinite(reference) or reference <= 0:
        reference = _finite_float(objective.get("base_objective_cost_usec"), float("inf"))
    ratio = reference / cost if math.isfinite(reference) and math.isfinite(cost) and cost > 0 else 0.0
    path_changed = bool(effective_path.get("selected_path_changed_vs_seed", False))
    improved = bool(correctness_gate.get("objective_improved_vs_seed", False))
    top_groups = list(trace_features.get("top_boundary_groups", []) or [])[:4]
    selected_sources = dict(trace_features.get("selected_source_counts", {}) or {})
    kind_counts = dict(trace_features.get("trace_kind_counts", {}) or {})
    directives: list[str] = []
    if not bool(correctness_gate.get("correct", False)):
        directives.append(
            "Reject this candidate: restore direct QBP coverage before optimizing latency."
        )
    elif improved:
        directives.append(
            "Keep this selected-path pattern and make small local mutations around the same boundary groups."
        )
    elif not path_changed:
        directives.append(
            "This is a seed-equivalent trace: mutate one costly boundary group overlay or action preset until selected_path_digest changes."
        )
    else:
        directives.append(
            "This path changed but is slower: use it only as an exploration trace and reduce the costly boundary groups before keeping the path."
        )
    if top_groups:
        directives.append(
            "Use the first top_boundary_groups selector as a boundary_group_policies entry; change boundary_scale_policy, scale_lattice, boundary_state_cap, or bootstrap_penalty for that group only."
        )
    if selected_sources:
        directives.append(
            "If selected_source_counts are unchanged from the seed, alter mcts_action_allowlist or a preset prior enough to change the selected source mix."
        )
    if kind_counts.get("bootstrap", 0) or kind_counts.get("rescale", 0):
        directives.append(
            "Compare orbit.trace.bootstrap/rescale node samples with the seed trace; avoid moves that add maintenance without reducing objective_cost_usec."
        )
    return {
        "objective": "learn_from_mlir_trace_then_minimize_latency",
        "objective_cost_usec": cost if math.isfinite(cost) else None,
        "reference_objective_cost_usec": reference if math.isfinite(reference) else None,
        "objective_cost_ratio_vs_seed": ratio,
        "correct": bool(correctness_gate.get("correct", False)),
        "objective_improved_vs_seed": improved,
        "selected_path_changed_vs_seed": path_changed,
        "selected_path_digest": str(effective_path.get("selected_path_digest", ""))[:32],
        "trace_kind_counts": kind_counts,
        "selected_source_counts": selected_sources,
        "top_boundary_groups": top_groups,
        "directives": directives[:6],
    }


def _trace_attr(line: str, name: str) -> str:
    marker = f'{name} = "'
    if marker not in line:
        return ""
    return line.split(marker, 1)[1].split('"', 1)[0]


def _trace_int_attr(line: str, name: str) -> int:
    marker = f"{name} = "
    if marker not in line:
        return 0
    tail = line.split(marker, 1)[1]
    digits = []
    for char in tail:
        if char.isdigit() or (char == "-" and not digits):
            digits.append(char)
        else:
            break
    try:
        return int("".join(digits))
    except ValueError:
        return 0


def _policy_bank_loss_reason(item: dict[str, Any]) -> str:
    if item.get("latency_improved"):
        return "beats_seed_latency"
    if not item.get("correct"):
        reasons = item.get("gate_reasons") or []
        return "incorrect_or_incomplete: " + ", ".join(str(reason) for reason in reasons[:3])
    delta = _finite_float(item.get("objective_delta_usec"), 0.0)
    return f"path_changed_but_slower_by_{delta:.3f}_usec"


def _experience_probe_enabled() -> bool:
    raw = os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _experience_inline_qbp_enabled() -> bool:
    raw = os.environ.get("ORBIT_OPENEVOLVE_INLINE_QBP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _experience_probe_limit() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_TASKS", "").strip()
    try:
        return max(1, min(16, int(raw))) if raw else 4
    except ValueError:
        return 4


def _experience_probe_every() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_EVERY", "").strip()
    try:
        return max(1, min(10_000, int(raw))) if raw else 16
    except ValueError:
        return 16


def _experience_probe_threshold() -> float:
    raw = os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_THRESHOLD", "").strip()
    return max(0.0, min(1.0, _finite_float(raw, 0.82))) if raw else 0.82


def _boundary_group_key_from_dict(item: dict[str, Any]) -> str:
    return _boundary_group_key_string(
        (
            _safe_int(item.get("in_lvl"), -1),
            _safe_int(item.get("in_scl"), -1),
            str(item.get("maino_v", "")),
            _safe_int(item.get("main_dag_size"), 0),
        )
    )


def _sampled_task_group_keys(task: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for item in task.get("group_keys", []) or []:
        if isinstance(item, dict):
            keys.add(_boundary_group_key_from_dict(item))
    if keys:
        return keys
    context = task.get("context", {})
    budgets = context.get("io_budgets", []) if isinstance(context, dict) else []
    try:
        grouped = _budget_boundary_groups(
            [
                _io_budget_from_json(item) if isinstance(item, dict) else item
                for item in budgets
                if isinstance(item, dict)
            ]
        )
    except Exception:
        return keys
    return {_boundary_group_key_string(key) for key in grouped}


def _sampled_task_metric_summary(
    *,
    task_position: int,
    task: dict[str, Any],
    diagnostics: dict[str, Any],
    costs: list[float],
) -> dict[str, Any]:
    boundary_groups = [
        item
        for item in diagnostics.get("boundary_group_summaries", []) or []
        if isinstance(item, dict)
    ]
    group_summary = _boundary_group_count_summary(boundary_groups)
    path_proxy = _sampled_path_proxy_from_boundary_groups(boundary_groups, costs)
    scored_groups = _scored_boundary_group_count(diagnostics)
    unsolved = _boundary_group_unsolved_summary(boundary_groups)
    top_costly = _boundary_group_top_cost_summary(boundary_groups, limit=8)
    return {
        "task_position": int(task_position),
        "task_index": _safe_int(task.get("index"), task_position),
        "kind": str(task.get("kind", "normal")),
        "group_keys": list(task.get("group_keys", []) or []),
        "requested_budgets": int(diagnostics.get("requested_budgets", 0) or 0),
        "solved_budgets": int(diagnostics.get("solved_budgets", 0) or 0),
        "requested_boundary_groups": int(
            diagnostics.get("requested_boundary_groups", 0) or 0
        ),
        "reachable_boundary_groups": int(scored_groups),
        "solved_boundary_groups": int(diagnostics.get("solved_boundary_groups", 0) or 0),
        "candidate_solved_boundary_groups": int(
            diagnostics.get("candidate_solved_boundary_groups", 0) or 0
        ),
        "fallback_selected_groups": int(
            diagnostics.get("fallback_selected_boundary_groups", 0) or 0
        ),
        "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0) or 0),
        "unreachable_boundary_groups": int(
            diagnostics.get("unreachable_boundary_groups", 0) or 0
        ),
        "sampled_dp_latency_usec": float(path_proxy["sampled_dp_latency_usec"]),
        "total_frontier_cost_usec": float(group_summary["frontier_total_cost_usec"]),
        "bootstrap_count": float(group_summary["frontier_total_bootstrap"]),
        "rescale_count": float(group_summary["frontier_total_rescale"]),
        "avg_bootstrap": float(group_summary["frontier_bootstrap"]),
        "avg_rescale": float(group_summary["frontier_rescale"]),
        "selected_path_digest": str(path_proxy["sampled_selected_path_digest"]),
        "selected_source_counts": dict(diagnostics.get("selected_source_counts", {}) or {}),
        "top_costly_boundary_groups": top_costly,
        "unsolved_boundary_groups": unsolved,
    }


def _sampled_task_metrics_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = result.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        metrics = diagnostics.get("sampled_task_metrics")
        if isinstance(metrics, list):
            return [item for item in metrics if isinstance(item, dict)]
    metrics = result.get("sampled_task_seed_metrics")
    if isinstance(metrics, list):
        return [item for item in metrics if isinstance(item, dict)]
    return []


def _top_costly_boundary_groups_from_task_metrics(
    metrics: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for metric in metrics:
        task_position = _safe_int(metric.get("task_position"), -1)
        task_index = _safe_int(metric.get("task_index"), task_position)
        for item in metric.get("top_costly_boundary_groups", []) or []:
            if not isinstance(item, dict):
                continue
            updated = dict(item)
            updated["task_position"] = task_position
            updated["task_index"] = task_index
            groups.append(updated)
    groups.sort(
        key=lambda item: (
            _finite_float(item.get("min_cost_usec"), 0.0),
            _finite_float(item.get("max_cost_usec"), 0.0),
        ),
        reverse=True,
    )
    return groups[: max(0, int(limit))]


def _annotate_sampled_budget_tasks_with_seed_metrics(
    tasks: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> None:
    by_position = {
        _safe_int(item.get("task_position"), -1): item
        for item in metrics
        if isinstance(item, dict)
    }
    by_index = {
        _safe_int(item.get("task_index"), -1): item
        for item in metrics
        if isinstance(item, dict)
    }
    for position, task in enumerate(tasks):
        if not isinstance(task, dict):
            continue
        task_index = _safe_int(task.get("index"), position)
        metric = by_position.get(position) or by_index.get(task_index)
        if metric is not None:
            task["seed_metrics"] = metric


def _task_seed_metric_latency(metric: dict[str, Any]) -> float:
    for key in ("sampled_dp_latency_usec", "objective_cost_usec", "total_frontier_cost_usec"):
        value = _finite_float(metric.get(key), float("inf"))
        if math.isfinite(value) and value > 0:
            return float(value)
    return float("inf")


def _promotion_probe_tasks_from_seed_metrics(
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], float] | None:
    tasks = [
        task
        for task in context.get("sampled_budget_tasks", []) or []
        if isinstance(task, dict) and isinstance(task.get("context"), dict)
    ]
    if not tasks:
        return None
    harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
    metrics = harness.get("sampled_task_seed_metrics")
    if not isinstance(metrics, list):
        reference = context.get("reference", {})
        if isinstance(reference, dict):
            metrics = reference.get("sampled_task_seed_metrics")
    if not isinstance(metrics, list):
        return None
    usable = [
        item
        for item in metrics
        if isinstance(item, dict) and math.isfinite(_task_seed_metric_latency(item))
    ]
    if not usable:
        return None
    usable.sort(key=_task_seed_metric_latency, reverse=True)
    selected: list[dict[str, Any]] = []
    selected_cost = 0.0
    seen_positions: set[int] = set()
    limit = _promotion_probe_limit()
    for metric in usable:
        position = _safe_int(metric.get("task_position"), -1)
        task_index = _safe_int(metric.get("task_index"), position)
        if position < 0 or position >= len(tasks):
            position = next(
                (
                    idx
                    for idx, task in enumerate(tasks)
                    if _safe_int(task.get("index"), idx) == task_index
                ),
                -1,
            )
        if position < 0 or position >= len(tasks) or position in seen_positions:
            continue
        selected.append(tasks[position])
        seen_positions.add(position)
        selected_cost += _task_seed_metric_latency(metric)
        if len(selected) >= limit:
            break
    if not selected or selected_cost <= 0:
        return None
    return selected, float(selected_cost)


def _experience_probe_tasks(context: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    tasks = [
        task
        for task in context.get("sampled_budget_tasks", []) or []
        if isinstance(task, dict) and isinstance(task.get("context"), dict)
    ]
    if not tasks:
        return [], float("inf")
    harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
    top_groups = [
        item
        for item in harness.get("top_costly_boundary_groups", []) or []
        if isinstance(item, dict)
    ]
    if not top_groups:
        return tasks[: _experience_probe_limit()], float("inf")
    top_cost_by_key = {}
    for item in top_groups:
        key = str(item.get("key") or "")
        if not key:
            group_key = item.get("group_key", {})
            if isinstance(group_key, dict):
                key = _boundary_group_key_from_dict(group_key)
        if key:
            top_cost_by_key[key] = _finite_float(item.get("min_cost_usec"), 0.0)
    wanted = {key for key, cost in top_cost_by_key.items() if key and cost > 0}
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for task in tasks:
        keys = _sampled_task_group_keys(task)
        if keys & wanted:
            selected.append(task)
            selected_keys |= keys & wanted
        if len(selected) >= _experience_probe_limit():
            break
    if not selected:
        return tasks[: _experience_probe_limit()], float("inf")
    reference_cost = sum(top_cost_by_key.get(key, 0.0) for key in selected_keys)
    return selected, float(reference_cost if reference_cost > 0 else float("inf"))


def _top_boundary_group_key_tuples(context: dict[str, Any], limit: int = 4) -> list[tuple[int, int, str, int]]:
    harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
    keys: list[tuple[int, int, str, int]] = []
    for item in harness.get("top_costly_boundary_groups", []) or []:
        if not isinstance(item, dict):
            continue
        group_key = item.get("group_key")
        if isinstance(group_key, dict):
            keys.append(
                (
                    int(group_key.get("in_lvl", -1)),
                    int(group_key.get("in_scl", -1)),
                    str(group_key.get("maino_v", "")),
                    int(group_key.get("main_dag_size", 0) or 0),
                )
            )
        if len(keys) >= limit:
            break
    return keys


def _boundary_group_policy_focus_score(
    context: dict[str, Any],
    hints: dict[str, Any],
) -> float:
    policies = [
        item
        for item in hints.get("boundary_group_policies", []) or []
        if isinstance(item, dict)
    ]
    if not policies:
        return 0.0
    top_keys = _top_boundary_group_key_tuples(context, limit=4)
    if not top_keys:
        return 0.25 if len(policies) <= 2 else 0.05
    matched = 0
    for item in policies:
        selector = item.get("selector", {})
        if any(_boundary_group_selector_matches_key(selector, key) for key in top_keys):
            matched += 1
    match_score = min(1.0, matched / max(1.0, min(2, len(top_keys))))
    focus_penalty = 1.0 if len(policies) <= 2 else max(0.0, 1.0 - 0.20 * (len(policies) - 2))
    return float(match_score * focus_penalty)


def _experience_surrogate_summary(
    context: dict[str, Any],
    eval_hints: dict[str, Any],
    policy_effect: dict[str, Any],
) -> dict[str, Any]:
    payload = _policy_effect_payload(eval_hints)
    policy = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
    actions = [
        item
        for item in payload.get("mcts_actions", []) or []
        if isinstance(item, dict)
    ]
    action_names = [str(item.get("name", "")) for item in actions if str(item.get("name", ""))]
    preferred_actions = {
        "wide_boundary_cost_beam",
        "budget_fulfillment_beam",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        "component_budget_repair",
        "latency_mcts_repair",
        "profile_waterline_repair",
    }
    action_overlap = len(preferred_actions & set(action_names)) / max(1, len(preferred_actions))
    targeted_count = int(payload.get("unit_policy_count", 0) or 0) + int(
        payload.get("boundary_group_policy_count", 0) or 0
    )
    lattice_keys = {
        "boundary_state_cap",
        "max_scale_candidates",
        "scale_floor_bits",
        "scale_lattice",
        "boundary_scale_policy",
        "bootstrap_anchor_selector",
        "bootstrap_penalty",
        "selection_bootstrap_penalty",
        "rescale_penalty",
        "level_drop_penalty",
    }
    lattice_changed = len(lattice_keys & set(policy))
    action_cost_objective = any(
        str(item.get("policy", {}).get("selection_objective", "")).lower() == "cost"
        for item in actions
        if isinstance(item.get("policy"), dict)
    )
    cost_objective = (
        str(policy.get("selection_objective", "")).lower() == "cost"
        or action_cost_objective
    )
    examples = context.get("harness", {}).get("candidate_examples", [])
    trace_example_bonus = 0.0
    if isinstance(examples, list) and examples:
        trace_example_bonus = 0.05
    effect_score = float(policy_effect.get("effect_score", 0.0) or 0.0)
    complexity_reasons = _promotion_complexity_reasons(eval_hints)
    focus_score = _boundary_group_policy_focus_score(context, eval_hints)
    target_focus = 1.0 if targeted_count <= 2 else max(0.0, 1.0 - 0.15 * (targeted_count - 2))
    raw_score = (
        0.16 * effect_score
        + 0.14 * min(1.0, action_overlap)
        + 0.22 * focus_score
        + 0.16 * target_focus
        + 0.12 * min(1.0, lattice_changed / 3.0)
        + (0.12 if cost_objective else 0.0)
        + (0.10 if not complexity_reasons else 0.0)
        + trace_example_bonus
    )
    score = min(
        1.0,
        raw_score,
    )
    if complexity_reasons:
        score = min(score, 0.18)
    digest = _hint_digest(payload)
    seed_equivalent = bool(policy_effect.get("seed_equivalent", False))
    every = _experience_probe_every()
    hash_promote = int(digest[:8], 16) % every == 0
    threshold = _experience_probe_threshold()
    promote = (not seed_equivalent) and not complexity_reasons and (score >= threshold or hash_promote)
    return {
        "score": float(score),
        "digest": digest[:24],
        "seed_equivalent": seed_equivalent,
        "promote_to_probe": bool(promote),
        "promotion_reason": (
            "surrogate_score_threshold"
            if score >= threshold
            else ("hash_exploration" if hash_promote else "feature_only")
        ),
        "promotion_threshold": float(threshold),
        "probe_every": int(every),
        "action_names": action_names[:12],
        "action_overlap": float(action_overlap),
        "targeted_policy_count": int(targeted_count),
        "lattice_changed_count": int(lattice_changed),
        "cost_objective": bool(cost_objective),
        "boundary_group_focus_score": float(focus_score),
        "target_focus_score": float(target_focus),
        "promotion_complexity_reasons": complexity_reasons,
    }


def _experience_surrogate_only_result(
    raw_hints: dict[str, Any],
    policy_effect: dict[str, Any],
    surrogate: dict[str, Any],
) -> dict[str, Any]:
    reason = "experience_surrogate_feature_only"
    result = _invalid_compile_result(reason, [reason])
    score = max(1e-6, min(0.49, 0.05 + 0.40 * float(surrogate.get("score", 0.0))))
    metrics = result.setdefault("metrics", {})
    metrics.update(
        {
            "combined_score": float(score),
            "experience_surrogate_score": float(surrogate.get("score", 0.0)),
            "experience_probe_promoted": 0.0,
            "experience_probe_task_count": 0.0,
            "candidate_stage_rank": 0.0,
            "policy_effect_score": float(policy_effect.get("effect_score", 0.0)),
            "seed_equivalent_policy": float(bool(policy_effect.get("seed_equivalent", False))),
        }
    )
    result.setdefault("artifacts", {})["candidate_stage"] = "surrogate"
    result.setdefault("artifacts", {})["experience_surrogate"] = json.dumps(
        {
            **surrogate,
            "policy_summary": json.loads(_compact_policy_summary(raw_hints)),
        },
        sort_keys=True,
    )
    return result


def _experience_probe_skip_result(
    context: dict[str, Any],
    raw_hints: dict[str, Any],
    eval_hints: dict[str, Any],
    eval_suite: str,
    policy_effect: dict[str, Any],
    *,
    suppress_output: bool,
    disabled: bool = False,
) -> dict[str, Any] | None:
    if disabled:
        return None
    if eval_suite == "polybert-full" or not context.get("sampled_budget_tasks"):
        return None
    surrogate = _experience_surrogate_summary(context, eval_hints, policy_effect)
    if not _experience_inline_qbp_enabled() or not _experience_probe_enabled():
        return _experience_surrogate_only_result(raw_hints, policy_effect, surrogate)
    if not bool(surrogate.get("promote_to_probe", False)):
        return _experience_surrogate_only_result(raw_hints, policy_effect, surrogate)
    selected_tasks, reference_cost = _experience_probe_tasks(context)
    if not selected_tasks or not math.isfinite(reference_cost):
        return None
    probe_context = deepcopy(context)
    probe_context["sampled_budget_tasks"] = deepcopy(selected_tasks)
    probe_context.setdefault("harness", {})["experience_probe"] = True
    probe_result = _evaluate_sampled_budget_tasks(
        probe_context,
        eval_hints,
        suppress_output=suppress_output,
    )
    probe_diagnostics = probe_result.get("diagnostics", {})
    if not isinstance(probe_diagnostics, dict):
        probe_diagnostics = {}
    probe_cost = _finite_float(
        probe_result.get("sampled_dp_latency_usec", probe_result.get("objective_cost_usec")),
        float("inf"),
    )
    direct_coverage = _finite_float(probe_result.get("candidate_qbp_coverage"), 0.0)
    boundary_validity = _finite_float(probe_result.get("boundary_group_validity"), 0.0)
    min_relative_improvement = _finite_float(
        os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_MIN_REL", 0.0),
        0.0,
    )
    improved = (
        math.isfinite(probe_cost)
        and probe_cost < reference_cost * (1.0 - max(0.0, min_relative_improvement))
    )
    direct_complete = direct_coverage >= 1.0 and boundary_validity >= 1.0
    if improved and direct_complete:
        return None
    if not direct_complete:
        reason = "experience_probe_incomplete_top_groups"
    else:
        reason = "experience_probe_no_latency_improvement"
    result = _invalid_compile_result(reason, [reason])
    metrics = result.setdefault("metrics", {})
    metrics.update(
        {
            "experience_probe_cost_usec": float(probe_cost if math.isfinite(probe_cost) else 0.0),
            "experience_probe_reference_cost_usec": float(reference_cost),
            "experience_probe_candidate_qbp_coverage": float(direct_coverage),
            "experience_probe_boundary_group_validity": float(boundary_validity),
            "experience_probe_task_count": float(len(selected_tasks)),
            "experience_probe_promoted": 1.0,
            "candidate_stage_rank": 1.0,
            "policy_effect_score": float(policy_effect.get("effect_score", 0.0)),
            "seed_equivalent_policy": float(bool(policy_effect.get("seed_equivalent", False))),
            "sampled_task_cache_hits": float(
                probe_result.get("sampled_task_cache_hits", 0.0)
            ),
            "sampled_task_cache_misses": float(
                probe_result.get("sampled_task_cache_misses", 0.0)
            ),
            "sampled_task_cache_writes": float(
                probe_result.get("sampled_task_cache_writes", 0.0)
            ),
        }
    )
    artifacts = result.setdefault("artifacts", {})
    artifacts["candidate_stage"] = "probe_qbp"
    artifacts["experience_probe"] = json.dumps(
        {
            "reason": reason,
            "candidate_cost_usec": probe_cost if math.isfinite(probe_cost) else None,
            "reference_cost_usec": reference_cost,
            "direct_coverage": direct_coverage,
            "boundary_validity": boundary_validity,
            "task_count": len(selected_tasks),
            "policy_summary": json.loads(_compact_policy_summary(raw_hints)),
            "surrogate": surrogate,
            "diagnostics": {
                "sampled_task_cache_hits": int(
                    probe_result.get("sampled_task_cache_hits", 0) or 0
                ),
                "sampled_task_cache_misses": int(
                    probe_result.get("sampled_task_cache_misses", 0) or 0
                ),
                "sampled_task_cache_writes": int(
                    probe_result.get("sampled_task_cache_writes", 0) or 0
                ),
                "selected_source_counts": dict(
                    probe_diagnostics.get("selected_source_counts", {}) or {}
                ),
            },
        },
        sort_keys=True,
    )
    return result


def _program_source_from_hints(hints: dict[str, Any], title: str) -> str:
    return (
        f'"""{title}"""\n\n'
        "def place(context):\n"
        f"    return {repr(hints)}\n"
    )


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _safe_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return cleaned[:80] or "candidate"


def evaluate_compile_candidate_program(
    context_path: str | Path, program_path: str | Path
) -> dict[str, Any]:
    total_start = time.perf_counter()
    last_mark = total_start
    timings: dict[str, float] = {}

    def mark_timing(name: str) -> None:
        nonlocal last_mark
        now = time.perf_counter()
        timings[name] = float(now - last_mark)
        last_mark = now

    def attach_timings(evaluation: dict[str, Any]) -> dict[str, Any]:
        timings["total"] = float(time.perf_counter() - total_start)
        metrics = evaluation.setdefault("metrics", {})
        for key, value in timings.items():
            metrics[f"timing_{key}_sec"] = float(value)
        artifacts = evaluation.setdefault("artifacts", {})
        artifacts["component_timing"] = json.dumps(
            {f"{key}_sec": float(value) for key, value in sorted(timings.items())},
            sort_keys=True,
        )
        return evaluation

    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    mark_timing("context_load")
    try:
        hints = _load_candidate_hints(Path(program_path), context)
        mark_timing("load_candidate")
        static = _static_validate_hints(context, hints)
        mark_timing("static_validate")
        if not static["valid"]:
            result = _invalid_compile_result("compile_static_gate", static["reasons"])
            attach_timings(result)
            trace_start = time.perf_counter()
            _record_compile_trace(context, Path(program_path), hints, result, "STATIC_ONLY")
            timings["record_trace"] = float(time.perf_counter() - trace_start)
            attach_timings(result)
            return result
        sampled_reasons = _sampled_policy_static_reasons(context, hints)
        mark_timing("sampled_static_gate")
        if sampled_reasons:
            result = _invalid_compile_result("compile_sampled_policy_gate", sampled_reasons)
            attach_timings(result)
            trace_start = time.perf_counter()
            _record_compile_trace(context, Path(program_path), hints, result, "STATIC_ONLY")
            timings["record_trace"] = float(time.perf_counter() - trace_start)
            attach_timings(result)
            return result
        eval_suite = str(context.get("harness", {}).get("eval_suite", "polybert-sampled"))
        eval_hints = _compile_hints_for_eval_suite(hints, eval_suite)
        mark_timing("compile_hints")
        policy_effect = _policy_effect_summary(context, hints, eval_hints, eval_suite)
        mark_timing("policy_effect")
        try:
            candidate_source_digest = _source_digest(
                Path(program_path).read_text(encoding="utf-8")
            )
        except Exception:
            candidate_source_digest = ""
        initial_source_digest = str(
            context.get("harness", {}).get("initial_program_digest", "") or ""
        )
        allow_seed_equivalent = (
            Path(program_path).name == "initial_program.py"
            or (
                bool(initial_source_digest)
                and candidate_source_digest == initial_source_digest
            )
            or os.environ.get("ORBIT_OPENEVOLVE_EVALUATE_SEED_EQUIV", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if bool(policy_effect.get("seed_equivalent", False)) and not allow_seed_equivalent:
            result = _invalid_compile_result(
                "seed_equivalent_policy_probe_skip",
                ["normalized_policy_matches_seed"],
            )
            result.setdefault("metrics", {})["seed_equivalent_policy"] = 1.0
            result.setdefault("metrics", {})["policy_effect_score"] = 0.0
            result.setdefault("artifacts", {})["policy_effect"] = json.dumps(
                policy_effect,
                sort_keys=True,
            )
            attach_timings(result)
            trace_start = time.perf_counter()
            _record_compile_trace(context, Path(program_path), hints, result, "STATIC_ONLY")
            timings["record_trace"] = float(time.perf_counter() - trace_start)
            attach_timings(result)
            return result
        probe_skip = _experience_probe_skip_result(
            context,
            hints,
            eval_hints,
            eval_suite,
            policy_effect,
            suppress_output=True,
            disabled=allow_seed_equivalent,
        )
        mark_timing("experience_probe")
        if probe_skip is not None:
            attach_timings(probe_skip)
            trace_start = time.perf_counter()
            _record_compile_trace(context, Path(program_path), hints, probe_skip, "CLEAR_ONLY")
            timings["record_trace"] = float(time.perf_counter() - trace_start)
            attach_timings(probe_skip)
            return probe_skip
        result = _evaluate_compile_hints(context, eval_hints, suppress_output=True)
        mark_timing("evaluate_compile_hints")
        result["static"] = static
        diagnostics = result.get("diagnostics", {})
        requested_budgets = max(1, int(diagnostics.get("requested_budgets", 1)))
        candidate_solved = int(diagnostics.get("candidate_solved_budgets", 0))
        solved_budgets = int(diagnostics.get("solved_budgets", requested_budgets if result["valid"] else 0))
        candidate_validity = candidate_solved / requested_budgets
        effective_validity = solved_budgets / requested_budgets
        requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
        has_boundary_group_diagnostics = int(
            diagnostics.get("requested_boundary_groups", 0) or 0
        ) > 0
        has_boundary_group_diagnostics = int(
            diagnostics.get("requested_boundary_groups", 0) or 0
        ) > 0
        solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
        fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
        invalid_groups = int(diagnostics.get("invalid_boundary_groups", 0) or 0)
        unreachable_groups = int(diagnostics.get("unreachable_boundary_groups", 0) or 0)
        scored_groups = _scored_boundary_group_count(diagnostics)
        if has_boundary_group_diagnostics:
            boundary_group_validity = min(1.0, solved_groups / scored_groups)
            candidate_qbp_coverage = min(1.0, candidate_groups / scored_groups)
        else:
            boundary_group_validity = min(1.0, effective_validity)
            candidate_qbp_coverage = min(1.0, candidate_validity)
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
        policy_effect_score = float(policy_effect.get("effect_score", 0.0))
        action_effect_score = _action_effect_score(diagnostics)
        target_bootstrap_count = _context_target_bootstrap_count(context)
        target_bootstrap_score = _contextual_target_bootstrap_score(
            context,
            target_bootstrap_count,
            result["bootstrap_count"],
            reference.get("bootstrap_count"),
        )
        component_bootstrap = _component_bootstrap_alignment_summary(
            context,
            result.get("bootstrap_locations", {}),
            result.get("bootstrap_count", 0),
        )
        component_bootstrap_score = float(component_bootstrap.get("score", 0.0))
        placement_effect = _placement_effect_summary(
            context,
            result,
            diagnostics,
            candidate_target_score=target_bootstrap_score,
        )
        effective_path = _result_effective_summary(context, result, diagnostics)
        placement_effect_score = float(placement_effect.get("effect_score", 0.0))
        path_effect_score = 1.0 if effective_path.get("selected_path_changed_vs_seed") else 0.0
        scale_floor_bits = _safe_int(result.get("scale_floor_bits"), int(_ckks_dict(context)["Sw"]))
        scale_floor_delta_bits = int(_ckks_dict(context)["Sw"]) - int(scale_floor_bits)
        scale_floor_summary = {
            "policy": context.get("params", {}).get("scale_floor_policy", "waterline"),
            "waterline_bits": int(_ckks_dict(context)["Sw"]),
            "active_floor_bits": int(scale_floor_bits),
            "scale_floor_delta_bits": int(scale_floor_delta_bits),
            "scale_floor_min_bits": context.get("params", {}).get("scale_floor_min_bits"),
            "candidates": _scale_floor_candidates_from_context(context),
            "relaxed": scale_floor_delta_bits > 0,
        }
        placement_effect_eps = 1e-6
        policy_tiebreak_score = (
            policy_effect_score if placement_effect_score > placement_effect_eps else 0.0
        )
        seed_bootstrap_count = _context_seed_bootstrap_count(context)
        bootstrap_delta_vs_seed = (
            float(result["bootstrap_count"] - seed_bootstrap_count)
            if seed_bootstrap_count is not None
            else 0.0
        )
        noise_estimate = _fast_compile_noise_estimate(context, result)
        noise_score = 1.0 if noise_estimate.get("valid", False) else 0.0
        reserve_score = _reserve_quality_score(result.get("reserve_summary", {}))
        objective = _cost_minimization_objective(
            context,
            result,
            diagnostics,
            effective_validity=effective_validity,
            repair_count=repair_count,
        )
        correctness_gate = _latency_only_correctness_gate(
            result,
            diagnostics,
            objective,
            static=static,
            boundary_group_validity=boundary_group_validity,
            candidate_qbp_coverage=candidate_qbp_coverage,
            effective_path=effective_path,
            policy_effect=policy_effect,
        )
        latency_only_score = _latency_only_combined_score(
            objective,
            correct=bool(correctness_gate["correct"]),
            correctness_gate=correctness_gate,
        )
        mark_timing("metrics_scoring")
        quality_score = (
            0.35 * latency_score
            + 0.08 * bootstrap_score
            + 0.05 * target_bootstrap_score
            + 0.10 * component_bootstrap_score
            + 0.05 * rescale_score
            + 0.08 * boundary_score
            + 0.07 * risk_score
            + 0.08 * noise_score
            + 0.09 * reserve_score
            + 0.04 * fallback_score
            + 0.04 * runtime_score
            + 0.08 * placement_effect_score
            + 0.04 * path_effect_score
            + 0.01 * policy_tiebreak_score
            + 0.02 * action_effect_score
        )
        cost_tiebreak_score = (
            0.68 * float(objective["objective_cost_score"])
            + 0.08 * bootstrap_score
            + 0.06 * rescale_score
            + 0.05 * reserve_score
            + 0.04 * placement_effect_score
            + 0.06 * path_effect_score
            + 0.02 * action_effect_score
            + 0.01 * policy_tiebreak_score
        )
        combined_score = latency_only_score
        execution_trace = _execution_trace_artifact(
            context,
            eval_hints,
            result,
            diagnostics,
            objective,
            policy_effect,
            placement_effect,
            effective_path,
            scale_floor_summary,
            correctness_gate,
        )
        mark_timing("execution_trace")
        candidate_artifacts = _maybe_write_candidate_mlir_artifacts(
            context,
            Path(program_path),
            eval_hints,
            result,
            objective,
            correctness_gate,
            execution_trace,
        )
        mark_timing("candidate_artifacts")
        if not str(candidate_artifacts.get("mlir_preview", "")).strip():
            trace_preview = _mlir_trace_preview_from_result(result, diagnostics)
            candidate_artifacts = dict(candidate_artifacts)
            candidate_artifacts["mlir_preview"] = trace_preview
            if not str(candidate_artifacts.get("mlir_digest", "")).strip():
                candidate_artifacts["mlir_digest"] = hashlib.sha256(
                    trace_preview.encode("utf-8")
                ).hexdigest()
            candidate_artifacts["trace_fallback"] = True
        candidate_trace_features = _mlir_trace_features(
            str(candidate_artifacts.get("mlir_preview", "")),
            result=result,
            diagnostics=diagnostics,
        )
        trace_learning_feedback = _trace_learning_feedback(
            context,
            objective,
            correctness_gate,
            effective_path,
            candidate_trace_features,
        )
        mark_timing("trace_features_feedback")
        candidate_stage = "full_replay" if eval_suite == "polybert-full" else "sampled_qbp"
        candidate_stage_rank = 3.0 if candidate_stage == "full_replay" else 2.0
        evaluation = {
            "metrics": {
                "combined_score": float(combined_score),
                "candidate_stage_rank": float(candidate_stage_rank),
                "experience_probe_promoted": 1.0,
                "latency_only_correct": float(bool(correctness_gate["correct"])),
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
                "placement_effect_score": float(placement_effect_score),
                "path_effect_score": float(path_effect_score),
                "seed_equivalent_path": float(
                    bool(effective_path.get("seed_equivalent_path", False))
                ),
                "objective_improved_vs_seed": float(
                    bool(correctness_gate.get("objective_improved_vs_seed", False))
                ),
                "policy_effect_score": float(policy_effect_score),
                "action_effect_score": float(action_effect_score),
                "seed_equivalent_policy": float(bool(policy_effect.get("seed_equivalent"))),
                "scale_floor_bits": float(scale_floor_bits),
                "scale_floor_delta_bits": float(scale_floor_delta_bits),
                "objective_cost_usec": float(objective["objective_cost_usec"]),
                "base_objective_cost_usec": float(objective["base_objective_cost_usec"]),
                "reference_objective_cost_usec": float(
                    objective["reference_objective_cost_usec"]
                ),
                "objective_cost_ratio_vs_seed": float(
                    objective["objective_cost_ratio_vs_seed"]
                ),
                "objective_cost_score": float(objective["objective_cost_score"]),
                "objective_tier": float(objective["objective_tier"]),
                "latency_only_score": float(latency_only_score),
                "total_frontier_cost_usec": float(objective["total_frontier_cost_usec"]),
                "sampled_dp_latency_usec": float(
                    _finite_float(
                        result.get("sampled_dp_latency_usec"),
                        objective["base_objective_cost_usec"],
                    )
                ),
                "sampled_selected_path_bootstraps": float(
                    _finite_float(result.get("sampled_selected_path_bootstraps"), 0.0)
                ),
                "sampled_selected_path_rescales": float(
                    _finite_float(result.get("sampled_selected_path_rescales"), 0.0)
                ),
                "reachable_boundary_groups": float(objective["reachable_boundary_groups"]),
                "unsolved_reachable_boundary_groups": float(
                    objective["unsolved_reachable_boundary_groups"]
                ),
                "candidate_direct_group_coverage": float(
                    objective["candidate_direct_group_coverage"]
                ),
                "direct_unsolved_boundary_groups": float(
                    objective["direct_unsolved_boundary_groups"]
                ),
                "objective_penalty_usec": float(objective["objective_penalty_usec"]),
                "bootstrap_score": float(bootstrap_score),
                "target_bootstrap_score": float(target_bootstrap_score),
                "component_bootstrap_score": float(component_bootstrap_score),
                "rescale_score": float(rescale_score),
                "final_latency_usec": float(result["final_latency_usec"] if result["valid"] else 0.0),
                "boundary_quality": float(boundary_score),
                "bootstrap_count": float(result["bootstrap_count"]),
                "reference_bootstrap_count": float(seed_bootstrap_count or 0),
                "bootstrap_delta_vs_seed": float(bootstrap_delta_vs_seed),
                "sampled_avg_bootstrap_count": float(
                    result.get("sampled_avg_bootstrap_count", result["bootstrap_count"])
                ),
                "sampled_frontier_total_bootstrap_count": float(
                    result.get(
                        "sampled_frontier_total_bootstrap_count",
                        result["bootstrap_count"],
                    )
                ),
                "target_bootstrap_count": float(target_bootstrap_count),
                "component_bootstrap_target": float(
                    component_bootstrap.get("target_total", 0)
                ),
                "rescale_count": float(result["rescale_count"]),
                "sampled_avg_rescale_count": float(
                    result.get("sampled_avg_rescale_count", result["rescale_count"])
                ),
                "sampled_frontier_total_rescale_count": float(
                    result.get(
                        "sampled_frontier_total_rescale_count",
                        result["rescale_count"],
                    )
                ),
                "profile_risk": float(result["profile_risk"]),
                "placement_runtime_sec": float(result["placement_runtime_sec"]),
                "sampled_task_total_runtime_sec": float(
                    result.get("sampled_task_total_runtime_sec", 0.0)
                ),
                "sampled_task_max_runtime_sec": float(
                    result.get("sampled_task_max_runtime_sec", 0.0)
                ),
                "sampled_task_solve_total_sec": float(
                    result.get("sampled_task_solve_total_sec", 0.0)
                ),
                "sampled_task_solve_max_sec": float(
                    result.get("sampled_task_solve_max_sec", 0.0)
                ),
                "sampled_task_cache_hits": float(
                    result.get("sampled_task_cache_hits", 0.0)
                ),
                "sampled_task_cache_misses": float(
                    result.get("sampled_task_cache_misses", 0.0)
                ),
                "sampled_task_cache_writes": float(
                    result.get("sampled_task_cache_writes", 0.0)
                ),
                "compile_replay_solve_partition_sec": float(
                    result.get("diagnostics", {})
                    .get("compile_replay_timing_sec", {})
                    .get("solve_partition", 0.0)
                ),
                "compile_replay_total_sec": float(
                    result.get("diagnostics", {})
                    .get("compile_replay_timing_sec", {})
                    .get("total", 0.0)
                ),
                "fallback_selected_budgets": float(result["fallback_selected_budgets"]),
                "requested_boundary_groups": float(requested_groups),
                "solved_boundary_groups": float(solved_groups),
                "candidate_solved_boundary_groups": float(candidate_groups),
                "fallback_selected_groups": float(fallback_groups),
                "invalid_boundary_groups": float(invalid_groups),
                "unreachable_boundary_groups": float(unreachable_groups),
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
                "candidate_stage": candidate_stage,
                "reference_final_latency_usec": str(reference.get("final_latency_usec", "none")),
                "objective_cost_usec": f"{objective['objective_cost_usec']:.3f}",
                "base_objective_cost_usec": f"{objective['base_objective_cost_usec']:.3f}",
                "reference_objective_cost_usec": (
                    f"{objective['reference_objective_cost_usec']:.3f}"
                ),
                "objective_cost_ratio_vs_seed": (
                    f"{objective['objective_cost_ratio_vs_seed']:.6f}"
                ),
                "objective_tier": str(int(objective["objective_tier"])),
                "correctness_gate": json.dumps(correctness_gate, sort_keys=True),
                "latency_only_objective": json.dumps(
                    {
                        "correct": bool(correctness_gate["correct"]),
                        "combined_score": float(combined_score),
                        "latency_only_score": float(latency_only_score),
                        "objective_cost_usec": float(objective["objective_cost_usec"]),
                        "reference_objective_cost_usec": float(
                            objective["reference_objective_cost_usec"]
                        ),
                        "selected_path_changed_vs_seed": bool(
                            effective_path.get("selected_path_changed_vs_seed", False)
                        ),
                        "effective_qbp_changed_vs_seed": bool(
                            effective_path.get("effective_qbp_changed_vs_seed", False)
                        ),
                        "objective_improved_vs_seed": bool(
                            correctness_gate.get("objective_improved_vs_seed", False)
                        ),
                        "failures": correctness_gate.get("reasons", []),
                    },
                    sort_keys=True,
                ),
                "total_frontier_cost_usec": f"{objective['total_frontier_cost_usec']:.3f}",
                "sampled_dp_latency_usec": str(result.get("sampled_dp_latency_usec", "none")),
                "scale_floor_summary": json.dumps(scale_floor_summary, sort_keys=True),
                "scale_floor_bits": str(scale_floor_bits),
                "scale_floor_delta_bits": str(scale_floor_delta_bits),
                "effective_qbp_digest": str(effective_path.get("effective_qbp_digest", "")),
                "selected_path_digest": str(effective_path.get("selected_path_digest", "")),
                "sampled_selected_path_bootstraps": str(
                    result.get("sampled_selected_path_bootstraps", "none")
                ),
                "sampled_selected_path_digest": str(
                    result.get("sampled_selected_path_digest", "")
                ),
                "reference_bootstrap_count": str(seed_bootstrap_count if seed_bootstrap_count is not None else "none"),
                "candidate_final_latency_usec": f"{result['final_latency_usec']:.3f}",
                "latency_delta_usec": (
                    f"{result['final_latency_usec'] - ref_latency:.3f}"
                    if math.isfinite(ref_latency) and result["valid"]
                    else "none"
                ),
                "bootstrap_count": str(result["bootstrap_count"]),
                "sampled_avg_bootstrap_count": str(
                    result.get("sampled_avg_bootstrap_count", result["bootstrap_count"])
                ),
                "sampled_frontier_total_bootstrap_count": str(
                    result.get(
                        "sampled_frontier_total_bootstrap_count",
                        result["bootstrap_count"],
                    )
                ),
                "bootstrap_delta_vs_seed": (
                    str(int(result["bootstrap_count"]) - seed_bootstrap_count)
                    if seed_bootstrap_count is not None
                    else "none"
                ),
                "rescale_count": str(result["rescale_count"]),
                "boundary_quality": f"{boundary_score:.6f}",
                "score_breakdown": json.dumps(
                    {
                        "objective": "latency_only_after_correctness_gate",
                        "correct": bool(correctness_gate["correct"]),
                        "seed_equivalent_path": bool(
                            effective_path.get("seed_equivalent_path", False)
                        ),
                        "objective_improved_vs_seed": bool(
                            correctness_gate.get("objective_improved_vs_seed", False)
                        ),
                        "latency_score": latency_score,
                        "latency_only_score": latency_only_score,
                        "correctness_failures": correctness_gate.get("reasons", []),
                        "bootstrap_score": bootstrap_score,
                        "component_bootstrap_score": component_bootstrap_score,
                        "rescale_score": rescale_score,
                        "boundary_score": boundary_score,
                        "risk_score": risk_score,
                        "noise_score": noise_score,
                        "reserve_score": reserve_score,
                        "fallback_score": fallback_score,
                        "runtime_score": runtime_score,
                        "placement_effect_score": placement_effect_score,
                        "path_effect_score": path_effect_score,
                        "policy_tiebreak_score": policy_tiebreak_score,
                        "objective_cost_score": objective["objective_cost_score"],
                        "cost_tiebreak_score": cost_tiebreak_score,
                    },
                    sort_keys=True,
                ),
                "trace_backed_candidate_examples": json.dumps(
                    context.get("harness", {}).get("candidate_examples", []),
                    sort_keys=True,
                    default=str,
                )[:TRACE_FEEDBACK_CHARS],
                "candidate_mlir_artifacts": json.dumps(
                    candidate_artifacts, sort_keys=True
                )[:4000],
                "candidate_mlir_path": str(candidate_artifacts.get("mlir_path", "")),
                "candidate_mlir_digest": str(candidate_artifacts.get("mlir_digest", "")),
                "candidate_mlir_preview": str(
                    candidate_artifacts.get("mlir_preview", "")
                )[:TRACE_PREVIEW_CHARS],
                "candidate_mlir_trace_features": json.dumps(
                    candidate_trace_features, sort_keys=True, default=str
                )[:8000],
                "trace_learning_feedback": json.dumps(
                    trace_learning_feedback, sort_keys=True, default=str
                )[:8000],
                "noise_estimator": json.dumps(noise_estimate, sort_keys=True)[:4000],
                "reserve_summary": json.dumps(
                    result.get("reserve_summary", {}), sort_keys=True
                )[:4000],
                "bootstrap_locations": json.dumps(result["bootstrap_locations"], sort_keys=True),
                "component_bootstrap_alignment": json.dumps(
                    component_bootstrap, sort_keys=True
                )[:4000],
                "component_bootstrap_budget": json.dumps(
                    context.get("unit_bootstrap_budget", {}), sort_keys=True
                )[:4000],
                "rescale_locations": json.dumps(result["rescale_locations"], sort_keys=True),
                "bottleneck_summary": json.dumps(result.get("bottleneck_summary", []), sort_keys=True),
                "selected_output_state": json.dumps(result["selected_output_state"], sort_keys=True),
                "boundary_group_validity": f"{solved_groups}/{scored_groups} scored ({requested_groups} raw)",
                "candidate_qbp_coverage": f"{candidate_groups}/{scored_groups} scored ({requested_groups} raw)",
                "reachable_boundary_groups": str(int(objective["reachable_boundary_groups"])),
                "unsolved_reachable_boundary_groups": str(
                    int(objective["unsolved_reachable_boundary_groups"])
                ),
                "candidate_direct_group_coverage": (
                    f"{objective['candidate_direct_group_coverage']:.6f}"
                ),
                "fallback_selected_groups": f"{fallback_groups}/{requested_groups}",
                "invalid_boundary_groups": f"{invalid_groups}/{requested_groups}",
                "unreachable_boundary_groups": f"{unreachable_groups}/{requested_groups}",
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
                "sampled_task_timing": json.dumps(
                    result["diagnostics"].get("sampled_task_timing_sec", {}),
                    sort_keys=True,
                )[:4000],
                "compile_replay_timing": json.dumps(
                    result["diagnostics"].get("compile_replay_timing_sec", {}),
                    sort_keys=True,
                )[:4000],
                "boundary_group_unsolved_summary": json.dumps(
                    _boundary_group_unsolved_summary(
                        result["diagnostics"].get("boundary_group_summaries", [])
                    ),
                    sort_keys=True,
                )[:4000],
                "policy_summary": _compact_policy_summary(eval_hints),
                "policy_effect_summary": json.dumps(policy_effect, sort_keys=True)[:4000],
                "placement_effect_summary": json.dumps(placement_effect, sort_keys=True)[:4000],
                "effective_qbp_digest": effective_path["effective_qbp_digest"],
                "selected_path_digest": effective_path["selected_path_digest"],
                "selected_path_changed_vs_seed": str(
                    bool(effective_path["selected_path_changed_vs_seed"])
                ),
                "effective_path_summary": json.dumps(effective_path, sort_keys=True)[:4000],
                "mcts_action_attempt_counts": json.dumps(
                    diagnostics.get("mcts_action_attempt_counts", {}), sort_keys=True
                ),
                "mcts_action_success_counts": json.dumps(
                    diagnostics.get("mcts_action_success_counts", {}), sort_keys=True
                ),
                "mcts_action_invalid_counts": json.dumps(
                    diagnostics.get("mcts_action_invalid_counts", {}), sort_keys=True
                ),
                "mcts_action_duplicate_skips": json.dumps(
                    diagnostics.get("mcts_action_duplicate_skips", {}), sort_keys=True
                ),
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
                "alphaevolve_feedback": json.dumps(
                    _alphaevolve_feedback(context, eval_hints, result),
                    sort_keys=True,
                )[:4000],
                "execution_trace": json.dumps(execution_trace, sort_keys=True)[:12000],
            },
        }
        attach_timings(evaluation)
        trace_start = time.perf_counter()
        _record_compile_trace(context, Path(program_path), eval_hints, evaluation, "CLEAR_ONLY")
        timings["record_trace"] = float(time.perf_counter() - trace_start)
        attach_timings(evaluation)
        return evaluation
    except Exception as exc:
        tb = traceback.format_exc()
        mark_timing("exception")
        return attach_timings({
            "metrics": {
                "combined_score": 0.0,
                "validity": 0.0,
                "latency_score": 0.0,
                "latency_only_correct": 0.0,
                "latency_only_score": 0.0,
                "objective_cost_usec": float("inf"),
                "base_objective_cost_usec": float("inf"),
                "reference_objective_cost_usec": 0.0,
                "objective_cost_ratio_vs_seed": 0.0,
                "objective_cost_score": 0.0,
                "objective_tier": 0.0,
                "total_frontier_cost_usec": float("inf"),
                "reachable_boundary_groups": 0.0,
                "unsolved_reachable_boundary_groups": 0.0,
                "candidate_direct_group_coverage": 0.0,
                "direct_unsolved_boundary_groups": 0.0,
                "objective_penalty_usec": float("inf"),
                "final_latency_usec": 0.0,
                "boundary_quality": 0.0,
                "bootstrap_count": 0.0,
                "component_bootstrap_score": 0.0,
                "candidate_qbp_coverage": 0.0,
                "boundary_group_validity": 0.0,
                "placement_effect_score": 0.0,
                "policy_effect_score": 0.0,
                "action_effect_score": 0.0,
                "seed_equivalent_policy": 0.0,
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
                "correctness_gate": json.dumps(
                    {"correct": False, "reasons": ["compile_harness"]},
                    sort_keys=True,
                ),
            },
        })


def _evaluate_compile_hints(
    context: dict[str, Any],
    hints: dict[str, Any],
    *,
    suppress_output: bool,
    evaluating_candidate: bool = True,
) -> dict[str, Any]:
    eval_suite = context.get("harness", {}).get("eval_suite", "polybert-sampled")
    if eval_suite != "polybert-full" and context.get("sampled_budget_tasks"):
        return _evaluate_sampled_budget_tasks(context, hints, suppress_output=suppress_output)

    from .iterative_partition import solve_partition
    from .qbp_manager import QBPManager

    timing_enabled = bool(context.get("harness", {}).get("timing_log_enabled", False))
    timing_start = time.perf_counter()
    timing_last = timing_start
    replay_timings: dict[str, float] = {}

    def mark_replay_timing(stage: str) -> None:
        nonlocal timing_last
        now = time.perf_counter()
        replay_timings[stage] = float(now - timing_last)
        timing_last = now
        if timing_enabled:
            print(
                "OpenEvolve timing: full_replay "
                f"{stage}={replay_timings[stage]:.3f}s "
                f"total={float(now - timing_start):.3f}s",
                flush=True,
            )

    tdag = tdag_from_context(context)
    params = tdag.params
    params.openevolve_iterations = 0
    params.openevolve_harness = "compile"
    params.openevolve_compile_hints = _compile_hints_for_eval_suite(hints, eval_suite)
    params.openevolve_evaluating_candidate = bool(evaluating_candidate)
    params.openevolve_collect_diagnostics = True
    params.openevolve_eval_suite = eval_suite
    scale_floor_bits = _apply_active_scale_floor_from_hints(
        params, params.openevolve_compile_hints
    )
    le = LatencyEstimator(params)
    qbp_manager = QBPManager(params, le)
    mark_replay_timing("setup")
    log_buffer = io.StringIO()
    start = time.time()
    try:
        stream = log_buffer if suppress_output else None
        try:
            if timing_enabled:
                print("OpenEvolve timing: full_replay solve_partition start", flush=True)
            solve_start = time.perf_counter()
            if stream is None:
                partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            else:
                with redirect_stdout(stream), redirect_stderr(stream):
                    partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            replay_timings["solve_partition"] = float(time.perf_counter() - solve_start)
            timing_last = time.perf_counter()
            if timing_enabled:
                print(
                    "OpenEvolve timing: full_replay "
                    f"solve_partition={replay_timings['solve_partition']:.3f}s "
                    f"total={float(timing_last - timing_start):.3f}s",
                    flush=True,
                )
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
            if timing_enabled:
                print(
                    "OpenEvolve timing: full_replay retry_solve_partition start",
                    flush=True,
                )
            solve_start = time.perf_counter()
            if stream is None:
                partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            else:
                with redirect_stdout(stream), redirect_stderr(stream):
                    partition_result = solve_partition(tdag, qbp_manager, {-1: {params.Sw: 0}}, le, params)
            replay_timings["retry_solve_partition"] = float(time.perf_counter() - solve_start)
            timing_last = time.perf_counter()
            if timing_enabled:
                print(
                    "OpenEvolve timing: full_replay "
                    f"retry_solve_partition={replay_timings['retry_solve_partition']:.3f}s "
                    f"total={float(timing_last - timing_start):.3f}s",
                    flush=True,
                )
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
        mark_replay_timing("collect_diagnostics")
        requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
        scored_groups = _scored_boundary_group_count(diagnostics)
        final_latency_usec = float(estimate_assign(assign, le))
        mark_replay_timing("estimate_final_assign")
        sample_tasks = _sampled_budget_tasks_from_qbp_manager(
            qbp_manager,
            params,
        )
        mark_replay_timing("sampled_budget_task_export")
        diagnostics["compile_replay_timing_sec"] = {
            **replay_timings,
            "total": float(time.perf_counter() - timing_start),
        }
        return {
            "valid": True,
            "validity": 1.0,
            "boundary_group_validity": float(
                min(1.0, int(diagnostics.get("solved_boundary_groups", 0) or 0) / scored_groups)
            ),
            "candidate_qbp_coverage": float(
                min(
                    1.0,
                    int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
                    / scored_groups,
                )
            ),
            "final_latency_usec": final_latency_usec,
            "aggregated_partition_cost_usec": float(final_cost),
            "objective_cost_usec": final_latency_usec,
            "total_frontier_cost_usec": float(final_cost),
            "scale_floor_bits": int(scale_floor_bits),
            "scale_floor_delta_bits": int(params.Sw) - int(scale_floor_bits),
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
            "unreachable_boundary_groups": int(
                diagnostics.get("unreachable_boundary_groups", 0)
            ),
            "selected_output_state": {
                "in_lvl": final_io_choice[0],
                "in_scl": final_io_choice[1],
                "out_lvl": final_io_choice[2],
                "out_scl": final_io_choice[3],
            },
            "reserve_summary": reserve_summary,
            "assignment": _serialize_assign(assign),
            "assignment_context": context,
            "bootstrap_locations": locations["bootstrap"],
            "rescale_locations": locations["rescale"],
            "bottleneck_summary": _bottleneck_summary(locations),
            "diagnostics": diagnostics,
            "sampled_budget_tasks": sample_tasks,
            "log_tail": log_buffer.getvalue()[-3000:],
        }
    except Exception as exc:
        tb = traceback.format_exc()
        diagnostics = _collect_qbp_diagnostics(qbp_manager)
        replay_timings["exception_total"] = float(time.perf_counter() - timing_start)
        diagnostics["compile_replay_timing_sec"] = {
            **replay_timings,
            "total": float(time.perf_counter() - timing_start),
        }
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
            "objective_cost_usec": float("inf"),
            "total_frontier_cost_usec": float("inf"),
            "scale_floor_bits": int(scale_floor_bits),
            "scale_floor_delta_bits": int(params.Sw) - int(scale_floor_bits),
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
            "unreachable_boundary_groups": int(
                diagnostics.get("unreachable_boundary_groups", 0)
            ),
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
    sampled_avg_bootstrap_count = (
        group_summary["frontier_bootstrap"]
        if group_summary["boundary_group_count"]
        else count_summary["avg_bootstrap"]
    )
    sampled_avg_rescale_count = (
        group_summary["frontier_rescale"]
        if group_summary["boundary_group_count"]
        else count_summary["avg_rescale"]
    )
    sampled_bootstrap_count = (
        group_summary["frontier_total_bootstrap"]
        if group_summary["boundary_group_count"]
        else count_summary["aggregate_bootstrap"]
    )
    sampled_rescale_count = (
        group_summary["frontier_total_rescale"]
        if group_summary["boundary_group_count"]
        else count_summary["aggregate_rescale"]
    )
    total_frontier_cost = (
        group_summary["frontier_total_cost_usec"]
        if group_summary["boundary_group_count"]
        else float(sum(costs))
    )
    sampled_path_proxy = _sampled_path_proxy_from_boundary_groups(
        list(diagnostics.get("boundary_group_summaries", [])),
        costs,
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
    scored_groups = _scored_boundary_group_count(diagnostics)
    scale_floor_bits = (
        params.active_scale_floor_bits()
        if hasattr(params, "active_scale_floor_bits")
        else params.Sw
    )
    return {
        "valid": bool(solved),
        "validity": float(solved / requested),
        "boundary_group_validity": float(min(1.0, solved_groups / scored_groups)),
        "candidate_qbp_coverage": float(min(1.0, candidate_groups / scored_groups)),
        "sampled_progress_only": True,
        "final_latency_usec": float(sum(costs) / len(costs)) if costs else 0.0,
        "aggregated_partition_cost_usec": float(sum(costs)) if costs else 0.0,
        "objective_cost_usec": float(sampled_path_proxy["sampled_dp_latency_usec"]),
        "total_frontier_cost_usec": float(total_frontier_cost),
        "scale_floor_bits": int(scale_floor_bits),
        "scale_floor_delta_bits": int(params.Sw) - int(scale_floor_bits),
        **sampled_path_proxy,
        "bootstrap_count": float(sampled_bootstrap_count),
        "rescale_count": float(sampled_rescale_count),
        "sampled_avg_bootstrap_count": float(sampled_avg_bootstrap_count),
        "sampled_avg_rescale_count": float(sampled_avg_rescale_count),
        "sampled_frontier_total_bootstrap_count": float(sampled_bootstrap_count),
        "sampled_frontier_total_rescale_count": float(sampled_rescale_count),
        "boundary_quality": 0.0,
        "profile_risk": float(_profile_risk(assignments, params)) if assignments else 1.0,
        "placement_runtime_sec": time.time() - start,
        "fallback_selected_budgets": int(diagnostics.get("fallback_selected_budgets", 0)),
        "fallback_selected_groups": int(
            diagnostics.get("fallback_selected_boundary_groups", 0)
        ),
        "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0)),
        "unreachable_boundary_groups": int(
            diagnostics.get("unreachable_boundary_groups", 0)
        ),
        "selected_output_state": {},
        "reserve_summary": reserve_summary,
        "assignment": _serialize_assign(best_assign) if best_assign is not None else {},
        "bootstrap_locations": locations["bootstrap"],
        "rescale_locations": locations["rescale"],
        "bottleneck_summary": _bottleneck_summary(locations),
        "diagnostics": diagnostics,
        "sampled_task_seed_metrics": list(diagnostics.get("sampled_task_metrics", [])),
        "sampled_budget_tasks": sampled_budget_tasks or [],
        "log_tail": log_buffer.getvalue()[-3000:],
    }


def _sampled_budget_tasks_from_qbp_manager(
    qbp_manager,
    params: Params,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, item in enumerate(getattr(qbp_manager, "openevolve_budget_tasks", [])):
        pdag = item.get("pdag") if isinstance(item, dict) else None
        budgets = item.get("io_budgets", []) if isinstance(item, dict) else []
        if pdag is None or not budgets:
            continue
        groups = _budget_boundary_groups([dict(budget) for budget in budgets])
        if not groups:
            continue
        records.append(
            {
                "index": index,
                "kind": str(item.get("kind", "normal")),
                "pdag": pdag,
                "groups": groups,
            }
        )
    if not records:
        return []

    max_groups = _sampled_compile_boundary_group_limit(params)
    selected_records = _evenly_spaced_items(records, min(len(records), max_groups))
    remaining = max_groups
    tasks: list[dict[str, Any]] = []
    for position, record in enumerate(selected_records):
        remaining_records = max(1, len(selected_records) - position)
        group_limit = max(1, remaining // remaining_records)
        if remaining % remaining_records:
            group_limit += 1
        group_keys = _evenly_spaced_items(
            _rank_sampled_boundary_group_keys(record["groups"], params),
            min(len(record["groups"]), group_limit),
        )
        sampled_budgets = [
            dict(budget)
            for key in group_keys
            for budget in record["groups"].get(key, [])
        ]
        if not sampled_budgets:
            continue
        task_context = build_context(record["pdag"], sampled_budgets, params)
        cache_dir = getattr(params, "openevolve_sampled_qbp_cache_dir", None)
        if cache_dir:
            task_context.setdefault("harness", {})["sampled_qbp_cache_dir"] = str(cache_dir)
        tasks.append(
            {
                "index": record["index"],
                "kind": record["kind"],
                "group_keys": [
                    {
                        "in_lvl": int(key[0]),
                        "in_scl": int(key[1]),
                        "maino_v": str(key[2]),
                        "main_dag_size": int(key[3]),
                    }
                    for key in group_keys
                ],
                "context": task_context,
            }
        )
        remaining -= len(group_keys)
        if remaining <= 0:
            break
    return tasks


def _sampled_compile_boundary_group_limit(params: Params) -> int:
    """Bound the OpenEvolve inner-loop QBP cache globally across partitions."""

    hint = int(getattr(params, "openevolve_max_unit_samples", 64) or 64)
    if getattr(params, "openevolve_eval_suite", "polybert-sampled") == "toy":
        return max(1, min(4, hint))
    parallel = max(1, int(getattr(params, "openevolve_parallel_evaluations", 1) or 1))
    if parallel >= 8:
        cap = 32
    elif parallel >= 4:
        cap = 24
    else:
        cap = 16
    return max(4, min(cap, hint))


def _rank_sampled_boundary_group_keys(
    groups: dict[tuple, list[dict]], params: Params
) -> list[tuple]:
    def group_cost(key: tuple) -> float:
        costs = []
        for budget in groups.get(key, []):
            raw = budget.get("main_qbp_cost")
            if isinstance(raw, dict) and raw:
                try:
                    costs.append(float(min(raw.values())))
                except (TypeError, ValueError):
                    pass
        return min(costs) if costs else 0.0

    def group_size(key: tuple) -> float:
        sizes = []
        for budget in groups.get(key, []):
            try:
                sizes.append(float(budget.get("main_dag_size", 0.0)))
            except (TypeError, ValueError):
                pass
        return max(sizes) if sizes else 0.0

    return sorted(
        groups,
        key=lambda key: (
            _boundary_group_input_cannot_refresh(params, key),
            bool(key[2]),
            group_cost(key),
            -group_size(key),
            int(key[0]),
            int(key[1]),
            str(key[2]),
            int(key[3]),
        ),
    )


def _evenly_spaced_items(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    if limit == 1:
        return [items[0]]
    indexes = {
        round(position * (len(items) - 1) / (limit - 1))
        for position in range(limit)
    }
    return [items[index] for index in sorted(indexes)]


def _run_sampled_task_payload(payload: tuple[dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    task_context, eval_hints, eval_suite = payload
    log_buffer = io.StringIO()
    total_start = time.perf_counter()
    timings: dict[str, float] = {}
    cache_path = _sampled_task_cache_path(task_context, eval_hints, eval_suite)
    cached = _read_sampled_task_cache(cache_path)
    if cached is not None:
        return cached

    def elapsed_since(start: float) -> float:
        return float(time.perf_counter() - start)

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            stage_start = time.perf_counter()
            task_tdag = tdag_from_context(task_context)
            task_params = task_tdag.params
            task_params.openevolve_iterations = 0
            task_params.openevolve_harness = "compile"
            task_params.openevolve_compile_hints = eval_hints
            task_params.openevolve_evaluating_candidate = True
            task_params.openevolve_eval_suite = eval_suite
            _apply_active_scale_floor_from_hints(task_params, eval_hints)
            timings["deserialize_context"] = elapsed_since(stage_start)
            stage_start = time.perf_counter()
            task_le = LatencyEstimator(task_params)
            timings["latency_estimator"] = elapsed_since(stage_start)
            stage_start = time.perf_counter()
            budgets = [
                _io_budget_from_json(item)
                for item in task_context.get("io_budgets", [])
                if isinstance(item, dict)
            ]
            timings["budget_parse"] = elapsed_since(stage_start)
            if not budgets:
                timings["total"] = elapsed_since(total_start)
                result = {
                    "task_context": task_context,
                    "diagnostics": {
                        "sampled_task_timing_sec": timings,
                        "sampled_task_cache_hit": False,
                    },
                    "assignments": [],
                    "costs": [],
                    "log_tail": log_buffer.getvalue()[-2000:],
                }
                _write_sampled_task_cache(
                    cache_path,
                    result,
                    validated_assignments=True,
                )
                return result
            task_diag: dict[str, Any] = {}
            stage_start = time.perf_counter()
            solve_budget_batch(
                task_tdag,
                budgets,
                task_le,
                task_params,
                eval_hints,
                task_diag,
            )
            timings["solve_budget_batch"] = elapsed_since(stage_start)
        stage_start = time.perf_counter()
        assignments = []
        validation_errors = []
        for assign in task_diag.get("assignments", []):
            if not isinstance(assign, Assign):
                continue
            try:
                assign.check_assign()
                assignments.append(_serialize_assign(assign))
            except Exception as exc:
                validation_errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
        timings["serialize_assignments"] = elapsed_since(stage_start)
        timings["total"] = elapsed_since(total_start)
        diagnostics = dict(task_diag)
        diagnostics.pop("assignments", None)
        diagnostics["sampled_task_timing_sec"] = timings
        diagnostics["sampled_task_cache_hit"] = False
        if validation_errors:
            diagnostics["sampled_task_assignment_validation_errors"] = validation_errors[:8]
        result = {
            "task_context": task_context,
            "diagnostics": diagnostics,
            "assignments": assignments,
            "costs": [float(item) for item in task_diag.get("costs", [])],
            "log_tail": log_buffer.getvalue()[-2000:],
        }
        _write_sampled_task_cache(
            cache_path,
            result,
            validated_assignments=not validation_errors,
        )
        return result
    except Exception as exc:
        timings["total"] = elapsed_since(total_start)
        return {
            "task_context": task_context,
            "diagnostics": {
                "invalid_reasons": {f"{type(exc).__name__}: {str(exc)[:240]}": 1},
                "traceback": traceback.format_exc()[-3000:],
                "sampled_task_timing_sec": timings,
                "sampled_task_cache_hit": False,
            },
            "assignments": [],
            "costs": [],
            "log_tail": (log_buffer.getvalue() + "\n" + traceback.format_exc())[-3000:],
        }


def _sampled_task_parallelism(context: dict[str, Any], task_count: int) -> int:
    if task_count <= 1:
        return 1
    env_requested = os.environ.get("ORBIT_OPENEVOLVE_EVALUATOR_QBP_WORKERS", "").strip()
    if env_requested:
        try:
            return max(1, min(task_count, int(env_requested)))
        except ValueError:
            pass
    params = context.get("params", {}) if isinstance(context.get("params"), dict) else {}
    threads = max(1, _safe_int(params.get("threads"), 1))
    outer_parallel = max(1, _safe_int(params.get("openevolve_parallel_evaluations"), 1))
    # Leave room for OpenEvolve's outer evaluator pool so OE100 does not create
    # hundreds of active workers per generation on large graphs.
    budget = max(1, threads // outer_parallel)
    return max(1, min(task_count, budget))


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
        "unreachable_boundary_groups": 0,
        "candidate_improved_budgets": 0,
        "candidate_invalid_reasons": {},
        "invalid_reasons": {},
        "candidate_costs": [],
        "costs": [],
        "assignments": [],
        "selected_source_counts": {},
        "mcts_action_attempt_counts": {},
        "mcts_action_success_counts": {},
        "mcts_action_invalid_counts": {},
        "mcts_action_duplicate_skips": {},
        "boundary_group_summaries": [],
        "sampled_task_metrics": [],
        "sampled_task_count": 0,
        "sampled_task_cache_hits": 0,
        "sampled_task_cache_misses": 0,
        "sampled_task_cache_writes": 0,
        "sampled_direct_budget_eval": False,
        "sampled_qbp_group_eval": True,
    }
    reserve_summaries: list[dict[str, Any]] = []
    best_assign: Assign | None = None
    best_cost = float("inf")
    best_task_context: dict[str, Any] | None = None
    task_timing_totals: dict[str, float] = {}
    task_timing_max: dict[str, float] = {}
    task_timing_rows: list[dict[str, Any]] = []
    log_buffer = io.StringIO()
    stdout_context = redirect_stdout(log_buffer) if suppress_output else nullcontext()
    stderr_context = redirect_stderr(log_buffer) if suppress_output else nullcontext()

    def merge_counts(key: str, item: dict[str, Any]) -> None:
        total[key] += int(item.get(key, 0))

    def decode_task_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], Tdag, Params, dict[str, Any], list[Assign], list[float]]:
        if payload.get("log_tail"):
            log_buffer.write(str(payload.get("log_tail"))[-2000:])
        task_context = payload["task_context"]
        task_tdag = tdag_from_context(task_context)
        task_params = task_tdag.params
        task_params.openevolve_eval_suite = eval_suite
        _apply_active_scale_floor_from_hints(task_params, eval_hints)
        task_assignments = [
            _assign_from_serialized(task_tdag, item)
            for item in payload.get("assignments", [])
            if isinstance(item, dict)
        ]
        return (
            task_context,
            task_tdag,
            task_params,
            dict(payload.get("diagnostics", {})),
            task_assignments,
            [float(item) for item in payload.get("costs", [])],
        )

    try:
        with stdout_context, stderr_context:
            tasks = [
                task
                for task in context.get("sampled_budget_tasks", [])
                if isinstance(task, dict) and isinstance(task.get("context"), dict)
            ]
            task_cache_dir = context.get("harness", {}).get("sampled_qbp_cache_dir")
            if task_cache_dir:
                for task in tasks:
                    task["context"].setdefault("harness", {})[
                        "sampled_qbp_cache_dir"
                    ] = str(task_cache_dir)
            workers = _sampled_task_parallelism(context, len(tasks))
            payloads = [(task["context"], eval_hints, eval_suite) for task in tasks]
            if workers > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                    raw_results = list(executor.map(_run_sampled_task_payload, payloads))
            else:
                raw_results = [_run_sampled_task_payload(payload) for payload in payloads]
            task_results = [decode_task_payload(payload) for payload in raw_results]
            total["sampled_task_parallelism"] = workers
            for task_position, (task, (
                task_context,
                task_tdag,
                task_params,
                task_diag,
                task_assignments,
                task_costs,
            )) in enumerate(zip(tasks, task_results)):
                if not task_diag:
                    continue
                total["sampled_task_count"] += 1
                total["sampled_task_metrics"].append(
                    _sampled_task_metric_summary(
                        task_position=task_position,
                        task=task,
                        diagnostics=task_diag,
                        costs=task_costs,
                    )
                )
                task_timing = task_diag.get("sampled_task_timing_sec", {})
                if isinstance(task_timing, dict):
                    normalized_timing = {
                        str(name): float(value)
                        for name, value in task_timing.items()
                        if isinstance(value, (int, float))
                    }
                    for name, value in normalized_timing.items():
                        task_timing_totals[name] = task_timing_totals.get(name, 0.0) + value
                        task_timing_max[name] = max(task_timing_max.get(name, 0.0), value)
                    task_timing_rows.append(
                        {
                            "task_index": len(task_timing_rows),
                            "total_sec": float(normalized_timing.get("total", 0.0)),
                            "solve_budget_batch_sec": float(
                                normalized_timing.get("solve_budget_batch", 0.0)
                            ),
                            "requested_budgets": int(task_diag.get("requested_budgets", 0)),
                            "requested_boundary_groups": int(
                                task_diag.get("requested_boundary_groups", 0)
                            ),
                            "cache_hit": bool(task_diag.get("sampled_task_cache_hit", False)),
                        }
                    )
                if bool(task_diag.get("sampled_task_cache_hit", False)):
                    total["sampled_task_cache_hits"] = (
                        int(total.get("sampled_task_cache_hits", 0) or 0) + 1
                    )
                else:
                    total["sampled_task_cache_misses"] = (
                        int(total.get("sampled_task_cache_misses", 0) or 0) + 1
                    )
                if bool(task_diag.get("sampled_task_cache_write", False)):
                    total["sampled_task_cache_writes"] = (
                        int(total.get("sampled_task_cache_writes", 0) or 0) + 1
                    )
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
                    "unreachable_boundary_groups",
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
                for dict_key in (
                    "mcts_action_attempt_counts",
                    "mcts_action_success_counts",
                    "mcts_action_invalid_counts",
                    "mcts_action_duplicate_skips",
                ):
                    for name, count in task_diag.get(dict_key, {}).items():
                        total[dict_key][name] = total[dict_key].get(name, 0) + int(count)
                for reason_key in ("invalid_reasons", "candidate_invalid_reasons"):
                    for reason, count in task_diag.get(reason_key, {}).items():
                        total[reason_key][reason] = total[reason_key].get(reason, 0) + int(count)
                if task_assignments:
                    reserve_summaries.append(
                        _aggregate_reserve_summary(task_assignments, task_tdag, task_params)
                    )
                if task_costs and len(task_costs) == len(task_assignments):
                    idx = min(range(len(task_costs)), key=lambda pos: task_costs[pos])
                    if task_costs[idx] < best_cost:
                        best_cost = task_costs[idx]
                        best_assign = task_assignments[idx]
                        best_task_context = task_context

        assignments = list(total["assignments"])
        costs = [float(item) for item in total["costs"]]
        count_summary = _assignment_count_summary(assignments)
        group_summary = _boundary_group_count_summary(
            list(total.get("boundary_group_summaries", []))
        )
        total["assignment_count_summary"] = count_summary
        total["boundary_group_count_summary"] = group_summary
        task_timing_top = sorted(
            task_timing_rows,
            key=lambda item: float(item.get("total_sec", 0.0)),
            reverse=True,
        )[:8]
        task_count = max(1, len(task_timing_rows))
        total["sampled_task_timing_sec"] = {
            "totals": task_timing_totals,
            "max": task_timing_max,
            "avg": {
                name: float(value / task_count)
                for name, value in task_timing_totals.items()
            },
            "top_slow_tasks": task_timing_top,
        }
        sampled_avg_bootstrap_count = (
            group_summary["frontier_bootstrap"]
            if group_summary["boundary_group_count"]
            else count_summary["avg_bootstrap"]
        )
        sampled_avg_rescale_count = (
            group_summary["frontier_rescale"]
            if group_summary["boundary_group_count"]
            else count_summary["avg_rescale"]
        )
        sampled_bootstrap_count = (
            group_summary["frontier_total_bootstrap"]
            if group_summary["boundary_group_count"]
            else count_summary["aggregate_bootstrap"]
        )
        sampled_rescale_count = (
            group_summary["frontier_total_rescale"]
            if group_summary["boundary_group_count"]
            else count_summary["aggregate_rescale"]
        )
        total_frontier_cost = (
            group_summary["frontier_total_cost_usec"]
            if group_summary["boundary_group_count"]
            else float(sum(costs))
        )
        sampled_path_proxy = _sampled_path_proxy_from_boundary_groups(
            list(total.get("boundary_group_summaries", [])),
            costs,
        )
        locations = (
            _maintenance_locations(best_assign)
            if best_assign is not None
            else {"bootstrap": {}, "rescale": {}}
        )
        params = tdag_from_context(context).params
        scale_floor_bits = _apply_active_scale_floor_from_hints(params, eval_hints)
        reserve_summary = _merge_reserve_summaries(reserve_summaries)
        requested = max(1, int(total.get("requested_budgets", 0) or 1))
        solved = int(total.get("solved_budgets", 0) or 0)
        requested_groups = max(1, int(total.get("requested_boundary_groups", 0) or 1))
        solved_groups = int(total.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(total.get("candidate_solved_boundary_groups", 0) or 0)
        scored_groups = _scored_boundary_group_count(total)
        return {
            "valid": bool(solved),
            "validity": float(solved / requested),
            "boundary_group_validity": float(min(1.0, solved_groups / scored_groups)),
            "candidate_qbp_coverage": float(min(1.0, candidate_groups / scored_groups)),
            "sampled_progress_only": True,
            "final_latency_usec": float(sum(costs) / len(costs)) if costs else 0.0,
            "aggregated_partition_cost_usec": float(sum(costs)) if costs else 0.0,
            "objective_cost_usec": float(sampled_path_proxy["sampled_dp_latency_usec"]),
            "total_frontier_cost_usec": float(total_frontier_cost),
            "scale_floor_bits": int(scale_floor_bits),
            "scale_floor_delta_bits": int(params.Sw) - int(scale_floor_bits),
            **sampled_path_proxy,
            "bootstrap_count": float(sampled_bootstrap_count),
            "rescale_count": float(sampled_rescale_count),
            "sampled_avg_bootstrap_count": float(sampled_avg_bootstrap_count),
            "sampled_avg_rescale_count": float(sampled_avg_rescale_count),
            "sampled_frontier_total_bootstrap_count": float(sampled_bootstrap_count),
            "sampled_frontier_total_rescale_count": float(sampled_rescale_count),
            "boundary_quality": 0.0,
            "profile_risk": float(_profile_risk(assignments, params)) if assignments else 1.0,
            "placement_runtime_sec": time.time() - start,
            "sampled_task_total_runtime_sec": float(task_timing_totals.get("total", 0.0)),
            "sampled_task_max_runtime_sec": float(task_timing_max.get("total", 0.0)),
            "sampled_task_solve_total_sec": float(
                task_timing_totals.get("solve_budget_batch", 0.0)
            ),
            "sampled_task_solve_max_sec": float(
                task_timing_max.get("solve_budget_batch", 0.0)
            ),
            "sampled_task_cache_hits": int(total.get("sampled_task_cache_hits", 0) or 0),
            "sampled_task_cache_misses": int(
                total.get("sampled_task_cache_misses", 0) or 0
            ),
            "sampled_task_cache_writes": int(
                total.get("sampled_task_cache_writes", 0) or 0
            ),
            "fallback_selected_budgets": int(total.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(total.get("fallback_selected_boundary_groups", 0)),
            "invalid_boundary_groups": int(total.get("invalid_boundary_groups", 0)),
            "unreachable_boundary_groups": int(total.get("unreachable_boundary_groups", 0)),
            "selected_output_state": {},
            "reserve_summary": reserve_summary,
            "assignment": _serialize_assign(best_assign) if best_assign is not None else {},
            "assignment_context": best_task_context or {},
            "bootstrap_locations": locations["bootstrap"],
            "rescale_locations": locations["rescale"],
            "bottleneck_summary": _bottleneck_summary(locations),
            "diagnostics": total,
            "sampled_task_seed_metrics": list(total.get("sampled_task_metrics", [])),
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
            "objective_cost_usec": float("inf"),
            "total_frontier_cost_usec": float("inf"),
            "scale_floor_bits": int(_scale_floor_bits_from_hints(tdag_from_context(context).params, eval_hints)),
            "scale_floor_delta_bits": int(_ckks_dict(context)["Sw"])
            - int(_scale_floor_bits_from_hints(tdag_from_context(context).params, eval_hints)),
            "bootstrap_count": 0.0,
            "rescale_count": 0.0,
            "boundary_quality": 0.0,
            "profile_risk": 1.0,
            "placement_runtime_sec": time.time() - start,
            "sampled_task_total_runtime_sec": float(task_timing_totals.get("total", 0.0)),
            "sampled_task_max_runtime_sec": float(task_timing_max.get("total", 0.0)),
            "sampled_task_solve_total_sec": float(
                task_timing_totals.get("solve_budget_batch", 0.0)
            ),
            "sampled_task_solve_max_sec": float(
                task_timing_max.get("solve_budget_batch", 0.0)
            ),
            "sampled_task_cache_hits": int(total.get("sampled_task_cache_hits", 0) or 0),
            "sampled_task_cache_misses": int(
                total.get("sampled_task_cache_misses", 0) or 0
            ),
            "sampled_task_cache_writes": int(
                total.get("sampled_task_cache_writes", 0) or 0
            ),
            "fallback_selected_budgets": int(total.get("fallback_selected_budgets", 0)),
            "fallback_selected_groups": int(total.get("fallback_selected_boundary_groups", 0)),
            "invalid_boundary_groups": int(total.get("invalid_boundary_groups", 0)),
            "unreachable_boundary_groups": int(total.get("unreachable_boundary_groups", 0)),
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


def _scored_boundary_group_count(diagnostics: dict[str, Any]) -> int:
    requested = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
    unreachable = max(0, int(diagnostics.get("unreachable_boundary_groups", 0) or 0))
    return max(1, requested - unreachable)


def _placement_baseline_from_result(result: dict[str, Any]) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    scored_groups = _scored_boundary_group_count(diagnostics)
    boundary_group_summaries = diagnostics.get("boundary_group_summaries", [])
    sampled_task_metrics = _sampled_task_metrics_from_result(result)
    return {
        "valid": bool(result.get("valid", False)),
        "final_latency_usec": _finite_float(result.get("final_latency_usec"), float("inf")),
        "objective_cost_usec": _finite_float(
            result.get("objective_cost_usec", result.get("final_latency_usec")),
            float("inf"),
        ),
        "total_frontier_cost_usec": _finite_float(
            result.get("total_frontier_cost_usec", result.get("aggregated_partition_cost_usec")),
            float("inf"),
        ),
        "sampled_dp_latency_usec": _finite_float(
            result.get("sampled_dp_latency_usec"),
            float("inf"),
        ),
        "effective_qbp_digest": _effective_qbp_digest(diagnostics),
        "selected_path_digest": _selected_path_digest(result, diagnostics),
        "bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0),
        "rescale_count": _finite_float(result.get("rescale_count"), 0.0),
        "fallback_selected_budgets": int(result.get("fallback_selected_budgets", 0) or 0),
        "fallback_selected_groups": int(
            result.get(
                "fallback_selected_groups",
                diagnostics.get("fallback_selected_boundary_groups", 0),
            )
            or 0
        ),
        "requested_boundary_groups": int(
            diagnostics.get("requested_boundary_groups", scored_groups) or scored_groups
        ),
        "reachable_boundary_groups": int(scored_groups),
        "scored_boundary_groups": int(scored_groups),
        "unsolved_reachable_boundary_groups": max(
            0,
            int(scored_groups) - int(diagnostics.get("solved_boundary_groups", 0) or 0),
        ),
        "solved_boundary_groups": int(
            diagnostics.get("solved_boundary_groups", 0) or 0
        ),
        "candidate_solved_boundary_groups": int(
            diagnostics.get("candidate_solved_boundary_groups", 0) or 0
        ),
        "unsolved_boundary_groups": _boundary_group_unsolved_summary(
            boundary_group_summaries
        ),
        "top_costly_boundary_groups": _boundary_group_top_cost_summary(
            boundary_group_summaries
        )
        or _top_costly_boundary_groups_from_task_metrics(sampled_task_metrics),
        "sampled_task_seed_metrics": sampled_task_metrics,
    }


def _baseline_objective_cost_usec(context: dict[str, Any], fallback: float) -> float:
    harness = context.get("harness", {})
    eval_suite = str(harness.get("eval_suite", "polybert-sampled"))
    candidates: list[Any] = []
    candidates.append(harness.get("active_seed_baseline"))
    if eval_suite != "polybert-full":
        candidates.append(harness.get("sampled_seed_baseline"))
    candidates.extend([harness.get("seed_baseline"), context.get("reference")])
    for item in candidates:
        if not isinstance(item, dict):
            continue
        for key in (
            "sampled_dp_latency_usec",
            "objective_cost_usec",
            "final_latency_usec",
            "total_frontier_cost_usec",
        ):
            value = _finite_float(item.get(key), float("inf"))
            if math.isfinite(value) and value > 0:
                return float(value)
    return fallback if math.isfinite(fallback) and fallback > 0 else 1_000_000.0


def _result_base_objective_cost_usec(result: dict[str, Any]) -> float:
    if result.get("sampled_progress_only"):
        for key in (
            "sampled_dp_latency_usec",
            "total_frontier_cost_usec",
            "aggregated_partition_cost_usec",
            "final_latency_usec",
        ):
            value = _finite_float(result.get(key), float("inf"))
            if math.isfinite(value) and value >= 0:
                return float(value)
    value = _finite_float(result.get("final_latency_usec"), float("inf"))
    if math.isfinite(value) and value >= 0:
        return float(value)
    value = _finite_float(result.get("aggregated_partition_cost_usec"), float("inf"))
    return float(value) if math.isfinite(value) and value >= 0 else float("inf")


def _cost_minimization_objective(
    context: dict[str, Any],
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    effective_validity: float,
    repair_count: int | float,
) -> dict[str, Any]:
    requested_groups = max(1, int(diagnostics.get("requested_boundary_groups", 0) or 1))
    reachable_groups = _scored_boundary_group_count(diagnostics)
    solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
    candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
    fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
    fallback_budgets = int(result.get("fallback_selected_budgets", 0) or 0)
    invalid_groups = int(diagnostics.get("invalid_boundary_groups", 0) or 0)
    unreachable_groups = int(diagnostics.get("unreachable_boundary_groups", 0) or 0)
    unsolved_reachable_groups = max(0, int(reachable_groups) - solved_groups)
    direct_unsolved_groups = max(0, int(reachable_groups) - candidate_groups)
    direct_coverage = min(1.0, max(0.0, candidate_groups / max(1, reachable_groups)))
    boundary_coverage = min(1.0, max(0.0, solved_groups / max(1, reachable_groups)))
    base_cost = _result_base_objective_cost_usec(result)
    if not math.isfinite(base_cost):
        base_cost = 0.0
    total_frontier_cost = _finite_float(
        result.get("total_frontier_cost_usec", result.get("aggregated_partition_cost_usec")),
        base_cost,
    )
    seed_cost = _baseline_objective_cost_usec(context, base_cost)
    penalty_unit = max(seed_cost, base_cost, 1_000_000.0)
    penalty = penalty_unit * (
        25.0 * unsolved_reachable_groups
        + 10.0 * direct_unsolved_groups
        + 8.0 * fallback_groups
        + 4.0 * fallback_budgets
        + 10.0 * invalid_groups
        + 2.0 * max(0.0, float(repair_count))
    )
    objective_cost = float(base_cost + penalty)
    ratio_vs_seed = (
        seed_cost / objective_cost
        if math.isfinite(seed_cost) and seed_cost > 0 and objective_cost > 0
        else 0.0
    )
    cost_score = 1.0 / (1.0 + max(0.0, objective_cost) / max(seed_cost, 1.0))
    complete_reachable = bool(result.get("valid")) and unsolved_reachable_groups == 0
    direct_complete = (
        complete_reachable
        and direct_unsolved_groups == 0
        and fallback_groups == 0
        and fallback_budgets == 0
        and int(repair_count) == 0
    )
    if direct_complete:
        tier = 2
    elif complete_reachable:
        tier = 1
    else:
        tier = 0
    incomplete_score = min(
        0.999,
        0.05
        + 0.32 * boundary_coverage
        + 0.36 * direct_coverage
        + 0.12 * max(0.0, min(1.0, effective_validity))
        + 0.08 * cost_score,
    )
    return {
        "objective_cost_usec": objective_cost,
        "base_objective_cost_usec": float(base_cost),
        "reference_objective_cost_usec": float(seed_cost),
        "objective_cost_ratio_vs_seed": float(ratio_vs_seed),
        "objective_cost_score": float(cost_score),
        "total_frontier_cost_usec": float(total_frontier_cost),
        "reachable_boundary_groups": float(reachable_groups),
        "unsolved_reachable_boundary_groups": float(unsolved_reachable_groups),
        "candidate_direct_group_coverage": float(direct_coverage),
        "boundary_group_validity": float(boundary_coverage),
        "complete_reachable": bool(complete_reachable),
        "direct_complete": bool(direct_complete),
        "objective_tier": float(tier),
        "incomplete_combined_score": float(incomplete_score),
        "requested_boundary_groups": float(requested_groups),
        "unreachable_boundary_groups": float(unreachable_groups),
        "direct_unsolved_boundary_groups": float(direct_unsolved_groups),
        "objective_penalty_usec": float(penalty),
    }


def _latency_only_correctness_gate(
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    objective: dict[str, Any],
    *,
    static: dict[str, Any],
    boundary_group_validity: float,
    candidate_qbp_coverage: float,
    effective_path: dict[str, Any] | None = None,
    policy_effect: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    effective_path = effective_path if isinstance(effective_path, dict) else {}
    policy_effect = policy_effect if isinstance(policy_effect, dict) else {}
    objective_improved = _objective_improved_vs_seed(objective)
    selected_path_changed = bool(
        effective_path.get("selected_path_changed_vs_seed", False)
    )
    seed_equivalent_path = bool(
        effective_path.get("reference_selected_path_digest")
        and not selected_path_changed
    )
    if not bool(static.get("valid", False)):
        reasons.append("static_policy_invalid")
    if not bool(result.get("valid", False)):
        reasons.append("orbit_assignment_invalid")
    if not bool(objective.get("direct_complete", False)):
        reasons.append("candidate_not_direct_complete")
    if float(boundary_group_validity) < 1.0:
        reasons.append("not_all_reachable_boundary_groups_solved")
    if float(candidate_qbp_coverage) < 1.0:
        reasons.append("not_all_reachable_boundary_groups_solved_by_candidate")
    if int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0) > 0:
        reasons.append("seed_fallback_selected_boundary_groups")
    if int(result.get("fallback_selected_budgets", 0) or 0) > 0:
        reasons.append("seed_fallback_selected_budgets")
    if int(diagnostics.get("invalid_boundary_groups", 0) or 0) > 0:
        reasons.append("invalid_boundary_groups")
    if _finite_float(objective.get("objective_cost_usec"), float("inf")) <= 0.0:
        reasons.append("missing_positive_latency_objective")
    elif not math.isfinite(_finite_float(objective.get("objective_cost_usec"), float("inf"))):
        reasons.append("nonfinite_latency_objective")
    return {
        "correct": not reasons,
        "reasons": reasons,
        "direct_complete": bool(objective.get("direct_complete", False)),
        "boundary_group_validity": float(boundary_group_validity),
        "candidate_qbp_coverage": float(candidate_qbp_coverage),
        "selected_path_changed_vs_seed": bool(
            effective_path.get("selected_path_changed_vs_seed", False)
        ),
        "effective_qbp_changed_vs_seed": bool(
            effective_path.get("effective_qbp_changed_vs_seed", False)
        ),
        "seed_equivalent_path": bool(seed_equivalent_path),
        "seed_equivalent_policy": bool(policy_effect.get("seed_equivalent", False)),
        "objective_improved_vs_seed": bool(objective_improved),
        "fallback_selected_boundary_groups": int(
            diagnostics.get("fallback_selected_boundary_groups", 0) or 0
        ),
        "fallback_selected_budgets": int(result.get("fallback_selected_budgets", 0) or 0),
        "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0) or 0),
    }


def _objective_improved_vs_seed(
    objective: dict[str, Any],
    *,
    min_relative_improvement: float = 0.0,
) -> bool:
    cost = _finite_float(objective.get("objective_cost_usec"), float("inf"))
    reference = _finite_float(objective.get("reference_objective_cost_usec"), float("inf"))
    if not math.isfinite(reference) or reference <= 0.0:
        reference = _finite_float(objective.get("base_objective_cost_usec"), float("inf"))
    if not math.isfinite(cost) or not math.isfinite(reference):
        return False
    if cost <= 0.0 or reference <= 0.0:
        return False
    return cost < reference * (1.0 - max(0.0, min_relative_improvement))


def _latency_only_combined_score(
    objective: dict[str, Any],
    *,
    correct: bool,
    correctness_gate: dict[str, Any] | None = None,
) -> float:
    if not correct:
        return 0.0
    correctness_gate = correctness_gate if isinstance(correctness_gate, dict) else {}
    cost = _finite_float(objective.get("objective_cost_usec"), float("inf"))
    reference = _finite_float(objective.get("reference_objective_cost_usec"), float("inf"))
    if not math.isfinite(reference) or reference <= 0:
        reference = _finite_float(objective.get("base_objective_cost_usec"), 1_000_000.0)
    if not math.isfinite(cost) or cost <= 0:
        return 0.0
    ratio = reference / cost
    improved = bool(correctness_gate.get("objective_improved_vs_seed", False))
    seed_equivalent_path = bool(correctness_gate.get("seed_equivalent_path", False))
    if seed_equivalent_path and not improved:
        # A syntactically changed policy that replays the seed path gives
        # OpenEvolve no useful placement signal. The seed itself is still
        # available through fail-open/recovery, but these clones should not win
        # sampled search or consume finalist slots.
        return 0.0
    if ratio < 1.0:
        # Slower path-changing candidates are useful trace examples. Give them
        # enough positive search credit to become mutation parents ahead of
        # seed-equivalent clones, but keep every latency-improving candidate
        # strictly above them. Final MLIR selection still rejects non-improving
        # sampled best programs through _sampled_best_invalid_reason().
        return max(1e-6, min(0.25, float(ratio) * 0.25))
    if math.isclose(ratio, 1.0, rel_tol=1e-12, abs_tol=1e-12):
        return 1.0
    # OpenEvolve maximizes a single score, and raw latency ratios for sampled
    # QBP improvements are often 1.00000x. Keep the ordering purely latency
    # based, but magnify the positive delta so small real improvements are not
    # numerically indistinguishable from the seed in MAP-Elites/selection logs.
    return 1.0 + max(0.0, ratio - 1.0) * 100_000.0


def _execution_trace_artifact(
    context: dict[str, Any],
    eval_hints: dict[str, Any],
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    objective: dict[str, Any],
    policy_effect: dict[str, Any],
    placement_effect: dict[str, Any],
    effective_path: dict[str, Any],
    scale_floor_summary: dict[str, Any],
    correctness_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "objective": "reject_invalid_then_minimize_latency",
        "correctness_gate": correctness_gate,
        "policy_summary": json.loads(_compact_policy_summary(eval_hints)),
        "policy_effect": {
            "seed_equivalent": bool(policy_effect.get("seed_equivalent", False)),
            "effect_score": _finite_float(policy_effect.get("effect_score"), 0.0),
            "changed_actions": list(policy_effect.get("changed_actions", []) or [])[:16],
            "changed_policy_keys": list(policy_effect.get("changed_policy_keys", []) or [])[:16],
        },
        "placement_effect": {
            "effect_score": _finite_float(placement_effect.get("effect_score"), 0.0),
            "selected_path_changed_vs_seed": bool(
                effective_path.get("selected_path_changed_vs_seed", False)
            ),
            "seed_equivalent_path": bool(
                effective_path.get("seed_equivalent_path", False)
            ),
            "objective_improved_vs_seed": bool(
                correctness_gate.get("objective_improved_vs_seed", False)
            ),
            "changed_boundary_groups": int(
                effective_path.get("changed_boundary_groups_vs_seed", 0) or 0
            ),
        },
        "latency": {
            "objective_cost_usec": _finite_float(objective.get("objective_cost_usec"), 0.0),
            "base_objective_cost_usec": _finite_float(
                objective.get("base_objective_cost_usec"), 0.0
            ),
            "reference_objective_cost_usec": _finite_float(
                objective.get("reference_objective_cost_usec"), 0.0
            ),
            "final_latency_usec": _finite_float(result.get("final_latency_usec"), 0.0),
            "sampled_dp_latency_usec": _finite_float(
                result.get("sampled_dp_latency_usec"), 0.0
            ),
            "total_frontier_cost_usec": _finite_float(
                objective.get("total_frontier_cost_usec"), 0.0
            ),
        },
        "coverage": {
            "reachable_boundary_groups": int(
                objective.get("reachable_boundary_groups", 0) or 0
            ),
            "unsolved_reachable_boundary_groups": int(
                objective.get("unsolved_reachable_boundary_groups", 0) or 0
            ),
            "candidate_direct_group_coverage": _finite_float(
                objective.get("candidate_direct_group_coverage"), 0.0
            ),
            "boundary_group_validity": _finite_float(
                objective.get("boundary_group_validity"), 0.0
            ),
            "fallback_selected_groups": int(
                diagnostics.get("fallback_selected_boundary_groups", 0) or 0
            ),
            "fallback_selected_budgets": int(result.get("fallback_selected_budgets", 0) or 0),
            "invalid_boundary_groups": int(diagnostics.get("invalid_boundary_groups", 0) or 0),
        },
        "path": {
            "effective_qbp_digest": effective_path.get("effective_qbp_digest", ""),
            "selected_path_digest": effective_path.get("selected_path_digest", ""),
            "sampled_selected_path_digest": result.get("sampled_selected_path_digest", ""),
            "selected_source_counts": diagnostics.get("selected_source_counts", {}),
        },
        "maintenance": {
            "bootstrap_count": int(result.get("bootstrap_count", 0) or 0),
            "rescale_count": int(result.get("rescale_count", 0) or 0),
            "bootstrap_locations": result.get("bootstrap_locations", {}),
            "rescale_locations": result.get("rescale_locations", {}),
            "bottleneck_summary": list(result.get("bottleneck_summary", []) or [])[:12],
        },
        "mcts_actions": {
            "attempt_counts": diagnostics.get("mcts_action_attempt_counts", {}),
            "success_counts": diagnostics.get("mcts_action_success_counts", {}),
            "invalid_counts": diagnostics.get("mcts_action_invalid_counts", {}),
            "duplicate_skips": diagnostics.get("mcts_action_duplicate_skips", {}),
        },
        "invalid_reasons": _compact_invalid_reasons(diagnostics),
        "candidate_invalid_reasons": _compact_invalid_reasons(
            {"invalid_reasons": diagnostics.get("candidate_invalid_reasons", {})}
        ),
        "unsolved_boundary_groups": _boundary_group_unsolved_summary(
            diagnostics.get("boundary_group_summaries", [])
        ),
        "top_costly_boundary_groups": _boundary_group_top_cost_summary(
            diagnostics.get("boundary_group_summaries", [])
        ),
        "scale_floor": scale_floor_summary,
        "model": context.get("model", {}),
    }


def _maybe_write_candidate_mlir_artifacts(
    context: dict[str, Any],
    program_path: Path,
    eval_hints: dict[str, Any],
    result: dict[str, Any],
    objective: dict[str, Any],
    correctness_gate: dict[str, Any],
    execution_trace: dict[str, Any],
) -> dict[str, Any]:
    if not bool(correctness_gate.get("correct", False)):
        return {"written": False, "reason": "correctness_gate_failed"}
    trace_dir = context.get("harness", {}).get("trace_dir")
    if not trace_dir:
        return {"written": False, "reason": "missing_trace_dir"}
    assignment = result.get("assignment")
    if not isinstance(assignment, dict) or not assignment:
        return {"written": False, "reason": "missing_assignment"}
    try:
        source = program_path.read_text(encoding="utf-8")
    except Exception:
        source = ""
    candidate_digest = hashlib.sha256(
        json.dumps(
            {
                "source": source,
                "policy": _compact_policy_summary(eval_hints),
                "assignment": assignment,
                "objective_cost_usec": objective.get("objective_cost_usec"),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    artifact_root = Path(trace_dir).parent / "candidate_mlirs" / candidate_digest
    try:
        artifact_root.mkdir(parents=True, exist_ok=True)
        assignment_context = (
            result.get("assignment_context")
            if isinstance(result.get("assignment_context"), dict) and result.get("assignment_context")
            else context
        )
        tdag = tdag_from_context(assignment_context)
        assign = _assign_from_serialized(tdag, assignment)
        assign.check_assign()
        decoded = decode_assign(assign, tdag, {str(node): str(node) for node in tdag.nodes})
        mlir_path = artifact_root / "candidate.mlir"
        tdag_to_mlir(decoded, str(mlir_path))
        mlir_text = mlir_path.read_text(encoding="utf-8")
        mlir_digest = hashlib.sha256(mlir_text.encode("utf-8")).hexdigest()
        summary = {
            "written": True,
            "candidate_digest": candidate_digest,
            "mlir_path": str(mlir_path),
            "mlir_digest": mlir_digest,
            "assignment_path": str(artifact_root / "assignment.json"),
            "policy_path": str(artifact_root / "policy.json"),
            "execution_trace_path": str(artifact_root / "execution_trace.json"),
            "mlir_bytes": len(mlir_text.encode("utf-8")),
            "context_name": assignment_context.get("tdag", {}).get("name", context.get("tdag", {}).get("name", "")),
            "sampled_partition_mlir": assignment_context is not context,
            "objective_cost_usec": _finite_float(objective.get("objective_cost_usec"), 0.0),
            "final_latency_usec": _finite_float(result.get("final_latency_usec"), 0.0),
            "bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0),
            "rescale_count": _finite_float(result.get("rescale_count"), 0.0),
            "scale_floor_bits": _finite_float(result.get("scale_floor_bits"), 0.0),
            "selected_path_digest": result.get("sampled_selected_path_digest", ""),
            "policy_summary": json.loads(_compact_policy_summary(eval_hints)),
            "mlir_preview": mlir_text[:6000],
        }
        (artifact_root / "summary.json").write_text(
            json.dumps({k: v for k, v in summary.items() if k != "mlir_preview"}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (artifact_root / "assignment.json").write_text(
            json.dumps(assignment, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_root / "policy.json").write_text(
            json.dumps(eval_hints, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        (artifact_root / "execution_trace.json").write_text(
            json.dumps(execution_trace, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        index_record = {k: v for k, v in summary.items() if k != "mlir_preview"}
        with (Path(trace_dir).parent / "candidate_mlirs" / "index.jsonl").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(json.dumps(index_record, sort_keys=True, default=str) + "\n")
        return summary
    except Exception as exc:
        return {
            "written": False,
            "reason": f"{type(exc).__name__}: {str(exc)[:240]}",
            "candidate_digest": candidate_digest,
        }


def _placement_effect_summary(
    context: dict[str, Any],
    result: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    candidate_target_score: float,
) -> dict[str, Any]:
    harness = context.get("harness", {})
    eval_suite = str(harness.get("eval_suite", "polybert-sampled"))
    baseline = harness.get("sampled_seed_baseline") if eval_suite != "polybert-full" else None
    if not isinstance(baseline, dict):
        baseline = harness.get("seed_baseline") or context.get("reference", {})
    if not isinstance(baseline, dict):
        baseline = {}
    scored_groups = _scored_boundary_group_count(diagnostics)
    candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
    solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
    fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
    fallback_budgets = int(result.get("fallback_selected_budgets", 0) or 0)
    baseline_scored_groups = max(1, int(baseline.get("scored_boundary_groups", scored_groups) or scored_groups))
    baseline_candidate_groups = int(
        baseline.get("candidate_solved_boundary_groups", 0) or 0
    )
    baseline_solved_groups = int(baseline.get("solved_boundary_groups", 0) or 0)
    baseline_fallback_groups = int(baseline.get("fallback_selected_groups", 0) or 0)
    baseline_fallback_budgets = int(baseline.get("fallback_selected_budgets", 0) or 0)
    baseline_latency = _finite_float(baseline.get("final_latency_usec"), float("inf"))
    candidate_latency = _finite_float(result.get("final_latency_usec"), float("inf"))
    baseline_bootstraps = _finite_float(baseline.get("bootstrap_count"), float("nan"))
    candidate_bootstraps = _finite_float(result.get("bootstrap_count"), float("nan"))
    baseline_rescales = _finite_float(baseline.get("rescale_count"), float("nan"))
    candidate_rescales = _finite_float(result.get("rescale_count"), float("nan"))
    target = _context_target_bootstrap_count(context)
    baseline_target_score = _contextual_target_bootstrap_score(
        context,
        target,
        baseline_bootstraps,
        baseline.get("reference_bootstrap_count"),
    )
    group_den = max(1, max(scored_groups, baseline_scored_groups))
    solved_delta = (solved_groups - baseline_solved_groups) / group_den
    candidate_group_delta = (candidate_groups - baseline_candidate_groups) / group_den
    fallback_delta = (
        (baseline_fallback_groups + baseline_fallback_budgets)
        - (fallback_groups + fallback_budgets)
    ) / max(1, baseline_fallback_groups + baseline_fallback_budgets + 1)
    latency_delta = (
        (baseline_latency - candidate_latency) / max(abs(baseline_latency), 1.0)
        if math.isfinite(baseline_latency) and math.isfinite(candidate_latency)
        else 0.0
    )
    bootstrap_delta = (
        (baseline_bootstraps - candidate_bootstraps) / max(abs(baseline_bootstraps), 1.0)
        if math.isfinite(baseline_bootstraps) and math.isfinite(candidate_bootstraps)
        else 0.0
    )
    rescale_delta = (
        (baseline_rescales - candidate_rescales) / max(abs(baseline_rescales), 1.0)
        if math.isfinite(baseline_rescales) and math.isfinite(candidate_rescales)
        else 0.0
    )
    target_delta = float(candidate_target_score) - float(baseline_target_score)
    effect_score = min(
        1.0,
        max(0.0, 0.30 * solved_delta)
        + max(0.0, 0.25 * candidate_group_delta)
        + max(0.0, 0.20 * latency_delta)
        + max(0.0, 0.15 * target_delta)
        + max(0.0, 0.05 * bootstrap_delta)
        + max(0.0, 0.05 * rescale_delta)
        + max(0.0, 0.10 * fallback_delta),
    )
    outcome_changed = any(
        abs(delta) > 1e-9
        for delta in (
            solved_delta,
            candidate_group_delta,
            fallback_delta,
            latency_delta,
            bootstrap_delta,
            rescale_delta,
            target_delta,
        )
    )
    return {
        "effect_score": float(effect_score),
        "outcome_changed": bool(outcome_changed),
        "baseline_source": (
            "sampled_seed_baseline"
            if eval_suite != "polybert-full" and isinstance(harness.get("sampled_seed_baseline"), dict)
            else "seed_baseline"
        ),
        "solved_group_delta": float(solved_delta),
        "candidate_group_delta": float(candidate_group_delta),
        "fallback_delta": float(fallback_delta),
        "latency_delta_ratio": float(latency_delta),
        "bootstrap_delta_ratio": float(bootstrap_delta),
        "rescale_delta_ratio": float(rescale_delta),
        "target_bootstrap_score_delta": float(target_delta),
        "baseline": {
            "final_latency_usec": baseline_latency if math.isfinite(baseline_latency) else "inf",
            "bootstrap_count": baseline_bootstraps if math.isfinite(baseline_bootstraps) else "nan",
            "rescale_count": baseline_rescales if math.isfinite(baseline_rescales) else "nan",
            "solved_boundary_groups": baseline_solved_groups,
            "candidate_solved_boundary_groups": baseline_candidate_groups,
            "fallback_selected_groups": baseline_fallback_groups,
            "fallback_selected_budgets": baseline_fallback_budgets,
            "target_bootstrap_score": float(baseline_target_score),
        },
    }


def _compile_hints_for_eval_suite(hints: dict[str, Any], eval_suite: str) -> dict[str, Any]:
    normalized = _with_default_policy(hints)
    if eval_suite == "polybert-full":
        return normalized
    return _bounded_sampled_policy(normalized)


_POLICY_EFFECT_KEYS = (
    "strategy",
    "budget_aggressive",
    "allow_bootstrap",
    "forbid_bootstrap",
    "allow_seed_fallback",
    "refresh_fanout_at_level_floor",
    "max_scale_candidates",
    "max_scale",
    "bootstrap_penalty",
    "selection_bootstrap_penalty",
    "rescale_penalty",
    "level_drop_penalty",
    "reserve_penalty",
    "min_transition_reserve",
    "min_decryptability_reserve",
    "noise_slack_model",
    "scale_floor_bits",
    "beam_width",
    "state_cap_per_node",
    "scale_lattice",
    "boundary_scale_policy",
    "boundary_state_cap",
    "mcts_rollout_budget",
    "mcts_exploration_weight",
    "mcts_max_repair_bootstraps",
    "mcts_action_cap",
    "mcts_prior_order",
    "mcts_action_allowlist",
    "mcts_action_blocklist",
    "target_bootstrap_count",
    "bootstrap_anchor_count",
    "bootstrap_anchor_level",
    "bootstrap_anchor_selector",
    "bootstrap_anchor_include_patterns",
    "bootstrap_anchor_exclude_patterns",
    "selection_objective",
    "prefer_component_budget_fit",
    "force_bootstrap_anchors",
    "enable_direct_budget_beam",
    "include_seed_repair_actions",
)


def _bootstrap_mcts_initial_policy_for_context(context: dict[str, Any]) -> dict[str, Any]:
    mcts = PlacementMCTS(context)
    actions = mcts.candidate_actions(action_cap=12)
    reference_action_names = [
        str(action.get("name"))
        for action in actions
        if str(action.get("name", "")) == "reference_boundary_cost_beam"
    ]
    for action in actions:
        policy = action.get("policy", {})
        name = str(action.get("name", ""))
        if name == "strict_no_bootstrap":
            action["prior"] = 0.05
            policy["boundary_scale_policy"] = "low"
            policy["max_scale_candidates"] = 24
        elif name == "budget_fulfillment_beam":
            action["prior"] = 0.54
            policy["beam_width"] = 8
            policy["state_cap_per_node"] = 32
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 40
            policy["bootstrap_penalty"] = 250_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "wide_boundary_cost_beam":
            action["prior"] = 0.36
            policy["beam_width"] = 8
            policy["state_cap_per_node"] = 32
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["boundary_scale_policy"] = "frontier"
            policy["bootstrap_penalty"] = 50_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "dense_boundary_cost_beam":
            action["prior"] = 0.40
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 16
            policy["max_scale_candidates"] = 96
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "dense"
            policy["bootstrap_penalty"] = 25_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "nonlinear_phase_boundary_beam":
            action["prior"] = 0.44
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 12
            policy["max_scale_candidates"] = 80
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["bootstrap_anchor_selector"] = "nonlinear_phase_boundaries"
            policy["bootstrap_anchor_count"] = 64
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_penalty"] = 35_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "reference_boundary_cost_beam":
            action["prior"] = 0.50
            policy["strategy"] = "latency_beam"
            policy["beam_width"] = 10
            policy["state_cap_per_node"] = 48
            policy["boundary_state_cap"] = 12
            policy["max_scale_candidates"] = 80
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["bootstrap_anchor_selector"] = "reference_bootstrap_locations"
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_penalty"] = 30_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "profile_waterline_repair":
            action["prior"] = 0.24
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "waterline"
            policy["scale_lattice"] = "waterline_sf"
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 48
            policy["state_cap_per_node"] = 24
            policy["beam_width"] = 6
            policy["bootstrap_penalty"] = 125_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
        elif name == "tuneinsight_avgcase_cost_beam":
            action["prior"] = 0.22
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "sf"
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["state_cap_per_node"] = 32
            policy["beam_width"] = 8
            policy["noise_slack_model"] = "tuneinsight_avgcase"
            policy["reserve_penalty"] = 0.0
            policy["min_transition_reserve"] = 0
            policy["min_decryptability_reserve"] = 0
            policy["bootstrap_penalty"] = 25_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "tuneinsight_deferred_bootstrap_beam":
            action["prior"] = 0.30
            policy["strategy"] = "latency_beam"
            policy["boundary_scale_policy"] = "frontier"
            policy["scale_lattice"] = "waterline_sf"
            policy["boundary_state_cap"] = 8
            policy["max_scale_candidates"] = 64
            policy["state_cap_per_node"] = 32
            policy["beam_width"] = 8
            policy["noise_slack_model"] = "tuneinsight_avgcase"
            policy["reserve_penalty"] = 0.0
            policy["min_transition_reserve"] = 0
            policy["min_decryptability_reserve"] = 0
            policy["bootstrap_penalty"] = 650_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "minimal_bootstrap_repair":
            action["prior"] = 0.18
            policy["bootstrap_anchor_count"] = 0
            policy["boundary_state_cap"] = 3
            policy["bootstrap_penalty"] = 2_500_000_000.0
            policy["selection_bootstrap_penalty"] = 250_000_000.0
            policy["selection_objective"] = "min_bootstrap"
        elif name == "component_budget_repair":
            action["prior"] = 0.28
            policy["selection_objective"] = "cost"
            policy["prefer_component_budget_fit"] = False
            policy["force_bootstrap_anchors"] = False
            policy["bootstrap_anchor_count"] = 0
            policy["boundary_state_cap"] = 6
            policy["max_scale_candidates"] = 48
            policy["bootstrap_penalty"] = 50_000_000.0
            policy["selection_bootstrap_penalty"] = 0.0
        elif name == "waterline_budget_repair":
            action["prior"] = 0.20
            policy["selection_bootstrap_penalty"] = 0.0
            policy["selection_objective"] = "cost"
        elif name == "latency_mcts_repair":
            action["prior"] = 0.19
        action["policy"] = policy
    policy = mcts.low_bootstrap_seed(
        rollout_budget=20,
        exploration_weight=1.15,
        max_repair_bootstraps=16,
        action_cap=len(actions),
    )
    policy["mcts_actions"] = actions
    policy["mcts_action_cap"] = 8
    policy["mcts_rollout_budget"] = 24
    relaxed_floor_action_names = [
        str(action.get("name"))
        for action in actions
        if str(action.get("name", "")).startswith("estimator_relaxed_floor_")
    ]
    policy["mcts_action_allowlist"] = [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "waterline_cost_beam",
        "dense_boundary_cost_beam",
        "nonlinear_phase_boundary_beam",
        *reference_action_names,
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        *relaxed_floor_action_names,
        "latency_mcts_repair",
        "component_budget_repair",
        "minimal_bootstrap_repair",
        "waterline_budget_repair",
    ]
    policy["mcts_exploration_weight"] = 1.15
    group_policies = []
    for idx, group in enumerate(mcts.top_costly_boundary_groups(limit=6)):
        if idx % 3 == 0:
            group_policies.append(
                mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=12,
                    max_scale_candidates=80,
                    boundary_scale_policy="frontier",
                    scale_lattice="waterline_sf",
                    bootstrap_penalty=35_000_000.0,
                    beam_width=10,
                    state_cap_per_node=48,
                )
            )
        elif idx % 3 == 1:
            group_policies.append(
                mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=8,
                    max_scale_candidates=64,
                    boundary_scale_policy="waterline",
                    scale_lattice="waterline_sf",
                    bootstrap_penalty=65_000_000.0,
                    beam_width=8,
                    state_cap_per_node=40,
                )
            )
        else:
            group_policies.append(
                mcts.boundary_group_policy(
                    group,
                    boundary_state_cap=16,
                    max_scale_candidates=96,
                    boundary_scale_policy="frontier",
                    scale_lattice="dense",
                    bootstrap_penalty=25_000_000.0,
                    beam_width=12,
                    state_cap_per_node=48,
                )
            )
    policy["boundary_group_policies"] = group_policies
    policy["include_seed_repair_actions"] = False
    policy["mcts_prior_order"] = True
    policy["mcts_action_presets"] = mcts.action_presets(
        budget_fulfillment_beam={
            "prior": 0.54,
            "policy": {
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 6,
                "max_scale_candidates": 40,
                "bootstrap_penalty": 250_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        wide_boundary_cost_beam={
            "prior": 0.36,
            "policy": {
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        dense_boundary_cost_beam={
            "prior": 0.40,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "dense",
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        nonlinear_phase_boundary_beam={
            "prior": 0.44,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                "bootstrap_anchor_count": 64,
                "force_bootstrap_anchors": False,
                "bootstrap_penalty": 35_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        **{
            name: {
                "prior": 0.50,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "reference_bootstrap_locations",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 30_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            }
            for name in reference_action_names
        },
        profile_waterline_repair={
            "prior": 0.24,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "state_cap_per_node": 24,
                "beam_width": 6,
                "bootstrap_penalty": 125_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
        },
        tuneinsight_avgcase_cost_beam={
            "prior": 0.22,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "sf",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "state_cap_per_node": 32,
                "beam_width": 8,
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        tuneinsight_deferred_bootstrap_beam={
            "prior": 0.30,
            "policy": {
                "strategy": "latency_beam",
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "state_cap_per_node": 32,
                "beam_width": 8,
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 650_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        **{
            name: {
                "prior": 0.34,
                "policy": {
                    "strategy": "latency_beam",
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "estimator_relaxed",
                    "scale_floor_bits": _safe_int(
                        name.rsplit("_", 1)[-1],
                        int(_ckks_dict(context)["Sw"]),
                    ),
                    "boundary_state_cap": 8,
                    "max_scale_candidates": 64,
                    "state_cap_per_node": 32,
                    "beam_width": 8,
                    "noise_slack_model": "tuneinsight_avgcase",
                    "reserve_penalty": 0.0,
                    "min_transition_reserve": 0,
                    "min_decryptability_reserve": 0,
                    "bootstrap_penalty": 35_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                },
            }
            for name in relaxed_floor_action_names
        },
        component_budget_repair={
            "prior": 0.28,
            "policy": {
                "selection_objective": "cost",
                "prefer_component_budget_fit": False,
                "force_bootstrap_anchors": False,
                "bootstrap_anchor_count": 0,
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
        },
    )
    policy.update(
        {
            "allow_seed_fallback": True,
            "allow_bootstrap": False,
            "refresh_fanout_at_level_floor": True,
            "max_scale_candidates": 24,
            "bootstrap_penalty": 25_000_000.0,
            "rescale_penalty": 0.0,
            "level_drop_penalty": 20_000_000.0,
            "reserve_penalty": 75_000.0,
            "min_transition_reserve": 1,
            "min_decryptability_reserve": 1,
            "boundary_scale_policy": "waterline",
            "beam_width": 10,
            "state_cap_per_node": 32,
            "boundary_state_cap": 12,
        }
    )
    for key in (
        "component_bootstrap_budgets",
        "enable_direct_budget_beam",
        "forbid_bootstrap",
        "scale_lattice",
        "selection_bootstrap_penalty",
        "target_bootstrap_count",
    ):
        policy.pop(key, None)
    policy["boundary_group_policies"] = []
    policy["mcts_action_cap"] = 10
    policy["mcts_rollout_budget"] = 16
    policy["mcts_exploration_weight"] = 1.25
    policy["mcts_max_repair_bootstraps"] = 128
    policy["mcts_prior_order"] = True
    policy["include_seed_repair_actions"] = True
    policy["selection_objective"] = "cost"
    active_action_names = [
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        *reference_action_names,
        "latency_mcts_repair",
    ]
    active_actions = [
        action for action in actions if str(action.get("name", "")) in set(active_action_names)
    ]
    for action in active_actions:
        name = str(action.get("name", ""))
        if name == "budget_fulfillment_beam":
            action["prior"] = 0.60
        elif name == "wide_boundary_cost_beam":
            action["prior"] = 0.40
        elif name == "dense_boundary_cost_beam":
            action["prior"] = 0.50
        elif name == "latency_mcts_repair":
            action["prior"] = 0.19
        elif name in reference_action_names:
            action["prior"] = 0.50
    policy["mcts_actions"] = active_actions
    policy["mcts_action_allowlist"] = active_action_names
    policy["mcts_action_presets"] = mcts.action_presets(
        budget_fulfillment_beam={
            "prior": 0.60,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 48,
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "boundary_scale_policy": "waterline",
                "target_bootstrap_count": 0,
            },
        },
        wide_boundary_cost_beam={
            "prior": 0.40,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 48,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "target_bootstrap_count": 0,
            },
        },
        dense_boundary_cost_beam={
            "prior": 0.50,
            "policy": {
                "strategy": "latency_beam",
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "scale_lattice": "dense",
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 25_000_000.0,
                "rescale_penalty": 0.0,
                "level_drop_penalty": 20_000_000.0,
                "selection_objective": "cost",
                "direct_budget_policy": True,
                "target_bootstrap_count": 0,
            },
        },
        **{
            name: {
                "prior": 0.50,
                "policy": {
                    "strategy": "latency_beam",
                    "beam_width": 10,
                    "state_cap_per_node": 48,
                    "boundary_state_cap": 12,
                    "max_scale_candidates": 80,
                    "boundary_scale_policy": "frontier",
                    "scale_lattice": "waterline_sf",
                    "bootstrap_anchor_selector": "reference_bootstrap_locations",
                    "force_bootstrap_anchors": False,
                    "bootstrap_penalty": 30_000_000.0,
                    "selection_bootstrap_penalty": 0.0,
                    "selection_objective": "cost",
                    "direct_budget_policy": True,
                    "target_bootstrap_count": 0,
                },
            }
            for name in reference_action_names
        },
    )
    return policy


def _seed_policy_for_effect(context: dict[str, Any]) -> dict[str, Any]:
    active_seed = _context_initial_policy_hints(context)
    if active_seed:
        return active_seed
    search_mode = str(context.get("harness", {}).get("search_mode", "bootstrap-mcts"))
    if search_mode == "bootstrap-mcts":
        return _bootstrap_mcts_initial_policy_for_context(context)
    return _zero_iteration_portfolio_hints()


def _policy_effect_summary(
    context: dict[str, Any],
    raw_hints: dict[str, Any],
    eval_hints: dict[str, Any],
    eval_suite: str,
) -> dict[str, Any]:
    seed_eval = _compile_hints_for_eval_suite(_seed_policy_for_effect(context), eval_suite)
    seed_payload = _policy_effect_payload(seed_eval)
    candidate_payload = _policy_effect_payload(eval_hints)
    seed_digest = _hint_digest(seed_payload)
    candidate_digest = _hint_digest(candidate_payload)
    changed_keys = sorted(
        key
        for key in set(seed_payload.get("policy", {})) | set(candidate_payload.get("policy", {}))
        if seed_payload.get("policy", {}).get(key) != candidate_payload.get("policy", {}).get(key)
    )
    seed_actions = {
        item.get("name"): item
        for item in seed_payload.get("mcts_actions", [])
        if isinstance(item, dict)
    }
    candidate_actions_payload = {
        item.get("name"): item
        for item in candidate_payload.get("mcts_actions", [])
        if isinstance(item, dict)
    }
    changed_actions = sorted(
        name
        for name in set(seed_actions) | set(candidate_actions_payload)
        if seed_actions.get(name) != candidate_actions_payload.get(name)
    )
    if seed_payload.get("mcts_action_presets_digest") != candidate_payload.get(
        "mcts_action_presets_digest"
    ):
        preset_names = set(seed_payload.get("mcts_action_preset_names", [])) | set(
            candidate_payload.get("mcts_action_preset_names", [])
        )
        changed_actions = sorted(set(changed_actions) | {str(name) for name in preset_names})
    changed_maps = sorted(
        key
        for key in (
            "preferred_node_levels_count",
            "preferred_node_scales_count",
            "preferred_edge_scales_count",
            "unit_policy_count",
            "boundary_group_policy_count",
            "portfolio_count",
        )
        if seed_payload.get(key) != candidate_payload.get(key)
    )
    repair_reasons = list(eval_hints.get("__repair_reasons", []) or [])
    change_units = len(changed_keys) + len(changed_actions) + len(changed_maps)
    effect_score = min(1.0, change_units / 8.0)
    if repair_reasons and candidate_digest == seed_digest:
        effect_score = 0.0
    return {
        "seed_digest": seed_digest[:16],
        "candidate_digest": candidate_digest[:16],
        "seed_equivalent": seed_digest == candidate_digest,
        "effect_score": float(effect_score),
        "changed_policy_keys": changed_keys[:16],
        "changed_actions": changed_actions[:16],
        "changed_sparse_maps": changed_maps,
        "repair_reasons": repair_reasons[:12],
        "raw_digest": _hint_digest(raw_hints)[:16],
        "eval_suite": eval_suite,
        "action_cap": candidate_payload.get("policy", {}).get("mcts_action_cap"),
        "action_names": [
            str(item.get("name"))
            for item in candidate_payload.get("mcts_actions", [])[:12]
            if isinstance(item, dict)
        ],
    }


def _policy_effect_payload(hints: dict[str, Any]) -> dict[str, Any]:
    policy = {
        key: hints.get(key)
        for key in _POLICY_EFFECT_KEYS
        if key in hints
    }
    payload: dict[str, Any] = {"policy": policy}
    actions = hints.get("mcts_actions")
    cap = max(1, min(64, _int_hint(hints.get("mcts_action_cap"), 64)))
    if isinstance(actions, list):
        payload["mcts_actions"] = [
            _mcts_action_effect_payload(item)
            for item in actions[:cap]
            if isinstance(item, dict)
        ]
    for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
        value = hints.get(key)
        if isinstance(value, dict) and value:
            payload[f"{key}_count"] = len(value)
            payload[f"{key}_digest"] = _hint_digest(value)[:16]
    if hints.get("unit_policies"):
        payload["unit_policy_count"] = len(hints.get("unit_policies") or [])
        payload["unit_policy_digest"] = _hint_digest(hints.get("unit_policies"))[:16]
    if hints.get("boundary_group_policies"):
        payload["boundary_group_policy_count"] = len(
            hints.get("boundary_group_policies") or []
        )
        payload["boundary_group_policy_digest"] = _hint_digest(
            hints.get("boundary_group_policies")
        )[:16]
    portfolio = hints.get("portfolio")
    if isinstance(portfolio, list) and portfolio:
        payload["portfolio_count"] = len(portfolio)
        payload["portfolio_digest"] = _hint_digest(
            [_policy_effect_payload(item) for item in portfolio if isinstance(item, dict)]
        )[:16]
    presets = hints.get("mcts_action_presets")
    if isinstance(presets, (dict, list)):
        payload["mcts_action_presets_digest"] = _hint_digest(presets)[:16]
        payload["mcts_action_preset_names"] = sorted(_mcts_action_preset_map(presets))
    return payload


def _effective_qbp_payload(diagnostics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    summaries = []
    for item in list(diagnostics.get("boundary_group_summaries", []))[:512]:
        if not isinstance(item, dict):
            continue
        summaries.append(
            {
                "group_key": item.get("group_key", {}),
                "requested_output_levels": item.get("requested_output_levels", []),
                "reachable_budgets": int(item.get("reachable_budgets", 0) or 0),
                "solved_budgets": int(item.get("solved_budgets", 0) or 0),
                "candidate_solved_budgets": int(
                    item.get("candidate_solved_budgets", 0) or 0
                ),
                "fallback_selected_budgets": int(
                    item.get("fallback_selected_budgets", 0) or 0
                ),
                "candidate_complete": bool(item.get("candidate_complete", False)),
                "complete": bool(item.get("complete", False)),
                "unsolved_reason": str(item.get("unsolved_reason", "")),
                "min_cost_usec": _digest_float(item.get("min_cost_usec")),
                "min_bootstrap": _digest_float(item.get("min_bootstrap")),
                "min_rescale": _digest_float(item.get("min_rescale")),
                "selected_source_counts": {
                    str(source): int(count or 0)
                    for source, count in sorted(
                        dict(item.get("selected_source_counts", {}) or {}).items()
                    )
                },
            }
        )
    return {
        "requested_boundary_groups": int(
            diagnostics.get("requested_boundary_groups", 0) or 0
        ),
        "solved_boundary_groups": int(diagnostics.get("solved_boundary_groups", 0) or 0),
        "candidate_solved_boundary_groups": int(
            diagnostics.get("candidate_solved_boundary_groups", 0) or 0
        ),
        "fallback_selected_boundary_groups": int(
            diagnostics.get("fallback_selected_boundary_groups", 0) or 0
        ),
        "unreachable_boundary_groups": int(
            diagnostics.get("unreachable_boundary_groups", 0) or 0
        ),
        "selected_source_counts": {
            str(source): int(count or 0)
            for source, count in sorted(
                dict(diagnostics.get("selected_source_counts", {}) or {}).items()
            )
        },
        "boundary_groups": summaries,
    }


def _effective_qbp_digest(diagnostics: dict[str, Any] | None) -> str:
    return _hint_digest(_effective_qbp_payload(diagnostics))


def _selected_path_payload(
    result: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        result = {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    assignment = result.get("assignment", {})
    return {
        "selected_output_state": result.get("selected_output_state", {}),
        "bootstrap_count": _digest_float(result.get("bootstrap_count")),
        "rescale_count": _digest_float(result.get("rescale_count")),
        "scale_floor_bits": _digest_float(result.get("scale_floor_bits")),
        "sampled_selected_path_bootstraps": _digest_float(
            result.get("sampled_selected_path_bootstraps")
        ),
        "sampled_selected_path_rescales": _digest_float(
            result.get("sampled_selected_path_rescales")
        ),
        "bootstrap_locations": result.get("bootstrap_locations", {}),
        "rescale_locations": result.get("rescale_locations", {}),
        "assignment_digest": _hint_digest(assignment)[:24] if assignment else "",
    }


def _selected_path_digest(
    result: dict[str, Any] | None,
    diagnostics: dict[str, Any] | None,
) -> str:
    return _hint_digest(_selected_path_payload(result, diagnostics))


def _digest_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, 3)


def _sampled_path_proxy_from_boundary_groups(
    summaries: list[dict[str, Any]],
    fallback_costs: list[float] | None = None,
) -> dict[str, Any]:
    usable = [
        item
        for item in summaries
        if isinstance(item, dict) and int(item.get("solved_budgets", 0) or 0) > 0
    ]
    if not usable:
        costs = [float(item) for item in fallback_costs or []]
        total_cost = float(sum(costs))
        return {
            "sampled_dp_latency_usec": total_cost,
            "sampled_selected_path_bootstraps": 0.0,
            "sampled_selected_path_rescales": 0.0,
            "sampled_selected_path_digest": _hint_digest(
                {"fallback_costs": [_digest_float(item) for item in costs[:256]]}
            ),
        }
    path_items = []
    total_cost = 0.0
    total_bootstrap = 0.0
    total_rescale = 0.0
    for item in usable:
        cost = _finite_float(item.get("min_cost_usec"), 0.0)
        bootstraps = _finite_float(item.get("min_bootstrap"), 0.0)
        rescales = _finite_float(item.get("min_rescale"), 0.0)
        total_cost += cost
        total_bootstrap += bootstraps
        total_rescale += rescales
        path_items.append(
            {
                "group_key": item.get("group_key", {}),
                "requested_output_levels": item.get("requested_output_levels", []),
                "candidate_complete": bool(item.get("candidate_complete", False)),
                "complete": bool(item.get("complete", False)),
                "min_cost_usec": _digest_float(cost),
                "min_bootstrap": _digest_float(bootstraps),
                "min_rescale": _digest_float(rescales),
                "selected_source_counts": item.get("selected_source_counts", {}),
            }
        )
    return {
        "sampled_dp_latency_usec": float(total_cost),
        "sampled_selected_path_bootstraps": float(total_bootstrap),
        "sampled_selected_path_rescales": float(total_rescale),
        "sampled_selected_path_digest": _hint_digest(path_items),
    }


def _result_effective_summary(
    context: dict[str, Any],
    result: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    qbp_digest = _effective_qbp_digest(diagnostics)
    path_digest = _selected_path_digest(result, diagnostics)
    reference = context.get("reference", {}) if isinstance(context, dict) else {}
    harness = context.get("harness", {}) if isinstance(context, dict) else {}
    if isinstance(harness, dict) and isinstance(harness.get("active_seed_baseline"), dict):
        reference = harness["active_seed_baseline"]
    if (
        isinstance(harness, dict)
        and not isinstance(harness.get("active_seed_baseline"), dict)
        and str(harness.get("eval_suite", "polybert-sampled")) != "polybert-full"
        and isinstance(harness.get("sampled_seed_baseline"), dict)
    ):
        reference = harness["sampled_seed_baseline"]
    reference_qbp = str(reference.get("effective_qbp_digest", "") or "")
    reference_path = str(reference.get("selected_path_digest", "") or "")
    qbp_cmp = qbp_digest[: len(reference_qbp)] if reference_qbp else qbp_digest
    path_cmp = path_digest[: len(reference_path)] if reference_path else path_digest
    changed_groups = 0
    if isinstance(diagnostics, dict):
        for item in diagnostics.get("boundary_group_summaries", []) or []:
            if not isinstance(item, dict):
                continue
            if int(item.get("candidate_solved_budgets", 0) or 0) > 0:
                changed_groups += 1
    return {
        "effective_qbp_digest": qbp_digest[:24],
        "selected_path_digest": path_digest[:24],
        "reference_qbp_digest": reference_qbp[:24],
        "reference_selected_path_digest": reference_path[:24],
        "effective_qbp_changed_vs_seed": bool(reference_qbp and qbp_cmp != reference_qbp),
        "selected_path_changed_vs_seed": bool(reference_path and path_cmp != reference_path),
        "seed_equivalent_path": bool(
            reference_path and path_cmp == reference_path
        ),
        "changed_boundary_groups_vs_seed": int(changed_groups),
        "selected_path_payload": _selected_path_payload(result, diagnostics),
    }


def _mcts_action_effect_payload(item: dict[str, Any]) -> dict[str, Any]:
    policy = item.get("policy", item)
    if not isinstance(policy, dict):
        policy = {}
    return {
        "name": str(item.get("name", "")),
        "prior": _float_hint(item.get("prior"), 0.0),
        "policy": {
            key: policy.get(key)
            for key in _POLICY_EFFECT_KEYS
            if key in policy
        },
    }


def _action_effect_score(diagnostics: dict[str, Any]) -> float:
    attempts = diagnostics.get("mcts_action_attempt_counts", {})
    successes = diagnostics.get("mcts_action_success_counts", {})
    selected = diagnostics.get("selected_source_counts", {})
    if not isinstance(attempts, dict):
        attempts = {}
    if not isinstance(successes, dict):
        successes = {}
    if not isinstance(selected, dict):
        selected = {}
    attempted_actions = len([name for name, count in attempts.items() if int(count or 0) > 0])
    successful_actions = len([name for name, count in successes.items() if int(count or 0) > 0])
    selected_candidate_actions = len(
        {
            str(source).split(":")[2]
            for source, count in selected.items()
            if str(source).startswith("candidate:boundary_mcts:")
            and int(count or 0) > 0
            and len(str(source).split(":")) >= 3
        }
    )
    if attempted_actions <= 0:
        return 0.0
    return min(
        1.0,
        0.35 * min(1.0, attempted_actions / 4.0)
        + 0.40 * min(1.0, successful_actions / max(1.0, attempted_actions))
        + 0.25 * min(1.0, selected_candidate_actions / 2.0),
    )


def _bounded_sampled_policy(hints: dict[str, Any]) -> dict[str, Any]:
    bounded = _with_default_policy(hints)
    portfolio = bounded.pop("portfolio", None)
    policy_bank_lightweight = _bool_hint(bounded.pop("policy_bank_lightweight", False), False)
    budget_aggressive = _bool_hint(bounded.get("budget_aggressive"), False)
    mcts_broad = str(bounded.get("strategy")) == "bootstrap_mcts"
    # Sampled evolution is intentionally lenient: failed direct candidate budgets
    # are repaired through the deterministic seed, then scored via
    # candidate_validity and fallback_selected_budgets. Finalist replay disables
    # this so selected MLIRs still come from a concrete policy.
    bounded["allow_seed_fallback"] = True
    bounded["allow_bootstrap"] = _bool_hint(bounded.get("allow_bootstrap"), budget_aggressive)
    bounded["refresh_fanout_at_level_floor"] = True
    bounded["max_scale_candidates"] = min(
        16 if policy_bank_lightweight else (48 if (budget_aggressive or mcts_broad) else 16),
        _int_hint(
            bounded.get("max_scale_candidates"),
            16
            if policy_bank_lightweight
            else (48 if (mcts_broad or budget_aggressive) else 16),
        ),
    )
    bounded["state_cap_per_node"] = min(
        8 if policy_bank_lightweight else (32 if (budget_aggressive or mcts_broad) else 8),
        _int_hint(
            bounded.get("state_cap_per_node"),
            8
            if policy_bank_lightweight
            else (24 if (mcts_broad or budget_aggressive) else 8),
        ),
    )
    bounded["beam_width"] = min(
        4 if policy_bank_lightweight else (8 if (budget_aggressive or mcts_broad) else 4),
        _int_hint(
            bounded.get("beam_width"),
            4
            if policy_bank_lightweight
            else (8 if (mcts_broad or budget_aggressive) else 4),
        ),
        )
    if str(bounded.get("strategy")) == "bootstrap_mcts":
        # Sampled evolution must stay a fast proxy. A candidate may request a
        # broader beam for full replay, but the sampled OpenEvolve evaluator
        # always projects MCTS actions onto bounded direct-budget variants.
        bounded["mcts_actions"] = _sampled_lightweight_mcts_actions(
            bounded.get("mcts_actions")
        )
        bounded["mcts_rollout_budget"] = min(
            4 if policy_bank_lightweight else (8 if budget_aggressive else 6),
            _int_hint(
                bounded.get("mcts_rollout_budget"),
                4 if policy_bank_lightweight else (8 if budget_aggressive else 6),
            ),
        )
        bounded["mcts_action_cap"] = min(
            2 if policy_bank_lightweight else (4 if budget_aggressive else 3),
            _int_hint(
                bounded.get("mcts_action_cap"),
                2 if policy_bank_lightweight else (4 if budget_aggressive else 3),
            ),
        )
        bounded["mcts_max_repair_bootstraps"] = min(
            4 if policy_bank_lightweight else (32 if budget_aggressive else 12),
            _int_hint(
                bounded.get("mcts_max_repair_bootstraps"),
                4 if policy_bank_lightweight else (32 if budget_aggressive else 12),
            ),
        )
        bounded["boundary_state_cap"] = min(
            2 if policy_bank_lightweight else (8 if budget_aggressive else 6),
            _int_hint(
                bounded.get("boundary_state_cap"),
                2 if policy_bank_lightweight else (8 if budget_aggressive else 6),
            ),
        )
    bounded = _clamp_sampled_policy_patch(
        bounded,
        policy_bank_lightweight=policy_bank_lightweight,
    )
    bounded["mcts_actions"] = _clamp_sampled_policy_items(
        bounded.get("mcts_actions"),
        policy_bank_lightweight=policy_bank_lightweight,
    )
    bounded["mcts_action_presets"] = _clamp_sampled_action_presets(
        bounded.get("mcts_action_presets"),
        policy_bank_lightweight=policy_bank_lightweight,
    )
    bounded["boundary_group_policies"] = _clamp_sampled_policy_items(
        bounded.get("boundary_group_policies"),
        policy_bank_lightweight=policy_bank_lightweight,
    )
    bounded["unit_policies"] = _clamp_sampled_policy_items(
        bounded.get("unit_policies"),
        policy_bank_lightweight=policy_bank_lightweight,
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


def _sampled_lightweight_mcts_actions(raw_actions: Any) -> list[dict[str, Any]] | Any:
    if not isinstance(raw_actions, list):
        return raw_actions
    filtered: list[dict[str, Any]] = []
    for item in raw_actions:
        if not isinstance(item, dict):
            continue
        policy = item.get("policy", item)
        if not isinstance(policy, dict):
            continue
        if str(policy.get("strategy", "")) == "latency_beam":
            # Keep one cheap direct budget-fulfillment beam in sampled MCTS.
            # The expensive global direct-beam pre-pass is still opt-in, but
            # without any beam-shaped action sampled candidates tend to solve
            # no QBP groups directly and only survive through seed fallback.
            if (
                str(item.get("name", "")) == "budget_fulfillment_beam"
                or str(item.get("name", "")) == "wide_boundary_cost_beam"
                or str(item.get("name", "")) == "waterline_cost_beam"
                or str(item.get("name", "")) == "dense_boundary_cost_beam"
                or str(item.get("name", "")) == "nonlinear_phase_boundary_beam"
                or str(item.get("name", "")) == "reference_boundary_cost_beam"
                or str(item.get("name", "")) == "profile_waterline_repair"
                or str(item.get("name", "")) == "tuneinsight_avgcase_cost_beam"
                or str(item.get("name", "")) == "tuneinsight_deferred_bootstrap_beam"
                or str(item.get("name", "")).startswith("estimator_relaxed_floor_")
                or str(item.get("name", "")) == "latency_mcts_repair"
                or _bool_hint(policy.get("direct_budget_policy"), False)
            ):
                cheap = dict(item)
                cheap_policy = dict(policy)
                dense_boundary = str(item.get("name", "")) == "dense_boundary_cost_beam"
                phase_boundary = str(item.get("name", "")) == "nonlinear_phase_boundary_beam"
                reference_boundary = str(item.get("name", "")) == "reference_boundary_cost_beam"
                waterline_cost = str(item.get("name", "")) == "waterline_cost_beam"
                cheap_policy["beam_width"] = min(
                    10 if dense_boundary or phase_boundary or reference_boundary else 8 if waterline_cost else 5,
                    _int_hint(
                        cheap_policy.get("beam_width"),
                        10 if dense_boundary or phase_boundary or reference_boundary else 8 if waterline_cost else 5,
                    ),
                )
                cheap_policy["state_cap_per_node"] = min(
                    48 if dense_boundary or phase_boundary or reference_boundary else 32 if waterline_cost else 16,
                    _int_hint(
                        cheap_policy.get("state_cap_per_node"),
                        48 if dense_boundary or phase_boundary or reference_boundary else 32 if waterline_cost else 16,
                    ),
                )
                cheap_policy["max_scale_candidates"] = min(
                    96 if dense_boundary else 80 if phase_boundary or reference_boundary else 48 if waterline_cost else 32,
                    _int_hint(
                        cheap_policy.get("max_scale_candidates"),
                        96 if dense_boundary else 80 if phase_boundary or reference_boundary else 48 if waterline_cost else 32,
                    ),
                )
                cheap_policy["boundary_state_cap"] = min(
                    16 if dense_boundary else 12 if phase_boundary or reference_boundary else 8 if waterline_cost else 6,
                    _int_hint(
                        cheap_policy.get("boundary_state_cap"),
                        16 if dense_boundary else 12 if phase_boundary or reference_boundary else 8 if waterline_cost else 6,
                    ),
                )
                cheap_policy["bootstrap_penalty"] = max(
                    0.0 if _selection_objective(cheap_policy) == "cost" else 750_000_000.0,
                    min(
                        1_500_000_000.0,
                        _float_hint(
                            cheap_policy.get("bootstrap_penalty"),
                            250_000_000.0
                            if _selection_objective(cheap_policy) == "cost"
                            else 1_500_000_000.0,
                        ),
                    ),
                )
                cheap_policy["selection_bootstrap_penalty"] = max(
                    0.0 if _selection_objective(cheap_policy) == "cost" else 250_000_000.0,
                    min(
                        750_000_000.0,
                        _float_hint(
                            cheap_policy.get("selection_bootstrap_penalty"),
                            0.0
                            if _selection_objective(cheap_policy) == "cost"
                            else 750_000_000.0,
                        ),
                    ),
                )
                cheap_policy["selection_objective"] = str(
                    cheap_policy.get("selection_objective", "cost")
                )
                cheap["policy"] = cheap_policy
                filtered.append(cheap)
            continue
        filtered.append(item)
    if filtered:
        # Sampled MCTS uses a small action cap. Keep one direct budget solver in
        # the first capped actions so evolution sees real QBP coverage instead
        # of only fallback-repaired seed assignments.
        original_order = {id(item): idx for idx, item in enumerate(filtered)}

        def sampled_action_rank(item: dict[str, Any]) -> tuple[int, int]:
            policy = item.get("policy", item)
            if not isinstance(policy, dict):
                policy = {}
            name = str(item.get("name", ""))
            strategy = str(policy.get("strategy", ""))
            is_budget_beam = (
                name == "budget_fulfillment_beam"
                or (strategy == "latency_beam" and _bool_hint(policy.get("direct_budget_policy"), False))
            )
            if is_budget_beam:
                return (0, original_order[id(item)])
            if name == "wide_boundary_cost_beam":
                return (1, original_order[id(item)])
            if name == "waterline_cost_beam":
                return (2, original_order[id(item)])
            if name == "dense_boundary_cost_beam":
                return (3, original_order[id(item)])
            if name == "nonlinear_phase_boundary_beam":
                return (3, original_order[id(item)])
            if name == "reference_boundary_cost_beam":
                return (3, original_order[id(item)])
            if name == "profile_waterline_repair":
                return (4, original_order[id(item)])
            if name == "tuneinsight_avgcase_cost_beam":
                return (5, original_order[id(item)])
            if name == "tuneinsight_deferred_bootstrap_beam":
                return (6, original_order[id(item)])
            if name.startswith("estimator_relaxed_floor_"):
                return (7, original_order[id(item)])
            if name == "latency_mcts_repair":
                return (8, original_order[id(item)])
            if name == "component_budget_repair":
                return (9, original_order[id(item)])
            if name == "waterline_budget_repair":
                return (10, original_order[id(item)])
            if name == "minimal_bootstrap_repair":
                return (11, original_order[id(item)])
            if name == "strict_no_bootstrap" or _bool_hint(policy.get("forbid_bootstrap"), False):
                return (12, original_order[id(item)])
            return (13, original_order[id(item)])

        filtered.sort(key=sampled_action_rank)
    return filtered or raw_actions


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
    raw_max_scale_candidates = _int_hint(
        hints.get("max_scale_candidates"),
        48 if budget_aggressive else 24,
    )
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
        bounded["max_scale_candidates"] = min(
            64 if budget_aggressive else 32,
            raw_max_scale_candidates,
        )
        bounded["mcts_rollout_budget"] = min(
            64 if budget_aggressive else 24,
            _int_hint(bounded.get("mcts_rollout_budget"), 32 if budget_aggressive else 12),
        )
        bounded["mcts_action_cap"] = min(
            16 if budget_aggressive else 8,
            _int_hint(bounded.get("mcts_action_cap"), 12 if budget_aggressive else 6),
        )
        bounded["mcts_max_repair_bootstraps"] = min(
            128 if budget_aggressive else 8,
            _int_hint(bounded.get("mcts_max_repair_bootstraps"), 128 if budget_aggressive else 6),
        )
        bounded["boundary_state_cap"] = min(
            12 if budget_aggressive else 8,
            _int_hint(bounded.get("boundary_state_cap"), 8 if budget_aggressive else 6),
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


def _sampled_policy_limits(policy_bank_lightweight: bool = False) -> dict[str, int]:
    if policy_bank_lightweight:
        return {
            "max_scale_candidates": 16,
            "state_cap_per_node": 8,
            "beam_width": 4,
            "boundary_state_cap": 2,
            "mcts_rollout_budget": 4,
            "mcts_action_cap": 2,
            "mcts_max_repair_bootstraps": 4,
        }
    return {
        "max_scale_candidates": 48,
        "state_cap_per_node": 32,
        "beam_width": 8,
        "boundary_state_cap": 8,
        "mcts_rollout_budget": 8,
        "mcts_action_cap": 4,
        "mcts_max_repair_bootstraps": 32,
    }


def _clamp_sampled_policy_patch(
    policy: dict[str, Any],
    *,
    policy_bank_lightweight: bool = False,
) -> dict[str, Any]:
    clamped = dict(policy)
    limits = _sampled_policy_limits(policy_bank_lightweight)
    defaults = {
        "max_scale_candidates": 24,
        "state_cap_per_node": 24,
        "beam_width": 4,
        "boundary_state_cap": 6,
        "mcts_rollout_budget": 6,
        "mcts_action_cap": 3,
        "mcts_max_repair_bootstraps": 12,
    }
    for key, limit in limits.items():
        if key in clamped:
            clamped[key] = min(limit, _int_hint(clamped.get(key), defaults[key]))
    return clamped


def _clamp_sampled_policy_items(
    value: Any,
    *,
    policy_bank_lightweight: bool = False,
) -> Any:
    if not isinstance(value, list):
        return value
    clamped_items: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            clamped_items.append(item)
            continue
        updated = dict(item)
        policy = updated.get("policy")
        if isinstance(policy, dict):
            updated["policy"] = _clamp_sampled_policy_patch(
                policy,
                policy_bank_lightweight=policy_bank_lightweight,
            )
        else:
            updated = _clamp_sampled_policy_patch(
                updated,
                policy_bank_lightweight=policy_bank_lightweight,
            )
        clamped_items.append(updated)
    return clamped_items


def _clamp_sampled_action_presets(
    value: Any,
    *,
    policy_bank_lightweight: bool = False,
) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for name, preset in value.items():
            if not isinstance(preset, dict):
                result[name] = preset
                continue
            updated = dict(preset)
            policy = updated.get("policy")
            if isinstance(policy, dict):
                updated["policy"] = _clamp_sampled_policy_patch(
                    policy,
                    policy_bank_lightweight=policy_bank_lightweight,
                )
            else:
                updated = _clamp_sampled_policy_patch(
                    updated,
                    policy_bank_lightweight=policy_bank_lightweight,
                )
            result[str(name)] = updated
        return result
    if isinstance(value, list):
        return _clamp_sampled_policy_items(
            value,
            policy_bank_lightweight=policy_bank_lightweight,
        )
    return value


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
    reference_hints = _finalist_hints_for_full_bundle(initial_hints or {})
    sampled_reference = sampled_context.get("reference")
    sampled_reference_source = (
        str(sampled_reference.get("source", ""))
        if isinstance(sampled_reference, dict)
        else ""
    )
    if (
        isinstance(sampled_reference, dict)
        and sampled_reference
        and sampled_reference_source not in {"sampled_initial_seed", "context_collection_seed"}
    ):
        full_context["reference"] = {
            "final_latency_usec": sampled_reference.get("final_latency_usec"),
            "objective_cost_usec": sampled_reference.get(
                "objective_cost_usec", sampled_reference.get("final_latency_usec")
            ),
            "total_frontier_cost_usec": sampled_reference.get("total_frontier_cost_usec"),
            "sampled_dp_latency_usec": sampled_reference.get("sampled_dp_latency_usec"),
            "bootstrap_count": sampled_reference.get("bootstrap_count"),
            "rescale_count": sampled_reference.get("rescale_count"),
            "effective_qbp_digest": _effective_qbp_digest(sampled_reference.get("diagnostics", {})),
            "selected_path_digest": _selected_path_digest(
                sampled_reference, sampled_reference.get("diagnostics", {})
            ),
            "valid": sampled_reference.get("valid", False),
            "source": "sampled_initial_seed",
            "policy_summary": sampled_reference.get(
                "policy_summary",
                _compact_policy_summary(reference_hints),
            ),
        }
    else:
        reference = _evaluate_compile_hints(full_context, reference_hints, suppress_output=True)
        reference_diagnostics = reference.get("diagnostics", {})
        if not isinstance(reference_diagnostics, dict):
            reference_diagnostics = {}
        full_context["reference"] = {
            "final_latency_usec": reference.get("final_latency_usec"),
            "objective_cost_usec": reference.get(
                "objective_cost_usec", reference.get("final_latency_usec")
            ),
            "total_frontier_cost_usec": reference.get("total_frontier_cost_usec"),
            "bootstrap_count": reference.get("bootstrap_count"),
            "rescale_count": reference.get("rescale_count"),
            "valid": reference.get("valid", False),
            "effective_qbp_digest": _effective_qbp_digest(reference_diagnostics),
            "selected_path_digest": _selected_path_digest(reference, reference_diagnostics),
            "source": "initial_seed",
            "policy_summary": _compact_policy_summary(reference_hints),
        }
    full_context.setdefault("harness", {})["seed_baseline"] = dict(full_context["reference"])
    candidate_limit = max(
        params.openevolve_finalists,
        params.openevolve_finalists * 4
        if getattr(params, "openevolve_search_mode", "") == "bootstrap-mcts"
        else params.openevolve_finalists,
    )
    candidates = _discover_finalist_codes(output_dir, best_code, candidate_limit)
    summaries = []
    best = None
    duplicate_best = None
    estimator_records = []
    scale_floor_records = []
    result_cache: dict[str, dict[str, Any]] = {}
    seen_effective_paths: dict[str, int] = {}
    progress_status: dict[str, Any] = {}

    def write_progress() -> None:
        try:
            (finalist_dir / "full_bundle_progress.json").write_text(
                json.dumps(
                    {
                        "reference": full_context["reference"],
                        "candidates": summaries,
                        "active": progress_status,
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

    candidate_items: list[tuple[str, str | None, dict[str, Any] | None]] = [
        (f"candidate_{idx}", code, None)
        for idx, code in enumerate(candidates)
    ]
    initial_items: list[tuple[str, str | None, dict[str, Any] | None]] = (
        [("initial_seed", None, initial_hints)] if initial_hints is not None else []
    )
    if str(full_context.get("harness", {}).get("search_mode", "")) == "bootstrap-mcts":
        # The reference seed was already replayed before evolution and is the
        # fail-open fallback. Replaying it again as finalist 0 can dominate the
        # post-evolution wall time on large graphs while adding no selectable
        # latency-improving candidate.
        reference = full_context.get("reference", {})
        if initial_hints is not None and isinstance(reference, dict):
            reference_path_digest = str(reference.get("selected_path_digest") or "")[:24]
            reference_qbp_digest = str(reference.get("effective_qbp_digest") or "")[:24]
            seed_summary = {
                "index": len(summaries),
                "label": "initial_seed_reference",
                "valid": bool(reference.get("valid", False)),
                "reference_only": True,
                "final_latency_usec": reference.get("final_latency_usec"),
                "objective_cost_usec": reference.get(
                    "objective_cost_usec", reference.get("final_latency_usec")
                ),
                "total_frontier_cost_usec": reference.get("total_frontier_cost_usec"),
                "bootstrap_count": reference.get("bootstrap_count"),
                "rescale_count": reference.get("rescale_count"),
                "fallback_selected_budgets": 0,
                "fallback_selected_groups": 0,
                "candidate_qbp_coverage": 1.0 if reference.get("valid", False) else 0.0,
                "effective_qbp_digest": reference_qbp_digest,
                "selected_path_digest": reference_path_digest,
                "selected_path_changed_vs_seed": False,
                "policy_effect_summary": {
                    "seed_equivalent": True,
                    "source": "precomputed_reference",
                },
                "finalist_gate": {
                    "latency_reject": True,
                    "latency_improvement_reject": True,
                    "final_latency_improved_vs_seed": False,
                    "reference_objective_cost_usec": reference.get(
                        "objective_cost_usec", reference.get("final_latency_usec")
                    ),
                    "effective_duplicate": False,
                    "seed_equivalent_policy": True,
                    "selected_path_changed_vs_seed": False,
                    "target_bootstrap_count": _context_target_bootstrap_count(full_context),
                    "seed_bootstrap_count": _context_seed_bootstrap_count(full_context),
                },
            }
            summaries.append(seed_summary)
            if reference_path_digest:
                seen_effective_paths[reference_path_digest] = seed_summary["index"]
            write_progress()
        finalist_items = candidate_items
    else:
        finalist_items = candidate_items + initial_items
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
                    progress_status = {
                        "phase": "evaluating_full_bundle_finalist",
                        "index": summary_index,
                        "label": summary_label,
                        "variant": variant_label,
                        "hint_digest": hint_digest,
                        "started_at": time.time(),
                    }
                    write_progress()
                    result = _evaluate_compile_hints(full_context, finalist_hints, suppress_output=True)
                    progress_status = {
                        "phase": "finished_full_bundle_finalist",
                        "index": summary_index,
                        "label": summary_label,
                        "variant": variant_label,
                        "hint_digest": hint_digest,
                        "finished_at": time.time(),
                    }
                    diagnostics = result.get("diagnostics", {})
                    effective_path = _result_effective_summary(
                        full_context, result, diagnostics if isinstance(diagnostics, dict) else {}
                    )
                    policy_effect = _policy_effect_summary(
                        full_context,
                        hints if isinstance(hints, dict) else {},
                        finalist_hints,
                        "polybert-full",
                    )
                    summary = {
                        "index": summary_index,
                        "label": summary_label,
                        "valid": result["valid"],
                        "final_latency_usec": result["final_latency_usec"],
                        "objective_cost_usec": result.get(
                            "objective_cost_usec", result["final_latency_usec"]
                        ),
                        "total_frontier_cost_usec": result.get(
                            "total_frontier_cost_usec",
                            result.get("aggregated_partition_cost_usec"),
                        ),
                        "boundary_quality": result.get("boundary_quality", 0.0),
                        "bootstrap_count": result["bootstrap_count"],
                        "rescale_count": result["rescale_count"],
                        "fallback_selected_budgets": result["fallback_selected_budgets"],
                        "fallback_selected_groups": result.get("fallback_selected_groups", 0),
                        "candidate_qbp_coverage": result.get("candidate_qbp_coverage", 0.0),
                        "effective_qbp_digest": effective_path["effective_qbp_digest"],
                        "selected_path_digest": effective_path["selected_path_digest"],
                        "selected_path_changed_vs_seed": effective_path[
                            "selected_path_changed_vs_seed"
                        ],
                        "policy_effect_summary": policy_effect,
                        "selected_output_state": result["selected_output_state"],
                        "reserve_summary": result.get("reserve_summary", {}),
                        "bootstrap_locations": result.get("bootstrap_locations", {}),
                        "rescale_locations": result.get("rescale_locations", {}),
                        "bootstrap_location_delta": _location_count_delta(
                            result.get("bootstrap_locations", {}),
                            full_context.get("reference", {}).get("bootstrap_locations", {}),
                        ),
                        "rescale_location_delta": _location_count_delta(
                            result.get("rescale_locations", {}),
                            full_context.get("reference", {}).get("rescale_locations", {}),
                        ),
                    }
                    noise = _noise_estimator_for_finalist(full_context, result, params)
                    if noise is not None:
                        summary["noise_estimator"] = noise
                        estimator_records.append({"index": summary_index, "label": summary_label, **noise})
                    scale_floor_record = {
                        "index": summary_index,
                        "label": summary_label,
                        **_scale_floor_summary(params, finalist_hints, noise),
                    }
                    summary["scale_floor_summary"] = scale_floor_record
                    scale_floor_records.append(scale_floor_record)
                    result_cache[hint_digest] = {
                        "index": summary_index,
                        "result": result,
                        "noise": noise,
                        "summary": dict(summary),
                    }
                if "effective_qbp_digest" not in summary:
                    diagnostics = result.get("diagnostics", {})
                    effective_path = _result_effective_summary(
                        full_context, result, diagnostics if isinstance(diagnostics, dict) else {}
                    )
                    policy_effect = _policy_effect_summary(
                        full_context,
                        hints if isinstance(hints, dict) else {},
                        finalist_hints,
                        "polybert-full",
                    )
                    summary["effective_qbp_digest"] = effective_path["effective_qbp_digest"]
                    summary["selected_path_digest"] = effective_path["selected_path_digest"]
                    summary["selected_path_changed_vs_seed"] = effective_path[
                        "selected_path_changed_vs_seed"
                    ]
                    summary["policy_effect_summary"] = policy_effect
                result_diagnostics = result.get("diagnostics", {})
                has_boundary_groups = bool(
                    isinstance(result_diagnostics, dict)
                    and int(result_diagnostics.get("requested_boundary_groups", 0) or 0) > 0
                )
                dedupe_effective_paths = (
                    str(full_context.get("harness", {}).get("search_mode", "")) == "bootstrap-mcts"
                    and has_boundary_groups
                )
                effective_key = str(
                    summary.get("selected_path_digest")
                    or summary.get("effective_qbp_digest")
                    or hint_digest
                )
                if dedupe_effective_paths:
                    duplicate_of = seen_effective_paths.get(effective_key)
                    if duplicate_of is None:
                        seen_effective_paths[effective_key] = summary_index
                        effective_duplicate = False
                    else:
                        summary["effective_duplicate_of_index"] = duplicate_of
                        effective_duplicate = True
                else:
                    effective_duplicate = False
                seed_equivalent_policy = bool(
                    dedupe_effective_paths
                    and summary.get("policy_effect_summary", {}).get("seed_equivalent", False)
                )
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
                    seed_bootstrap_count = _context_seed_bootstrap_count(full_context)
                    bootstrap_delta_vs_seed = (
                        int(result["bootstrap_count"]) - seed_bootstrap_count
                        if seed_bootstrap_count is not None
                        else None
                    )
                    bootstrap_regression = int(
                        bootstrap_delta_vs_seed is not None and bootstrap_delta_vs_seed > 0
                    )
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
                    scale_gate = summary.get("scale_floor_summary", {})
                    relaxed_floor_reject = bool(
                        scale_gate.get("relaxed")
                        and (
                            params.noise_estimator != "finalists"
                            or noise is None
                            or not noise.get("valid", False)
                            or not math.isfinite(_finite_float(noise.get("output_margin_bits"), float("nan")))
                        )
                    )
                    relaxed_floor_reason = (
                        "relaxed scale floors require --noise-estimator finalists with a valid finite output margin"
                        if relaxed_floor_reject
                        else None
                    )
                    if relaxed_floor_reject and not profile_noise_reject:
                        profile_noise_reject = True
                        profile_noise_reason = relaxed_floor_reason
                    finalist_objective = {
                        "objective_cost_usec": result.get(
                            "objective_cost_usec",
                            result.get("final_latency_usec", float("inf")),
                        ),
                        "reference_objective_cost_usec": _baseline_objective_cost_usec(
                            full_context,
                            _finite_float(result.get("final_latency_usec"), float("inf")),
                        ),
                    }
                    finalist_latency_improved = _objective_improved_vs_seed(
                        finalist_objective
                    )
                    latency_improvement_reject = int(not finalist_latency_improved)
                    latency_reject = int(
                        latency_improvement_reject
                        or (
                            latency_target is not None
                            and float(result["final_latency_usec"]) > latency_target
                        )
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
                        "latency_improvement_reject": bool(latency_improvement_reject),
                        "final_latency_improved_vs_seed": bool(finalist_latency_improved),
                        "reference_objective_cost_usec": finalist_objective[
                            "reference_objective_cost_usec"
                        ],
                        "effective_duplicate": bool(effective_duplicate),
                        "seed_equivalent_policy": bool(seed_equivalent_policy),
                        "selected_path_changed_vs_seed": bool(
                            summary.get("selected_path_changed_vs_seed", False)
                        ),
                        "output_margin_target_bits": margin_target,
                        "output_margin_reject": bool(margin_reject),
                        "forced_bootstrap_floor": forced_bootstrap_floor,
                        "target_bootstrap_count": target_bootstrap_count,
                        "target_bootstrap_excess": bootstrap_excess,
                        "seed_bootstrap_count": seed_bootstrap_count,
                        "bootstrap_delta_vs_seed": bootstrap_delta_vs_seed,
                        "bootstrap_regression": bool(bootstrap_regression),
                        "bootstrap_reject": bool(bootstrap_reject),
                        "plaintext_quality_reject": plaintext_quality_reject,
                        "plaintext_quality_reason": plaintext_quality_reason,
                        "profile_noise_reject": profile_noise_reject,
                        "profile_noise_reason": profile_noise_reason,
                        "relaxed_floor_reject": relaxed_floor_reject,
                        "relaxed_floor_reason": relaxed_floor_reason,
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
                        int(effective_duplicate),
                        int(seed_equivalent_policy),
                        bootstrap_reject,
                        latency_reject,
                        float(result["final_latency_usec"]),
                        bootstrap_excess,
                        bootstrap_regression,
                        int(result.get("rescale_count", 0) or 0),
                        -float(margin),
                        summary_index,
                        finalist_hints,
                        summary,
                    )
                    if effective_duplicate or seed_equivalent_policy:
                        if duplicate_best is None or item[:16] < duplicate_best[:16]:
                            duplicate_best = item
                    elif best is None or item[:16] < best[:16]:
                        best = item
            except Exception as exc:
                progress_status = {
                    "phase": "failed_full_bundle_finalist",
                    "index": summary_index,
                    "label": summary_label,
                    "variant": variant_label,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "finished_at": time.time(),
                }
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
                "selected_index": None if best is None else best[15],
                "effective_dedupe": {
                    "unique_effective_paths": len(seen_effective_paths),
                    "duplicate_candidates": len(
                        [item for item in summaries if item.get("effective_duplicate_of_index") is not None]
                    ),
                    "duplicate_best_index": None if duplicate_best is None else duplicate_best[15],
                    "duplicate_best_not_selectable": bool(duplicate_best is not None),
                },
                "finalist_gate": {
                    "latency_target_usec": _finalist_latency_target_usec(full_context),
                    "output_margin_target_bits": _finalist_output_margin_target_bits(full_context),
                    "forced_bootstrap_floor": _finalist_forced_bootstrap_floor(full_context),
                    "target_bootstrap_count": _context_target_bootstrap_count(full_context),
                    "seed_bootstrap_count": _context_seed_bootstrap_count(full_context),
                },
                "noise_estimator": {
                    "mode": params.noise_estimator,
                    "min_output_margin_bits": params.noise_estimator_min_output_margin_bits,
                    "records": estimator_records,
                },
                "scale_floor": {
                    "policy": getattr(params, "scale_floor_policy", "waterline"),
                    "records": scale_floor_records,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if best is None:
        if duplicate_best is not None:
            _write_finalist_rejection_summary(
                finalist_dir,
                duplicate_best[17],
                {
                    "effective_duplicate_reject": True,
                    "seed_equivalent_policy_reject": bool(
                        duplicate_best[17].get("finalist_gate", {}).get(
                            "seed_equivalent_policy"
                        )
                    ),
                    "selected_index": duplicate_best[15],
                },
            )
        return _bounded_fail_open_hints(initial_hints)
    write_noise_summary(
        finalist_dir / "scale_floor_summary.json",
        {
            "selected_index": best[15],
            "selected": best[17].get("scale_floor_summary", {}),
            "records": scale_floor_records,
        },
    )
    fallback_reject = bool(best[5])
    if (
        best[0]
        or best[1]
        or fallback_reject
        or bool(best[9])
        or ((best[2] or best[3]) and params.noise_estimator == "finalists")
    ):
        _write_finalist_rejection_summary(
            finalist_dir,
            best[17],
            {
                "plaintext_quality_reject": bool(best[0]),
                "profile_noise_reject": bool(best[1]),
                "fallback_selected_reject": fallback_reject,
                "latency_reject": bool(best[9]),
                "latency_improvement_reject": bool(
                    best[17].get("finalist_gate", {}).get(
                        "latency_improvement_reject"
                    )
                ),
                "noise_warning_reject": bool(best[2]),
                "noise_estimator_reject": bool(best[3]),
                "relaxed_floor_reject": bool(
                    best[17].get("finalist_gate", {}).get("relaxed_floor_reject")
                ),
                "selected_index": best[15],
            },
        )
        return _bounded_fail_open_hints(initial_hints)
    write_noise_summary(
        finalist_dir / "noise_estimator_summary.json",
        {
            "selected_index": best[15],
            "selected": best[17],
            "records": estimator_records,
        },
    )
    return best[16]


def _context_collection_seed_hints(
    initial_hints: dict[str, Any], params: Params
) -> dict[str, Any]:
    """Use a cheap QBP-shaped replay to discover sampled boundary groups.

    Positive bootstrap-MCTS runs evolve a broad action policy. Replaying that
    broad policy over the whole graph just to discover sampled boundary groups
    is expensive and can dominate short OpenEvolve runs. A plain deterministic
    smoke policy is too narrow, though: it can collapse the cached context to a
    single boundary group and leave OpenEvolve with no useful placement surface.
    Collect tasks with a lightweight bootstrap-MCTS policy so the cached sample
    still has Orbit-shaped boundary groups, then evaluate the actual seed on
    those cached tasks.
    """

    if (
        getattr(params, "openevolve_search_mode", "") == "bootstrap-mcts"
        and getattr(params, "openevolve_eval_suite", "polybert-sampled") != "polybert-full"
    ):
        hints = _bootstrap_mcts_seed_policy(params)
        hints.update(
            {
                "context_collection_seed": True,
                "policy_bank_lightweight": True,
                "budget_aggressive": True,
                "allow_seed_fallback": True,
                "allow_bootstrap": True,
                "include_seed_repair_actions": False,
                "mcts_rollout_budget": 4,
                "mcts_action_cap": 2,
                "mcts_max_repair_bootstraps": 4,
                "boundary_state_cap": 2,
                "max_scale_candidates": 16,
                "state_cap_per_node": 8,
                "beam_width": 4,
                "mcts_action_allowlist": [
                    "budget_fulfillment_beam",
                    "wide_boundary_cost_beam",
                ],
            }
        )
        return hints
    return initial_hints


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

    if initial_hints is not None:
        if _bool_hint(initial_hints.get("policy_bank_full_validated_initial"), False):
            fallback = _policy_bank_full_compile_replay_hints(initial_hints)
            fallback.pop("policy_bank_lightweight", None)
            fallback["fail_open_reason"] = "policy_bank_full_validated_seed"
            return fallback
    fallback = _zero_iteration_portfolio_hints()
    fallback["fail_open_reason"] = "zero_iteration_seed_portfolio"
    if initial_hints is not None:
        fallback["recovered_from_initial_digest"] = _hint_digest(initial_hints)
        if _bool_hint(initial_hints.get("policy_bank_validated_initial"), False):
            fallback["rejected_initial_reason"] = "sampled_policy_bank_not_full_validated"
            if initial_hints.get("policy_bank_selected_label") is not None:
                fallback["rejected_policy_bank_selected_label"] = str(
                    initial_hints.get("policy_bank_selected_label")
                )
    return fallback


def _full_validate_policy_bank_seed(
    root: Path,
    sampled_context: dict[str, Any],
    params: Params,
    initial_hints: dict[str, Any] | None,
    sampled_reject_reason: str,
) -> dict[str, Any] | None:
    """Promote a sampled policy-bank seed only after a full replay succeeds.

    The policy-bank prepass is intentionally sampled and can find useful path
    changes before OpenEvolve has a better candidate. It is not sufficient for
    final MLIR generation, though: a sampled winner may omit downstream
    boundary states that the full Orbit DP later needs. When the sampled best is
    seed-equivalent to the policy-bank initial program, run one full replay and
    only return those hints if the full assignment is valid.
    """

    if not initial_hints:
        return None
    if not _bool_hint(initial_hints.get("policy_bank_validated_initial"), False):
        return None
    if _bool_hint(initial_hints.get("policy_bank_full_validated_initial"), False):
        return initial_hints
    if sampled_reject_reason not in {
        "sampled_best_seed_equivalent_path",
        "sampled_best_not_latency_improved",
    }:
        return None
    if int(getattr(params, "openevolve_finalists", 0) or 0) <= 0:
        return None

    finalist_dir = root / "finalists"
    finalist_dir.mkdir(parents=True, exist_ok=True)
    full_context = json.loads(json.dumps(sampled_context))
    full_context.setdefault("harness", {})["eval_suite"] = "polybert-full"
    summary_path = finalist_dir / "policy_bank_full_validation.json"
    progress_path = finalist_dir / "policy_bank_full_validation_progress.json"
    records: list[dict[str, Any]] = []
    best: tuple[float, int, int, dict[str, Any], dict[str, Any]] | None = None
    variants = _policy_bank_full_validation_variants(initial_hints)
    timeout_sec = _policy_bank_full_validation_timeout_sec(params)
    active: dict[str, Any] = {}

    def write_progress() -> None:
        progress = {
            "valid": best is not None,
            "sampled_reject_reason": sampled_reject_reason,
            "timeout_sec": timeout_sec,
            "active": active,
            "records": records,
            "updated_at": time.time(),
        }
        progress_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    for idx, (label, replay_hints) in enumerate(variants):
        active.clear()
        active.update(
            {
                "index": idx,
                "label": label,
                "status": "running",
                "started_at": time.time(),
            }
        )
        write_progress()
        print(
            "OpenEvolve compile harness: full-validating sampled policy-bank "
            f"variant {idx + 1}/{len(variants)} {label} timeout={timeout_sec}s.",
            flush=True,
        )
        record = _run_policy_bank_full_validation_variant(
            idx,
            label,
            full_context,
            replay_hints,
            sampled_reject_reason,
            timeout_sec,
        )
        record["elapsed_sec"] = time.time() - float(active.get("started_at", time.time()))
        records.append(record)
        active["status"] = "done"
        active["valid"] = bool(record.get("valid", False))
        active["error"] = record.get("error")
        active["elapsed_sec"] = record["elapsed_sec"]
        write_progress()
        print(
            "OpenEvolve compile harness: full-validation "
            f"{label} valid={record.get('valid')} "
            f"latency={record.get('final_latency_usec')} "
            f"bootstraps={record.get('bootstrap_count')} "
            f"error={record.get('error')}.",
            flush=True,
        )
        valid = bool(record.get("valid", False))
        fallback_groups = int(record.get("fallback_selected_groups", 0) or 0)
        fallback_budgets = int(record.get("fallback_selected_budgets", 0) or 0)
        latency = _finite_float(record.get("final_latency_usec"), float("inf"))
        if valid and fallback_groups == 0 and fallback_budgets == 0 and math.isfinite(latency):
            item = (
                latency,
                int(record.get("bootstrap_count", 0) or 0),
                int(record.get("rescale_count", 0) or 0),
                replay_hints,
                record,
            )
            if best is None or item[:3] < best[:3]:
                best = item

    summary = {
        "valid": best is not None,
        "sampled_reject_reason": sampled_reject_reason,
        "selected_index": None if best is None else best[4]["index"],
        "selected_label": None if best is None else best[4]["label"],
        "records": records,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if best is None:
        return None

    result_record = best[4]
    promoted = deepcopy(best[3])
    promoted["policy_bank_full_validated_initial"] = True
    promoted["policy_bank_full_validation_variant"] = result_record["label"]
    promoted["policy_bank_full_validation_latency_usec"] = result_record.get("final_latency_usec")
    promoted["policy_bank_full_validation_bootstrap_count"] = result_record.get("bootstrap_count")
    promoted["policy_bank_full_validation_rescale_count"] = result_record.get("rescale_count")
    promoted["policy_bank_full_validation_digest"] = result_record["selected_path_digest"]
    return promoted


def _policy_bank_full_validation_variants(
    initial_hints: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    bounded = _policy_bank_full_compile_replay_hints(initial_hints)
    broad = deepcopy(initial_hints)
    broad.pop("portfolio", None)
    broad.pop("policy_bank_lightweight", None)
    broad["policy_bank_full_compile_replay"] = True
    broad["policy_bank_full_validation_variant"] = "broad_action_replay"
    broad["allow_seed_fallback"] = True
    broad["mcts_rollout_budget"] = max(4, _int_hint(broad.get("mcts_rollout_budget"), 4))
    broad["mcts_action_cap"] = max(2, _int_hint(broad.get("mcts_action_cap"), 2))
    broad["boundary_state_cap"] = max(6, _int_hint(broad.get("boundary_state_cap"), 6))
    broad["max_scale_candidates"] = max(24, _int_hint(broad.get("max_scale_candidates"), 24))
    broad["state_cap_per_node"] = max(16, _int_hint(broad.get("state_cap_per_node"), 16))
    broad["mcts_action_allowlist"] = [
        "budget_fulfillment_beam",
        "dense_boundary_cost_beam",
        "wide_boundary_cost_beam",
    ]
    if not broad.get("mcts_actions") and broad.get("mcts_action_presets"):
        presets = broad.get("mcts_action_presets")
        if isinstance(presets, dict):
            broad["mcts_actions"] = [
                {"name": name, **value}
                for name, value in presets.items()
                if isinstance(value, dict)
            ]

    return [
        ("sampled_overlay_bounded", bounded),
        ("broad_action_replay", broad),
    ]


def _policy_bank_full_compile_replay_hints(hints: dict[str, Any]) -> dict[str, Any]:
    """Bound a validated sampled policy-bank winner for production replay.

    Sampled policy-bank evaluation tries several MCTS actions per boundary group
    so it can discover which direct budget policy wins. The production Orbit
    compile intentionally runs a single action per group for speed. Preserve the
    sampled winner by adding boundary-specific action allowlists for the costly
    groups where the trace shows a non-default action won, and keep the global
    replay bounded to one deterministic action.
    """

    replay = deepcopy(hints)
    replay.pop("portfolio", None)
    replay.pop("policy_bank_lightweight", None)
    replay["policy_bank_full_compile_replay"] = True
    replay["allow_seed_fallback"] = True
    replay["mcts_rollout_budget"] = 1
    replay["mcts_action_cap"] = 1
    replay["mcts_prior_order"] = True

    default_action = _first_available_action_name(
        replay,
        preferred=(
            "budget_fulfillment_beam",
            "wide_boundary_cost_beam",
            "dense_boundary_cost_beam",
            "waterline_cost_beam",
        ),
    )
    if default_action:
        replay["mcts_action_allowlist"] = [default_action]

    overlays = list(replay.get("boundary_group_policies", []) or [])
    for item in list(replay.get("policy_bank_selected_top_costly_boundary_groups", []) or [])[:8]:
        if not isinstance(item, dict):
            continue
        selector = item.get("group_key")
        if not isinstance(selector, dict):
            continue
        action_name = _dominant_boundary_mcts_action(item.get("selected_source_counts", {}))
        if not action_name or action_name == default_action:
            continue
        overlays.append(
            {
                "selector": dict(selector),
                "policy": {
                    "mcts_action_allowlist": [action_name],
                    "mcts_action_cap": 1,
                    "mcts_rollout_budget": 1,
                },
            }
        )
    if overlays:
        replay["boundary_group_policies"] = overlays[:32]
    replay["policy_bank_full_compile_overlay_count"] = len(overlays)
    return replay


def _dominant_boundary_mcts_action(selected_source_counts: Any) -> str | None:
    if not isinstance(selected_source_counts, dict):
        return None
    best_name = None
    best_count = 0
    for source, count in selected_source_counts.items():
        action_name = _boundary_mcts_action_name_from_source(str(source))
        if not action_name:
            continue
        try:
            numeric = int(count or 0)
        except (TypeError, ValueError):
            numeric = 0
        if numeric > best_count:
            best_name = action_name
            best_count = numeric
    return best_name


def _boundary_mcts_action_name_from_source(source: str) -> str | None:
    parts = str(source).split(":")
    if len(parts) >= 3 and parts[0] == "candidate" and parts[1] == "boundary_mcts":
        return parts[2]
    return None


def _first_available_action_name(
    hints: dict[str, Any],
    *,
    preferred: tuple[str, ...],
) -> str | None:
    names: set[str] = set()
    raw_actions = hints.get("mcts_actions")
    if isinstance(raw_actions, list):
        for item in raw_actions:
            if isinstance(item, dict) and item.get("name") is not None:
                names.add(str(item.get("name")))
    presets = hints.get("mcts_action_presets")
    if isinstance(presets, dict):
        names.update(str(name) for name in presets)
    elif isinstance(presets, list):
        for item in presets:
            if isinstance(item, dict) and item.get("name") is not None:
                names.add(str(item.get("name")))
    allowlist = [
        str(item)
        for item in hints.get("mcts_action_allowlist", []) or []
        if str(item)
    ]
    if allowlist:
        names = {name for name in names if name in allowlist} or set(allowlist)
    for name in preferred:
        if name in names:
            return name
    return sorted(names)[0] if names else None


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
    if _finite_float(metrics.get("experience_surrogate_score"), 0.0) > 0.0 and (
        _finite_float(metrics.get("experience_probe_promoted"), 0.0) <= 0.0
    ):
        return "sampled_best_surrogate_only"
    if (
        _finite_float(metrics.get("experience_probe_task_count"), 0.0) > 0.0
        and _finite_float(metrics.get("latency_only_correct"), 0.0) <= 0.0
    ):
        return "sampled_best_probe_only_not_selectable"
    validity = _finite_float(metrics.get("validity"), 0.0)
    effective_validity = _finite_float(metrics.get("effective_validity"), validity)
    combined_score = _finite_float(metrics.get("combined_score"), 0.0)
    if validity <= 0.0 and effective_validity <= 0.0 and combined_score < 1.0:
        return "sampled_best_solved_no_budgets"
    if (
        _finite_float(metrics.get("seed_equivalent_path"), 0.0) >= 1.0
        and _finite_float(metrics.get("objective_improved_vs_seed"), 0.0) <= 0.0
    ):
        return "sampled_best_seed_equivalent_path"
    if (
        _finite_float(metrics.get("latency_only_correct"), 0.0) >= 1.0
        and _finite_float(metrics.get("objective_improved_vs_seed"), 0.0) <= 0.0
    ):
        return "sampled_best_not_latency_improved"
    candidate_qbp_coverage = _finite_float(metrics.get("candidate_qbp_coverage"), 0.0)
    candidate_groups = _finite_float(metrics.get("candidate_solved_boundary_groups"), 0.0)
    fallback_budgets = _finite_float(metrics.get("fallback_selected_budgets"), 0.0)
    fallback_groups = _finite_float(metrics.get("fallback_selected_groups"), 0.0)
    if (
        candidate_qbp_coverage <= 0.0
        and candidate_groups <= 0.0
        and (fallback_budgets > 0.0 or fallback_groups > 0.0)
    ):
        return "sampled_best_has_no_direct_qbp_coverage"
    return None


def _promotion_candidate_limit(params: Params) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PROMOTION_CANDIDATES", "").strip()
    try:
        value = int(raw) if raw else 16
    except ValueError:
        value = 16
    return max(0, min(128, value))


def _promotion_timeout_sec(params: Params) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PROMOTION_TIMEOUT_SEC", "").strip()
    try:
        value = int(raw) if raw else 300
    except ValueError:
        value = 300
    return max(0, min(86_400, value))


def _promotion_eval_timeout_sec(params: Params) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PROMOTION_EVAL_TIMEOUT_SEC", "").strip()
    try:
        value = int(raw) if raw else int(getattr(params, "openevolve_evaluator_timeout_sec", 180) or 180)
    except (TypeError, ValueError):
        value = 180
    return max(0, min(3_600, value))


def _strategy_discovery_enabled() -> bool:
    raw = os.environ.get("ORBIT_OPENEVOLVE_STRATEGY_DISCOVERY", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _strategy_discovery_limit() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_STRATEGY_DISCOVERY_VARIANTS", "").strip()
    try:
        value = int(raw) if raw else 8
    except ValueError:
        value = 8
    return max(0, min(64, value))


def _strategy_discovery_timeout_sec(promotion_timeout_sec: int) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_STRATEGY_DISCOVERY_TIMEOUT_SEC", "").strip()
    try:
        value = int(raw) if raw else min(120, max(30, promotion_timeout_sec // 3))
    except ValueError:
        value = min(120, max(30, promotion_timeout_sec // 3))
    return max(0, min(max(0, promotion_timeout_sec), value))


def _strategy_discovery_eval_timeout_sec(promotion_eval_timeout_sec: int) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_STRATEGY_DISCOVERY_EVAL_TIMEOUT_SEC", "").strip()
    try:
        value = int(raw) if raw else min(15, max(5, promotion_eval_timeout_sec // 3))
    except ValueError:
        value = min(15, max(5, promotion_eval_timeout_sec // 3))
    return max(0, min(max(0, promotion_eval_timeout_sec), value))


def _promotion_bounded_retry_enabled() -> bool:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PROMOTION_BOUNDED_RETRY", "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _promotion_probe_limit() -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PROMOTION_PROBE_TASKS", "").strip()
    try:
        value = int(raw) if raw else 1
    except ValueError:
        value = 1
    return max(1, min(16, value))


def _promotion_probe_tasks(context: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    from_metrics = _promotion_probe_tasks_from_seed_metrics(context)
    if from_metrics is not None:
        return from_metrics
    old_value = os.environ.get("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_TASKS")
    os.environ["ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_TASKS"] = str(_promotion_probe_limit())
    try:
        selected, reference_cost = _experience_probe_tasks(context)
    finally:
        if old_value is None:
            os.environ.pop("ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_TASKS", None)
        else:
            os.environ["ORBIT_OPENEVOLVE_EXPERIENCE_PROBE_TASKS"] = old_value
    if selected and not math.isfinite(reference_cost):
        reference_cost = _promotion_fallback_probe_reference_cost(context, len(selected))
    return selected, reference_cost


def _promotion_fallback_probe_reference_cost(
    context: dict[str, Any],
    selected_task_count: int,
) -> float:
    harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
    for source in (
        harness.get("sampled_seed_baseline"),
        context.get("reference"),
        harness.get("seed_baseline"),
    ):
        if not isinstance(source, dict):
            continue
        for key in ("sampled_dp_latency_usec", "objective_cost_usec", "final_latency_usec"):
            value = _finite_float(source.get(key), float("inf"))
            if math.isfinite(value) and value > 0:
                return float(value)
    full_reference = _promotion_reference_latency(context)
    tasks = context.get("sampled_budget_tasks")
    task_count = len(tasks) if isinstance(tasks, list) and tasks else max(1, selected_task_count)
    if math.isfinite(full_reference) and full_reference > 0:
        return float(full_reference * max(1, selected_task_count) / max(1, task_count))
    return float("inf")


def _promotion_result_direct_valid(result: dict[str, Any]) -> bool:
    if not bool(result.get("valid", False)):
        return False
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    if _finite_float(result.get("boundary_group_validity"), 0.0) < 1.0:
        return False
    if _finite_float(result.get("candidate_qbp_coverage"), 0.0) < 1.0:
        return False
    if int(result.get("fallback_selected_budgets", 0) or 0) > 0:
        return False
    if int(result.get("fallback_selected_groups", 0) or 0) > 0:
        return False
    if int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0) > 0:
        return False
    if int(result.get("invalid_boundary_groups", 0) or 0) > 0:
        return False
    if int(diagnostics.get("invalid_boundary_groups", 0) or 0) > 0:
        return False
    return True


def _openevolve_qbp_engine(params: Params, hints: dict[str, Any] | None = None) -> str:
    if isinstance(hints, dict) and hints.get("qbp_engine") in {"mcts", "dp"}:
        return str(hints["qbp_engine"])
    return str(getattr(params, "openevolve_qbp_engine", "mcts"))


def _use_qbp_dp_engine(params: Params, hints: dict[str, Any] | None = None) -> bool:
    return (
        getattr(params, "openevolve_search_mode", "bootstrap-mcts") == "bootstrap-mcts"
        and _openevolve_qbp_engine(params, hints) == "dp"
    )


def _promotion_latency(result: dict[str, Any]) -> float:
    for key in ("sampled_dp_latency_usec", "objective_cost_usec", "final_latency_usec"):
        value = _finite_float(result.get(key), float("inf"))
        if math.isfinite(value) and value > 0:
            return float(value)
    return float("inf")


def _promotion_reference_latency(context: dict[str, Any]) -> float:
    fallback = _finite_float(
        context.get("reference", {}).get("sampled_dp_latency_usec")
        if isinstance(context.get("reference"), dict)
        else None,
        float("inf"),
    )
    value = _baseline_objective_cost_usec(context, fallback)
    return float(value) if math.isfinite(value) and value > 0 else float("inf")


def _promotion_record_summary(
    *,
    index: int,
    stage: str,
    hints: dict[str, Any] | None,
    result: dict[str, Any] | None = None,
    reason: str = "",
    code_digest: str = "",
) -> dict[str, Any]:
    result = result if isinstance(result, dict) else {}
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return {
        "index": int(index),
        "stage": stage,
        "reason": reason,
        "code_digest": code_digest,
        "valid": bool(result.get("valid", False)),
        "direct_valid": bool(_promotion_result_direct_valid(result)) if result else False,
        "latency_usec": _promotion_latency(result) if result else None,
        "candidate_qbp_coverage": _finite_float(
            result.get("candidate_qbp_coverage"), 0.0
        )
        if result
        else 0.0,
        "boundary_group_validity": _finite_float(
            result.get("boundary_group_validity"), 0.0
        )
        if result
        else 0.0,
        "fallback_selected_budgets": int(result.get("fallback_selected_budgets", 0) or 0)
        if result
        else 0,
        "fallback_selected_groups": int(result.get("fallback_selected_groups", 0) or 0)
        if result
        else 0,
        "selected_path_digest": str(result.get("sampled_selected_path_digest", "")),
        "bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0)
        if result
        else 0.0,
        "rescale_count": _finite_float(result.get("rescale_count"), 0.0)
        if result
        else 0.0,
        "sampled_task_cache_hits": int(result.get("sampled_task_cache_hits", 0) or 0)
        if result
        else 0,
        "sampled_task_cache_misses": int(result.get("sampled_task_cache_misses", 0) or 0)
        if result
        else 0,
        "timed_out": bool(result.get("timed_out", False)) if result else False,
        "timeout_sec": result.get("timeout_sec") if result else None,
        "error": str(result.get("error", ""))[:500] if result else "",
        "worker_exitcode": result.get("worker_exitcode") if result else None,
        "selected_source_counts": dict(diagnostics.get("selected_source_counts", {}) or {}),
        "policy_summary": _compact_policy_summary(hints or {}),
    }


def _sampled_tasks_digest(tasks: list[dict[str, Any]]) -> str:
    compact = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        compact.append(
            {
                "index": _safe_int(task.get("index"), -1),
                "group_keys": task.get("group_keys", []),
                "seed_metrics": task.get("seed_metrics", {}),
            }
        )
    return _hint_digest(compact)


def _source_counts_digest(counts: dict[str, Any]) -> str:
    normalized = {
        str(key): int(value or 0)
        for key, value in dict(counts or {}).items()
        if int(value or 0) != 0
    }
    return _hint_digest(normalized)


def _selected_source_counts_from_result(result: dict[str, Any]) -> dict[str, int]:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return {
        str(key): int(value or 0)
        for key, value in dict(diagnostics.get("selected_source_counts", {}) or {}).items()
    }


def _counter_delta(candidate: dict[str, int], reference: dict[str, int]) -> dict[str, Any]:
    keys = sorted(set(candidate) | set(reference))
    delta = {
        key: int(candidate.get(key, 0)) - int(reference.get(key, 0))
        for key in keys
        if int(candidate.get(key, 0)) != int(reference.get(key, 0))
    }
    return {
        "delta": delta,
        "delta_nonzero": bool(delta),
        "candidate_total": int(sum(int(value or 0) for value in candidate.values())),
        "reference_total": int(sum(int(value or 0) for value in reference.values())),
    }


def _probe_seed_summary(
    selected_probe_tasks: list[dict[str, Any]],
    reference_latency_usec: float,
) -> dict[str, Any]:
    path_parts: list[str] = []
    source_counts: Counter[str] = Counter()
    group_keys: list[Any] = []
    bootstrap = 0.0
    rescale = 0.0
    for task in selected_probe_tasks:
        if not isinstance(task, dict):
            continue
        metric = task.get("seed_metrics")
        if isinstance(metric, dict):
            digest = str(metric.get("selected_path_digest", ""))
            if digest:
                path_parts.append(digest)
            for key, value in dict(metric.get("selected_source_counts", {}) or {}).items():
                source_counts[str(key)] += int(value or 0)
            bootstrap += _finite_float(metric.get("bootstrap_count"), 0.0)
            rescale += _finite_float(metric.get("rescale_count"), 0.0)
        group_keys.extend(list(task.get("group_keys", []) or []))
    selected_digest = (
        path_parts[0]
        if len(path_parts) == 1
        else _hint_digest(path_parts) if path_parts else ""
    )
    return {
        "selected_path_digest": selected_digest,
        "selected_source_counts": dict(source_counts),
        "latency_usec": float(reference_latency_usec),
        "bootstrap_count": float(bootstrap),
        "rescale_count": float(rescale),
        "group_keys": group_keys[:16],
        "task_digest": _sampled_tasks_digest(selected_probe_tasks),
    }


def _first_changed_boundary_group(
    seed_summary: dict[str, Any],
    result: dict[str, Any],
) -> Any:
    diagnostics = result.get("diagnostics", {})
    if isinstance(diagnostics, dict):
        for item in diagnostics.get("boundary_group_summaries", []) or []:
            if isinstance(item, dict):
                return item.get("group_key") or item.get("key") or item
    groups = seed_summary.get("group_keys", [])
    if isinstance(groups, list) and groups:
        return groups[0]
    return None


def _path_differential_summary(
    seed_summary: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    candidate_digest = str(result.get("sampled_selected_path_digest", ""))
    seed_digest = str(seed_summary.get("selected_path_digest", ""))
    candidate_sources = _selected_source_counts_from_result(result)
    seed_sources = {
        str(key): int(value or 0)
        for key, value in dict(seed_summary.get("selected_source_counts", {}) or {}).items()
    }
    candidate_latency = _promotion_latency(result)
    seed_latency = _finite_float(seed_summary.get("latency_usec"), float("inf"))
    source_delta = _counter_delta(candidate_sources, seed_sources)
    selected_path_changed = bool(candidate_digest and candidate_digest != seed_digest)
    source_changed = bool(source_delta.get("delta_nonzero", False))
    direct_valid = _promotion_result_direct_valid(result)
    candidate_solved = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
    reason = "unknown"
    if bool(result.get("timed_out", False)):
        reason = "timed_out_before_qbp"
    elif not direct_valid and candidate_solved <= 0:
        reason = "invalid_no_candidate_attempts"
    elif not selected_path_changed:
        reason = "same_action_order" if not source_changed else "same_boundary_states"
    elif math.isfinite(candidate_latency) and math.isfinite(seed_latency):
        if math.isclose(candidate_latency, seed_latency, rel_tol=1e-12, abs_tol=1e-9):
            reason = "probe_tied_seed"
        elif candidate_latency > seed_latency:
            reason = "changed_but_slower"
        else:
            reason = "changed_and_faster"
    elif selected_path_changed:
        reason = "changed_path_unknown_cost"
    return {
        "collapse_reason": reason,
        "seed_selected_path_digest": seed_digest,
        "candidate_selected_path_digest": candidate_digest,
        "selected_path_changed": bool(selected_path_changed),
        "selected_source_counts_changed": bool(source_changed),
        "seed_source_counts": seed_sources,
        "candidate_source_counts": candidate_sources,
        "source_count_delta": source_delta,
        "seed_latency_usec": seed_latency,
        "candidate_latency_usec": candidate_latency,
        "latency_delta_usec": (
            float(candidate_latency - seed_latency)
            if math.isfinite(candidate_latency) and math.isfinite(seed_latency)
            else None
        ),
        "seed_bootstrap_count": _finite_float(seed_summary.get("bootstrap_count"), 0.0),
        "candidate_bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0),
        "seed_rescale_count": _finite_float(seed_summary.get("rescale_count"), 0.0),
        "candidate_rescale_count": _finite_float(result.get("rescale_count"), 0.0),
        "first_changed_boundary_group": _first_changed_boundary_group(seed_summary, result),
    }


def _promotion_path_source_digest(result: dict[str, Any]) -> str:
    return _hint_digest(
        {
            "selected_path_digest": str(result.get("sampled_selected_path_digest", "")),
            "selected_source_counts": _selected_source_counts_from_result(result),
        }
    )


def _dp_probe_reason(diff: dict[str, Any], direct_valid: bool, improved: bool) -> str:
    if not direct_valid:
        return "dp_incomplete"
    if improved:
        return "dp_changed_faster"
    reason = str(diff.get("collapse_reason", ""))
    if reason in {"same_action_order", "same_boundary_states", "probe_tied_seed"}:
        return "dp_seed_equivalent"
    if reason == "changed_but_slower":
        return "dp_changed_slower"
    return reason or "dp_not_latency_improved"


def _promotion_timeout_context(
    hints: dict[str, Any],
    selected_probe_tasks: list[dict[str, Any]],
    *,
    place_timeout_sec: int,
    qbp_timeout_sec: int,
    total_timeout_sec: int,
    last_phase: str,
) -> dict[str, Any]:
    return {
        "place_timeout_sec": int(place_timeout_sec),
        "qbp_solve_timeout_sec": int(qbp_timeout_sec),
        "total_probe_timeout_sec": int(total_timeout_sec),
        "last_timing_phase": str(last_phase),
        "action_allowlist": list(hints.get("mcts_action_allowlist", []) or [])[:8],
        "action_presets": _promotion_probe_action_names(hints, limit=8),
        "task_digest": _sampled_tasks_digest(selected_probe_tasks),
        "boundary_group_keys": [
            key
            for task in selected_probe_tasks
            if isinstance(task, dict)
            for key in list(task.get("group_keys", []) or [])[:4]
        ][:8],
    }


def _promotion_effective_probe_digest(result: dict[str, Any]) -> str:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    return _hint_digest(
        {
            "selected_path_digest": str(result.get("sampled_selected_path_digest", "")),
            "candidate_qbp_coverage": _digest_float(
                _finite_float(result.get("candidate_qbp_coverage"), 0.0)
            ),
            "boundary_group_validity": _digest_float(
                _finite_float(result.get("boundary_group_validity"), 0.0)
            ),
            "latency_usec": _digest_float(_promotion_latency(result)),
            "bootstrap_count": _digest_float(
                _finite_float(result.get("bootstrap_count"), 0.0)
            ),
            "rescale_count": _digest_float(
                _finite_float(result.get("rescale_count"), 0.0)
            ),
            "selected_source_counts": dict(
                diagnostics.get("selected_source_counts", {}) or {}
            ),
        }
    )


def _promotion_complexity_reasons(hints: dict[str, Any]) -> list[str]:
    limits = _sampled_policy_limits(False)
    max_seen = {key: 0 for key in limits}
    policy_like_count = 0

    def visit(value: Any) -> None:
        nonlocal policy_like_count
        if isinstance(value, dict):
            if any(key in value for key in limits) or "strategy" in value:
                policy_like_count += 1
                for key in limits:
                    if key in value:
                        max_seen[key] = max(
                            max_seen[key],
                            _int_hint(value.get(key), 0),
                        )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(hints)
    reasons = [
        f"{key}>{limit} ({max_seen[key]})"
        for key, limit in limits.items()
        if max_seen[key] > limit
    ]
    if policy_like_count > 48:
        reasons.append(f"policy_like_count>48 ({policy_like_count})")
    return reasons[:8]


def _promotion_probe_limits() -> dict[str, int]:
    return {
        "max_scale_candidates": 48,
        "state_cap_per_node": 32,
        "beam_width": 8,
        "boundary_state_cap": 8,
        "mcts_rollout_budget": 6,
        "mcts_action_cap": 3,
        "mcts_max_repair_bootstraps": 12,
    }


def _clamp_promotion_probe_patch(policy: dict[str, Any]) -> dict[str, Any]:
    clamped = dict(policy)
    defaults = {
        "max_scale_candidates": 32,
        "state_cap_per_node": 24,
        "beam_width": 6,
        "boundary_state_cap": 6,
        "mcts_rollout_budget": 4,
        "mcts_action_cap": 2,
        "mcts_max_repair_bootstraps": 8,
    }
    for key, limit in _promotion_probe_limits().items():
        if key in clamped:
            clamped[key] = min(limit, _int_hint(clamped.get(key), defaults[key]))
    return clamped


def _clamp_promotion_probe_items(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    result: list[Any] = []
    for item in value:
        if not isinstance(item, dict):
            result.append(item)
            continue
        updated = dict(item)
        policy = updated.get("policy")
        if isinstance(policy, dict):
            updated["policy"] = _clamp_promotion_probe_patch(policy)
        else:
            updated = _clamp_promotion_probe_patch(updated)
        result.append(updated)
    return result


def _clamp_promotion_probe_presets(value: Any) -> Any:
    allowed = {
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        "reference_boundary_cost_beam",
        "nonlinear_phase_boundary_beam",
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        "component_budget_repair",
        "latency_mcts_repair",
    }
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for name, preset in value.items():
            if str(name) not in allowed:
                continue
            if not isinstance(preset, dict):
                result[str(name)] = preset
                continue
            updated = dict(preset)
            policy = updated.get("policy")
            if isinstance(policy, dict):
                updated["policy"] = _clamp_promotion_probe_patch(policy)
            else:
                updated = _clamp_promotion_probe_patch(updated)
            result[str(name)] = updated
        return result
    if isinstance(value, list):
        return [
            item
            for item in _clamp_promotion_probe_items(value)
            if isinstance(item, dict) and str(item.get("name", "")) in allowed
        ]
    return value


def _promotion_probe_action_names(hints: dict[str, Any], limit: int = 3) -> list[str]:
    scored: list[tuple[float, int, str]] = []
    raw_actions = hints.get("mcts_actions")
    if isinstance(raw_actions, list):
        for idx, item in enumerate(raw_actions):
            if not isinstance(item, dict) or item.get("name") is None:
                continue
            scored.append(
                (
                    _finite_float(item.get("prior"), 0.0),
                    -idx,
                    str(item.get("name")),
                )
            )
    presets = hints.get("mcts_action_presets")
    if isinstance(presets, dict):
        for idx, (name, item) in enumerate(presets.items()):
            prior = (
                _finite_float(item.get("prior"), 0.0)
                if isinstance(item, dict)
                else 0.0
            )
            scored.append((prior, -idx, str(name)))
    elif isinstance(presets, list):
        for idx, item in enumerate(presets):
            if not isinstance(item, dict) or item.get("name") is None:
                continue
            scored.append(
                (
                    _finite_float(item.get("prior"), 0.0),
                    -idx,
                    str(item.get("name")),
                )
            )
    seen: set[str] = set()
    names: list[str] = []
    for _prior, _idx, name in sorted(scored, reverse=True):
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= limit:
            break
    if not names:
        names = ["budget_fulfillment_beam", "wide_boundary_cost_beam"]
    return names


def _boundary_group_selector_matches_sampled_tasks(
    selector: Any,
    tasks: list[dict[str, Any]],
) -> bool:
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for group_key in task.get("group_keys", []) or []:
            if isinstance(group_key, dict):
                key_tuple = (
                    int(group_key.get("in_lvl", -1)),
                    int(group_key.get("in_scl", -1)),
                    str(group_key.get("maino_v", "")),
                    int(group_key.get("main_dag_size", 0) or 0),
                )
                if _boundary_group_selector_matches_key(selector, key_tuple):
                    return True
        task_context = task.get("context", {})
        budgets = task_context.get("io_budgets", []) if isinstance(task_context, dict) else []
        if isinstance(budgets, list):
            groups = _budget_boundary_groups(
                [item for item in budgets if isinstance(item, dict)]
            )
            if any(_boundary_group_selector_matches_key(selector, key) for key in groups):
                return True
    return False


def _promotion_probe_hints(
    eval_hints: dict[str, Any],
    selected_probe_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    probe = deepcopy(eval_hints)
    probe.pop("portfolio", None)
    probe["allow_seed_fallback"] = True
    probe["budget_aggressive"] = False
    probe["mcts_action_presets"] = _clamp_promotion_probe_presets(
        probe.get("mcts_action_presets")
    )
    probe["mcts_actions"] = _clamp_promotion_probe_items(probe.get("mcts_actions"))
    probe["mcts_action_allowlist"] = _promotion_probe_action_names(probe, limit=3)
    probe["mcts_action_cap"] = min(
        3,
        max(
            len(probe["mcts_action_allowlist"]),
            _int_hint(probe.get("mcts_action_cap"), 2),
        ),
    )
    probe["mcts_rollout_budget"] = min(6, max(3, _int_hint(probe.get("mcts_rollout_budget"), 3)))
    probe["mcts_max_repair_bootstraps"] = min(
        12,
        _int_hint(probe.get("mcts_max_repair_bootstraps"), 8),
    )
    probe = _clamp_promotion_probe_patch(probe)
    policies = []
    for item in probe.get("boundary_group_policies", []) or []:
        if not isinstance(item, dict):
            continue
        selector = item.get("selector", {})
        if not _boundary_group_selector_matches_sampled_tasks(selector, selected_probe_tasks):
            continue
        updated = dict(item)
        policy = updated.get("policy")
        if isinstance(policy, dict):
            updated["policy"] = _clamp_promotion_probe_patch(policy)
        policies.append(updated)
        if len(policies) >= 1:
            break
    probe["boundary_group_policies"] = policies
    probe["unit_policies"] = []
    return probe


def _strategy_discovery_base_hints(
    initial_hints: dict[str, Any] | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(initial_hints, dict) and initial_hints:
        return deepcopy(initial_hints)
    seed = _context_initial_policy_hints(context)
    if seed:
        return deepcopy(seed)
    return {"strategy": "bootstrap_mcts"}


def _strategy_discovery_variants(
    initial_hints: dict[str, Any] | None,
    context: dict[str, Any],
    selected_probe_tasks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base = _strategy_discovery_base_hints(initial_hints, context)
    variants: list[dict[str, Any]] = []
    labels: set[str] = set()

    def add(
        label: str,
        dimension: str,
        patch: dict[str, Any],
        *,
        changed_keys: list[str] | None = None,
    ) -> None:
        if label in labels:
            return
        labels.add(label)
        hints = deepcopy(base)
        hints.update(patch)
        variants.append(
            {
                "label": label,
                "dimension": dimension,
                "changed_keys": changed_keys or sorted(patch.keys()),
                "hints": hints,
            }
        )

    top_groups = (
        context.get("harness", {}).get("top_costly_boundary_groups", [])
        if isinstance(context.get("harness", {}), dict)
        else []
    )
    if selected_probe_tasks:
        matching_groups = []
        for item in top_groups or []:
            if not isinstance(item, dict):
                continue
            group_key = item.get("group_key", item)
            if _boundary_group_selector_matches_sampled_tasks(group_key, selected_probe_tasks):
                matching_groups.append(item)
        top_groups = matching_groups or top_groups

    # Ensure the default bounded discovery limit still covers every strategy
    # dimension once before exploring additional values.
    add(
        "boundary_scale_policy_waterline",
        "boundary_scale_policy",
        {"boundary_scale_policy": "waterline"},
    )
    add("scale_lattice_dense", "scale_lattice", {"scale_lattice": "dense"})
    add("boundary_state_cap_8", "boundary_state_cap", {"boundary_state_cap": 8})
    add("beam_width_8", "beam_width", {"beam_width": 8})
    add(
        "bootstrap_penalty_25000000",
        "bootstrap_penalty",
        {"bootstrap_penalty": 25_000_000.0, "selection_bootstrap_penalty": 0.0},
        changed_keys=["bootstrap_penalty", "selection_bootstrap_penalty"],
    )
    add(
        "action_allowlist_dense_boundary_cost_beam",
        "action_allowlist",
        {"mcts_action_allowlist": ["dense_boundary_cost_beam"], "mcts_action_cap": 1},
        changed_keys=["mcts_action_allowlist", "mcts_action_cap"],
    )
    for idx, item in enumerate([group for group in top_groups or [] if isinstance(group, dict)][:1]):
        selector = item.get("group_key", item)
        if isinstance(selector, dict):
            add(
                f"top_boundary_{idx}_frontier_8_48",
                "per_top_boundary_override",
                {
                    "boundary_group_policies": [
                        {
                            "selector": selector,
                            "policy": {
                                "strategy": "latency_beam",
                                "direct_budget_policy": True,
                                "allow_bootstrap": True,
                                "allow_seed_fallback": False,
                                "boundary_scale_policy": "frontier",
                                "boundary_state_cap": 8,
                                "max_scale_candidates": 48,
                                "beam_width": 6,
                                "state_cap_per_node": 24,
                                "bootstrap_penalty": 35_000_000.0,
                                "selection_objective": "cost",
                            },
                        }
                    ]
                },
                changed_keys=["boundary_group_policies"],
            )

    for value in ("waterline", "frontier", "sf", "low"):
        add(
            f"boundary_scale_policy_{value}",
            "boundary_scale_policy",
            {"boundary_scale_policy": value},
        )
    for value in ("waterline_sf", "dense"):
        add(f"scale_lattice_{value}", "scale_lattice", {"scale_lattice": value})
    for value in (4, 6, 8):
        add(
            f"boundary_state_cap_{value}",
            "boundary_state_cap",
            {"boundary_state_cap": value},
        )
    for value in (4, 6, 8):
        add(f"beam_width_{value}", "beam_width", {"beam_width": value})
    for value in (25_000_000.0, 125_000_000.0, 650_000_000.0):
        add(
            f"bootstrap_penalty_{int(value)}",
            "bootstrap_penalty",
            {"bootstrap_penalty": value, "selection_bootstrap_penalty": 0.0},
            changed_keys=["bootstrap_penalty", "selection_bootstrap_penalty"],
        )
    for names in (
        ["budget_fulfillment_beam"],
        ["wide_boundary_cost_beam"],
        ["dense_boundary_cost_beam"],
        ["nonlinear_phase_boundary_beam"],
        ["tuneinsight_avgcase_cost_beam"],
    ):
        add(
            "action_allowlist_" + "_".join(names),
            "action_allowlist",
            {"mcts_action_allowlist": names, "mcts_action_cap": len(names)},
            changed_keys=["mcts_action_allowlist", "mcts_action_cap"],
        )

    for idx, item in enumerate([group for group in top_groups or [] if isinstance(group, dict)][:2]):
        selector = item.get("group_key", item)
        if not isinstance(selector, dict):
            continue
        for boundary_policy, state_cap, scale_count in (
            ("frontier", 8, 48),
            ("waterline", 6, 32),
            ("frontier", 8, 64),
        ):
            add(
                f"top_boundary_{idx}_{boundary_policy}_{state_cap}_{scale_count}",
                "per_top_boundary_override",
                {
                    "boundary_group_policies": [
                        {
                            "selector": selector,
                            "policy": {
                                "strategy": "latency_beam",
                                "direct_budget_policy": True,
                                "allow_bootstrap": True,
                                "allow_seed_fallback": False,
                                "boundary_scale_policy": boundary_policy,
                                "boundary_state_cap": state_cap,
                                "max_scale_candidates": scale_count,
                                "beam_width": 6,
                                "state_cap_per_node": 24,
                                "bootstrap_penalty": 35_000_000.0,
                                "selection_objective": "cost",
                            },
                        }
                    ]
                },
                changed_keys=["boundary_group_policies"],
            )
    return variants[: _strategy_discovery_limit()]


def _strategy_discovery_program_source(label: str, hints: dict[str, Any]) -> str:
    return _program_source_from_hints(hints, f"Strategy discovery seed: {label}.")


def _bounded_retry_probe_hints(hints: dict[str, Any]) -> dict[str, Any]:
    tight = {
        "max_scale_candidates": 24,
        "state_cap_per_node": 16,
        "beam_width": 4,
        "boundary_state_cap": 4,
        "mcts_rollout_budget": 3,
        "mcts_action_cap": 2,
        "mcts_max_repair_bootstraps": 6,
    }

    def clamp_policy(policy: dict[str, Any]) -> dict[str, Any]:
        updated = dict(policy)
        for key, limit in tight.items():
            if key in updated:
                updated[key] = min(limit, _int_hint(updated.get(key), limit))
        return updated

    def recursive_clamp(value: Any) -> Any:
        if isinstance(value, dict):
            updated = clamp_policy(value)
            return {key: recursive_clamp(child) for key, child in updated.items()}
        if isinstance(value, list):
            return [recursive_clamp(item) for item in value]
        return value

    retry = recursive_clamp(deepcopy(hints))
    retry["mcts_action_allowlist"] = list(retry.get("mcts_action_allowlist", []) or [])[:2]
    retry["mcts_action_cap"] = min(2, max(1, len(retry["mcts_action_allowlist"]) or 1))
    if isinstance(retry.get("mcts_action_presets"), dict):
        presets = {}
        for name, preset in retry["mcts_action_presets"].items():
            if not isinstance(preset, dict):
                presets[name] = preset
                continue
            item = dict(preset)
            if isinstance(item.get("policy"), dict):
                item["policy"] = clamp_policy(item["policy"])
            presets[name] = item
        retry["mcts_action_presets"] = presets
    retry["mcts_actions"] = _clamp_promotion_probe_items(retry.get("mcts_actions"))
    if isinstance(retry.get("mcts_actions"), list):
        retry["mcts_actions"] = retry["mcts_actions"][:2]
    if isinstance(retry.get("boundary_group_policies"), list):
        groups = []
        for item in retry["boundary_group_policies"][:1]:
            if not isinstance(item, dict):
                continue
            updated = dict(item)
            if isinstance(updated.get("policy"), dict):
                updated["policy"] = clamp_policy(updated["policy"])
            groups.append(updated)
        retry["boundary_group_policies"] = groups
    return retry


def _candidate_intent_survived_retry(original: dict[str, Any], retry: dict[str, Any]) -> bool:
    original_actions = set(_promotion_probe_action_names(original, limit=8))
    retry_actions = set(_promotion_probe_action_names(retry, limit=8))
    if original_actions and retry_actions and not (original_actions & retry_actions):
        return False
    original_groups = len(original.get("boundary_group_policies", []) or [])
    retry_groups = len(retry.get("boundary_group_policies", []) or [])
    return retry_groups >= min(1, original_groups)


def _strategy_discovery_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    ranked = sorted(
        records,
        key=lambda item: (
            not bool(item.get("path_differential", {}).get("selected_path_changed")),
            not bool(item.get("latency_improved")),
            _finite_float(item.get("latency_usec"), float("inf")),
        ),
    )
    for item in ranked[:6]:
        diff = item.get("path_differential", {})
        examples.append(
            {
                "kind": "strategy_discovery_probe",
                "label": item.get("label"),
                "dimension": item.get("dimension"),
                "direct_valid": bool(item.get("direct_valid")),
                "latency_improved": bool(item.get("latency_improved")),
                "collapse_reason": diff.get("collapse_reason"),
                "selected_path_digest": item.get("selected_path_digest", ""),
                "reference_latency_usec": item.get("reference_latency_usec"),
                "latency_usec": item.get("latency_usec"),
                "selected_source_counts": item.get("selected_source_counts", {}),
                "changed_keys": item.get("changed_keys", []),
            }
        )
    return examples


def _write_strategy_discovery_artifacts(
    discovery_dir: Path,
    summary: dict[str, Any],
    path_rows: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> None:
    (discovery_dir / "strategy_table.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (discovery_dir / "path_delta_table.json").write_text(
        json.dumps(path_rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (discovery_dir / "prompt_examples.json").write_text(
        json.dumps(examples, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _run_strategy_discovery_prepass(
    promotion_dir: Path,
    context: dict[str, Any],
    initial_hints: dict[str, Any] | None,
    params: Params,
    selected_probe_tasks: list[dict[str, Any]],
    probe_reference_cost: float,
    *,
    eval_timeout_sec: int,
    total_timeout_sec: int,
    started_at: float,
) -> dict[str, Any]:
    discovery_dir = promotion_dir / "strategy_discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    discovery_started_at = time.monotonic()
    discovery_timeout_sec = _strategy_discovery_timeout_sec(total_timeout_sec)
    discovery_eval_timeout_sec = _strategy_discovery_eval_timeout_sec(eval_timeout_sec)
    seed_summary = _probe_seed_summary(selected_probe_tasks, probe_reference_cost)
    has_seed_hints = bool(initial_hints)
    variants = (
        _strategy_discovery_variants(initial_hints, context, selected_probe_tasks)
        if has_seed_hints
        else []
    )
    records: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    path_moving_codes: list[str] = []
    skip_policy_digests: set[str] = set()
    timed_out = False
    if not _strategy_discovery_enabled() or not variants:
        summary = {
            "enabled": bool(_strategy_discovery_enabled()),
            "variant_count": 0,
            "records": [],
            "path_moving_code_count": 0,
            "skip_policy_digest_count": 0,
            "timeout_sec": discovery_timeout_sec,
            "eval_timeout_sec": discovery_eval_timeout_sec,
        }
        _write_strategy_discovery_artifacts(discovery_dir, summary, [], [])
        return {
            "records": records,
            "path_moving_codes": path_moving_codes,
            "skip_policy_digests": skip_policy_digests,
            "examples": [],
            "summary": summary,
        }

    probe_context = deepcopy(context)
    probe_context["sampled_budget_tasks"] = deepcopy(selected_probe_tasks)
    probe_context.setdefault("harness", {})["experience_probe"] = True
    probe_context.setdefault("harness", {})["strategy_discovery"] = True
    place_timeout = _candidate_place_timeout_sec(context)
    for index, variant in enumerate(variants):
        if (
            discovery_timeout_sec > 0
            and time.monotonic() - discovery_started_at >= discovery_timeout_sec
        ):
            timed_out = True
            records.append(
                {
                    "index": index,
                    "label": variant.get("label"),
                    "dimension": variant.get("dimension"),
                    "reason": f"strategy discovery timeout after {discovery_timeout_sec}s",
                }
            )
            break
        hints = deepcopy(variant["hints"])
        eval_hints = _compile_hints_for_eval_suite(hints, params.openevolve_eval_suite)
        probe_hints = _promotion_probe_hints(eval_hints, selected_probe_tasks)
        policy_digest = _hint_digest(_policy_effect_payload(probe_hints))
        complexity_reasons = _promotion_complexity_reasons(probe_hints)
        if complexity_reasons:
            bounded = _bounded_retry_probe_hints(probe_hints)
            bounded_reasons = _promotion_complexity_reasons(bounded)
            if not bounded_reasons:
                probe_hints = bounded
                policy_digest = _hint_digest(_policy_effect_payload(probe_hints))
                complexity_reasons = []
        if complexity_reasons:
            record = {
                "index": index,
                "label": variant.get("label"),
                "dimension": variant.get("dimension"),
                "changed_keys": variant.get("changed_keys", []),
                "reason": "strategy_policy_too_expensive:" + ",".join(complexity_reasons[:3]),
                "probe_policy_digest": policy_digest,
                "direct_valid": False,
                "latency_improved": False,
            }
            records.append(record)
            continue
        result = _evaluate_promotion_probe(
            probe_context,
            probe_hints,
            params,
            selected_probe_tasks,
            timeout_sec=discovery_eval_timeout_sec,
        )
        retry_summary: dict[str, Any] | None = None
        active_hints = probe_hints
        if (
            bool(result.get("timed_out", False))
            and _promotion_bounded_retry_enabled()
            and not _use_qbp_dp_engine(params, probe_hints)
        ):
            retry_hints = _bounded_retry_probe_hints(probe_hints)
            retry_result = _evaluate_promotion_probe(
                probe_context,
                retry_hints,
                params,
                selected_probe_tasks,
                timeout_sec=min(
                    discovery_eval_timeout_sec,
                    max(5, discovery_eval_timeout_sec // 2 if discovery_eval_timeout_sec else 0),
                ),
            )
            retry_summary = {
                "attempted": True,
                "intent_survived": _candidate_intent_survived_retry(probe_hints, retry_hints),
                "latency_usec": _promotion_latency(retry_result),
                "direct_valid": _promotion_result_direct_valid(retry_result),
                "timed_out": bool(retry_result.get("timed_out", False)),
                "policy_digest": _hint_digest(_policy_effect_payload(retry_hints)),
            }
            if _promotion_result_direct_valid(retry_result):
                result = retry_result
                active_hints = retry_hints
                policy_digest = retry_summary["policy_digest"]
        diff = _path_differential_summary(seed_summary, result)
        latency = _promotion_latency(result)
        direct_valid = _promotion_result_direct_valid(result)
        improved = bool(direct_valid and math.isfinite(latency) and latency < probe_reference_cost)
        source_counts = _selected_source_counts_from_result(result)
        record = {
            "index": index,
            "label": variant.get("label"),
            "dimension": variant.get("dimension"),
            "changed_keys": variant.get("changed_keys", []),
            "direct_valid": bool(direct_valid),
            "valid": bool(result.get("valid", False)),
            "latency_usec": latency,
            "reference_latency_usec": probe_reference_cost,
            "latency_improved": improved,
            "selected_path_digest": str(result.get("sampled_selected_path_digest", "")),
            "selected_source_counts": source_counts,
            "probe_policy_digest": policy_digest,
            "probe_effective_digest": _promotion_effective_probe_digest(result),
            "path_source_digest": _promotion_path_source_digest(result),
            "path_differential": diff,
            "bootstrap_count": _finite_float(result.get("bootstrap_count"), 0.0),
            "rescale_count": _finite_float(result.get("rescale_count"), 0.0),
            "timed_out": bool(result.get("timed_out", False)),
            "timeout_context": _promotion_timeout_context(
                active_hints,
                selected_probe_tasks,
                place_timeout_sec=place_timeout,
                qbp_timeout_sec=eval_timeout_sec,
                total_timeout_sec=total_timeout_sec,
                last_phase=str(result.get("timeout_phase", "qbp_solve" if result.get("timed_out") else "done")),
            ),
            "bounded_retry": retry_summary,
            "policy_summary": _compact_policy_summary(active_hints),
        }
        if _use_qbp_dp_engine(params, active_hints):
            record["reason"] = _dp_probe_reason(diff, direct_valid, improved)
        elif not direct_valid:
            record["reason"] = diff["collapse_reason"]
        elif not improved:
            record["reason"] = diff["collapse_reason"]
        else:
            record["reason"] = "latency_improved"
        records.append(record)
        path_rows.append(
            {
                "label": record["label"],
                "dimension": record["dimension"],
                "collapse_reason": diff["collapse_reason"],
                "selected_path_changed": diff["selected_path_changed"],
                "selected_source_counts_changed": diff["selected_source_counts_changed"],
                "latency_delta_usec": diff["latency_delta_usec"],
                "selected_path_digest": record["selected_path_digest"],
                "path_source_digest": record["path_source_digest"],
            }
        )
        if diff["collapse_reason"] in {"same_action_order", "same_boundary_states", "probe_tied_seed"}:
            skip_policy_digests.add(policy_digest)
        if bool(diff.get("selected_path_changed")):
            path_moving_codes.append(
                _strategy_discovery_program_source(str(variant.get("label", "strategy")), active_hints)
            )
        partial_summary = {
            "enabled": True,
            "variant_count": len(variants),
            "evaluated_count": len(records),
            "path_moving_count": sum(
                1 for item in records if item.get("path_differential", {}).get("selected_path_changed")
            ),
            "latency_improved_count": sum(1 for item in records if item.get("latency_improved")),
            "skip_policy_digest_count": len(skip_policy_digests),
            "path_moving_code_count": len(path_moving_codes),
            "timed_out": False,
            "timeout_sec": discovery_timeout_sec,
            "eval_timeout_sec": discovery_eval_timeout_sec,
            "records": records,
        }
        _write_strategy_discovery_artifacts(
            discovery_dir,
            partial_summary,
            path_rows,
            _strategy_discovery_examples(records),
        )
    summary = {
        "enabled": True,
        "variant_count": len(variants),
        "evaluated_count": len(records),
        "path_moving_count": sum(
            1 for item in records if item.get("path_differential", {}).get("selected_path_changed")
        ),
        "latency_improved_count": sum(1 for item in records if item.get("latency_improved")),
        "skip_policy_digest_count": len(skip_policy_digests),
        "path_moving_code_count": len(path_moving_codes),
        "timed_out": timed_out,
        "timeout_sec": discovery_timeout_sec,
        "eval_timeout_sec": discovery_eval_timeout_sec,
        "elapsed_sec": round(time.monotonic() - discovery_started_at, 6),
        "records": records,
    }
    examples = _strategy_discovery_examples(records)
    _write_strategy_discovery_artifacts(discovery_dir, summary, path_rows, examples)
    return {
        "records": records,
        "path_moving_codes": path_moving_codes[:16],
        "skip_policy_digests": skip_policy_digests,
        "examples": examples,
        "summary": summary,
    }


def _evaluate_promotion_probe(
    context: dict[str, Any],
    hints: dict[str, Any],
    params: Params,
    selected_probe_tasks: list[dict[str, Any]],
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    if _use_qbp_dp_engine(params, hints):
        return _evaluate_dp_probe_for_promotion(
            context,
            hints,
            selected_probe_tasks,
            timeout_sec=timeout_sec,
        )
    return _evaluate_sampled_budget_tasks_for_promotion(
        context,
        hints,
        timeout_sec=timeout_sec,
    )


def _apply_dp_probe_seed_reference(
    context: dict[str, Any],
    initial_hints: dict[str, Any] | None,
    params: Params,
    selected_probe_tasks: list[dict[str, Any]],
    fallback_reference_cost: float,
) -> tuple[list[dict[str, Any]], float, dict[str, Any] | None]:
    if not initial_hints or not _use_qbp_dp_engine(params, initial_hints):
        return selected_probe_tasks, fallback_reference_cost, None
    probe_context = deepcopy(context)
    probe_context["sampled_budget_tasks"] = deepcopy(selected_probe_tasks)
    result = _evaluate_sampled_budget_tasks_dp_probe(
        probe_context,
        initial_hints,
        selected_tasks=probe_context["sampled_budget_tasks"],
    )
    latency = _promotion_latency(result)
    if not (math.isfinite(latency) and latency > 0):
        return selected_probe_tasks, fallback_reference_cost, result
    updated_tasks = deepcopy(selected_probe_tasks)
    metrics = list(result.get("diagnostics", {}).get("sampled_task_metrics", []) or [])
    for idx, task in enumerate(updated_tasks):
        if not isinstance(task, dict):
            continue
        metric = metrics[min(idx, len(metrics) - 1)] if metrics else {}
        if isinstance(metric, dict):
            task["seed_metrics"] = {
                "selected_path_digest": metric.get("selected_path_digest", ""),
                "selected_source_counts": {"candidate:qbp_dp": int(metric.get("boundary_group_count", 0) or 0)},
                "sampled_dp_latency_usec": metric.get("sampled_dp_latency_usec", latency),
                "objective_cost_usec": metric.get("sampled_dp_latency_usec", latency),
                "bootstrap_count": metric.get("bootstrap_count", result.get("bootstrap_count", 0.0)),
                "rescale_count": metric.get("rescale_count", result.get("rescale_count", 0.0)),
            }
    return updated_tasks, latency, result


def _write_promotion_progress(
    promotion_dir: Path,
    *,
    started_at: float,
    timeout_sec: int,
    current_stage: str,
    current_index: int | None,
    total_candidates: int,
    records: list[dict[str, Any]],
    selected: bool,
    timed_out: bool = False,
) -> None:
    payload = {
        "current_stage": current_stage,
        "current_index": current_index,
        "total_candidates": total_candidates,
        "record_count": len(records),
        "selected": selected,
        "timed_out": timed_out,
        "elapsed_sec": round(time.monotonic() - started_at, 6),
        "timeout_sec": timeout_sec,
        "latest_records": records[-8:],
    }
    (promotion_dir / "promotion_progress.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _promotion_sampled_eval_worker(
    result_queue,
    sampled_context: dict[str, Any],
    hints: dict[str, Any],
) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    try:
        result = _evaluate_sampled_budget_tasks(
            sampled_context,
            hints,
            suppress_output=True,
        )
        result_queue.put({"ok": True, "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "result": {
                    "valid": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "traceback": traceback.format_exc()[-4000:],
                },
            }
        )


def _promotion_dp_probe_worker(
    result_queue,
    probe_context: dict[str, Any],
    hints: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
) -> None:
    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            pass
    try:
        context = deepcopy(probe_context)
        context["sampled_budget_tasks"] = deepcopy(selected_tasks)
        context.setdefault("harness", {})["dp_table_probe"] = True
        result = _evaluate_sampled_budget_tasks_dp_probe(
            context,
            hints,
            selected_tasks=context["sampled_budget_tasks"],
        )
        result_queue.put({"ok": True, "result": result})
    except BaseException as exc:
        result_queue.put(
            {
                "ok": False,
                "result": {
                    "valid": False,
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                    "traceback": traceback.format_exc()[-4000:],
                },
            }
        )


def _evaluate_sampled_budget_tasks_dp_probe(
    context: dict[str, Any],
    hints: dict[str, Any],
    *,
    selected_tasks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fast no-materialization QBP-DP probe used before promotion replay."""

    start = time.time()
    eval_suite = str(context.get("harness", {}).get("eval_suite", "polybert-sampled"))
    eval_hints = _compile_hints_for_eval_suite(hints, eval_suite)
    tasks = selected_tasks
    if tasks is None:
        tasks = [
            task
            for task in context.get("sampled_budget_tasks", [])
            if isinstance(task, dict) and isinstance(task.get("context"), dict)
        ]
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
        "unreachable_boundary_groups": 0,
        "candidate_improved_budgets": 0,
        "candidate_invalid_reasons": {},
        "invalid_reasons": {},
        "candidate_costs": [],
        "costs": [],
        "selected_source_counts": {},
        "boundary_group_summaries": [],
        "sampled_task_metrics": [],
        "sampled_task_count": 0,
        "sampled_direct_budget_eval": False,
        "sampled_qbp_group_eval": True,
        "qbp_engine": "dp",
        "dp_table_probe": True,
    }
    for task_position, task in enumerate(tasks or []):
        task_context = task.get("context") if isinstance(task, dict) else None
        if not isinstance(task_context, dict):
            continue
        task_tdag = tdag_from_context(task_context)
        task_params = task_tdag.params
        task_params.openevolve_iterations = 0
        task_params.openevolve_harness = "compile"
        task_params.openevolve_compile_hints = eval_hints
        task_params.openevolve_evaluating_candidate = True
        task_params.openevolve_eval_suite = eval_suite
        task_params.openevolve_qbp_engine = "dp"
        _apply_active_scale_floor_from_hints(task_params, eval_hints)
        task_le = LatencyEstimator(task_params)
        budgets = [
            _io_budget_from_json(item)
            for item in task_context.get("io_budgets", [])
            if isinstance(item, dict)
        ]
        task_group_summaries: list[dict[str, Any]] = []
        total["sampled_task_count"] += 1
        for group_key, group_budgets in _budget_boundary_groups(budgets).items():
            selected_results: list[_QBPDPBudgetResult] = []
            reachable = 0
            for budget in group_budgets:
                total["requested_budgets"] += 1
                result = _qbp_dp_solve_budget(
                    task_tdag,
                    task_params,
                    budget,
                    task_le,
                    eval_hints,
                    materialize=False,
                )
                if result.valid and result.out_key is not None:
                    selected_results.append(result)
                    reachable += 1
                    total["solved_budgets"] += 1
                    total["candidate_solved_budgets"] += 1
                    total["candidate_costs"].append(float(result.cost))
                    total["costs"].append(float(result.cost))
                elif _boundary_budget_output_cannot_refresh(task_params, group_key, budget):
                    continue
                else:
                    reachable += 1
                    reason = result.reason or "qbp-dp found no feasible state"
                    total["candidate_invalid_reasons"][reason] = (
                        total["candidate_invalid_reasons"].get(reason, 0) + 1
                    )
            total["requested_boundary_groups"] += 1
            requested = len(group_budgets)
            solved = len(selected_results)
            if requested > 0 and reachable > 0 and solved >= reachable:
                total["solved_boundary_groups"] += 1
                total["candidate_solved_boundary_groups"] += 1
                reason = "complete_candidate"
            elif solved > 0:
                total["partial_boundary_groups"] += 1
                reason = "partial_output_levels"
            elif reachable <= 0 or _boundary_group_input_cannot_refresh(task_params, group_key):
                total["unreachable_boundary_groups"] += 1
                reason = "unreachable_input_cannot_refresh"
            else:
                total["invalid_boundary_groups"] += 1
                reason = "no_feasible_attempt"
            costs = [float(item.cost) for item in selected_results]
            bootstraps = [float(item.bootstrap_count) for item in selected_results]
            rescales = [float(item.rescale_count) for item in selected_results]
            selected_source_counts = {"candidate:qbp_dp": solved} if solved else {}
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
                        for budget in group_budgets
                        if int(budget.get("out_lvl", -1)) >= 0
                    }
                ),
                "requested_budgets": requested,
                "reachable_budgets": int(reachable),
                "unreachable_budgets": int(max(0, requested - reachable)),
                "solved_budgets": solved,
                "candidate_solved_budgets": solved,
                "fallback_selected_budgets": 0,
                "complete": bool(requested > 0 and reachable > 0 and solved >= reachable),
                "candidate_complete": bool(requested > 0 and reachable > 0 and solved >= reachable),
                "selected_source_counts": selected_source_counts,
                "unsolved_reason": reason,
                "unreachable_input": bool(reachable <= 0),
                "avg_bootstrap": float(sum(bootstraps) / len(bootstraps)) if bootstraps else 0.0,
                "min_bootstrap": float(min(bootstraps)) if bootstraps else 0.0,
                "max_bootstrap": float(max(bootstraps)) if bootstraps else 0.0,
                "avg_rescale": float(sum(rescales) / len(rescales)) if rescales else 0.0,
                "min_rescale": float(min(rescales)) if rescales else 0.0,
                "max_rescale": float(max(rescales)) if rescales else 0.0,
                "avg_cost_usec": float(sum(costs) / len(costs)) if costs else 0.0,
                "min_cost_usec": float(min(costs)) if costs else 0.0,
                "max_cost_usec": float(max(costs)) if costs else 0.0,
            }
            task_group_summaries.append(summary)
            if len(total["boundary_group_summaries"]) < 512:
                total["boundary_group_summaries"].append(summary)
            for source, count in selected_source_counts.items():
                total["selected_source_counts"][source] = (
                    total["selected_source_counts"].get(source, 0) + int(count)
                )
        task_proxy = _sampled_path_proxy_from_boundary_groups(task_group_summaries)
        total["sampled_task_metrics"].append(
            {
                "task_position": task_position,
                "task_index": task.get("index") if isinstance(task, dict) else task_position,
                "group_keys": task.get("group_keys", []) if isinstance(task, dict) else [],
                "selected_path_digest": task_proxy["sampled_selected_path_digest"],
                "sampled_dp_latency_usec": task_proxy["sampled_dp_latency_usec"],
                "bootstrap_count": task_proxy["sampled_selected_path_bootstraps"],
                "rescale_count": task_proxy["sampled_selected_path_rescales"],
                "boundary_group_count": len(task_group_summaries),
            }
        )
    group_summary = _boundary_group_count_summary(
        list(total.get("boundary_group_summaries", []))
    )
    sampled_path_proxy = _sampled_path_proxy_from_boundary_groups(
        list(total.get("boundary_group_summaries", [])),
        list(total.get("costs", [])),
    )
    requested = max(1, int(total.get("requested_budgets", 0) or 1))
    solved = int(total.get("solved_budgets", 0) or 0)
    scored_groups = _scored_boundary_group_count(total)
    solved_groups = int(total.get("solved_boundary_groups", 0) or 0)
    candidate_groups = int(total.get("candidate_solved_boundary_groups", 0) or 0)
    params = tdag_from_context(context).params
    scale_floor_bits = _apply_active_scale_floor_from_hints(params, eval_hints)
    return {
        "valid": bool(solved),
        "validity": float(solved / requested),
        "boundary_group_validity": float(min(1.0, solved_groups / scored_groups)),
        "candidate_qbp_coverage": float(min(1.0, candidate_groups / scored_groups)),
        "sampled_progress_only": True,
        "dp_table_probe": True,
        "final_latency_usec": float(sum(total["costs"]) / len(total["costs"])) if total["costs"] else 0.0,
        "aggregated_partition_cost_usec": float(sum(total["costs"])) if total["costs"] else 0.0,
        "objective_cost_usec": float(sampled_path_proxy["sampled_dp_latency_usec"]),
        "total_frontier_cost_usec": float(group_summary["frontier_total_cost_usec"]),
        "scale_floor_bits": int(scale_floor_bits),
        "scale_floor_delta_bits": int(params.Sw) - int(scale_floor_bits),
        **sampled_path_proxy,
        "bootstrap_count": float(group_summary["frontier_total_bootstrap"]),
        "rescale_count": float(group_summary["frontier_total_rescale"]),
        "sampled_frontier_total_bootstrap_count": float(group_summary["frontier_total_bootstrap"]),
        "sampled_frontier_total_rescale_count": float(group_summary["frontier_total_rescale"]),
        "boundary_quality": 0.0,
        "profile_risk": 0.0 if solved else 1.0,
        "placement_runtime_sec": time.time() - start,
        "fallback_selected_budgets": 0,
        "fallback_selected_groups": 0,
        "invalid_boundary_groups": int(total.get("invalid_boundary_groups", 0)),
        "unreachable_boundary_groups": int(total.get("unreachable_boundary_groups", 0)),
        "selected_output_state": {},
        "reserve_summary": {},
        "assignment": {},
        "bootstrap_locations": {},
        "rescale_locations": {},
        "bottleneck_summary": [],
        "diagnostics": total,
        "sampled_task_seed_metrics": list(total.get("sampled_task_metrics", [])),
        "sampled_budget_tasks": [],
        "log_tail": "",
    }


def _evaluate_sampled_budget_tasks_for_promotion(
    sampled_context: dict[str, Any],
    hints: dict[str, Any],
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    if timeout_sec <= 0:
        return _evaluate_sampled_budget_tasks(
            sampled_context,
            hints,
            suppress_output=True,
        )
    ctx = _mp_context()
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_promotion_sampled_eval_worker,
        args=(result_queue, sampled_context, hints),
    )
    proc.start()
    deadline = time.time() + float(timeout_sec)
    payload: dict[str, Any] | None = None
    while time.time() < deadline:
        try:
            payload = result_queue.get(timeout=min(0.5, max(0.0, deadline - time.time())))
            break
        except queue_module.Empty:
            if not proc.is_alive():
                break
    if payload is None:
        proc.join(0)
        try:
            payload = result_queue.get_nowait()
        except queue_module.Empty:
            payload = None
    if payload is None and proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        else:
            proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.join(5)
        return {
            "valid": False,
            "timed_out": True,
            "timeout_sec": int(timeout_sec),
            "timeout_phase": "qbp_solve",
            "last_timing_phase": "qbp_solve",
            "error": f"promotion QBP evaluation timed out after {timeout_sec}s",
        }
    proc.join(5)
    if payload is None:
        return {
            "valid": False,
            "error": f"promotion QBP worker exited with code {proc.exitcode}",
            "worker_exitcode": proc.exitcode,
        }
    result = payload.get("result") if isinstance(payload, dict) else None
    return result if isinstance(result, dict) else {"valid": False, "error": "promotion QBP worker returned no result"}


def _evaluate_dp_probe_for_promotion(
    context: dict[str, Any],
    hints: dict[str, Any],
    selected_probe_tasks: list[dict[str, Any]],
    *,
    timeout_sec: int,
) -> dict[str, Any]:
    probe_context = deepcopy(context)
    probe_context["sampled_budget_tasks"] = deepcopy(selected_probe_tasks)
    probe_context.setdefault("harness", {})["dp_table_probe"] = True
    if timeout_sec <= 0:
        return _evaluate_sampled_budget_tasks_dp_probe(
            probe_context,
            hints,
            selected_tasks=probe_context["sampled_budget_tasks"],
        )
    ctx = _mp_context()
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_promotion_dp_probe_worker,
        args=(result_queue, probe_context, hints, selected_probe_tasks),
    )
    proc.start()
    deadline = time.time() + float(timeout_sec)
    payload: dict[str, Any] | None = None
    while time.time() < deadline:
        try:
            payload = result_queue.get(timeout=min(0.5, max(0.0, deadline - time.time())))
            break
        except queue_module.Empty:
            if not proc.is_alive():
                break
    if payload is None:
        proc.join(0)
        try:
            payload = result_queue.get_nowait()
        except queue_module.Empty:
            payload = None
    if payload is None and proc.is_alive():
        if hasattr(os, "killpg"):
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        else:
            proc.terminate()
        proc.join(5)
        if proc.is_alive() and hasattr(proc, "kill"):
            if hasattr(os, "killpg"):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            proc.join(5)
        return {
            "valid": False,
            "timed_out": True,
            "timeout_sec": int(timeout_sec),
            "timeout_phase": "dp_table_probe",
            "last_timing_phase": "dp_table_probe",
            "dp_table_probe": True,
            "error": f"promotion DP probe timed out after {timeout_sec}s",
        }
    proc.join(5)
    if payload is None:
        return {
            "valid": False,
            "dp_table_probe": True,
            "error": f"promotion DP probe worker exited with code {proc.exitcode}",
            "worker_exitcode": proc.exitcode,
        }
    result = payload.get("result") if isinstance(payload, dict) else None
    return result if isinstance(result, dict) else {"valid": False, "dp_table_probe": True, "error": "promotion DP probe worker returned no result"}


def _run_sampled_promotion_pass(
    root: Path,
    output_dir: Path,
    context: dict[str, Any],
    best_code: str,
    params: Params,
    initial_hints: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if params.openevolve_eval_suite == "polybert-full":
        return None
    if not context.get("sampled_budget_tasks"):
        return None
    limit = _promotion_candidate_limit(params)
    if limit <= 0:
        return None
    promotion_dir = root / "sampled_promotion"
    promotion_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    timeout_sec = _promotion_timeout_sec(params)
    eval_timeout_sec = _promotion_eval_timeout_sec(params)
    selected_probe_tasks, probe_reference_cost = _promotion_probe_tasks(context)
    if not selected_probe_tasks or not math.isfinite(probe_reference_cost):
        return None
    selected_probe_tasks, probe_reference_cost, dp_seed_probe = _apply_dp_probe_seed_reference(
        context,
        initial_hints,
        params,
        selected_probe_tasks,
        probe_reference_cost,
    )

    full_reference_cost = _promotion_reference_latency(context)
    discovery = _run_strategy_discovery_prepass(
        promotion_dir,
        context,
        initial_hints,
        params,
        selected_probe_tasks,
        probe_reference_cost,
        eval_timeout_sec=eval_timeout_sec,
        total_timeout_sec=timeout_sec,
        started_at=started_at,
    )
    if discovery.get("examples"):
        context.setdefault("harness", {})["candidate_examples"] = _merge_candidate_examples(
            context.get("harness", {}).get("candidate_examples", []),
            discovery.get("examples", []),
        )
    discovery_limit = min(128, max(limit, limit * 4))
    codes = list(discovery.get("path_moving_codes", []) or []) + _discover_finalist_codes(
        output_dir, best_code, discovery_limit
    )
    records: list[dict[str, Any]] = []
    best: tuple[float, float, float, int, dict[str, Any], dict[str, Any], str] | None = None
    seen: set[str] = set()
    seen_policy_digests: set[str] = set()
    seen_effective_digests: set[str] = set()
    seen_path_source_digests: set[str] = set()
    discovery_skip_policy_digests = set(discovery.get("skip_policy_digests", set()) or set())
    duplicate_policy_count = 0
    duplicate_effective_count = 0
    duplicate_path_source_count = 0
    discovery_seed_equivalent_skip_count = 0
    probe_evaluated_count = 0
    timed_out = False
    seed_probe_summary = _probe_seed_summary(selected_probe_tasks, probe_reference_cost)

    _write_promotion_progress(
        promotion_dir,
        started_at=started_at,
        timeout_sec=timeout_sec,
        current_stage="start",
        current_index=None,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=False,
    )

    for index, code in enumerate(codes):
        if len(seen_effective_digests) >= limit:
            break
        if timeout_sec > 0 and time.monotonic() - started_at >= timeout_sec:
            timed_out = True
            records.append(
                {
                    "index": index,
                    "stage": "timeout",
                    "reason": f"promotion timeout after {timeout_sec}s",
                }
            )
            break
        code_digest = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if code_digest in seen:
            continue
        seen.add(code_digest)
        print(
            "OpenEvolve compile harness: sampled promotion "
            f"candidate {len(seen)}/{len(codes)} static digest={code_digest[:12]}.",
            flush=True,
        )
        _write_promotion_progress(
            promotion_dir,
            started_at=started_at,
            timeout_sec=timeout_sec,
            current_stage="static",
            current_index=index,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=best is not None,
        )
        hints = _hints_from_code(code, context)
        if hints is None:
            records.append(
                _promotion_record_summary(
                    index=index,
                    stage="static",
                    hints=None,
                    reason="candidate_hints_not_loadable",
                    code_digest=code_digest,
                )
            )
            _write_promotion_progress(
                promotion_dir,
                started_at=started_at,
                timeout_sec=timeout_sec,
                current_stage="static_rejected",
                current_index=index,
                total_candidates=min(limit, len(codes)),
                records=records,
                selected=best is not None,
            )
            continue
        static = _static_validate_hints(context, hints)
        if not static.get("valid", False):
            records.append(
                _promotion_record_summary(
                    index=index,
                    stage="static",
                    hints=hints,
                    reason="static_invalid:" + ",".join(static.get("reasons", [])[:3]),
                    code_digest=code_digest,
                )
            )
            _write_promotion_progress(
                promotion_dir,
                started_at=started_at,
                timeout_sec=timeout_sec,
                current_stage="static_rejected",
                current_index=index,
                total_candidates=min(limit, len(codes)),
                records=records,
                selected=best is not None,
            )
            continue
        eval_hints = _compile_hints_for_eval_suite(hints, params.openevolve_eval_suite)
        probe_hints = _promotion_probe_hints(eval_hints, selected_probe_tasks)
        complexity_reasons = _promotion_complexity_reasons(probe_hints)
        if complexity_reasons:
            bounded = _bounded_retry_probe_hints(probe_hints)
            bounded_reasons = _promotion_complexity_reasons(bounded)
            if not bounded_reasons:
                probe_hints = bounded
                complexity_reasons = []
        if complexity_reasons:
            records.append(
                _promotion_record_summary(
                    index=index,
                    stage="static",
                    hints=probe_hints,
                    reason="promotion_policy_too_expensive:"
                    + ",".join(complexity_reasons[:3]),
                    code_digest=code_digest,
                )
            )
            _write_promotion_progress(
                promotion_dir,
                started_at=started_at,
                timeout_sec=timeout_sec,
                current_stage="static_rejected",
                current_index=index,
                total_candidates=min(limit, len(codes)),
                records=records,
                selected=best is not None,
            )
            continue
        policy_digest = _hint_digest(_policy_effect_payload(probe_hints))
        if policy_digest in discovery_skip_policy_digests:
            discovery_seed_equivalent_skip_count += 1
            records.append(
                _promotion_record_summary(
                    index=index,
                    stage="static",
                    hints=probe_hints,
                    reason="strategy_discovery_seed_equivalent_probe_skip",
                    code_digest=code_digest,
                )
            )
            _write_promotion_progress(
                promotion_dir,
                started_at=started_at,
                timeout_sec=timeout_sec,
                current_stage="static_rejected",
                current_index=index,
                total_candidates=min(limit, len(codes)),
                records=records,
                selected=best is not None,
            )
            continue
        if policy_digest in seen_policy_digests:
            duplicate_policy_count += 1
            records.append(
                _promotion_record_summary(
                    index=index,
                    stage="static",
                    hints=probe_hints,
                    reason="duplicate_probe_policy_digest",
                    code_digest=code_digest,
                )
            )
            _write_promotion_progress(
                promotion_dir,
                started_at=started_at,
                timeout_sec=timeout_sec,
                current_stage="static_rejected",
                current_index=index,
                total_candidates=min(limit, len(codes)),
                records=records,
                selected=best is not None,
            )
            continue
        seen_policy_digests.add(policy_digest)
        probe_evaluated_count += 1
        probe_context = deepcopy(context)
        probe_context["sampled_budget_tasks"] = deepcopy(selected_probe_tasks)
        probe_context.setdefault("harness", {})["experience_probe"] = True
        if _use_qbp_dp_engine(params, probe_hints):
            (promotion_dir / f"dp_probe_hints_{index}.json").write_text(
                json.dumps(
                    _jsonable_policy_hints(probe_hints),
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        probe_stage_name = (
            "dp_table_probe" if _use_qbp_dp_engine(params, probe_hints) else "probe_qbp"
        )
        print(
            "OpenEvolve compile harness: sampled promotion "
            f"candidate {len(seen)}/{len(codes)} {probe_stage_name} timeout={eval_timeout_sec}s.",
            flush=True,
        )
        _write_promotion_progress(
            promotion_dir,
            started_at=started_at,
            timeout_sec=timeout_sec,
            current_stage=probe_stage_name,
            current_index=index,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=best is not None,
        )
        probe_result = _evaluate_promotion_probe(
            probe_context,
            probe_hints,
            params,
            selected_probe_tasks,
            timeout_sec=eval_timeout_sec,
        )
        active_probe_hints = probe_hints
        bounded_retry_summary: dict[str, Any] | None = None
        if (
            bool(probe_result.get("timed_out", False))
            and _promotion_bounded_retry_enabled()
            and not _use_qbp_dp_engine(params, probe_hints)
        ):
            retry_hints = _bounded_retry_probe_hints(probe_hints)
            retry_timeout = min(
                eval_timeout_sec,
                max(10, eval_timeout_sec // 2 if eval_timeout_sec else 0),
            )
            retry_result = _evaluate_promotion_probe(
                probe_context,
                retry_hints,
                params,
                selected_probe_tasks,
                timeout_sec=retry_timeout,
            )
            bounded_retry_summary = {
                "attempted": True,
                "intent_survived": _candidate_intent_survived_retry(probe_hints, retry_hints),
                "timeout_sec": retry_timeout,
                "timed_out": bool(retry_result.get("timed_out", False)),
                "direct_valid": _promotion_result_direct_valid(retry_result),
                "latency_usec": _promotion_latency(retry_result),
                "policy_digest": _hint_digest(_policy_effect_payload(retry_hints)),
            }
            if _promotion_result_direct_valid(retry_result):
                probe_result = retry_result
                active_probe_hints = retry_hints
                policy_digest = str(bounded_retry_summary["policy_digest"])
        probe_latency = _promotion_latency(probe_result)
        path_diff = _path_differential_summary(seed_probe_summary, probe_result)
        probe_record = _promotion_record_summary(
            index=index,
            stage=probe_stage_name,
            hints=active_probe_hints,
            result=probe_result,
            reason="",
            code_digest=code_digest,
        )
        probe_record["reference_latency_usec"] = probe_reference_cost
        probe_record["full_policy_summary"] = _compact_policy_summary(eval_hints)
        probe_record["probe_policy_digest"] = policy_digest
        effective_digest = _promotion_effective_probe_digest(probe_result)
        probe_record["probe_effective_digest"] = effective_digest
        path_source_digest = _promotion_path_source_digest(probe_result)
        probe_record["path_source_digest"] = path_source_digest
        probe_record["path_differential"] = path_diff
        probe_record["path_collapse_reason"] = path_diff.get("collapse_reason")
        probe_record["timeout_context"] = _promotion_timeout_context(
            active_probe_hints,
            selected_probe_tasks,
            place_timeout_sec=_candidate_place_timeout_sec(context),
            qbp_timeout_sec=eval_timeout_sec,
            total_timeout_sec=timeout_sec,
            last_phase=str(
                probe_result.get(
                    "last_timing_phase",
                    probe_result.get("timeout_phase", "done"),
                )
            ),
        )
        if bounded_retry_summary is not None:
            probe_record["bounded_retry"] = bounded_retry_summary
        records.append(probe_record)
        _write_promotion_progress(
            promotion_dir,
            started_at=started_at,
            timeout_sec=timeout_sec,
            current_stage="probe_qbp_done",
            current_index=index,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=best is not None,
        )
        if not _promotion_result_direct_valid(probe_result):
            probe_record["reason"] = (
                "dp_incomplete"
                if _use_qbp_dp_engine(params, active_probe_hints)
                else "probe_not_direct_valid"
            )
            continue
        if effective_digest in seen_effective_digests:
            duplicate_effective_count += 1
            probe_record["reason"] = "duplicate_probe_effective_path"
            continue
        seen_effective_digests.add(effective_digest)
        if path_source_digest in seen_path_source_digests:
            duplicate_path_source_count += 1
            probe_record["reason"] = "duplicate_probe_path_source"
            continue
        seen_path_source_digests.add(path_source_digest)
        if not (math.isfinite(probe_latency) and probe_latency < probe_reference_cost):
            probe_record["reason"] = (
                _dp_probe_reason(path_diff, True, False)
                if _use_qbp_dp_engine(params, active_probe_hints)
                else "probe_not_latency_improved"
            )
            continue

        if timeout_sec > 0 and time.monotonic() - started_at >= timeout_sec:
            timed_out = True
            records.append(
                {
                    "index": index,
                    "stage": "timeout",
                    "reason": f"promotion timeout before sampled replay after {timeout_sec}s",
                }
            )
            break
        print(
            "OpenEvolve compile harness: sampled promotion "
            f"candidate {len(seen)}/{len(codes)} sampled_qbp timeout={eval_timeout_sec}s.",
            flush=True,
        )
        _write_promotion_progress(
            promotion_dir,
            started_at=started_at,
            timeout_sec=timeout_sec,
            current_stage="sampled_qbp",
            current_index=index,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=best is not None,
        )
        full_result = _evaluate_sampled_budget_tasks_for_promotion(
            context,
            active_probe_hints,
            timeout_sec=eval_timeout_sec,
        )
        full_latency = _promotion_latency(full_result)
        full_record = _promotion_record_summary(
            index=index,
            stage="sampled_qbp",
            hints=active_probe_hints,
            result=full_result,
            reason="",
            code_digest=code_digest,
        )
        full_record["reference_latency_usec"] = full_reference_cost
        full_record["probe_latency_usec"] = probe_latency
        full_record["path_differential"] = _path_differential_summary(
            _probe_seed_summary(context.get("sampled_budget_tasks", []), full_reference_cost),
            full_result,
        )
        records.append(full_record)
        _write_promotion_progress(
            promotion_dir,
            started_at=started_at,
            timeout_sec=timeout_sec,
            current_stage="sampled_qbp_done",
            current_index=index,
            total_candidates=min(limit, len(codes)),
            records=records,
            selected=best is not None,
        )
        if not _promotion_result_direct_valid(full_result):
            full_record["reason"] = "sampled_not_direct_valid"
            continue
        if not (math.isfinite(full_latency) and full_latency < full_reference_cost):
            full_record["reason"] = "sampled_not_latency_improved"
            continue
        item = (
            full_latency,
            _finite_float(full_result.get("bootstrap_count"), 0.0),
            _finite_float(full_result.get("rescale_count"), 0.0),
            index,
            active_probe_hints,
            full_record,
            code,
        )
        if best is None or item[:4] < best[:4]:
            best = item

    summary = {
        "candidate_count": len(codes),
        "evaluated_count": len(seen),
        "probe_evaluated_count": probe_evaluated_count,
        "duplicate_policy_count": duplicate_policy_count,
        "duplicate_effective_path_count": duplicate_effective_count,
        "duplicate_path_source_count": duplicate_path_source_count,
        "strategy_discovery_seed_equivalent_skip_count": discovery_seed_equivalent_skip_count,
        "distinct_probe_policy_count": len(seen_policy_digests),
        "distinct_probe_effective_path_count": len(seen_effective_digests),
        "distinct_probe_path_source_count": len(seen_path_source_digests),
        "strategy_discovery": {
            "enabled": bool(_strategy_discovery_enabled()),
            "summary_path": str(promotion_dir / "strategy_discovery" / "strategy_table.json"),
            "path_delta_path": str(promotion_dir / "strategy_discovery" / "path_delta_table.json"),
            "path_moving_code_count": len(discovery.get("path_moving_codes", []) or []),
            "latency_improved_count": int(
                discovery.get("summary", {}).get("latency_improved_count", 0)
                if isinstance(discovery.get("summary"), dict)
                else 0
            ),
            "path_moving_count": int(
                discovery.get("summary", {}).get("path_moving_count", 0)
                if isinstance(discovery.get("summary"), dict)
                else 0
            ),
        },
        "probe_task_count": len(selected_probe_tasks),
        "probe_reference_latency_usec": probe_reference_cost,
        "sampled_reference_latency_usec": full_reference_cost,
        "qbp_engine": getattr(params, "openevolve_qbp_engine", "mcts"),
        "dp_seed_probe": {
            "valid": bool(dp_seed_probe.get("valid", False)),
            "latency_usec": _promotion_latency(dp_seed_probe),
            "selected_path_digest": str(dp_seed_probe.get("sampled_selected_path_digest", "")),
            "candidate_qbp_coverage": _finite_float(dp_seed_probe.get("candidate_qbp_coverage"), 0.0),
            "boundary_group_validity": _finite_float(dp_seed_probe.get("boundary_group_validity"), 0.0),
        }
        if isinstance(dp_seed_probe, dict)
        else None,
        "selected": best is not None,
        "selected_index": None if best is None else best[3],
        "selected_latency_usec": None if best is None else best[0],
        "selected_record": None if best is None else best[5],
        "timed_out": timed_out,
        "elapsed_sec": round(time.monotonic() - started_at, 6),
        "timeout_sec": timeout_sec,
        "eval_timeout_sec": eval_timeout_sec,
        "records": records,
    }
    (promotion_dir / "promotion_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    if getattr(params, "openevolve_qbp_engine", "mcts") == "dp":
        (promotion_dir / "dp_probe_summary.json").write_text(
            json.dumps(
                {
                    "qbp_engine": "dp",
                    "probe_task_count": len(selected_probe_tasks),
                    "probe_reference_latency_usec": probe_reference_cost,
                    "selected": best is not None,
                    "selected_latency_usec": None if best is None else best[0],
                    "distinct_probe_policy_count": len(seen_policy_digests),
                    "distinct_probe_effective_path_count": len(seen_effective_digests),
                    "distinct_probe_path_source_count": len(seen_path_source_digests),
                    "records": [
                        {
                            key: record.get(key)
                            for key in (
                                "index",
                                "stage",
                                "reason",
                                "direct_valid",
                                "latency_usec",
                                "reference_latency_usec",
                                "selected_path_digest",
                                "path_collapse_reason",
                                "probe_policy_digest",
                                "probe_effective_digest",
                                "path_source_digest",
                            )
                        }
                        for record in records
                    ],
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    _write_promotion_progress(
        promotion_dir,
        started_at=started_at,
        timeout_sec=timeout_sec,
        current_stage="done",
        current_index=None,
        total_candidates=len(codes),
        records=records,
        selected=best is not None,
        timed_out=timed_out,
    )
    print(
        "OpenEvolve compile harness: sampled promotion "
        f"candidates={len(codes)} selected={best is not None} "
        f"sampled_reference={full_reference_cost}.",
        flush=True,
    )
    if best is None:
        return None
    selected_hints = deepcopy(best[4])
    selected_hints["openevolve_promotion_validated"] = True
    selected_hints["openevolve_promotion_stage"] = "sampled_qbp"
    selected_hints["openevolve_promotion_latency_usec"] = float(best[0])
    selected_hints["openevolve_promotion_reference_latency_usec"] = float(full_reference_cost)
    selected_hints["openevolve_promotion_code_digest"] = hashlib.sha256(
        best[6].encode("utf-8")
    ).hexdigest()
    (promotion_dir / "selected_program.py").write_text(best[6], encoding="utf-8")
    return selected_hints


def _write_sampled_selection_summary(
    root: Path,
    output_dir: Path,
    reason: str,
    *,
    fail_open: bool,
    selected_sampled_best: bool = False,
) -> None:
    info_path = output_dir / "best" / "best_program_info.json"
    info: dict[str, Any] = {}
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            info = {}
    try:
        (root / "sampled_selection_summary.json").write_text(
            json.dumps(
                {
                    "selected_sampled_best": bool(selected_sampled_best),
                    "fail_open": bool(fail_open),
                    "reason": reason,
                    "best_metrics": info.get("metrics", {}),
                    "best_artifacts": {
                        key: value
                        for key, value in info.get("artifacts", {}).items()
                        if key
                        in {
                            "candidate_mlir_path",
                            "candidate_mlir_digest",
                            "sampled_selected_path_digest",
                            "latency_only_objective",
                            "correctness_gate",
                        }
                    },
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass


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
    candidates = _discover_finalist_codes(output_dir, "", 1)
    if candidates:
        return candidates[0]
    return None


def _load_reusable_policy_bank_initial_hints(
    output_dir: Path, context: dict[str, Any]
) -> dict[str, Any] | None:
    """Recover a sampled policy-bank seed from a reused OpenEvolve workspace."""

    bank_dir = output_dir / "policy_bank"
    summary_path = bank_dir / "policy_bank_summary.json"
    if not summary_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    label = str(summary.get("selected_initial_label") or "").strip()
    if not label:
        return None
    safe_label = _safe_filename(label)
    candidates = sorted(bank_dir.glob(f"*_{safe_label}.py"))
    if not candidates:
        candidates = [
            path
            for path in sorted(bank_dir.glob("*.py"))
            if safe_label in path.name or label in path.name
        ]
    for program_path in candidates:
        try:
            hints = _load_candidate_hints(program_path, context)
        except Exception:
            continue
        if not isinstance(hints, dict):
            continue
        hints = deepcopy(hints)
        hints["policy_bank_validated_initial"] = True
        hints["policy_bank_selected_label"] = label
        hints["policy_bank_reused_initial"] = True
        objective = summary.get("selected_initial_objective_cost_usec")
        if objective is not None:
            hints["policy_bank_selected_objective_cost_usec"] = objective
        return hints
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
        "unreachable_boundary_groups": 0,
        "candidate_improved_budgets": 0,
        "invalid_reasons": {},
        "candidate_invalid_reasons": {},
        "candidate_costs": [],
        "costs": [],
        "assignments": [],
        "selected_source_counts": {},
        "mcts_action_attempt_counts": {},
        "mcts_action_success_counts": {},
        "mcts_action_invalid_counts": {},
        "mcts_action_duplicate_skips": {},
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
            "unreachable_boundary_groups",
            "candidate_improved_budgets",
        ):
            totals[key] += int(item.get(key, 0))
        totals["candidate_costs"].extend(item.get("candidate_costs", []))
        for source, count in item.get("selected_source_counts", {}).items():
            totals["selected_source_counts"][source] = (
                totals["selected_source_counts"].get(source, 0) + int(count)
            )
        for dict_key in (
            "mcts_action_attempt_counts",
            "mcts_action_success_counts",
            "mcts_action_invalid_counts",
            "mcts_action_duplicate_skips",
        ):
            for name, count in item.get(dict_key, {}).items():
                totals[dict_key][name] = totals[dict_key].get(name, 0) + int(count)
        totals["costs"].extend(item.get("costs", []))
        totals["assignments"].extend(item.get("assignments", []))
        for dict_key in ("invalid_reasons", "candidate_invalid_reasons"):
            for reason, count in item.get(dict_key, {}).items():
                totals[dict_key][reason] = totals[dict_key].get(reason, 0) + int(count)
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
            "component_bootstrap_score": 0.0,
            "candidate_qbp_coverage": 0.0,
            "boundary_group_validity": 0.0,
            "placement_effect_score": 0.0,
            "policy_effect_score": 0.0,
            "action_effect_score": 0.0,
            "seed_equivalent_policy": 0.0,
            "placement_runtime_sec": 0.0,
            "estimated_precision_bits": 0.0,
            "output_margin_bits": 0.0,
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
            "combined_score": 0.0,
            "validity": 0.0,
            "latency_score": 0.0,
            "latency_only_correct": 0.0,
            "latency_only_score": 0.0,
            "objective_cost_usec": float("inf"),
            "base_objective_cost_usec": float("inf"),
            "reference_objective_cost_usec": 0.0,
            "objective_cost_ratio_vs_seed": 0.0,
            "objective_cost_score": 0.0,
            "objective_tier": 0.0,
            "total_frontier_cost_usec": float("inf"),
            "reachable_boundary_groups": 0.0,
            "unsolved_reachable_boundary_groups": 0.0,
            "candidate_direct_group_coverage": 0.0,
            "direct_unsolved_boundary_groups": 0.0,
            "objective_penalty_usec": float("inf"),
            "final_latency_usec": 0.0,
            "boundary_quality": 0.0,
            "bootstrap_count": 0.0,
            "component_bootstrap_score": 0.0,
            "rescale_count": 0.0,
            "profile_risk": 1.0,
            "placement_runtime_sec": 0.0,
            "fallback_selected_budgets": 0.0,
            "boundary_group_validity": 0.0,
            "candidate_qbp_coverage": 0.0,
            "placement_effect_score": 0.0,
            "policy_effect_score": 0.0,
            "action_effect_score": 0.0,
            "seed_equivalent_policy": 0.0,
            "fallback_selected_groups": 0.0,
            "invalid_boundary_groups": 0.0,
            "estimated_precision_bits": 0.0,
            "output_margin_bits": 0.0,
        },
        "artifacts": {
            "failure_stage": stage,
            "invalid_reasons": json.dumps(reason_counts, sort_keys=True),
            "correctness_gate": json.dumps(
                {"correct": False, "reasons": [stage, *reasons[:8]]},
                sort_keys=True,
            ),
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
    if hints.get("scale_floor_bits") is not None:
        floor = _int_hint(hints.get("scale_floor_bits"), int(ckks["Sw"]))
        if floor not in _scale_floor_candidates_from_context(context):
            reasons.append(f"scale_floor_bits {floor} is not one of configured floor candidates")
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
                "sampled_dp_latency_usec",
                "scale_floor_summary",
                "scale_floor_bits",
                "scale_floor_delta_bits",
                "sampled_selected_path_bootstraps",
                "sampled_selected_path_digest",
                "repair_summary",
                "per_unit_score_table",
                "selected_source_counts",
                "assignment_count_summary",
                "boundary_group_count_summary",
                "candidate_qbp_coverage",
                "boundary_group_validity",
                "fallback_selected_groups",
                "unreachable_boundary_groups",
                "component_bootstrap_alignment",
                "component_bootstrap_budget",
                "alphaevolve_feedback",
                "policy_effect_summary",
                "placement_effect_summary",
                "effective_qbp_digest",
                "selected_path_digest",
                "selected_path_changed_vs_seed",
                "effective_path_summary",
                "mcts_action_attempt_counts",
                "mcts_action_success_counts",
                "mcts_action_invalid_counts",
                "mcts_action_duplicate_skips",
                "boundary_group_unsolved_summary",
                "correctness_gate",
                "latency_only_objective",
                "trace_backed_candidate_examples",
                "execution_trace",
                "candidate_mlir_artifacts",
                "candidate_mlir_path",
                "candidate_mlir_digest",
                "candidate_mlir_preview",
                "candidate_mlir_trace_features",
                "trace_learning_feedback",
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


def _alphaevolve_feedback(
    context: dict[str, Any],
    hints: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    """Compact profiling feedback for the next OpenEvolve mutation.

    This mirrors the paper's use of execution traces: it converts Orbit's
    selected-action and invalid-action diagnostics into concrete policy moves.
    """

    metrics = result.get("metrics", result) if isinstance(result, dict) else {}
    artifacts = result.get("artifacts", {}) if isinstance(result, dict) else {}
    diagnostics = result.get("diagnostics", {}) if isinstance(result, dict) else {}
    selected = _dict_from_jsonish(
        artifacts.get("selected_source_counts", diagnostics.get("selected_source_counts", {}))
    )
    action_attempts = _dict_from_jsonish(
        artifacts.get(
            "mcts_action_attempt_counts",
            diagnostics.get("mcts_action_attempt_counts", {}),
        )
    )
    action_successes = _dict_from_jsonish(
        artifacts.get(
            "mcts_action_success_counts",
            diagnostics.get("mcts_action_success_counts", {}),
        )
    )
    invalid = _dict_from_jsonish(
        artifacts.get("candidate_invalid_reasons", diagnostics.get("candidate_invalid_reasons", {}))
    )
    policy_effect = _dict_from_jsonish(artifacts.get("policy_effect_summary", {}))
    placement_effect = _dict_from_jsonish(artifacts.get("placement_effect_summary", {}))
    component = result.get("component_bootstrap_alignment")
    if not isinstance(component, dict):
        component = _dict_from_jsonish(artifacts.get("component_bootstrap_alignment", {}))

    target = _context_target_bootstrap_count(context)
    bootstrap_count = _finite_float(
        metrics.get("bootstrap_count", result.get("bootstrap_count", 0.0)),
        0.0,
    )
    component_score = _finite_float(
        metrics.get(
            "component_bootstrap_score",
            component.get("score") if isinstance(component, dict) else None,
        ),
        0.0,
    )
    qbp_coverage = _finite_float(metrics.get("candidate_qbp_coverage"), 0.0)
    boundary_validity = _finite_float(metrics.get("boundary_group_validity"), 0.0)
    fallback_groups = _finite_float(metrics.get("fallback_selected_groups"), 0.0)

    selected_top = sorted(
        selected.items(), key=lambda item: (-_safe_int(item[1], 0), item[0])
    )[:5]
    invalid_top = sorted(
        invalid.items(), key=lambda item: (-_safe_int(item[1], 0), item[0])
    )[:5]
    suggestions: list[str] = []

    if any("component_budget_repair" in reason and "no feasible" in reason for reason, _count in invalid_top):
        suggestions.append(
            "component pressure repair is too strict; set force_bootstrap_anchors=False and increase boundary_state_cap/max_scale_candidates before adding anchors."
        )
    if any("latency_beam found no feasible" in reason for reason, _count in invalid_top):
        suggestions.append(
            "latency_beam is missing legal boundary states; try boundary_scale_policy='waterline' or 'frontier' and increase max_scale_candidates before widening beam_width."
        )
    if qbp_coverage < 1.0 or boundary_validity < 1.0:
        suggestions.append(
            "QBP coverage is incomplete; preserve whole boundary groups and sweep boundary_state_cap/max_scale_candidates rather than optimizing individual budgets."
        )
    if fallback_groups > 0:
        suggestions.append(
            "fallback groups were selected; make the candidate action solve those groups directly because fallback-heavy candidates cannot win finalist replay."
        )
    if selected_top and all("component_budget_repair" not in source for source, _count in selected_top) and component_score < 0.75:
        suggestions.append(
            "selected_source_counts miss high-maintenance actions; raise their priors or reduce cost penalties only if objective_cost_usec improves."
        )
    if policy_effect.get("seed_equivalent") is True:
        suggestions.append(
            "normalized policy is seed-equivalent after sampled bounding; mutate mcts_action_presets or high-impact MCTS keys so candidate_digest changes."
        )
    if _finite_float(metrics.get("seed_equivalent_path"), 0.0) >= 1.0:
        suggestions.append(
            "selected_path_digest is seed-equivalent; change boundary_state_cap, boundary_scale_policy, scale_floor_bits, or bootstrap anchor placement until the selected QBP path digest changes."
        )
    elif _finite_float(placement_effect.get("effect_score"), 0.0) <= 0.0:
        suggestions.append(
            "policy changed but QBP/DP placement metrics did not improve; retarget boundary_state_cap, max_scale_candidates, boundary scale policy, or selection_objective until solved groups, latency, or fallback counts move."
        )
    if action_attempts and not action_successes:
        suggestions.append(
            "MCTS actions are attempted but none solve budgets; use the invalid-action reasons to retarget anchors or switch boundary_scale_policy before increasing rollout_budget."
        )
    if not suggestions:
        suggestions.append(
            "Current candidate passed the main gates; mutate latency/rescale knobs conservatively while preserving QBP coverage."
        )

    return {
        "trace_model": "alphaevolve-fhe-style",
        "selected_source_counts_top": dict(selected_top),
        "candidate_invalid_reasons_top": dict(invalid_top),
        "mcts_action_attempt_counts_top": dict(
            sorted(action_attempts.items(), key=lambda item: (-_safe_int(item[1], 0), item[0]))[:5]
        ),
        "mcts_action_success_counts_top": dict(
            sorted(action_successes.items(), key=lambda item: (-_safe_int(item[1], 0), item[0]))[:5]
        ),
        "policy_effect": policy_effect,
        "placement_effect": placement_effect,
        "target_bootstrap_count": target,
        "bootstrap_count": bootstrap_count,
        "component_bootstrap_score": component_score,
        "candidate_qbp_coverage": qbp_coverage,
        "boundary_group_validity": boundary_validity,
        "fallback_selected_groups": fallback_groups,
        "policy_focus": {
            "mcts_action_cap": hints.get("mcts_action_cap"),
            "mcts_rollout_budget": hints.get("mcts_rollout_budget"),
            "component_budget_budgets_present": bool(_policy_unit_bootstrap_targets(hints)),
        },
        "suggestions": suggestions[:6],
    }


def _dict_from_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


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
        has_boundary_group_diagnostics = int(
            diagnostics.get("requested_boundary_groups", 0) or 0
        ) > 0
        solved_groups = int(diagnostics.get("solved_boundary_groups", 0) or 0)
        candidate_groups = int(diagnostics.get("candidate_solved_boundary_groups", 0) or 0)
        fallback_groups = int(diagnostics.get("fallback_selected_boundary_groups", 0) or 0)
        invalid_groups = int(diagnostics.get("invalid_boundary_groups", 0) or 0)
        unreachable_groups = int(diagnostics.get("unreachable_boundary_groups", 0) or 0)
        scored_groups = _scored_boundary_group_count(diagnostics)
        candidate_improved = int(diagnostics.get("candidate_improved_budgets", 0))
        requested = max(1, len(io_budgets))
        effective_validity = solved / requested
        validity = candidate_solved / requested
        if has_boundary_group_diagnostics:
            boundary_group_validity = min(1.0, solved_groups / scored_groups)
            candidate_qbp_coverage = min(1.0, candidate_groups / scored_groups)
        else:
            boundary_group_validity = min(1.0, effective_validity)
            candidate_qbp_coverage = min(1.0, validity)
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
        target_bootstrap_score = _contextual_target_bootstrap_score(
            context,
            target_bootstrap_count,
            counts["bootstrap"],
            reference["bootstrap_count"],
        )
        component_bootstrap = _component_bootstrap_alignment_summary(
            context,
            _maintenance_locations(assignments[0])["bootstrap"] if assignments else {},
            counts["bootstrap"],
        )
        component_bootstrap_score = float(component_bootstrap.get("score", 0.0))
        rescale_score = _relative_reduction(reference["rescale_count"], counts["rescale"])
        risk_score = 1.0 / (1.0 + profile_risk)
        fallback_score = 1.0 / (1.0 + fallback_selected + fallback_groups)
        quality_score = (
            0.39 * latency_score
            + 0.12 * bootstrap_score
            + 0.06 * target_bootstrap_score
            + 0.09 * component_bootstrap_score
            + 0.08 * rescale_score
            + 0.13 * risk_score
            + 0.12 * reserve_score
        )
        latency_correct = (
            bool(costs)
            and math.isfinite(avg_cost)
            and avg_cost > 0.0
            and effective_validity >= 1.0
            and boundary_group_validity >= 1.0
            and candidate_solved > 0
            and fallback_selected == 0
            and fallback_groups == 0
            and invalid_groups == 0
        )
        if latency_correct:
            reference_for_score = (
                reference_avg
                if math.isfinite(reference_avg) and reference_avg > 0.0
                else avg_cost
            )
            combined_score = max(0.0, reference_for_score / avg_cost)
        else:
            combined_score = 0.0
        return {
            "metrics": {
                "combined_score": float(combined_score),
                "latency_only_correct": float(latency_correct),
                "validity": float(validity),
                "effective_validity": float(effective_validity),
                "candidate_validity": float(validity),
                "boundary_group_validity": float(boundary_group_validity),
                "candidate_qbp_coverage": float(candidate_qbp_coverage),
                "placement_effect_score": 0.0,
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
                "component_bootstrap_score": float(component_bootstrap_score),
                "component_bootstrap_target": float(
                    component_bootstrap.get("target_total", 0)
                ),
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
                "unreachable_boundary_groups": float(unreachable_groups),
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
                "boundary_group_validity": f"{solved_groups}/{scored_groups} scored ({requested_groups} raw)",
                "candidate_qbp_coverage": f"{candidate_groups}/{scored_groups} scored ({requested_groups} raw)",
                "fallback_selected_groups": f"{fallback_groups}/{requested_groups}",
                "invalid_boundary_groups": f"{invalid_groups}/{requested_groups}",
                "unreachable_boundary_groups": f"{unreachable_groups}/{requested_groups}",
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
                "component_bootstrap_alignment": json.dumps(
                    component_bootstrap, sort_keys=True
                )[:4000],
                "reserve_summary": json.dumps(reserve_summary, sort_keys=True)[:4000],
                "policy_summary": _compact_policy_summary(hints),
                "repair_summary": json.dumps(_unit_policy_summary(context, hints), sort_keys=True),
                "unmatched_resilience_targets": json.dumps(
                    _profile_unmatched_targets(context), sort_keys=True
                ),
                "profiler_plan_hints": _compact_profile_plan(context),
                "alphaevolve_feedback": json.dumps(
                    _alphaevolve_feedback(
                        context,
                        hints,
                        {
                            "diagnostics": diagnostics,
                            "bootstrap_count": counts["bootstrap"],
                            "rescale_count": counts["rescale"],
                            "component_bootstrap_alignment": component_bootstrap,
                            "candidate_qbp_coverage": candidate_qbp_coverage,
                            "boundary_group_validity": boundary_group_validity,
                            "fallback_selected_groups": fallback_groups,
                            "valid": bool(solved),
                        },
                    ),
                    sort_keys=True,
                )[:4000],
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


def _qbp_dp_state_key(state: _QBPDPState) -> tuple[int, int]:
    return (int(state.level), int(state.scale))


def _qbp_dp_assign_data() -> dict[str, Any]:
    return {
        "v_lvl_out": {},
        "v_scl_out": {},
        "v_lvl_in": {},
        "v_scl_in": {},
        "e_lvl_out": {},
        "e_scl_out": {},
    }


def _qbp_dp_copy_assign_data(data: dict[str, Any] | None) -> dict[str, Any]:
    result = _qbp_dp_assign_data()
    if not isinstance(data, dict):
        return result
    for key in result:
        value = data.get(key, {})
        result[key] = dict(value) if isinstance(value, dict) else {}
    return result


def _qbp_dp_set_assign_value(
    data: dict[str, Any],
    table: str,
    key: Any,
    value: int,
    *,
    allow_overwrite: bool = False,
) -> bool:
    target = data.setdefault(table, {})
    if key in target and int(target[key]) != int(value) and not allow_overwrite:
        return False
    target[key] = int(value)
    return True


def _qbp_dp_merge_assign_data(
    states: list[_QBPDPState],
    *,
    materialize: bool,
) -> dict[str, Any] | None:
    if not materialize:
        return None
    merged = _qbp_dp_assign_data()
    for state in states:
        data = state.assign_data
        if not isinstance(data, dict):
            continue
        for table in merged:
            for key, value in dict(data.get(table, {}) or {}).items():
                if not _qbp_dp_set_assign_value(merged, table, key, int(value)):
                    return None
    return merged


def _qbp_dp_transition_metrics(
    params: Params,
    le: LatencyEstimator,
    in_lvl: int,
    in_scl: int,
    out_lvl: int,
    out_scl: int,
) -> tuple[float, float, float] | None:
    if not params.check_resbts(in_lvl, in_scl, out_lvl, out_scl):
        return None
    try:
        cost = float(le.resbts_cost(in_lvl, in_scl, out_lvl, out_scl))
    except Exception:
        return None
    uses_bootstrap = not params.check_res(in_lvl, in_scl, out_lvl, out_scl)
    if uses_bootstrap:
        before = max(0, int(in_lvl) - int(params.bts_lb))
        after = max(0, int(out_lvl) - int(params.bts_lb))
        rescale_count = float(before + after)
    else:
        rescale_count = float(max(0, int(in_lvl) - int(out_lvl)))
    return cost, 1.0 if uses_bootstrap else 0.0, rescale_count


def _qbp_dp_op_cost(
    tdag: Tdag,
    node: str,
    input_level: int,
    le: LatencyEstimator,
) -> float:
    op = tdag.nodes[node].get("op")
    if op in {"input", "output", "dummy", "constant"}:
        return 0.0
    single_cnt, double_cnt = tdag.get_v_weights(node)
    cost = 0.0
    if single_cnt > 0:
        cost += float(le.op_lmaps[f"{op}_single"][input_level]) * single_cnt
    if double_cnt > 0:
        cost += float(le.op_lmaps[f"{op}_double"][input_level]) * double_cnt
    return cost


def _qbp_dp_state_sort_key(state: _QBPDPState) -> tuple[float, float, float, int, int]:
    return (
        float(state.cost),
        float(state.bootstrap_count),
        float(state.rescale_count),
        int(state.level),
        int(state.scale),
    )


def _qbp_dp_prune_states(
    states: list[_QBPDPState],
    cap: int,
) -> list[_QBPDPState]:
    best_by_key: dict[tuple[int, int], _QBPDPState] = {}
    for state in states:
        key = _qbp_dp_state_key(state)
        prev = best_by_key.get(key)
        if prev is None or _qbp_dp_state_sort_key(state) < _qbp_dp_state_sort_key(prev):
            best_by_key[key] = state
    ranked = sorted(best_by_key.values(), key=_qbp_dp_state_sort_key)
    return ranked[: max(1, int(cap))]


def _qbp_dp_main_qbp_choices(io_budget: dict[str, Any]) -> list[tuple[tuple[int, int] | None, float]]:
    raw = io_budget.get("main_qbp_cost")
    if not isinstance(raw, dict) or not raw:
        return [(None, 0.0)]
    choices: list[tuple[tuple[int, int], float]] = []
    for key, value in raw.items():
        try:
            if isinstance(key, tuple):
                lvl, scl = int(key[0]), int(key[1])
            elif isinstance(key, list):
                lvl, scl = int(key[0]), int(key[1])
            else:
                text = str(key).strip().strip("()[]")
                parts = [part.strip() for part in text.split(",") if part.strip()]
                lvl, scl = int(parts[0]), int(parts[1])
            choices.append(((lvl, scl), float(value)))
        except Exception:
            continue
    choices.sort(key=lambda item: item[1])
    return choices or [(None, 0.0)]


def _qbp_dp_input_state(
    tdag: Tdag,
    node: str,
    params: Params,
    io_budget: dict[str, Any],
    fixed_inputs: dict[str, tuple[int, int]],
    *,
    materialize: bool,
) -> _QBPDPState:
    if node in fixed_inputs:
        level, scale = fixed_inputs[node]
    else:
        scale = int(io_budget.get("in_scl", params.scale_lower_bound(node, tdag.nodes[node], "in")))
        if scale < 0:
            scale = params.scale_lower_bound(node, tdag.nodes[node], "in")
        scale = max(scale, params.scale_lower_bound(node, tdag.nodes[node], "in"))
        level = int(io_budget.get("in_lvl", -1))
        if level < 0:
            level = _highest_decryptable_level(params, scale)
    _assert_decryptable(params, level, scale)
    assign_data = _qbp_dp_assign_data() if materialize else None
    if assign_data is not None:
        assign_data["v_lvl_in"][node] = int(level)
        assign_data["v_scl_in"][node] = int(scale)
        assign_data["v_lvl_out"][node] = int(level)
        assign_data["v_scl_out"][node] = int(scale)
    return _QBPDPState(int(level), int(scale), 0.0, 0.0, 0.0, assign_data)


def _qbp_dp_constant_state(
    tdag: Tdag,
    node: str,
    params: Params,
    *,
    materialize: bool,
) -> _QBPDPState:
    assign_data = _qbp_dp_assign_data() if materialize else None
    if assign_data is not None:
        assign_data["v_lvl_out"][node] = int(params.lvl_ub)
        assign_data["v_scl_out"][node] = int(params.Csw)
    return _QBPDPState(int(params.lvl_ub), int(params.Csw), 0.0, 0.0, 0.0, assign_data)


def _qbp_dp_level_candidates(
    params: Params,
    preferred: int | None,
    policy: dict[str, Any],
) -> list[int]:
    values = {params.lvl_lb, params.lvl_ub, params.bts_lb, params.bts_lb + 1}
    if preferred is not None:
        values.add(int(preferred))
    min_level = _effective_min_internal_level(policy, params)
    return [
        level
        for level in sorted(v for v in values | set(range(params.lvl_lb, params.lvl_ub + 1)) if params.lvl_lb <= v <= params.lvl_ub)
        if level >= min_level
    ] or list(range(params.lvl_lb, params.lvl_ub + 1))


def _qbp_dp_scale_candidates(
    values: list[Any],
    lower: int,
    upper: int,
    params: Params,
    policy: dict[str, Any],
    cap: int | None = None,
) -> list[int]:
    local_policy = dict(policy)
    if cap is not None:
        local_policy["max_scale_candidates"] = min(
            _int_hint(local_policy.get("max_scale_candidates"), 32),
            int(cap),
        )
    return _scale_candidates(values, int(lower), int(upper), params, local_policy)


def _qbp_dp_output_states(
    tdag: Tdag,
    node: str,
    params: Params,
    io_budget: dict[str, Any],
    policy: dict[str, Any],
    input_level: int,
    input_scale: int,
    node_level_hint: int | None,
    node_scale_hint: int | None,
) -> list[tuple[int, int]]:
    is_output = node in tdag.outputs
    if is_output and int(io_budget.get("out_lvl", -1)) >= 0:
        levels = [int(io_budget["out_lvl"])]
    else:
        levels = _qbp_dp_level_candidates(params, node_level_hint, policy)
    lower = params.scale_lower_bound(node, tdag.nodes[node], "out")
    values = [lower, input_scale, params.Sw, params.Csw, params.Sf, node_scale_hint]
    results: list[tuple[int, int]] = []
    for level in levels:
        upper = min(int(policy.get("max_scale", _max_scale(params))), params.decryptable_scale_bound(level))
        if is_output:
            upper = min(upper, params.boundary_output_scale_bound(level))
        if upper < lower:
            continue
        for scale in _qbp_dp_scale_candidates(values, lower, upper, params, policy):
            if params.is_decryptable_state(level, scale):
                results.append((int(level), int(scale)))
    return results


def _qbp_dp_incoming_options(
    tdag: Tdag,
    node: str,
    pred_states: list[tuple[str, _QBPDPState]],
    params: Params,
    policy: dict[str, Any],
    node_scale_hint: int | None,
    edge_scale_hints: dict[str, int],
) -> list[tuple[int, dict[str, int], int]]:
    preds = [pred for pred, _state in pred_states]
    lower = max(
        params.scale_lower_bound(node, tdag.nodes[node], "in"),
        int(node_scale_hint or 0),
    )
    max_scale = int(policy.get("max_scale", _max_scale(params)))
    if max_scale < lower:
        return []
    op = tdag.nodes[node].get("op")
    options: list[tuple[int, dict[str, int], int]] = []
    if op != "mul":
        values: list[Any] = [lower, params.Sw, params.Csw, params.Sf, node_scale_hint]
        for pred, state in pred_states:
            values.extend([state.scale, edge_scale_hints.get(_edge_key(pred, node))])
            if tdag.nodes[pred].get("op") == "constant":
                values.append(params.Csw)
        for scale in _qbp_dp_scale_candidates(values, lower, max_scale, params, policy):
            options.append((int(scale), {pred: int(scale) for pred in preds}, int(scale)))
        return options

    per_pred: list[tuple[str, list[int]]] = []
    for pred, state in pred_states:
        if tdag.nodes[pred].get("op") == "constant":
            per_pred.append((pred, [int(params.Csw)]))
            continue
        edge_lower = _edge_scale_lb(tdag, pred, node, params)
        edge_upper = max(edge_lower, max_scale // 2 if len(preds) == 1 else max_scale)
        values = [
            edge_lower,
            edge_scale_hints.get(_edge_key(pred, node)),
            state.scale,
            math.ceil(state.scale / 2),
            params.Sw,
            params.Csw,
            params.Sf,
        ]
        per_pred.append(
            (
                pred,
                _qbp_dp_scale_candidates(values, edge_lower, edge_upper, params, policy),
            )
        )
    if len(per_pred) == 1:
        pred, scales = per_pred[0]
        for scale in scales:
            node_scale = 2 * int(scale)
            if lower <= node_scale <= max_scale:
                options.append((node_scale, {pred: int(scale)}, node_scale))
        return options
    if len(per_pred) > 2:
        return []
    (p0, s0s), (p1, s1s) = per_pred
    for s0 in s0s:
        for s1 in s1s:
            node_scale = int(s0) + int(s1)
            if lower <= node_scale <= max_scale:
                options.append((node_scale, {p0: int(s0), p1: int(s1)}, node_scale))
    return options


def _qbp_dp_add_assignment_for_transition(
    data: dict[str, Any],
    tdag: Tdag,
    node: str,
    pred_states: list[tuple[str, _QBPDPState]],
    edge_levels: dict[str, int],
    edge_scales: dict[str, int],
    input_level: int,
    input_scale: int,
    output_level: int,
    output_scale: int,
    params: Params,
) -> bool:
    for pred, _state in pred_states:
        if tdag.nodes[pred].get("op") == "constant":
            const_scale = params.Csw if tdag.nodes[node].get("op") == "mul" else edge_scales[pred]
            if not _qbp_dp_set_assign_value(data, "v_lvl_out", pred, input_level, allow_overwrite=True):
                return False
            if not _qbp_dp_set_assign_value(data, "v_scl_out", pred, const_scale, allow_overwrite=True):
                return False
            continue
        edge = (pred, node)
        if not _qbp_dp_set_assign_value(data, "e_lvl_out", edge, edge_levels[pred]):
            return False
        if not _qbp_dp_set_assign_value(data, "e_scl_out", edge, edge_scales[pred]):
            return False
    for table, value in (
        ("v_lvl_in", input_level),
        ("v_scl_in", input_scale),
        ("v_lvl_out", output_level),
        ("v_scl_out", output_scale),
    ):
        if not _qbp_dp_set_assign_value(data, table, node, value):
            return False
    return True


def _qbp_dp_node_states(
    tdag: Tdag,
    node: str,
    pred_state_lists: list[list[_QBPDPState]],
    params: Params,
    le: LatencyEstimator,
    io_budget: dict[str, Any],
    hints: dict[str, Any],
    *,
    materialize: bool,
) -> list[_QBPDPState]:
    node_hints = _policy_hints_for_node(hints, list(hints.get("unit_policies", []) or []), str(node), dict(tdag.nodes[node]))
    policy = _policy_options(node_hints, params)
    node_level_hints = _int_map(hints.get("preferred_node_levels", {}))
    node_scale_hints = _int_map(hints.get("preferred_node_scales", {}))
    edge_scale_hints = _int_map(hints.get("preferred_edge_scales", {}))
    state_cap = max(1, min(64, _int_hint(policy.get("state_cap_per_node"), 16)))
    candidate_states: list[_QBPDPState] = []
    preds = list(tdag.predecessors(node))
    for combo in itertools.product(*[states[:state_cap] for states in pred_state_lists]):
        pred_states = list(zip(preds, combo))
        for input_scale, edge_scales, node_scale in _qbp_dp_incoming_options(
            tdag,
            node,
            pred_states,
            params,
            policy,
            node_scale_hints.get(node),
            edge_scale_hints,
        ):
            for input_level in _qbp_dp_level_candidates(params, node_level_hints.get(node), policy):
                edge_cost = 0.0
                edge_bootstrap = 0.0
                edge_rescale = 0.0
                edge_levels: dict[str, int] = {}
                feasible = True
                for pred, state in pred_states:
                    if tdag.nodes[pred].get("op") == "constant":
                        edge_levels[pred] = int(input_level)
                        continue
                    metrics = _qbp_dp_transition_metrics(
                        params,
                        le,
                        state.level,
                        state.scale,
                        input_level,
                        edge_scales[pred],
                    )
                    if metrics is None:
                        feasible = False
                        break
                    cost, bootstraps, rescales = metrics
                    edge_cost += float(tdag.edges[pred, node].get("weight", 1)) * cost
                    edge_bootstrap += bootstraps
                    edge_rescale += rescales
                    edge_levels[pred] = int(input_level)
                if not feasible or not params.is_decryptable_state(input_level, node_scale):
                    continue
                for output_level, output_scale in _qbp_dp_output_states(
                    tdag,
                    node,
                    params,
                    io_budget,
                    policy,
                    input_level,
                    node_scale,
                    node_level_hints.get(node),
                    node_scale_hints.get(node),
                ):
                    metrics = _qbp_dp_transition_metrics(
                        params,
                        le,
                        input_level,
                        node_scale,
                        output_level,
                        output_scale,
                    )
                    if metrics is None:
                        continue
                    vertex_cost, vertex_bootstrap, vertex_rescale = metrics
                    op_cost = _qbp_dp_op_cost(tdag, node, input_level, le)
                    total_cost = (
                        sum(state.cost for state in combo)
                        + edge_cost
                        + op_cost
                        + float(tdag.nodes[node].get("weight", 1)) * vertex_cost
                    )
                    assign_data = _qbp_dp_merge_assign_data(list(combo), materialize=materialize)
                    if materialize:
                        if assign_data is None:
                            continue
                        if not _qbp_dp_add_assignment_for_transition(
                            assign_data,
                            tdag,
                            node,
                            pred_states,
                            edge_levels,
                            edge_scales,
                            input_level,
                            node_scale,
                            output_level,
                            output_scale,
                            params,
                        ):
                            continue
                    candidate_states.append(
                        _QBPDPState(
                            int(output_level),
                            int(output_scale),
                            float(total_cost),
                            float(sum(state.bootstrap_count for state in combo) + edge_bootstrap + vertex_bootstrap),
                            float(sum(state.rescale_count for state in combo) + edge_rescale + vertex_rescale),
                            assign_data,
                        )
                    )
    return _qbp_dp_prune_states(candidate_states, state_cap)


def _qbp_dp_solve_budget(
    tdag: Tdag,
    params: Params,
    io_budget: dict[str, Any],
    le: LatencyEstimator,
    hints: dict[str, Any],
    *,
    materialize: bool,
) -> _QBPDPBudgetResult:
    hints = _with_default_policy(hints)
    main_choices = _qbp_dp_main_qbp_choices(io_budget)
    best: _QBPDPBudgetResult | None = None
    input_node = list(tdag.inputs)[0]
    output_node = list(tdag.outputs)[0]
    for fixed_main, main_cost in main_choices:
        fixed_inputs = {}
        if fixed_main is not None and io_budget.get("maino_v"):
            fixed_inputs[str(io_budget["maino_v"])] = fixed_main
        states_by_node: dict[str, list[_QBPDPState]] = {}
        try:
            for node in nx.topological_sort(tdag):
                op = tdag.nodes[node].get("op")
                if op == "constant":
                    states_by_node[node] = [_qbp_dp_constant_state(tdag, node, params, materialize=materialize)]
                    continue
                if node in tdag.inputs or tdag.in_degree(node) == 0:
                    states_by_node[node] = [
                        _qbp_dp_input_state(
                            tdag,
                            node,
                            params,
                            io_budget,
                            fixed_inputs,
                            materialize=materialize,
                        )
                    ]
                    continue
                pred_lists = [states_by_node.get(pred, []) for pred in tdag.predecessors(node)]
                if any(not states for states in pred_lists):
                    states_by_node[node] = []
                    continue
                states_by_node[node] = _qbp_dp_node_states(
                    tdag,
                    node,
                    pred_lists,
                    params,
                    le,
                    io_budget,
                    hints,
                    materialize=materialize,
                )
            output_states = states_by_node.get(output_node, [])
        except Exception as exc:
            return _QBPDPBudgetResult(False, (-1, -1), None, float("inf"), 0.0, 0.0, reason=str(exc)[:240])
        for state in output_states:
            if int(io_budget.get("out_lvl", -1)) >= 0 and state.level != int(io_budget["out_lvl"]):
                continue
            input_state = states_by_node[input_node][0]
            in_key = (int(input_state.level), int(input_state.scale))
            out_key = (int(state.level), int(state.scale))
            assign = None
            actual_cost = float(state.cost + main_cost)
            if materialize:
                if state.assign_data is None:
                    continue
                try:
                    assign = Assign.from_dict(tdag, state.assign_data)
                    assign.check_assign()
                    actual_cost = float(estimate_assign(assign, le) + main_cost)
                except Exception as exc:
                    continue
            item = _QBPDPBudgetResult(
                True,
                in_key,
                out_key,
                actual_cost,
                float(state.bootstrap_count),
                float(state.rescale_count),
                assign,
            )
            if best is None or (item.cost, item.bootstrap_count, item.rescale_count) < (
                best.cost,
                best.bootstrap_count,
                best.rescale_count,
            ):
                best = item
    return best or _QBPDPBudgetResult(False, (-1, -1), None, float("inf"), 0.0, 0.0, reason="qbp-dp found no feasible boundary state")


def _solve_budget_batch_qbp_dp(
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
    for group_key, budgets in _budget_boundary_groups(io_budgets_list).items():
        selected_attempts: list[_BudgetAttempt] = []
        candidate_solved_count = 0
        fallback_selected_count = 0
        reachable_budget_indices: set[int] = set()
        for budget_idx, io_budget in enumerate(budgets):
            result = _qbp_dp_solve_budget(
                pdag,
                params,
                io_budget,
                le,
                hints,
                materialize=True,
            )
            attempts: list[_BudgetAttempt] = []
            if result.valid and result.assign is not None and result.out_key is not None:
                attempts.append(
                    _BudgetAttempt(
                        "candidate:qbp_dp",
                        result.assign,
                        result.cost,
                        result.in_key,
                        result.out_key,
                        result.cost,
                        dict(hints),
                    )
                )
                candidate_solved_count += 1
            elif diagnostics is not None:
                _record_invalid_reason(
                    diagnostics,
                    "candidate_invalid_reasons",
                    PlacementError(result.reason or "qbp-dp found no feasible assignment"),
                )
            if not attempts and _should_use_seed_fallback(hints):
                for source, policy_hints in _seed_fallback_attempts(params):
                    try:
                        attempts.append(_solve_one_budget_attempt(pdag, params, io_budget, le, source, policy_hints))
                    except Exception as exc:
                        if diagnostics is not None:
                            _record_invalid_reason(diagnostics, "invalid_reasons", exc)
            if not attempts:
                if (
                    not _boundary_budget_output_cannot_refresh(params, group_key, io_budget)
                    and _probe_boundary_budget_reachable(pdag, params, io_budget, le)
                ):
                    reachable_budget_indices.add(budget_idx)
                continue
            reachable_budget_indices.add(budget_idx)
            best_attempt = _best_attempt(attempts)
            if best_attempt is None:
                continue
            if best_attempt.source.startswith("seed_fallback"):
                fallback_selected_count += 1
            selected_attempts.append(best_attempt)
            if diagnostics is not None:
                candidate_attempt = _best_candidate_attempt(attempts, params, hints)
                fallback_attempt = _best_attempt(
                    attempt for attempt in attempts if attempt.source.startswith("seed_fallback")
                )
                _record_selected_attempt(diagnostics, best_attempt, candidate_attempt, fallback_attempt)
            current = io_to_cost.get(best_attempt.in_key, {}).get(best_attempt.out_key)
            if current is None or best_attempt.cost < current:
                io_to_cost.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.cost
                io_to_assign.setdefault(best_attempt.in_key, {})[best_attempt.out_key] = best_attempt.assign
        if diagnostics is not None:
            _record_boundary_group_result(
                diagnostics,
                params,
                group_key,
                budgets,
                selected_attempts,
                candidate_solved_count,
                fallback_selected_count,
                len(reachable_budget_indices),
            )
    return io_to_assign, io_to_cost


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
        if _use_qbp_dp_engine(params, hints):
            return _solve_budget_batch_qbp_dp(
                pdag,
                io_budgets_list,
                le,
                params,
                hints,
                diagnostics,
            )
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
        if (
            not getattr(params, "openevolve_evaluating_candidate", False)
            and getattr(params, "openevolve_compile_hints", None) is not None
        ):
            parallel = _solve_budget_batch_boundary_mcts_parallel(
                pdag,
                io_budgets_list,
                params,
                hints,
                diagnostics,
            )
            if parallel is not None:
                return parallel
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

        candidate_attempt = _best_candidate_attempt(attempts, params, hints)
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


def _solve_budget_batch_boundary_mcts_parallel(
    pdag: Tdag,
    io_budgets_list: list[dict],
    params: Params,
    hints: dict[str, Any],
    diagnostics: dict[str, Any] | None,
) -> tuple[
    dict[tuple[int, int], dict[tuple[int, int], Assign]],
    dict[tuple[int, int], dict[tuple[int, int], float]],
] | None:
    grouped_budgets = list(_budget_boundary_groups(io_budgets_list).values())
    if len(grouped_budgets) > 1:
        task_budgets = grouped_budgets
        task_kind = "boundary_groups"
    elif len(io_budgets_list) > 1:
        task_budgets = [[dict(budget)] for budget in io_budgets_list]
        task_kind = "output_budgets"
    else:
        task_budgets = grouped_budgets
        task_kind = "single"
    workers = _parallel_qbp_worker_count(params, len(task_budgets))
    if workers <= 1 or len(task_budgets) <= 1:
        return None
    print(
        "OpenEvolve QBP parallel boundary replay: "
        f"pdag={pdag.name} tasks={len(task_budgets)} kind={task_kind} workers={workers}",
        flush=True,
    )
    payloads = [
        (build_context(pdag, [dict(budget) for budget in budgets], params), hints)
        for budgets in task_budgets
    ]
    io_to_assign: dict[tuple[int, int], dict[tuple[int, int], Assign]] = {}
    io_to_cost: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for payload in executor.map(_run_boundary_group_replay_payload, payloads):
            if payload.get("error"):
                if diagnostics is not None:
                    _record_invalid_reason(
                        diagnostics,
                        "invalid_reasons",
                        PlacementError(str(payload.get("error"))),
                    )
                continue
            context = payload.get("context")
            if not isinstance(context, dict):
                continue
            task_tdag = tdag_from_context(context)
            for record in payload.get("records", []) or []:
                if not isinstance(record, dict):
                    continue
                try:
                    in_key = (int(record["in_key"][0]), int(record["in_key"][1]))
                    out_key = (int(record["out_key"][0]), int(record["out_key"][1]))
                    cost = float(record["cost"])
                    assign = _assign_from_serialized(task_tdag, record["assign"])
                except Exception as exc:
                    if diagnostics is not None:
                        _record_invalid_reason(diagnostics, "invalid_reasons", exc)
                    continue
                current = io_to_cost.get(in_key, {}).get(out_key)
                if current is None or cost < current:
                    io_to_cost.setdefault(in_key, {})[out_key] = cost
                    io_to_assign.setdefault(in_key, {})[out_key] = assign
            if diagnostics is not None:
                _merge_boundary_replay_diagnostics(
                    diagnostics,
                    payload.get("diagnostics", {}),
                )
    return io_to_assign, io_to_cost


def _parallel_qbp_worker_count(params: Params, group_count: int) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_QBP_WORKERS", "").strip()
    cap_raw = os.environ.get("ORBIT_OPENEVOLVE_MAX_QBP_WORKERS", "").strip()
    try:
        if raw:
            requested = int(raw)
        else:
            requested = int(getattr(params, "threads", 1) or 1)
    except ValueError:
        requested = int(getattr(params, "threads", 1) or 1)
    try:
        worker_cap = int(cap_raw) if cap_raw else 96
    except ValueError:
        worker_cap = 96
    worker_cap = max(1, min(worker_cap, 128))
    return max(1, min(int(group_count or 0), requested, worker_cap))


def _run_boundary_group_replay_payload(
    payload: tuple[dict[str, Any], dict[str, Any]]
) -> dict[str, Any]:
    context, hints = payload
    try:
        task_tdag = tdag_from_context(context)
        task_params = task_tdag.params
        task_params.openevolve_iterations = 0
        task_params.openevolve_harness = "compile"
        task_params.openevolve_compile_hints = hints
        task_params.openevolve_evaluating_candidate = False
        _apply_active_scale_floor_from_hints(task_params, hints)
        task_le = LatencyEstimator(task_params)
        budgets = [
            _io_budget_from_json(item)
            for item in context.get("io_budgets", [])
            if isinstance(item, dict)
        ]
        diagnostics: dict[str, Any] = {}
        _init_budget_diagnostics(diagnostics, len(budgets))
        io_to_assign, io_to_cost = _solve_budget_batch_boundary_mcts(
            task_tdag,
            budgets,
            task_le,
            task_params,
            hints,
            diagnostics,
        )
        records = []
        for in_key, out_to_cost in io_to_cost.items():
            for out_key, cost in out_to_cost.items():
                assign = io_to_assign.get(in_key, {}).get(out_key)
                if assign is None:
                    continue
                records.append(
                    {
                        "in_key": [int(in_key[0]), int(in_key[1])],
                        "out_key": [int(out_key[0]), int(out_key[1])],
                        "cost": float(cost),
                        "assign": _serialize_assign(assign),
                    }
                )
        diagnostics.pop("assignments", None)
        return {
            "context": context,
            "records": records,
            "diagnostics": diagnostics,
        }
    except Exception as exc:
        return {
            "context": context,
            "records": [],
            "diagnostics": locals().get("diagnostics", {}),
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def _merge_boundary_replay_diagnostics(
    total: dict[str, Any], item: dict[str, Any] | None
) -> None:
    if not isinstance(item, dict):
        return
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
        "unreachable_boundary_groups",
        "candidate_improved_budgets",
    ):
        total[key] = int(total.get(key, 0) or 0) + int(item.get(key, 0) or 0)
    for key in ("costs", "candidate_costs"):
        total.setdefault(key, []).extend(item.get(key, []) or [])
    for dict_key in (
        "selected_source_counts",
        "mcts_action_attempt_counts",
        "mcts_action_success_counts",
        "mcts_action_invalid_counts",
        "mcts_action_duplicate_skips",
        "candidate_invalid_reasons",
        "invalid_reasons",
    ):
        dest = total.setdefault(dict_key, {})
        for name, count in dict(item.get(dict_key, {}) or {}).items():
            dest[str(name)] = int(dest.get(str(name), 0) or 0) + int(count or 0)
    summaries = total.setdefault("boundary_group_summaries", [])
    if len(summaries) < 512:
        summaries.extend(list(item.get("boundary_group_summaries", []) or [])[: 512 - len(summaries)])


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
    diagnostics["unreachable_boundary_groups"] = 0
    diagnostics["candidate_improved_budgets"] = 0
    diagnostics["costs"] = []
    diagnostics["candidate_costs"] = []
    diagnostics["assignments"] = []
    diagnostics["selected_source_counts"] = {}
    diagnostics["mcts_action_attempt_counts"] = {}
    diagnostics["mcts_action_success_counts"] = {}
    diagnostics["mcts_action_invalid_counts"] = {}
    diagnostics["mcts_action_duplicate_skips"] = {}
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
    params: Params,
    group_key: tuple,
    budgets: list[dict],
    selected_attempts: list[_BudgetAttempt],
    candidate_solved_count: int,
    fallback_selected_count: int,
    reachable_budget_count: int | None = None,
) -> None:
    requested = len(budgets)
    solved = len(selected_attempts)
    unreachable_budget_count = sum(
        1
        for budget in budgets
        if _boundary_budget_output_cannot_refresh(params, group_key, budget)
    )
    if reachable_budget_count is None:
        reachable_requested = max(solved, requested - unreachable_budget_count)
    else:
        reachable_requested = max(solved, min(requested, int(reachable_budget_count)))
        unreachable_budget_count = max(0, requested - reachable_requested)
    diagnostics["requested_boundary_groups"] += 1
    if requested > 0 and reachable_requested > 0 and solved >= reachable_requested:
        diagnostics["solved_boundary_groups"] += 1
    elif solved > 0:
        diagnostics["partial_boundary_groups"] += 1
    elif reachable_requested <= 0 or _boundary_group_input_cannot_refresh(params, group_key):
        diagnostics["unreachable_boundary_groups"] += 1
    else:
        diagnostics["invalid_boundary_groups"] += 1
    if requested > 0 and reachable_requested > 0 and candidate_solved_count >= reachable_requested:
        diagnostics["candidate_solved_boundary_groups"] += 1
    if fallback_selected_count > 0:
        diagnostics["fallback_selected_boundary_groups"] += 1

    selected_counts = _assignment_count_summary([attempt.assign for attempt in selected_attempts])
    selected_costs = [_attempt_actual_cost(attempt) for attempt in selected_attempts]
    selected_sources = Counter(attempt.source for attempt in selected_attempts)
    if selected_costs:
        avg_cost = float(sum(selected_costs) / len(selected_costs))
        min_cost = float(min(selected_costs))
        max_cost = float(max(selected_costs))
    else:
        avg_cost = min_cost = max_cost = 0.0
    if requested > 0 and reachable_requested > 0 and solved >= reachable_requested and candidate_solved_count >= reachable_requested:
        unsolved_reason = "complete_candidate"
    elif requested > 0 and reachable_requested > 0 and solved >= reachable_requested:
        unsolved_reason = "complete_with_fallback"
    elif solved > 0:
        unsolved_reason = "partial_output_levels"
    elif reachable_requested <= 0 or _boundary_group_input_cannot_refresh(params, group_key):
        unsolved_reason = "unreachable_input_cannot_refresh"
    else:
        unsolved_reason = "no_feasible_attempt"
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
        "reachable_budgets": int(reachable_requested),
        "unreachable_budgets": int(max(0, requested - reachable_requested)),
        "solved_budgets": solved,
        "candidate_solved_budgets": int(candidate_solved_count),
        "fallback_selected_budgets": int(fallback_selected_count),
        "complete": bool(
            requested > 0 and reachable_requested > 0 and solved >= reachable_requested
        ),
        "candidate_complete": bool(
            requested > 0
            and reachable_requested > 0
            and candidate_solved_count >= reachable_requested
        ),
        "selected_source_counts": dict(selected_sources),
        "unsolved_reason": unsolved_reason,
        "unreachable_input": bool(
            solved == 0
            and (
                reachable_requested <= 0
                or _boundary_group_input_cannot_refresh(params, group_key)
            )
        ),
        "avg_bootstrap": float(selected_counts["avg_bootstrap"]),
        "min_bootstrap": float(selected_counts["min_bootstrap"]),
        "max_bootstrap": float(selected_counts["max_bootstrap"]),
        "avg_rescale": float(selected_counts["avg_rescale"]),
        "min_rescale": float(selected_counts["min_rescale"]),
        "max_rescale": float(selected_counts["max_rescale"]),
        "avg_cost_usec": avg_cost,
        "min_cost_usec": min_cost,
        "max_cost_usec": max_cost,
    }
    summaries = diagnostics.setdefault("boundary_group_summaries", [])
    if len(summaries) < 256:
        summaries.append(summary)


def _boundary_group_input_cannot_refresh(params: Params, group_key: tuple) -> bool:
    """Return true for sampled input states that cannot legally bootstrap.

    QBP budget tables can contain boundary states that are syntactically present
    but unusable for a deep downstream partition. Original Orbit's DP can ignore
    those states; sampled OpenEvolve feedback should not treat them the same as
    a reachable group that the candidate failed to place.
    """

    try:
        in_lvl = int(group_key[0])
        in_scl = int(group_key[1])
    except (TypeError, ValueError, IndexError):
        return False
    if in_lvl < 0:
        return False
    return not params.check_res(in_lvl, in_scl, params.bts_lb, params.Sf)


def _boundary_budget_output_cannot_refresh(
    params: Params, group_key: tuple, budget: dict[str, Any]
) -> bool:
    """Return true when a sampled output state is unreachable from this input.

    Some sampled QBP groups include high output levels for an input state that
    cannot legally bootstrap. Original Orbit's DP can simply avoid those output
    states; OpenEvolve scoring should not require a candidate to solve them.
    """

    if not _boundary_group_input_cannot_refresh(params, group_key):
        return False
    try:
        in_lvl = int(group_key[0])
        in_scl = int(group_key[1])
        out_lvl = int(budget.get("out_lvl", -1))
    except (TypeError, ValueError, IndexError):
        return False
    if out_lvl < 0:
        return False
    if out_lvl < params.lvl_lb or out_lvl > params.lvl_ub:
        return True
    # Without a bootstrap, Orbit's rescale/modswitch legality is:
    # Sf*in_lvl - in_scl >= Sf*out_lvl - out_scl. If the minimum output
    # scale that satisfies this is above the legal/decryptable scale range,
    # this output state is unreachable for the sampled input boundary.
    min_out_scl = max(0, params.Sf * out_lvl - (params.Sf * in_lvl - in_scl))
    max_out_scl = min(params.max_scale(), params.decryptable_scale_bound(out_lvl))
    return min_out_scl > max_out_scl


def _probe_boundary_budget_reachable(
    pdag: Tdag,
    params: Params,
    budget: dict[str, Any],
    le: LatencyEstimator,
) -> bool:
    """Check candidate-independent reachability for sampled QBP feedback.

    This deliberately does not add a fallback assignment to the candidate. It
    only separates "candidate failed" from "the sampled output state is not
    materializable by Orbit's deterministic no-solver placement helpers."
    """

    for _source, policy_hints in _seed_fallback_attempts(params):
        try:
            _solve_one_budget_attempt(
                pdag,
                params,
                budget,
                le,
                "reachability_probe",
                policy_hints,
            )
            return True
        except Exception:
            continue
    return False


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
        reachable_budget_indices: set[int] = set()
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
            has_candidate_attempt = any(
                attempt.source.startswith("candidate") for attempt in attempts
            )
            if not has_candidate_attempt and _should_use_seed_fallback(hints):
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
                if (
                    not _boundary_budget_output_cannot_refresh(params, _group_key, io_budget)
                    and _probe_boundary_budget_reachable(pdag, params, io_budget, le)
                ):
                    reachable_budget_indices.add(index)
                continue
            reachable_budget_indices.add(index)
            candidate_attempt = _best_candidate_attempt(attempts, params, hints)
            fallback_attempt = _best_attempt(
                attempt for attempt in attempts if attempt.source.startswith("seed_fallback")
            )
            best_attempt = candidate_attempt or fallback_attempt or _best_attempt(attempts)
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
                params,
                _group_key,
                budgets,
                selected_attempts,
                group_candidate_solved,
                group_fallback_selected,
                len(reachable_budget_indices),
            )
    return io_to_assign, io_to_cost


def _budget_boundary_groups(io_budgets_list: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for budget in io_budgets_list:
        key = _budget_boundary_group_key(budget)
        groups.setdefault(key, []).append(budget)
    for budgets in groups.values():
        budgets.sort(key=lambda item: int(item.get("out_lvl", -1)))
    return groups


def _budget_boundary_group_key(budget: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(budget.get("in_lvl", -1)),
        int(budget.get("in_scl", -1)),
        str(budget.get("maino_v", "")),
        int(budget.get("main_dag_size", 0) or 0),
    )


def _apply_boundary_group_policy_overlays(
    hints: dict[str, Any],
    group_key: tuple[int, int, str, int],
) -> dict[str, Any]:
    policies = hints.get("boundary_group_policies")
    if not isinstance(policies, list) or not policies:
        return hints
    merged = dict(hints)
    for item in policies:
        if not isinstance(item, dict):
            continue
        selector = item.get("selector", {})
        if not _boundary_group_selector_matches_key(selector, group_key):
            continue
        patch = item.get("policy", {})
        if not isinstance(patch, dict):
            continue
        presets = patch.get("mcts_action_presets")
        if isinstance(presets, (dict, list)):
            merged["mcts_action_presets"] = _merge_mcts_action_presets(
                merged.get("mcts_action_presets"),
                presets,
                replace=_bool_hint(patch.get("replace_mcts_action_presets"), False),
            )
        action_patch = _boundary_group_action_patch(patch)
        if action_patch:
            action_names = _boundary_group_overlay_action_names(merged)
            if action_names:
                merged["mcts_action_presets"] = _merge_mcts_action_presets(
                    merged.get("mcts_action_presets"),
                    {
                        name: {"policy": dict(action_patch)}
                        for name in action_names
                    },
                    replace=False,
                )
        for key, value in patch.items():
            if key in {
                "selector",
                "policy",
                "mcts_action_presets",
                "replace_mcts_action_presets",
                "boundary_group_policies",
            }:
                continue
            if key in _PATCHABLE_POLICY_KEYS or key in {
                "direct_budget_policy",
                "forbid_bootstrap",
                "budget_aggressive",
                "boundary_scale_policy",
                "boundary_state_cap",
                "bootstrap_anchor_count",
                "bootstrap_anchor_level",
                "bootstrap_anchor_selector",
                "bootstrap_anchor_include_patterns",
                "bootstrap_anchor_exclude_patterns",
                "force_bootstrap_anchors",
                "selection_objective",
                "prefer_component_budget_fit",
                "noise_slack_model",
                "mcts_action_cap",
                "mcts_rollout_budget",
                "mcts_prior_order",
                "mcts_action_allowlist",
                "mcts_action_blocklist",
            }:
                merged[key] = value
    return merged


def _boundary_group_action_patch(patch: dict[str, Any]) -> dict[str, Any]:
    """Project a local boundary overlay onto direct MCTS action policies."""

    if not isinstance(patch, dict):
        return {}
    action_patch: dict[str, Any] = {}
    for key, value in patch.items():
        if key in {
            "selector",
            "policy",
            "mcts_action_presets",
            "replace_mcts_action_presets",
            "boundary_group_policies",
            "mcts_action_allowlist",
            "mcts_action_blocklist",
            "mcts_action_cap",
            "mcts_rollout_budget",
            "mcts_prior_order",
        }:
            continue
        if key in _PATCHABLE_POLICY_KEYS or key in {
            "direct_budget_policy",
            "forbid_bootstrap",
            "budget_aggressive",
            "boundary_scale_policy",
            "boundary_state_cap",
            "bootstrap_anchor_count",
            "bootstrap_anchor_level",
            "bootstrap_anchor_selector",
            "bootstrap_anchor_include_patterns",
            "bootstrap_anchor_exclude_patterns",
            "force_bootstrap_anchors",
            "selection_objective",
            "prefer_component_budget_fit",
            "noise_slack_model",
        }:
            action_patch[key] = value
    return action_patch


def _boundary_group_overlay_action_names(hints: dict[str, Any]) -> list[str]:
    direct_actions = {
        "budget_fulfillment_beam",
        "wide_boundary_cost_beam",
        "dense_boundary_cost_beam",
        "reference_boundary_cost_beam",
        "waterline_cost_beam",
        "nonlinear_phase_boundary_beam",
        "profile_waterline_repair",
        "tuneinsight_avgcase_cost_beam",
        "tuneinsight_deferred_bootstrap_beam",
        "latency_mcts_repair",
    }
    allowlist = [
        str(item)
        for item in hints.get("mcts_action_allowlist", []) or []
        if str(item)
    ]
    if allowlist:
        names = [name for name in allowlist if name in direct_actions]
        if names:
            return names
    presets = hints.get("mcts_action_presets")
    if isinstance(presets, dict):
        names = [str(name) for name in presets if str(name) in direct_actions]
        if names:
            return names
    return ["budget_fulfillment_beam"]


def _seed_fallback_attempts(params: Params) -> list[tuple[str, dict[str, Any]]]:
    seed = _default_policy_hints()
    beam = _budget_fulfillment_beam_policy()
    if not getattr(params, "openevolve_evaluating_candidate", False):
        return [("seed_fallback_latency_beam", beam)]
    if getattr(params, "openevolve_eval_suite", "polybert-sampled") != "polybert-full":
        return [
            ("seed_fallback", seed),
            ("seed_fallback_relaxed", _relaxed_scheduler_policy(seed, params)),
            ("seed_fallback_waterline", _waterline_seed_policy()),
        ]
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
            "bootstrap_penalty": 250_000_000.0,
            "selection_bootstrap_penalty": 0.0,
            "selection_objective": "cost",
            "beam_width": 8,
            "state_cap_per_node": 32,
            "boundary_state_cap": 6,
            "max_scale_candidates": 40,
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
    group_key = _budget_boundary_group_key(budgets[0]) if budgets else (-1, -1, "", 0)
    hints = _apply_boundary_group_policy_overlays(hints, group_key)
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
    direct_attempts = {}
    if _use_direct_budget_beam(hints, policy):
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
        _record_diag_count(diagnostics, "mcts_action_attempt_counts", action.name)
        action_digest = json.dumps(action.policy, sort_keys=True, default=str)
        if action_digest in seen_actions and step >= len(actions):
            _record_diag_count(diagnostics, "mcts_action_duplicate_skips", action.name)
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
                    if best_for_budget is None or _attempt_selection_rank(
                        attempt, variant, params
                    ) > _attempt_selection_rank(
                        best_for_budget, best_for_budget.policy or variant, params
                    ):
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
            _record_diag_count(
                diagnostics,
                "mcts_action_success_counts",
                action.name,
                len(action_attempts),
            )
            best_cost = min(attempt.cost for attempt in action_attempts)
            best_assign = min(action_attempts, key=lambda attempt: attempt.cost).assign
            if reward > stats[idx].best_reward:
                stats[idx].best_reward = reward
                stats[idx].best_assign = best_assign
                stats[idx].best_cost = best_cost
        else:
            stats[idx].reward_sum += _mcts_invalid_reward(action, step)
            if diagnostics is not None:
                _record_diag_count(
                    diagnostics,
                    "mcts_action_invalid_counts",
                    action.name,
                    sum(invalid_reasons.values()) or 1,
                )
                for reason, count in invalid_reasons.most_common(3):
                    diagnostics.setdefault("candidate_invalid_reasons", {})[
                        f"{action.name}: {reason}"
                    ] = diagnostics.setdefault("candidate_invalid_reasons", {}).get(
                        f"{action.name}: {reason}", 0
                    ) + count
        if _boundary_mcts_group_target_met(action_attempts, len(budgets), params, hints):
            break
    if _bool_hint(hints.get("include_seed_repair_actions"), False) or _bool_hint(
        policy.get("include_seed_repair_actions"), False
    ):
        candidate_attempts = [
            attempt
            for attempts in attempts_by_budget.values()
            for attempt in attempts
            if attempt.source.startswith("candidate:")
        ]
        if _boundary_mcts_group_target_met(candidate_attempts, len(budgets), params, hints):
            return attempts_by_budget
        for label, repair_policy in _seed_fallback_attempts(params):
            for budget_idx, budget in enumerate(budgets):
                try:
                    attempts_by_budget[budget_idx].append(
                        _solve_one_budget_attempt(
                            pdag,
                            params,
                            budget,
                            le,
                            f"candidate:seed_repair:{label}",
                            repair_policy,
                        )
                    )
                except Exception as exc:
                    if diagnostics is not None:
                        _record_invalid_reason(diagnostics, "candidate_invalid_reasons", exc)
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


def _use_direct_budget_beam(hints: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Opt in to the expensive direct beam pre-pass.

    The beam is useful as a budget-fulfillment repair action, but running it
    before every MCTS action made compile-level startup spend minutes in the
    seed replay before OpenEvolve got a chance to mutate anything.
    """

    return _bool_hint(hints.get("enable_direct_budget_beam"), False) or _bool_hint(
        policy.get("enable_direct_budget_beam"), False
    )


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
        dict(policy_hints),
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
    return _BudgetAttempt("candidate_record", assign, actual_cost, in_key, out_key, actual_cost, None)


def _best_attempt(attempts) -> _BudgetAttempt | None:
    best = None
    for attempt in attempts:
        if best is None or attempt.cost < best.cost:
            best = attempt
    return best


def _candidate_attempt_objective(attempts: list[_BudgetAttempt]) -> bool:
    return any(
        _selection_objective(attempt.policy or {}) != "cost"
        for attempt in attempts
    )


def _best_candidate_attempt(
    attempts,
    params: Params,
    root_hints: dict[str, Any] | None = None,
) -> _BudgetAttempt | None:
    candidate_attempts = [
        attempt for attempt in attempts if attempt.source.startswith("candidate")
    ]
    if not candidate_attempts:
        return None
    root_objective = _selection_objective(root_hints or {})
    honor_action_objectives = _bool_hint(
        (root_hints or {}).get("honor_action_selection_objectives"),
        False,
    )
    if root_objective == "cost" and not honor_action_objectives:
        return _best_attempt(candidate_attempts)
    if not _candidate_attempt_objective(candidate_attempts):
        return _best_attempt(candidate_attempts)
    return max(
        candidate_attempts,
        key=lambda attempt: _attempt_selection_rank(
            attempt,
            attempt.policy or {},
            params,
        ),
    )


def _selection_objective(policy: dict[str, Any]) -> str:
    objective = str(policy.get("selection_objective", "cost")).strip().lower()
    if objective in {"cost", "latency", "runtime"}:
        return "cost"
    return "cost"


def _attempt_selection_rank(
    attempt: _BudgetAttempt,
    policy: dict[str, Any],
    params: Params,
) -> tuple[float, float]:
    objective = _selection_objective(policy)
    if objective == "cost":
        return (-float(attempt.cost), -float(_attempt_actual_cost(attempt)))

    counts = _maintenance_counts(attempt.assign)
    locations = _maintenance_locations(attempt.assign)
    bootstrap_count = float(counts["bootstrap"])
    target_score = _policy_target_bootstrap_score(policy, params, bootstrap_count, None)
    component_context = _component_context_from_policy(policy)
    component_score = _component_bootstrap_alignment_score(
        component_context,
        locations.get("bootstrap", {}),
        bootstrap_count,
    )
    rescale_score = 1.0 / (1.0 + float(counts["rescale"]) / 32.0)
    latency_score = 1.0 / (1.0 + max(0.0, _attempt_actual_cost(attempt)) / 1_000_000_000.0)
    if objective == "min_bootstrap":
        return (
            -bootstrap_count,
            0.20 * rescale_score + 0.80 * latency_score,
        )
    if objective == "target_bootstrap_fit":
        score = 0.62 * target_score + 0.18 * component_score + 0.12 * rescale_score + 0.08 * latency_score
    else:
        score = 0.42 * target_score + 0.38 * component_score + 0.12 * rescale_score + 0.08 * latency_score
    return (float(score), -float(_attempt_actual_cost(attempt)))


def _component_context_from_policy(policy: dict[str, Any]) -> dict[str, Any]:
    targets = _policy_unit_bootstrap_targets(policy)
    return {
        "unit_bootstrap_budget": {
            "unit_budgets": {
                unit_id: {"target": target}
                for unit_id, target in targets.items()
            }
        },
        "placement_units": [],
    }


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


def _record_diag_count(
    diagnostics: dict[str, Any] | None,
    key: str,
    name: str,
    count: int = 1,
) -> None:
    if diagnostics is None:
        return
    values = diagnostics.setdefault(key, {})
    values[str(name)] = values.get(str(name), 0) + int(count)


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
            16,
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
        if anchors and _bool_hint(base.get("force_bootstrap_anchors"), False):
            forced = dict(variant)
            forced["force_bootstrap_nodes"] = list(anchors)
            variants.append(_with_default_policy(forced))
            soft = dict(variant)
            soft["force_bootstrap_nodes"] = []
            variants.append(_with_default_policy(soft))
        else:
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
    selector = str(policy.get("bootstrap_anchor_selector", "")).strip().lower()
    if selector in {
        "nonlinear_phase_boundaries",
        "nonlinear_phase_boundary",
        "phase_boundaries",
        "phase_boundary",
    }:
        anchors = _nonlinear_phase_boundary_anchor_nodes(tdag, count, policy)
        if anchors:
            return anchors
    if selector in {
        "reference_bootstrap_locations",
        "reference_bootstrap_location",
        "reference_locations",
    }:
        anchors = _reference_bootstrap_anchor_nodes(tdag, count, policy)
        if anchors:
            return anchors
    depths = _node_depths(tdag)
    component_targets = _policy_unit_bootstrap_targets(policy)
    if not component_targets:
        component_targets = _policy_unit_bootstrap_targets(root_hints)

    def rank(node: str) -> tuple[float, int, int, str]:
        attrs = dict(tdag.nodes[node])
        unit_budget = max(
            [component_targets.get(unit_id, 0) for unit_id in _node_unit_ids(attrs)] or [0]
        )
        nonlinear_bonus = 1 if _node_nonlinear_kind(attrs) else 0
        op_bonus = 1 if attrs.get("op") == "mul" else 0
        fanout = int(tdag.out_degree(node))
        return (-(4 * unit_budget + 2 * nonlinear_bonus + op_bonus), -fanout, -depths.get(node, 0), node)

    candidates = [
        str(node)
        for node in tdag.nodes
        if tdag.nodes[node].get("op") not in {"input", "constant"}
    ]
    return sorted(candidates, key=rank)[:count]


_NONLINEAR_PHASE_INCLUDE_PATTERNS: tuple[tuple[str, int], ...] = (
    ("layernorm_avg_res_by", 110),
    ("layernorm_avg_res", 108),
    ("inv_sqrt_seed", 106),
    ("inv_sqrt_newton_update", 104),
    ("layernorm_centered_sigma", 102),
    ("qk_numerator_sum", 98),
    ("qk_row_sum_add", 96),
    ("approx_exp_rescale", 94),
    ("reciprocal_newton_update", 92),
    ("attention_value_combined_output", 88),
    ("attention_value_matmul_output", 86),
    ("split_output_bias_add", 82),
    ("pooler_dense", 78),
)


_NONLINEAR_PHASE_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "variance_residual",
    "approx_exp_square",
    "newton_xy",
    "newton_yx",
    "newton_yyx",
    "y_square",
    "output_rescale",
    "denominator_rescale",
    "numerator_rescale",
    "softmax_mul",
    "packed_ctpt_linear_rescale",
    "residual_add",
)


def _string_list_hint(value: Any, limit: int = 32) -> list[str]:
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple)):
        raw_items = list(value)
    else:
        return []
    result: list[str] = []
    for item in raw_items[:limit]:
        text = str(item).strip().lower()
        if text:
            result.append(text)
    return result


def _phase_boundary_score(attrs: dict[str, Any], policy: dict[str, Any]) -> int | None:
    text = f"{_node_scope(attrs)};{_node_op_tag(attrs)};{attrs.get('comment', '')}".lower()
    excludes = list(_NONLINEAR_PHASE_EXCLUDE_PATTERNS)
    excludes.extend(_string_list_hint(policy.get("bootstrap_anchor_exclude_patterns")))
    if any(pattern and pattern in text for pattern in excludes):
        return None

    includes = list(_NONLINEAR_PHASE_INCLUDE_PATTERNS)
    includes.extend(
        (pattern, 72)
        for pattern in _string_list_hint(policy.get("bootstrap_anchor_include_patterns"))
    )
    best = max((weight for pattern, weight in includes if pattern in text), default=0)
    nonlinear_kind = _node_nonlinear_kind(attrs)
    if nonlinear_kind == "attention_softmax" and ("sum" in text or "rescale" in text):
        best = max(best, 70)
    if nonlinear_kind == "reciprocal" and ("seed" in text or "update" in text):
        best = max(best, 68)
    if nonlinear_kind == "norm" and ("avg" in text or "sigma" in text or "inv_sqrt" in text):
        best = max(best, 66)
    if "attention_value" in text and ("output" in text or "matmul" in text):
        best = max(best, 64)
    if ("bias_add" in text or "dense" in text) and ("output" in text or "pooler" in text):
        best = max(best, 58)
    return best or None


def _nonlinear_phase_boundary_anchor_nodes(
    tdag: Tdag,
    count: int,
    policy: dict[str, Any],
) -> list[str]:
    depths = _node_depths(tdag)

    def rank(node: str) -> tuple[int, int, int, str]:
        attrs = dict(tdag.nodes[node])
        score = _phase_boundary_score(attrs, policy) or 0
        fanout = int(tdag.out_degree(node))
        return (-score, -fanout, -depths.get(str(node), 0), str(node))

    candidates = [
        str(node)
        for node in tdag.nodes
        if tdag.nodes[node].get("op") not in {"input", "constant"}
        and _phase_boundary_score(dict(tdag.nodes[node]), policy) is not None
    ]
    return sorted(candidates, key=rank)[: max(0, min(count, 64))]


def _reference_bootstrap_anchor_nodes(
    tdag: Tdag,
    count: int,
    policy: dict[str, Any],
) -> list[str]:
    patterns = _string_list_hint(policy.get("bootstrap_anchor_include_patterns"), limit=64)
    if not patterns:
        return []
    depths = _node_depths(tdag)

    def pattern_score(attrs: dict[str, Any]) -> int | None:
        text = f"{_node_location(attrs)};{_node_scope(attrs)};{_node_op_tag(attrs)};{attrs.get('comment', '')}".lower()
        for idx, pattern in enumerate(patterns):
            if pattern and pattern in text:
                return 1000 - idx
        return None

    def rank(node: str) -> tuple[int, int, int, str]:
        attrs = dict(tdag.nodes[node])
        score = pattern_score(attrs) or 0
        fanout = int(tdag.out_degree(node))
        return (-score, -fanout, -depths.get(str(node), 0), str(node))

    candidates = [
        str(node)
        for node in tdag.nodes
        if tdag.nodes[node].get("op") not in {"input", "constant"}
        and pattern_score(dict(tdag.nodes[node])) is not None
    ]
    return sorted(candidates, key=rank)[: max(0, min(count, 64))]


def _policy_unit_bootstrap_targets(policy: dict[str, Any]) -> dict[str, int]:
    raw = policy.get("component_bootstrap_budgets") or policy.get("unit_bootstrap_budgets")
    if not isinstance(raw, dict):
        return {}
    if "unit_budgets" in raw and isinstance(raw.get("unit_budgets"), dict):
        raw = raw["unit_budgets"]
    targets: dict[str, int] = {}
    for unit_id, value in raw.items():
        if isinstance(value, dict):
            value = value.get("target")
        target = max(0, _safe_int(value, 0))
        if target > 0:
            targets[str(unit_id)] = target
    return targets


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
    bootstrap_score = _policy_target_bootstrap_score(
        hints,
        params,
        summary["avg_bootstrap"],
        None,
    )
    rescale_score = 1.0 / (1.0 + summary["avg_rescale"] / 32.0)
    cost_score = 1.0 / (1.0 + (sum(attempt.cost for attempt in attempts) / len(attempts)) / 1_000_000_000.0)
    objective = _selection_objective(hints)
    if objective == "cost":
        return 0.72 * coverage + 0.20 * cost_score + 0.05 * bootstrap_score + 0.03 * rescale_score
    if objective == "min_bootstrap":
        low_bootstrap_score = 1.0 / (1.0 + summary["avg_bootstrap"])
        return 0.58 * coverage + 0.30 * low_bootstrap_score + 0.07 * cost_score + 0.05 * rescale_score
    return 0.50 * coverage + 0.30 * bootstrap_score + 0.12 * rescale_score + 0.08 * cost_score


def _boundary_mcts_group_target_met(
    attempts: list[_BudgetAttempt],
    requested: int,
    params: Params,
    hints: dict[str, Any],
) -> bool:
    if requested <= 0 or len(attempts) < requested:
        return False
    if _selection_objective(hints) == "cost":
        return False
    summary = _assignment_count_summary([attempt.assign for attempt in attempts])
    return _policy_bootstrap_target_met(hints, params, summary["avg_bootstrap"])


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
            strategy = str(policy.get("strategy"))
            if strategy not in {"latency_beam", "waterline_seed"}:
                strategy = "level_preserving"
            policy["strategy"] = strategy
            actions.append(
                MCTSAction(
                    name=str(item.get("name", f"candidate_action_{idx}")),
                    policy=policy,
                    prior=_float_hint(item.get("prior"), 0.0),
                )
            )
    if not actions:
        actions = _default_bootstrap_mcts_actions(hints, params)
    actions = _apply_mcts_action_presets(actions, hints)
    allowlist = {
        str(item)
        for item in hints.get("mcts_action_allowlist", []) or []
        if str(item)
    }
    blocklist = {
        str(item)
        for item in hints.get("mcts_action_blocklist", []) or []
        if str(item)
    }
    if allowlist:
        actions = [action for action in actions if action.name in allowlist]
    if blocklist:
        actions = [action for action in actions if action.name not in blocklist]
    if _bool_hint(hints.get("mcts_prior_order"), True):
        actions = sorted(actions, key=lambda action: (-float(action.prior), action.name))
    if not actions:
        actions = _apply_mcts_action_presets(_default_bootstrap_mcts_actions(hints, params), hints)
    return actions[: max(1, min(64, _int_hint(hints.get("mcts_action_cap"), 24)))]


def _apply_mcts_action_presets(
    actions: list[MCTSAction],
    hints: dict[str, Any],
) -> list[MCTSAction]:
    preset_map = _mcts_action_preset_map(hints.get("mcts_action_presets"))
    if not preset_map:
        return actions
    updated: list[MCTSAction] = []
    for action in actions:
        preset = preset_map.get(action.name)
        if not isinstance(preset, dict):
            updated.append(action)
            continue
        if _bool_hint(preset.get("disabled"), False) or preset.get("enabled") is False:
            continue
        policy = dict(action.policy)
        policy_overrides = preset.get("policy", preset)
        if isinstance(policy_overrides, dict):
            for key, value in policy_overrides.items():
                if key in _PATCHABLE_POLICY_KEYS or key in {
                    "direct_budget_policy",
                    "forbid_bootstrap",
                    "budget_aggressive",
                    "boundary_scale_policy",
                    "boundary_state_cap",
                    "bootstrap_anchor_count",
                    "bootstrap_anchor_level",
                    "bootstrap_anchor_selector",
                    "bootstrap_anchor_include_patterns",
                    "bootstrap_anchor_exclude_patterns",
                    "force_bootstrap_anchors",
                    "component_bootstrap_budgets",
                    "unit_bootstrap_budgets",
                    "selection_objective",
                    "prefer_component_budget_fit",
                    "noise_slack_model",
                }:
                    policy[key] = value
        updated.append(
            MCTSAction(
                action.name,
                _with_default_policy(policy),
                _float_hint(preset.get("prior"), action.prior),
            )
        )
    return updated


def _mcts_action_preset_map(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        result = {}
        for name, preset in value.items():
            if isinstance(preset, dict):
                result[str(name)] = dict(preset)
        return result
    if isinstance(value, list):
        result = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if name is None:
                continue
            result[str(name)] = dict(item)
        return result
    return {}


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
            "boundary_state_cap": max(1, min(16, _int_hint(base.get("boundary_state_cap"), 3))),
            "bootstrap_anchor_count": max(0, _int_hint(base.get("bootstrap_anchor_count"), 0)),
        }
    )
    target = max(0, _int_hint(base.get("target_bootstrap_count"), 0))
    explicit_anchor_count = max(0, min(8, target)) if target > 0 else 0
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
            0.05,
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
            0.12,
        ),
        (
            "minimal_bootstrap_repair",
            {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "bootstrap_penalty": max(2_500_000_000.0, _float_hint(base.get("bootstrap_penalty"), 0.0)),
                "selection_bootstrap_penalty": max(
                    250_000_000.0, _float_hint(base.get("selection_bootstrap_penalty"), 0.0)
                ),
                "level_drop_penalty": 0.0,
                "rescale_penalty": max(25_000.0, _float_hint(base.get("rescale_penalty"), 0.0)),
                "boundary_scale_policy": "frontier",
                "bootstrap_anchor_count": explicit_anchor_count,
                "boundary_state_cap": max(3, _int_hint(base.get("boundary_state_cap"), 3)),
                "selection_objective": "min_bootstrap",
            },
            0.18,
        ),
        (
            "waterline_budget_repair",
            {
                **_waterline_seed_policy(),
                "direct_budget_policy": True,
                "allow_seed_fallback": False,
                "target_bootstrap_count": target,
                "selection_bootstrap_penalty": max(
                    250_000_000.0,
                    _float_hint(base.get("selection_bootstrap_penalty"), 0.0),
                ),
            },
            0.20,
        ),
        (
            "budget_fulfillment_beam",
            {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.54,
        ),
        (
            "wide_boundary_cost_beam",
            {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.36,
        ),
        (
            "dense_boundary_cost_beam",
            {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "dense",
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.40,
        ),
        (
            "nonlinear_phase_boundary_beam",
            {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                "bootstrap_anchor_count": 64,
                "force_bootstrap_anchors": False,
                "bootstrap_penalty": 35_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.44,
        ),
        (
            "profile_waterline_repair",
            {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "state_cap_per_node": 24,
                "beam_width": 6,
                "bootstrap_penalty": 125_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
            0.24,
        ),
        (
            "tuneinsight_avgcase_cost_beam",
            {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "sf",
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.22,
        ),
        (
            "tuneinsight_deferred_bootstrap_beam",
            {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 650_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
            0.30,
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
                "bootstrap_anchor_count": explicit_anchor_count,
            },
            0.19,
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
                "bootstrap_anchor_count": explicit_anchor_count,
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
    if _selection_objective(root_hints) == "cost":
        # This is an internal rollout heuristic, not the OpenEvolve fitness.
        # The external evaluator still gates correctness and ranks by raw
        # latency. A small latency-priced maintenance regularizer keeps local
        # rollouts from choosing paths whose isolated transition costs look
        # cheap but explode global bootstrap/rescale maintenance.
        op_penalty = 0.0002 * counts["rescale"] + 0.004 * counts["bootstrap"]
        latency_score = 1.0 / (1.0 + max(0.0, cost) / 1_000_000_000.0)
        return latency_score - op_penalty
    bootstrap_score = _policy_target_bootstrap_score(
        root_hints,
        params,
        counts["bootstrap"],
        None,
    )
    op_penalty = 0.0005 * counts["rescale"] + 0.02 * counts["bootstrap"]
    latency_penalty = min(0.25, max(0.0, cost) / 1_000_000_000_000.0)
    no_bootstrap_bonus = (
        0.08
        if counts["bootstrap"] == 0
        and not _policy_unit_bootstrap_targets(root_hints)
        else 0.0
    )
    strict_bonus = 0.04 if action_policy.get("forbid_bootstrap") else 0.0
    return bootstrap_score + no_bootstrap_bonus + strict_bonus - op_penalty - latency_penalty


def _mcts_invalid_reward(action: MCTSAction, step: int) -> float:
    return -0.08 + min(0.04, step * 0.001) + max(-0.05, min(0.05, action.prior))


def _mcts_target_met(assign: Assign, params: Params, hints: dict[str, Any]) -> bool:
    return _policy_bootstrap_target_met(
        hints,
        params,
        _aggregate_counts([assign])["bootstrap"],
    )


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


def _scale_floor_candidates_from_context(context: dict[str, Any]) -> list[int]:
    ckks = _ckks_dict(context)
    params_data = context.get("params", {}) if isinstance(context, dict) else {}
    constraints = context.get("constraints", {}).get("ckks", {}) if isinstance(context, dict) else {}
    raw = (
        params_data.get("openevolve_scale_floor_candidates")
        or constraints.get("scale_floor_candidates")
        or [int(ckks["Sw"])]
    )
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, (list, tuple)):
        items = list(raw)
    else:
        items = [raw]
    min_floor = _safe_int(
        params_data.get("scale_floor_min_bits", constraints.get("scale_floor_min_bits")),
        max(24, int(ckks["Sw"]) - 12),
    )
    candidates: list[int] = []
    for item in items:
        value = max(min_floor, min(int(ckks["Sw"]), _safe_int(item, int(ckks["Sw"]))))
        if value not in candidates:
            candidates.append(value)
    if int(ckks["Sw"]) not in candidates:
        candidates.insert(0, int(ckks["Sw"]))
    return candidates or [int(ckks["Sw"])]


def _scale_floor_bits_from_hints(params: Params, hints: dict[str, Any] | None) -> int:
    if getattr(params, "scale_floor_policy", "waterline") != "estimator-relaxed":
        return int(params.Sw)
    allowed = list(getattr(params, "openevolve_scale_floor_candidates", [int(params.Sw)]) or [int(params.Sw)])
    if int(params.Sw) not in allowed:
        allowed.insert(0, int(params.Sw))
    raw = int(params.Sw)
    if isinstance(hints, dict) and hints.get("scale_floor_bits") is not None:
        raw = _int_hint(hints.get("scale_floor_bits"), int(params.Sw))
    min_floor = int(getattr(params, "scale_floor_min_bits", max(24, int(params.Sw) - 12)))
    raw = max(min_floor, min(int(params.Sw), raw))
    return min(allowed, key=lambda value: (abs(int(value) - raw), -int(value)))


def _apply_active_scale_floor_from_hints(params: Params, hints: dict[str, Any] | None) -> int:
    floor = _scale_floor_bits_from_hints(params, hints)
    if getattr(params, "scale_floor_policy", "waterline") == "estimator-relaxed":
        params._active_scale_floor_bits = int(floor)
    else:
        params._active_scale_floor_bits = None
    if hasattr(params, "active_scale_floor_bits"):
        return int(params.active_scale_floor_bits())
    return int(getattr(params, "Sw", floor))


def _scale_floor_summary(params: Params, hints: dict[str, Any] | None, noise: dict[str, Any] | None = None) -> dict[str, Any]:
    floor = _scale_floor_bits_from_hints(params, hints)
    summary = {
        "policy": getattr(params, "scale_floor_policy", "waterline"),
        "waterline_bits": int(params.Sw),
        "active_floor_bits": int(floor),
        "scale_floor_delta_bits": int(params.Sw) - int(floor),
        "scale_floor_min_bits": int(getattr(params, "scale_floor_min_bits", int(params.Sw))),
        "candidates": list(getattr(params, "openevolve_scale_floor_candidates", [int(params.Sw)])),
        "relaxed": int(floor) < int(params.Sw),
        "estimator_mode": getattr(params, "noise_estimator", "off"),
    }
    if noise is not None:
        summary["estimator_valid"] = bool(noise.get("valid", False))
        summary["estimator_margin_bits"] = noise.get("output_margin_bits")
    return summary


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
    rollout_budget = (
        24
        if params is None
        else max(8, min(64, int(getattr(params, "openevolve_mcts_rollout_budget", 24))))
    )
    budget_aggressive = bool(getattr(params, "openevolve_budget_aggressive", False)) if params else False
    sampled_repair_cap = 128 if budget_aggressive else max(16, target)
    policy = _low_scale_frontier_policy()
    policy.update(
        {
            "strategy": "bootstrap_mcts",
            "budget_aggressive": budget_aggressive,
            "prefer_level_preservation": True,
            "allow_bootstrap": True,
            "forbid_bootstrap": False,
            "allow_seed_fallback": True,
            "refresh_fanout_at_level_floor": False,
            "max_scale_candidates": 24,
            "bootstrap_penalty": 250_000_000.0,
            "selection_bootstrap_penalty": 0.0,
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
            "mcts_action_cap": 6,
            "boundary_state_cap": 6,
            "enable_direct_budget_beam": False,
            "include_seed_repair_actions": False,
            "selection_objective": "cost",
        }
    )
    if params is not None:
        relaxed_floor_action_names = [
            f"estimator_relaxed_floor_{floor}"
            for floor in getattr(params, "openevolve_scale_floor_candidates", [])
            if int(floor) < int(params.Sw)
        ]
        policy["mcts_action_cap"] = 10
        policy["mcts_action_allowlist"] = [
            "budget_fulfillment_beam",
            "wide_boundary_cost_beam",
            "waterline_cost_beam",
            "profile_waterline_repair",
            "tuneinsight_avgcase_cost_beam",
            "tuneinsight_deferred_bootstrap_beam",
            *relaxed_floor_action_names,
            "latency_mcts_repair",
            "component_budget_repair",
            "minimal_bootstrap_repair",
            "waterline_budget_repair",
        ]
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
        "max_scale_candidates": max(3, min(128, _int_hint(hints.get("max_scale_candidates"), 32))),
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
        "beam_width": max(1, min(12, _int_hint(hints.get("beam_width"), 4))),
        "state_cap_per_node": max(1, min(64, _int_hint(hints.get("state_cap_per_node"), 8))),
        "scale_lattice": str(hints.get("scale_lattice", "default")),
        "mcts_rollout_budget": max(1, min(256, _int_hint(hints.get("mcts_rollout_budget"), 48))),
        "mcts_exploration_weight": max(
            0.0, min(8.0, _float_hint(hints.get("mcts_exploration_weight"), 1.4))
        ),
        "mcts_max_repair_bootstraps": max(
            0, min(1024, _int_hint(hints.get("mcts_max_repair_bootstraps"), 16))
        ),
        "enable_direct_budget_beam": _bool_hint(hints.get("enable_direct_budget_beam"), False),
        "force_bootstrap_nodes": {
            str(node) for node in hints.get("force_bootstrap_nodes", []) or []
        }
        if isinstance(hints.get("force_bootstrap_nodes", []), (list, tuple, set))
        else set(),
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
    lattice = str(policy.get("scale_lattice", "default"))
    for value in values:
        try:
            base = int(value)
        except (TypeError, ValueError):
            continue
        for delta in (0, -params.Sf, -2 * params.Sf, params.Sf):
            candidates.add(base + delta)
        candidates.add(math.ceil(base / 2))
        candidates.add(max(lower_bound, min(upper_bound, base)))
    max_count = int(policy["max_scale_candidates"])
    if lattice in {"dense", "dense_integer", "full"}:
        span = int(upper_bound) - int(lower_bound)
        if span <= max(0, max_count - len(candidates)):
            candidates.update(range(int(lower_bound), int(upper_bound) + 1))
        else:
            slots = max(2, max_count - len(candidates))
            step = max(1.0, span / max(1, slots - 1))
            for idx in range(slots):
                candidates.add(int(round(int(lower_bound) + idx * step)))
            # CKKS rescale boundaries often sit just around Sf/Sw offsets, so
            # include a small local neighborhood around each seed instead of
            # relying only on exact multiples.
            for seed in list(candidates):
                for delta in (-2, -1, 1, 2):
                    candidates.add(seed + delta)
    filtered = sorted({v for v in candidates if lower_bound <= v <= upper_bound})
    if not filtered:
        return [int(lower_bound)]
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
    if (
        str(v) in policy.get("force_bootstrap_nodes", set())
        and not policy.get("forbid_bootstrap")
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
        slack_multiplier = _noise_slack_reserve_multiplier(policy)
        reserve_deficit += slack_multiplier * max(
            0.0,
            float(policy.get("min_transition_reserve", 0))
            - float(_transition_reserve_bits(params, in_lvl, in_scl, out_lvl, out_scl)),
        )
        reserve_deficit += slack_multiplier * max(
            0.0,
            float(policy.get("min_decryptability_reserve", 0))
            - float(_decryptability_reserve_bits(params, out_lvl, out_scl)),
        )
    # Search heuristic only. OpenEvolve-visible combined_score and finalist
    # selection remain correctness-gated raw latency minimization.
    score = (
        cost
        + policy["bootstrap_penalty"] * counts["bootstrap"]
        + policy["rescale_penalty"] * counts["rescale"]
        + policy["level_drop_penalty"] * max(0, in_lvl - out_lvl)
        + reserve_penalty * reserve_deficit
    )
    cache[cache_key] = score
    return score


def _noise_slack_reserve_multiplier(policy: dict[str, Any]) -> float:
    """Scale deterministic reserve slack for average-case CKKS noise policies.

    TuneInsight's estimator propagates operation-level noise and precision
    directly, including rounding and key-switching noise. When a policy opts
    into that model, reserve slack should guide ranking but should not force
    early refreshes as aggressively as the old worst-case level/scale proxy.
    """

    model = str(policy.get("noise_slack_model", "worst_case")).strip().lower()
    if model in {"tuneinsight", "tuneinsight_avgcase", "average_case", "avgcase"}:
        return 0.25
    if model in {"off", "none", "ignore"}:
        return 0.0
    return 1.0


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


def _contextual_target_bootstrap_score(
    context: dict[str, Any],
    target: int | float | None,
    candidate: int | float,
    reference: int | float | None = None,
) -> float:
    component_target = _safe_int(
        context.get("unit_bootstrap_budget", {}).get("component_budget_total"),
        0,
    )
    if component_target <= 0:
        return _target_bootstrap_score(target, candidate, reference)
    target_int = max(component_target, _safe_int(target, 0))
    return _bootstrap_budget_fit_score(
        max(0.0, _finite_float(candidate, float("inf"))),
        target_int,
        reference,
        component_target=component_target,
    )


def _policy_target_bootstrap_score(
    hints: dict[str, Any],
    params: Params,
    candidate: int | float,
    reference: int | float | None = None,
) -> float:
    target = max(
        0,
        _int_hint(
            hints.get("target_bootstrap_count"),
            int(getattr(params, "openevolve_target_bootstrap_count", 0)),
        ),
    )
    component_target = sum(_policy_unit_bootstrap_targets(hints).values())
    if component_target <= 0:
        return _target_bootstrap_score(target, candidate, reference)
    return _bootstrap_budget_fit_score(
        max(0.0, _finite_float(candidate, float("inf"))),
        max(target, component_target),
        reference,
        component_target=component_target,
    )


def _bootstrap_budget_fit_score(
    candidate: float,
    target: int | float,
    reference: int | float | None = None,
    *,
    component_target: int | float = 0,
) -> float:
    target_value = max(1.0, float(target or 0))
    if not math.isfinite(candidate):
        return 0.0
    if component_target > 0 and candidate < float(component_target):
        # Component budgets are soft, but under-maintaining softmax/norm style
        # regions should not score as perfect merely because it is fewer
        # bootstraps globally.
        return max(0.05, min(0.82, candidate / max(1.0, float(component_target))))
    if candidate <= target_value:
        return 1.0
    return _target_bootstrap_score(int(target_value), candidate, reference)


def _policy_bootstrap_target_met(
    hints: dict[str, Any],
    params: Params,
    candidate: int | float,
) -> bool:
    target = max(
        0,
        _int_hint(
            hints.get("target_bootstrap_count"),
            int(getattr(params, "openevolve_target_bootstrap_count", 0)),
        ),
    )
    component_target = sum(_policy_unit_bootstrap_targets(hints).values())
    candidate_value = max(0.0, _finite_float(candidate, float("inf")))
    if component_target > 0:
        return candidate_value >= max(1.0, 0.65 * component_target) and candidate_value <= max(
            component_target,
            target,
        ) * 1.35
    return target > 0 and candidate_value <= target


def _component_bootstrap_alignment_summary(
    context: dict[str, Any],
    locations: dict[str, Any] | None,
    total_bootstraps: int | float | None,
) -> dict[str, Any]:
    budget_info = context.get("unit_bootstrap_budget", {})
    raw_targets = budget_info.get("unit_budgets", {}) if isinstance(budget_info, dict) else {}
    targets = {
        str(unit_id): max(0, _safe_int(item.get("target"), 0))
        for unit_id, item in raw_targets.items()
        if isinstance(item, dict) and _safe_int(item.get("target"), 0) > 0
    }
    if not targets:
        return {
            "score": 1.0,
            "target_total": 0,
            "matched_bootstraps": 0,
            "outside_bootstraps": max(0, _safe_int(total_bootstraps, 0)),
            "unit_actuals": {},
        }
    unit_actuals: Counter[str] = Counter()
    outside = 0
    known_units = {str(unit.get("id", "")) for unit in context.get("placement_units", []) if isinstance(unit, dict)}
    for raw_location, raw_count in (locations or {}).items():
        count = max(0, _safe_int(raw_count, 0))
        if count <= 0:
            continue
        candidate_ids = _unit_ids_for_location(str(raw_location))
        matched = next((unit_id for unit_id in candidate_ids if unit_id in targets), None)
        if matched is None:
            matched = next((unit_id for unit_id in candidate_ids if unit_id in known_units), None)
        if matched is None:
            outside += count
        else:
            unit_actuals[matched] += count
    per_unit_scores = []
    for unit_id, target in targets.items():
        actual = int(unit_actuals.get(unit_id, 0))
        if actual <= target:
            per_unit_scores.append(0.45 + 0.55 * (actual / max(1, target)))
        else:
            per_unit_scores.append(max(0.0, target / max(1, actual)))
    matched_total = sum(unit_actuals.values())
    total = max(matched_total + outside, _safe_int(total_bootstraps, matched_total + outside), 1)
    outside_score = 1.0 / (1.0 + outside / total)
    score = 0.82 * (sum(per_unit_scores) / max(1, len(per_unit_scores))) + 0.18 * outside_score
    return {
        "score": float(max(0.0, min(1.0, score))),
        "target_total": int(sum(targets.values())),
        "matched_bootstraps": int(matched_total),
        "outside_bootstraps": int(outside),
        "unit_actuals": dict(sorted(unit_actuals.items())),
        "unit_targets": targets,
    }


def _component_bootstrap_alignment_score(
    context: dict[str, Any],
    locations: dict[str, Any] | None,
    total_bootstraps: int | float | None,
) -> float:
    return float(
        _component_bootstrap_alignment_summary(context, locations, total_bootstraps)["score"]
    )


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


def _context_seed_bootstrap_count(context: dict[str, Any]) -> int | None:
    candidates = [
        context.get("reference", {}).get("bootstrap_count"),
        context.get("harness", {}).get("seed_baseline", {}).get("bootstrap_count"),
    ]
    for value in candidates:
        if value is None:
            continue
        try:
            result = int(value)
        except (TypeError, ValueError):
            continue
        if result >= 0:
            return result
    return None


def _context_budget_aggressive(context: dict[str, Any]) -> bool:
    value = context.get("harness", {}).get(
        "budget_aggressive",
        context.get("params", {}).get("openevolve_budget_aggressive", True),
    )
    return _bool_hint(value, True)


def _alphaevolve_guidance(params: Params, unit_bootstrap_budget: dict[str, Any]) -> dict[str, Any]:
    """Paper-inspired operating instructions for the evolved program.

    The AlphaEvolve TPU/FHE paper emphasizes narrow high-impact mutation
    surfaces, full-module scoring, correctness/security gates, and compact
    execution-trace feedback. Keep the same shape here: OpenEvolve mutates the
    placement policy, while Orbit owns CKKS legality, repair, and final DP.
    """

    requested_target = int(getattr(params, "openevolve_target_bootstrap_count", 0) or 0)
    return {
        "source": "Adapting AlphaEvolve to Optimize Fully Homomorphic Encryption on TPUs",
        "principles": [
            "Mutate only the exposed placement policy and bounded action parameters.",
            "Optimize broad QBP/compile behavior; isolated budget wins do not matter unless final DP improves.",
            "Validity and CKKS safety are hard gates before latency minimization.",
            "Use evaluator trace artifacts as profiling feedback, especially selected_source_counts and candidate_invalid_reasons.",
            "Prefer implementation/scheduling choices over inventing new cryptographic semantics.",
            "Candidate programs may return sparse policy overlays; omitted MCTS preset fields inherit the current best seed unless replace_mcts_action_presets=True is explicit.",
        ],
        "high_impact_knobs": [
            "mcts_actions[*].prior",
            "mcts_actions[*].policy.selection_objective",
            "mcts_actions[*].policy.bootstrap_anchor_count",
            "mcts_actions[*].policy.force_bootstrap_anchors",
            "mcts_actions[*].policy.boundary_state_cap",
            "mcts_actions[*].policy.boundary_scale_policy",
            "mcts_actions[*].policy.max_scale_candidates",
            "mcts_actions[*].policy.bootstrap_penalty",
            "mcts_actions[*].policy.selection_bootstrap_penalty",
            "mcts_actions[*].policy.noise_slack_model",
            "boundary_group_policies[*].selector",
            "boundary_group_policies[*].policy.boundary_state_cap",
            "boundary_group_policies[*].policy.boundary_scale_policy",
            "boundary_group_policies[*].policy.max_scale_candidates",
            "mcts_rollout_budget",
            "mcts_action_cap",
        ],
        "strategy_pool": [
            {
                "name": "component_pressure_repair",
                "when": "candidate coverage is low in nonlinear or high-maintenance graph regions",
                "mutations": [
                    "keep selection_objective='cost'",
                    "try force_bootstrap_anchors=false before adding anchors",
                    "use boundary_state_cap and max_scale_candidates before count-like knobs",
                    "lower bootstrap penalties only when it reduces objective_cost_usec",
                ],
            },
            {
                "name": "budget_fulfillment_beam",
                "when": "candidate_qbp_coverage or boundary_group_validity is low",
                "mutations": [
                    "increase boundary_state_cap before increasing beam_width",
                    "try boundary_scale_policy in ['frontier', 'waterline', 'sf']",
                    "avoid excessive bootstrap penalties that erase legal refresh candidates",
                ],
            },
            {
                "name": "strict_no_bootstrap",
                "when": "latency is high and trace artifacts show unnecessary maintenance",
                "mutations": [
                    "keep as a baseline action, but do not let it dominate when QBP coverage drops",
                ],
            },
        ],
        "tiered_evaluator": {
            "static_only": "normalize and clamp candidate policy; reject source mutation or solver calls",
            "clear_only": "replay QBP boundary groups with deterministic Orbit repair",
            "finalist": "full-bundle replay; fallback-heavy candidates cannot win",
        },
        "requested_bootstrap_target": requested_target if requested_target > 0 else None,
        "maintenance_pressure_summary": {
            "kind_pressure_totals": unit_bootstrap_budget.get("kind_pressure_totals", {}),
            "unit_pressure_count": len(unit_bootstrap_budget.get("unit_maintenance_pressure", {}) or {}),
            "notes": unit_bootstrap_budget.get("notes", []),
        },
    }


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
        "noise_slack_model",
        "scale_floor_bits",
        "beam_width",
        "state_cap_per_node",
        "scale_lattice",
        "budget_aggressive",
        "selection_bootstrap_penalty",
        "selection_objective",
        "prefer_component_budget_fit",
        "force_bootstrap_anchors",
        "force_bootstrap_nodes",
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
        "component_bootstrap_budgets",
        "unit_bootstrap_budgets",
        "mcts_action_presets",
        "boundary_group_policies",
    ]
    summary = {key: hints.get(key) for key in keys if key in hints}
    for key in ("preferred_node_levels", "preferred_node_scales", "preferred_edge_scales"):
        value = hints.get(key)
        if isinstance(value, dict) and value:
            summary[f"{key}_count"] = len(value)
    if hints.get("unit_policies"):
        summary["unit_policy_count"] = len(hints.get("unit_policies") or [])
    if hints.get("boundary_group_policies"):
        summary["boundary_group_policy_count"] = len(
            hints.get("boundary_group_policies") or []
        )
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


def _jsonable_policy_hints(hints: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(hints, default=str))


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


def _histogram_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _waterline_profile_summary(
    pdag: Tdag,
    params: Params,
    placement_units: list[dict[str, Any]],
) -> dict[str, Any]:
    by_layer: dict[str, dict[str, Any]] = {}
    whole = {
        "node_count": 0,
        "op_histogram": Counter(),
        "nonlinear_histogram": Counter(),
        "input_scale_lower_bounds": Counter(),
        "output_scale_lower_bounds": Counter(),
        "min_input_scale_lower_bound": None,
        "max_input_scale_lower_bound": None,
        "min_output_scale_lower_bound": None,
        "max_output_scale_lower_bound": None,
    }

    def bucket_for(layer: str) -> dict[str, Any]:
        return by_layer.setdefault(
            layer,
            {
                "node_count": 0,
                "op_histogram": Counter(),
                "nonlinear_histogram": Counter(),
                "input_scale_lower_bounds": Counter(),
                "output_scale_lower_bounds": Counter(),
                "min_input_scale_lower_bound": None,
                "max_input_scale_lower_bound": None,
                "min_output_scale_lower_bound": None,
                "max_output_scale_lower_bound": None,
            },
        )

    def record_bound(bucket: dict[str, Any], key: str, value: int) -> None:
        hist_key = f"{key}_scale_lower_bounds"
        min_key = f"min_{key}_scale_lower_bound"
        max_key = f"max_{key}_scale_lower_bound"
        bucket[hist_key][int(value)] += 1
        bucket[min_key] = value if bucket[min_key] is None else min(int(bucket[min_key]), value)
        bucket[max_key] = value if bucket[max_key] is None else max(int(bucket[max_key]), value)

    for node, attrs in pdag.nodes(data=True):
        op = str(attrs.get("op", ""))
        layer = _node_layer(dict(attrs))
        nonlinear = _node_nonlinear_kind(dict(attrs)) or "linear/other"
        for bucket in (whole, bucket_for(layer)):
            bucket["node_count"] += 1
            bucket["op_histogram"][op] += 1
            bucket["nonlinear_histogram"][nonlinear] += 1
        if op == "constant":
            continue
        in_lb = int(params.scale_lower_bound(str(node), attrs, "in"))
        out_lb = int(params.scale_lower_bound(str(node), attrs, "out"))
        for bucket in (whole, by_layer[layer]):
            record_bound(bucket, "input", in_lb)
            record_bound(bucket, "output", out_lb)

    def freeze(bucket: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_count": int(bucket["node_count"]),
            "op_histogram": _histogram_dict(bucket["op_histogram"]),
            "nonlinear_histogram": _histogram_dict(bucket["nonlinear_histogram"]),
            "input_scale_lower_bounds": _histogram_dict(bucket["input_scale_lower_bounds"]),
            "output_scale_lower_bounds": _histogram_dict(bucket["output_scale_lower_bounds"]),
            "min_input_scale_lower_bound": bucket["min_input_scale_lower_bound"],
            "max_input_scale_lower_bound": bucket["max_input_scale_lower_bound"],
            "min_output_scale_lower_bound": bucket["min_output_scale_lower_bound"],
            "max_output_scale_lower_bound": bucket["max_output_scale_lower_bound"],
        }

    node_lookup = {str(node): (node, attrs) for node, attrs in pdag.nodes(data=True)}
    by_unit = {}
    for unit in placement_units:
        unit_id = str(unit.get("id", ""))
        selector = unit.get("selector", {}) if isinstance(unit, dict) else {}
        unit_bucket = {
            "input_scale_lower_bounds": Counter(),
            "output_scale_lower_bounds": Counter(),
            "min_input_scale_lower_bound": None,
            "max_input_scale_lower_bound": None,
            "min_output_scale_lower_bound": None,
            "max_output_scale_lower_bound": None,
        }
        for node_id in unit.get("node_ids", []) or []:
            item = node_lookup.get(str(node_id))
            if item is None:
                continue
            _, attrs = item
            if attrs.get("op") == "constant":
                continue
            record_bound(
                unit_bucket,
                "input",
                int(params.scale_lower_bound(str(node_id), attrs, "in")),
            )
            record_bound(
                unit_bucket,
                "output",
                int(params.scale_lower_bound(str(node_id), attrs, "out")),
            )
        by_unit[unit_id] = {
            "layer": selector.get("layer"),
            "nonlinear_kind": selector.get("nonlinear_kind"),
            "node_count": len(unit.get("node_ids", []) or []),
            "min_scale_lb_out": unit.get("min_scale_lb_out"),
            "op_histogram": unit.get("op_histogram", {}),
            "input_scale_lower_bounds": _histogram_dict(
                unit_bucket["input_scale_lower_bounds"]
            ),
            "output_scale_lower_bounds": _histogram_dict(
                unit_bucket["output_scale_lower_bounds"]
            ),
            "min_input_scale_lower_bound": unit_bucket["min_input_scale_lower_bound"],
            "max_input_scale_lower_bound": unit_bucket["max_input_scale_lower_bound"],
            "min_output_scale_lower_bound": unit_bucket["min_output_scale_lower_bound"],
            "max_output_scale_lower_bound": unit_bucket["max_output_scale_lower_bound"],
        }

    return {
        "ckks": {
            "input_waterline": int(params.Sw),
            "constant_waterline": int(params.Csw),
            "rescaling_factor": int(params.Sf),
            "level_lower_bound": int(params.lvl_lb),
            "level_upper_bound": int(params.lvl_ub),
            "bootstrap_level_lower_bound": int(params.bts_lb),
            "bootstrap_level_upper_bound": int(params.bts_ub),
            "max_scale": int(_max_scale(params)),
        },
        "profile": {
            "resilience_profile_path": getattr(params, "resilience_profile_path", None),
            "resilience_mode": getattr(params, "resilience_mode", "waterline"),
            "resilience_constraint_policy": getattr(
                params, "resilience_constraint_policy", "relax-only"
            ),
        },
        "whole_graph": freeze(whole),
        "by_layer": {layer: freeze(bucket) for layer, bucket in sorted(by_layer.items())},
        "by_unit": by_unit,
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


_NONLINEAR_MAINTENANCE_PRIORS: dict[str, dict[str, Any]] = {
    "attention_softmax": {
        "pressure": 1.0,
        "reason": "softmax/normalize-and-square regions often dominate maintenance cost",
    },
    "reciprocal": {
        "pressure": 0.8,
        "reason": "reciprocal/Newton-style approximations need noise reserve across iterations",
    },
    "norm": {
        "pressure": 0.8,
        "reason": "norm or inverse-sqrt regions have iterative polynomial maintenance",
    },
    "activation": {
        "pressure": 0.35,
        "reason": "activation pressure depends on polynomial degree and local fanout",
    },
}


def _unit_bootstrap_budget_summary(
    units: list[dict[str, Any]],
    params: Params,
    global_target: int | float | None,
) -> dict[str, Any]:
    unit_pressure: dict[str, dict[str, Any]] = {}
    kind_pressure_totals: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    for unit in units:
        pressure, reason = _unit_maintenance_pressure(unit, params)
        selector = unit.get("selector", {}) if isinstance(unit, dict) else {}
        kind = str(selector.get("nonlinear_kind") or selector.get("kind") or "layer")
        if pressure > 0:
            unit_id = str(unit.get("id", ""))
            unit_pressure[unit_id] = {
                "pressure": float(pressure),
                "kind": kind,
                "selector": dict(selector),
                "reason": reason,
            }
            kind_pressure_totals[kind] += float(pressure)
            kind_counts[kind] += 1
    requested_global = max(0, _safe_int(global_target, 0))
    return {
        "unit_budgets": {},
        "unit_maintenance_pressure": unit_pressure,
        "kind_pressure_totals": dict(sorted(kind_pressure_totals.items())),
        "kind_unit_counts": dict(sorted(kind_counts.items())),
        "kind_priors": _NONLINEAR_MAINTENANCE_PRIORS,
        "component_budget_total": 0,
        "requested_global_target": requested_global,
        "effective_target_bootstrap_count": requested_global,
        "notes": [
            "Maintenance pressure is a relative graph prior, not a bootstrap-count target.",
            "Final Orbit validation and latency decide whether bootstraps are legal and useful.",
        ],
    }


def _unit_maintenance_pressure(unit: dict[str, Any], params: Params) -> tuple[float, str]:
    selector = unit.get("selector", {}) if isinstance(unit, dict) else {}
    kind = str(selector.get("nonlinear_kind") or "")
    if not kind:
        return 0.0, "linear/layer aggregation unit"
    op_hist = unit.get("op_histogram", {}) if isinstance(unit, dict) else {}
    mul_count = max(0, _safe_int(op_hist.get("mul"), 0))
    fanout = max(1, _safe_int(unit.get("max_out_degree"), 1))
    base = _finite_float(
        _NONLINEAR_MAINTENANCE_PRIORS.get(kind, {}).get("pressure"),
        0.25,
    )
    pressure = base * (1.0 + math.log1p(mul_count) / 4.0) * (1.0 + min(fanout - 1, 4) / 8.0)
    if kind == "attention_softmax":
        return pressure, "attention softmax maintenance pressure"
    if kind in {"reciprocal", "norm"}:
        return pressure, f"{kind} iterative approximation maintenance pressure"
    if kind == "activation":
        return pressure if mul_count >= 4 else 0.0, "activation polynomial maintenance pressure"
    return pressure, f"unknown nonlinear kind {kind}"


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
    return _node_comment_metadata(attrs).get("scope", "")


def _node_op_tag(attrs: dict[str, Any]) -> str:
    return _node_comment_metadata(attrs).get("op", str(attrs.get("op", "")))


def _node_layer(attrs: dict[str, Any]) -> str:
    metadata = _node_comment_metadata(attrs)
    if metadata.get("layer"):
        return metadata["layer"]
    scope = _node_scope(attrs)
    pieces = [piece for piece in scope.split(".") if piece]
    if "layer" in pieces:
        idx = pieces.index("layer")
        if idx + 1 < len(pieces):
            return f"layer.{pieces[idx + 1]}"
    if ".pooler" in scope or scope.endswith("pooler") or scope.startswith("bert.pooler"):
        return "pooler"
    if ".classifier" in scope or scope.endswith("classifier") or scope.startswith("bert.classifier"):
        return "classifier"
    return "global"


def _node_comment_metadata(attrs: dict[str, Any]) -> dict[str, str]:
    """Extract Rotom comment metadata even when comments have numeric prefixes.

    Rotom comments often look like ``"7716 scope=...;op=..."`` after graph
    compression.  The previous parser only accepted fields at the beginning of
    a semicolon chunk, which collapsed those nodes to ``layer:global`` and made
    layer/nonlinear OpenEvolve budgeting ineffective.
    """

    comment = str(attrs.get("comment", ""))
    metadata: dict[str, str] = {}
    for part in comment.split(";"):
        text = part.strip()
        for key in ("scope", "op", "layer"):
            needle = f"{key}="
            idx = text.find(needle)
            if idx < 0:
                continue
            value = text[idx + len(needle) :].strip()
            if value:
                metadata[key] = value
    return metadata


def _node_nonlinear_kind(attrs: dict[str, Any]) -> str | None:
    text = f"{_node_scope(attrs)};{_node_op_tag(attrs)};{attrs.get('comment', '')}".lower()
    return _infer_nonlinear_kind(text)


def _infer_nonlinear_kind(text: str) -> str | None:
    text = text.lower()
    if "reciprocal" in text or "newton" in text:
        return "reciprocal"
    if "qk_softmax" in text or "softmax" in text:
        return "attention_softmax"
    if "activation" in text or "gelu" in text or "silu" in text or "quadratic" in text:
        return "activation"
    if "layernorm" in text or "layer_norm" in text or "polynorm" in text or "norm" in text:
        return "norm"
    return None


def _unit_ids_for_location(location: str) -> list[str]:
    parts = {}
    for item in location.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            parts[key.strip()] = value.strip()
    layer = parts.get("layer", "")
    op = parts.get("op", "")
    unit_ids: list[str] = []
    nonlinear = _infer_nonlinear_kind(location)
    if layer and layer != "none" and nonlinear:
        unit_ids.append(f"nonlinear:{layer}:{nonlinear}")
    if layer and layer != "none":
        unit_ids.append(f"layer:{layer}")
    if op and nonlinear and layer in {"", "none"}:
        unit_ids.append(f"nonlinear:global:{nonlinear}")
    return unit_ids


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


def _context_unit_bootstrap_targets(context: dict[str, Any]) -> dict[str, int]:
    budget_info = context.get("unit_bootstrap_budget", {})
    raw = budget_info.get("unit_budgets", {}) if isinstance(budget_info, dict) else {}
    return {
        str(unit_id): max(0, _safe_int(item.get("target"), 0))
        for unit_id, item in raw.items()
        if isinstance(item, dict) and _safe_int(item.get("target"), 0) > 0
    }


def _reference_bootstrap_patterns_from_context(context: dict[str, Any]) -> list[str]:
    reference = context.get("harness", {}).get("reference_json", {})
    if not isinstance(reference, dict):
        return []
    raw_locations = (
        reference.get("bootstrap_locations")
        or reference.get("bootstrap_locations_by_location")
        or reference.get("maintenance_locations", {}).get("bootstrap")
    )
    if not isinstance(raw_locations, dict):
        return []
    weighted: list[tuple[int, str]] = []
    for location, count in raw_locations.items():
        text = str(location).strip().lower()
        if not text:
            continue
        weight = max(1, _safe_int(count, 1))
        weighted.append((weight, text))
        fields = {}
        for part in text.split(";"):
            if "=" in part:
                key, value = part.split("=", 1)
                fields[key.strip()] = value.strip()
        op = fields.get("op")
        if op:
            weighted.append((weight, op))
            for token in op.replace("-", "_").split("_"):
                if len(token) >= 5:
                    weighted.append((max(1, weight - 1), token))
        layer = fields.get("layer")
        if layer and op:
            weighted.append((weight, f"layer={layer};op={op}"))
    best: dict[str, int] = {}
    for weight, pattern in weighted:
        if not pattern or pattern in {"bootstrap", "rescale", "mul", "add"}:
            continue
        best[pattern] = max(best.get(pattern, 0), int(weight))
    return [
        pattern
        for pattern, _weight in sorted(best.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    ][:32]


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
    component_targets = _context_unit_bootstrap_targets(context)
    component_anchor_count = _graph_maintenance_anchor_hint(context)
    base = {
        "strategy": "level_preserving",
        "prefer_level_preservation": True,
        "allow_seed_fallback": False,
        "refresh_fanout_at_level_floor": False,
        "max_scale_candidates": 24,
        "bootstrap_penalty": 250_000_000.0,
        "selection_bootstrap_penalty": 0.0,
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
        "boundary_state_cap": 4,
        "bootstrap_anchor_count": 0,
        "component_bootstrap_budgets": component_targets,
        "selection_objective": "cost",
        "prefer_component_budget_fit": False,
    }
    actions: list[dict[str, Any]] = [
        {
            "name": "strict_no_bootstrap",
            "prior": 0.05,
            "policy": {
                **base,
                "forbid_bootstrap": True,
                "allow_bootstrap": False,
                "boundary_scale_policy": "low",
            },
        },
        {
            "name": "budget_fulfillment_beam",
            "prior": 0.54,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 6,
                "max_scale_candidates": 40,
                "bootstrap_penalty": 250_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "wide_boundary_cost_beam",
            "prior": 0.36,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "profile_waterline_repair",
            "prior": 0.24,
            "policy": {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "boundary_state_cap": 6,
                "max_scale_candidates": 48,
                "state_cap_per_node": 24,
                "beam_width": 6,
                "bootstrap_penalty": 125_000_000.0,
                "selection_bootstrap_penalty": 0.0,
            },
        },
        {
            "name": "tuneinsight_avgcase_cost_beam",
            "prior": 0.22,
            "policy": {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "sf",
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "tuneinsight_deferred_bootstrap_beam",
            "prior": 0.30,
            "policy": {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "beam_width": 8,
                "state_cap_per_node": 32,
                "boundary_state_cap": 8,
                "max_scale_candidates": 64,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "noise_slack_model": "tuneinsight_avgcase",
                "reserve_penalty": 0.0,
                "min_transition_reserve": 0,
                "min_decryptability_reserve": 0,
                "bootstrap_penalty": 650_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "waterline_budget_repair",
            "prior": 0.20,
            "policy": {
                **_waterline_seed_policy(),
                "direct_budget_policy": True,
                "allow_seed_fallback": False,
                "target_bootstrap_count": target,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "component_budget_repair",
            "prior": 0.28,
            "policy": {
                **base,
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "frontier",
                "bootstrap_anchor_count": component_anchor_count,
                "boundary_state_cap": 4,
                "max_scale_candidates": 32,
                "bootstrap_penalty": 50_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
                "prefer_component_budget_fit": False,
                "force_bootstrap_anchors": False,
                "min_transition_reserve": 2,
                "min_decryptability_reserve": 2,
            },
        },
        {
            "name": "waterline_cost_beam",
            "prior": 0.58,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 12,
                "state_cap_per_node": 40,
                "boundary_state_cap": 8,
                "max_scale_candidates": 32,
                "boundary_scale_policy": "waterline",
                "scale_lattice": "waterline_sf",
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "dense_boundary_cost_beam",
            "prior": 0.40,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 16,
                "max_scale_candidates": 96,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "dense",
                "bootstrap_penalty": 25_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "nonlinear_phase_boundary_beam",
            "prior": 0.44,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "bootstrap_anchor_selector": "nonlinear_phase_boundaries",
                "bootstrap_anchor_count": 64,
                "force_bootstrap_anchors": False,
                "bootstrap_penalty": 35_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        },
        {
            "name": "latency_mcts_repair",
            "prior": 0.19,
            "policy": {
                **base,
                "strategy": "latency_beam",
                "forbid_bootstrap": False,
                "allow_bootstrap": True,
                "boundary_scale_policy": "sf",
                "bootstrap_anchor_count": component_anchor_count,
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
                "bootstrap_anchor_count": 0,
                "boundary_state_cap": 3,
                "bootstrap_penalty": 2_500_000_000.0,
                "selection_bootstrap_penalty": 250_000_000.0,
                "selection_objective": "min_bootstrap",
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
            "name": "candidate_low_scale_frontier",
            "prior": 0.04,
            "policy": {
                **_low_scale_frontier_policy(),
                "allow_seed_fallback": False,
                "target_bootstrap_count": target,
                "selection_bootstrap_penalty": 900_000_000.0,
            },
        },
    ]
    if str(context.get("params", {}).get("scale_floor_policy", "waterline")) == "estimator-relaxed":
        relaxed_actions: list[dict[str, Any]] = []
        for floor in _scale_floor_candidates_from_context(context):
            if int(floor) >= int(ckks["Sw"]):
                continue
            relaxed_actions.append(
                {
                    "name": f"estimator_relaxed_floor_{int(floor)}",
                    "prior": 0.34 + 0.01 * max(0, int(ckks["Sw"]) - int(floor)),
                    "policy": {
                        **base,
                        "strategy": "latency_beam",
                        "forbid_bootstrap": False,
                        "allow_bootstrap": True,
                        "beam_width": 8,
                        "state_cap_per_node": 32,
                        "boundary_state_cap": 8,
                        "max_scale_candidates": 64,
                        "boundary_scale_policy": "frontier",
                        "scale_lattice": "waterline_sf",
                        "noise_slack_model": "tuneinsight_avgcase",
                        "scale_floor_bits": int(floor),
                        "reserve_penalty": 0.0,
                        "min_transition_reserve": 0,
                        "min_decryptability_reserve": 0,
                        "bootstrap_penalty": 650_000_000.0,
                        "selection_bootstrap_penalty": 0.0,
                        "selection_objective": "cost",
                    },
                }
            )
        if relaxed_actions:
            # Keep graph-derived phase-boundary actions inside small sampled
            # action caps; relaxed floors are still available but should not
            # push structural placement policies out of the candidate surface.
            phase_actions = [
                action
                for action in actions
                if str(action.get("name", "")) == "nonlinear_phase_boundary_beam"
            ]
            if phase_actions:
                actions = [
                    action
                    for action in actions
                    if str(action.get("name", "")) != "nonlinear_phase_boundary_beam"
                ]
                dense_index = next(
                    (
                        idx
                        for idx, action in enumerate(actions)
                        if str(action.get("name", "")) == "dense_boundary_cost_beam"
                    ),
                    len(actions),
                )
                actions[dense_index:dense_index] = phase_actions
            insert_at = min(len(actions), 9)
            actions[insert_at:insert_at] = relaxed_actions
    reference_patterns = _reference_bootstrap_patterns_from_context(context)
    if reference_patterns:
        action = {
            "name": "reference_boundary_cost_beam",
            "prior": 0.50,
            "policy": {
                **_budget_fulfillment_beam_policy(),
                "target_bootstrap_count": target,
                "direct_budget_policy": True,
                "beam_width": 10,
                "state_cap_per_node": 48,
                "boundary_state_cap": 12,
                "max_scale_candidates": 80,
                "boundary_scale_policy": "frontier",
                "scale_lattice": "waterline_sf",
                "bootstrap_anchor_selector": "reference_bootstrap_locations",
                "bootstrap_anchor_include_patterns": reference_patterns,
                "bootstrap_anchor_count": max(8, min(64, len(reference_patterns) * 2)),
                "force_bootstrap_anchors": False,
                "bootstrap_penalty": 30_000_000.0,
                "selection_bootstrap_penalty": 0.0,
                "selection_objective": "cost",
            },
        }
        insert_at = min(len(actions), 8)
        actions[insert_at:insert_at] = [action]
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


def score_rollout(metrics: dict[str, Any], *, target_bootstraps: int | None = None) -> float:
    latency = _finite_float(
        metrics.get("objective_cost_usec", metrics.get("final_latency_usec")),
        float("inf"),
    )
    reference = _finite_float(metrics.get("reference_objective_cost_usec"), float("inf"))
    validity = _finite_float(metrics.get("candidate_validity", metrics.get("validity")), 0.0)
    repairs = _finite_float(metrics.get("repair_count"), 0.0)
    latency_score = (
        max(0.0, min(2.0, reference / latency)) / 2.0
        if math.isfinite(latency) and latency > 0 and math.isfinite(reference) and reference > 0
        else 0.0
    )
    target_score = (
        _target_bootstrap_score(
            int(target_bootstraps),
            _finite_float(metrics.get("bootstrap_count"), float("inf")),
            None,
        )
        if target_bootstraps
        else 1.0
    )
    return (
        0.55 * latency_score
        + 0.30 * max(0.0, min(1.0, validity))
        + 0.10 / (1.0 + repairs)
        + 0.05 * target_score
    )


class PlacementMCTS:
    """Constrained API exposed to OpenEvolve for low-bootstrap placement search."""

    def __init__(self, context: dict[str, Any]):
        self.context = context

    def low_bootstrap_seed(
        self,
        *,
        target_bootstraps: int = 0,
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
        component_targets = _context_unit_bootstrap_targets(self.context)
        policy.update(
            {
                "budget_aggressive": bool(
                    self.context.get("harness", {}).get("budget_aggressive", False)
                ),
                "target_bootstrap_count": int(target_bootstraps),
                "component_bootstrap_budgets": component_targets,
                "mcts_rollout_budget": int(rollout_budget),
                "mcts_exploration_weight": float(exploration_weight),
                "mcts_max_repair_bootstraps": int(max_repair_bootstraps),
                "mcts_action_cap": int(action_cap),
                "mcts_prior_order": True,
                "enable_direct_budget_beam": False,
                "include_seed_repair_actions": False,
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

    def action_focus(
        self,
        *names: str,
        cap: int | None = None,
        block: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return root-level action selection knobs for ablation-style search."""

        result: dict[str, Any] = {"mcts_prior_order": True}
        if names:
            result["mcts_action_allowlist"] = [str(name) for name in names]
            result["mcts_action_cap"] = int(cap if cap is not None else len(names))
        elif cap is not None:
            result["mcts_action_cap"] = int(cap)
        if block:
            result["mcts_action_blocklist"] = [str(name) for name in block]
        return result

    def action_presets(self, **presets: Any) -> dict[str, dict[str, Any]]:
        """Return a declarative action override table for stable evolution.

        Candidate programs can set only the high-impact knobs for named MCTS
        actions. Orbit keeps the action implementations fixed and validates the
        resulting placement, so small model mutations are less likely to be
        erased by Python control-flow changes that do not affect scheduling.
        """

        normalized: dict[str, dict[str, Any]] = {}
        for name, preset in presets.items():
            if not isinstance(preset, dict):
                continue
            policy = preset.get("policy", preset)
            item: dict[str, Any] = {}
            if "prior" in preset:
                item["prior"] = _float_hint(preset.get("prior"), 0.0)
            if "enabled" in preset:
                item["enabled"] = bool(preset.get("enabled"))
            if "disabled" in preset:
                item["disabled"] = bool(preset.get("disabled"))
            if isinstance(policy, dict):
                item["policy"] = {
                    key: value
                    for key, value in policy.items()
                    if key in _PATCHABLE_POLICY_KEYS
                    or key
                    in {
                        "direct_budget_policy",
                        "boundary_scale_policy",
                        "boundary_state_cap",
                        "bootstrap_anchor_count",
                        "bootstrap_anchor_level",
                        "force_bootstrap_anchors",
                        "selection_objective",
                        "prefer_component_budget_fit",
                    }
                }
            normalized[str(name)] = item
        return normalized

    def boundary_policy(
        self,
        *,
        boundary_state_cap: int = 6,
        max_scale_candidates: int = 48,
        boundary_scale_policy: str = "frontier",
        bootstrap_penalty: float = 125_000_000.0,
        selection_bootstrap_penalty: float = 0.0,
        beam_width: int = 6,
        state_cap_per_node: int = 24,
    ) -> dict[str, Any]:
        """Return high-impact boundary-state knobs for cost-focused MCTS actions."""

        return {
            "strategy": "latency_beam",
            "allow_bootstrap": True,
            "allow_seed_fallback": False,
            "refresh_fanout_at_level_floor": True,
            "direct_budget_policy": True,
            "min_transition_reserve": 0,
            "min_decryptability_reserve": 0,
            "boundary_state_cap": int(boundary_state_cap),
            "max_scale_candidates": int(max_scale_candidates),
            "boundary_scale_policy": str(boundary_scale_policy),
            "bootstrap_penalty": float(bootstrap_penalty),
            "selection_bootstrap_penalty": float(selection_bootstrap_penalty),
            "beam_width": int(beam_width),
            "state_cap_per_node": int(state_cap_per_node),
            "selection_objective": "cost",
        }

    def target_units(self, limit: int | None = None) -> list[dict[str, Any]]:
        return target_units(self.context, limit)

    def top_costly_boundary_groups(self, limit: int | None = None) -> list[dict[str, Any]]:
        groups = (
            self.context.get("harness", {}).get("top_costly_boundary_groups", [])
            if isinstance(self.context.get("harness", {}), dict)
            else []
        )
        result = [dict(item) for item in groups if isinstance(item, dict)]
        return result[:limit] if limit is not None else result

    def boundary_group_policy(
        self,
        group: dict[str, Any],
        **overrides: Any,
    ) -> dict[str, Any]:
        selector = group.get("group_key", group) if isinstance(group, dict) else {}
        policy = self.boundary_policy(**{
            key: value
            for key, value in overrides.items()
            if key in {
                "boundary_state_cap",
                "max_scale_candidates",
                "boundary_scale_policy",
                "bootstrap_penalty",
                "selection_bootstrap_penalty",
                "beam_width",
                "state_cap_per_node",
            }
        })
        for key, value in overrides.items():
            if key not in policy:
                policy[key] = value
        return {"selector": selector, "policy": policy}

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
    hints = _call_candidate_place(module.place, context)
    return _normalize_candidate_hints(hints, context)


def _candidate_place_timeout_sec(context: dict[str, Any] | None) -> int:
    raw = os.environ.get("ORBIT_OPENEVOLVE_PLACE_TIMEOUT_SEC", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if isinstance(context, dict):
        harness = context.get("harness", {}) if isinstance(context.get("harness"), dict) else {}
        params = context.get("params", {}) if isinstance(context.get("params"), dict) else {}
        for value in (
            harness.get("candidate_place_timeout_sec"),
            params.get("openevolve_candidate_place_timeout_sec"),
        ):
            if value is not None:
                return max(1, _safe_int(value, 10))
    return 10


def _call_candidate_place(place_fn: Any, context: dict[str, Any]) -> Any:
    """Call candidate place(context) with a tight timeout.

    OpenEvolve-generated programs should describe compact policies, not run
    graph search inside Python before Orbit can clamp and validate the result.
    This prevents a single pathological mutation from consuming the whole
    evaluator timeout.
    """

    timeout = _candidate_place_timeout_sec(context)
    if timeout <= 0 or not hasattr(signal, "SIGALRM"):
        return place_fn(context)
    old_handler = signal.getsignal(signal.SIGALRM)

    def _timeout(_signum: int, _frame: Any) -> None:
        raise PlacementError(f"candidate place(context) timed out after {timeout}s")

    try:
        signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, float(timeout))
        return place_fn(context)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _context_initial_policy_hints(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    harness = context.get("harness", {})
    if not isinstance(harness, dict):
        return {}
    hints = harness.get("initial_policy_hints")
    return deepcopy(hints) if isinstance(hints, dict) else {}


def _merge_mcts_action_presets(
    seed_presets: Any,
    candidate_presets: Any,
    *,
    replace: bool = False,
) -> dict[str, dict[str, Any]]:
    merged = {} if replace else deepcopy(_mcts_action_preset_map(seed_presets))
    for name, patch in _mcts_action_preset_map(candidate_presets).items():
        base = {} if replace else deepcopy(merged.get(name, {}))
        if _bool_hint(patch.get("replace"), False):
            base = {}
        base_policy = deepcopy(base.get("policy", {})) if isinstance(base.get("policy"), dict) else {}
        patch_policy = patch.get("policy", patch)
        if isinstance(patch_policy, dict):
            base_policy.update(
                {
                    key: value
                    for key, value in patch_policy.items()
                    if key not in {"name", "prior", "enabled", "disabled", "replace", "policy"}
                }
            )
        if base_policy:
            base["policy"] = base_policy
        for key in ("prior", "enabled", "disabled"):
            if key in patch:
                base[key] = patch[key]
        merged[str(name)] = base
    return merged


def _validate_candidate_source(program_path: Path) -> None:
    source = program_path.read_text(encoding="utf-8")
    for token in BANNED_CANDIDATE_TOKENS:
        if token in source:
            raise PlacementError(f"candidate program uses banned token {token!r}")


def _normalize_candidate_hints(value: Any, context: dict[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    seed_hints = _context_initial_policy_hints(context)
    inherit_initial = _bool_hint(value.get("inherit_initial_policy", True), True)
    seed_base = _with_default_policy(seed_hints) if inherit_initial and seed_hints else {}
    structured_keys = {
        "policy",
        "global_policy",
        "placement_records",
        "portfolio",
        "patches",
        "unit_policies",
        "boundary_group_policies",
        "mcts_actions",
        "mcts_action_presets",
        "mcts_action_allowlist",
        "mcts_action_blocklist",
    }
    if not any(key in value for key in structured_keys):
        if seed_base:
            merged = dict(seed_base)
            merged.update(value)
            if "mcts_action_presets" in value:
                merged["mcts_action_presets"] = _merge_mcts_action_presets(
                    seed_base.get("mcts_action_presets"),
                    value.get("mcts_action_presets"),
                    replace=_bool_hint(value.get("replace_mcts_action_presets"), False),
                )
            return _sanitize_candidate_hints(
                _apply_patch_vocabulary(merged, value.get("patches", []), context),
                context,
            )
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
    result = _with_default_policy(seed_base)
    result.update(policy)
    if isinstance(value.get("mcts_actions"), list):
        result["mcts_actions"] = value["mcts_actions"]
    if isinstance(value.get("mcts_action_presets"), (dict, list)):
        result["mcts_action_presets"] = _merge_mcts_action_presets(
            seed_base.get("mcts_action_presets"),
            value["mcts_action_presets"],
            replace=_bool_hint(value.get("replace_mcts_action_presets"), False),
        )
    if "mcts_action_cap" in value:
        result["mcts_action_cap"] = value["mcts_action_cap"]
    if "mcts_prior_order" in value:
        result["mcts_prior_order"] = value["mcts_prior_order"]
    if isinstance(value.get("mcts_action_allowlist"), list):
        result["mcts_action_allowlist"] = value["mcts_action_allowlist"]
    if isinstance(value.get("mcts_action_blocklist"), list):
        result["mcts_action_blocklist"] = value["mcts_action_blocklist"]
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
    if "unit_policies" in value:
        result["unit_policies"] = _normalize_unit_policies(value.get("unit_policies"), context)
    elif "unit_policies" in seed_base:
        result["unit_policies"] = deepcopy(seed_base.get("unit_policies", []))
    if "boundary_group_policies" in value:
        result["boundary_group_policies"] = _normalize_boundary_group_policies(
            value.get("boundary_group_policies"), context
        )
    elif "boundary_group_policies" in seed_base:
        result["boundary_group_policies"] = deepcopy(seed_base.get("boundary_group_policies", []))
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


def _normalize_boundary_group_policies(
    value: Any, context: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Normalize sparse QBP boundary-group policy overlays.

    Candidate programs get group selectors from execution traces. Selectors are
    intentionally data-shaped dictionaries so this remains graph-general and
    does not depend on BERT layer labels.
    """

    if isinstance(value, dict):
        raw_items = []
        for key, policy in value.items():
            selector: dict[str, Any]
            if isinstance(key, str) and key.strip().startswith("{"):
                try:
                    selector = json.loads(key)
                except Exception:
                    selector = {"key": key}
            elif isinstance(key, str):
                selector = {"key": key}
            else:
                selector = {}
            raw_items.append({"selector": selector, "policy": policy})
    elif isinstance(value, list):
        raw_items = value
    else:
        return []

    normalized: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        selector = item.get("selector", item.get("group_key", {}))
        if isinstance(selector, str):
            selector = {"key": selector}
        if not isinstance(selector, dict):
            continue
        policy = item.get("policy", {})
        if not isinstance(policy, dict):
            policy = {}
        if not _boundary_group_selector_is_well_formed(selector):
            continue
        if context is not None and not _selector_matches_any_boundary_group(selector, context):
            # Keep group overlays only when they can affect the sampled context.
            continue
        normalized.append({"selector": dict(selector), "policy": _with_default_policy(policy)})
    return normalized[:32]


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
    sanitized_boundary_groups = []
    for idx, item in enumerate(result.get("boundary_group_policies", []) or []):
        if not isinstance(item, dict):
            repairs.append(f"boundary_group_policies[{idx}] dropped non-dict item")
            continue
        selector = item.get("selector", {})
        if not _selector_matches_any_boundary_group(selector, context):
            repairs.append(f"boundary_group_policies[{idx}] dropped unmatched selector")
            continue
        policy = _with_default_policy(item.get("policy", {}))
        _sanitize_policy_values(policy, context, repairs, f"boundary_group_policies[{idx}]")
        sanitized_boundary_groups.append({"selector": selector, "policy": policy})
    result["boundary_group_policies"] = sanitized_boundary_groups
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
    if "selection_objective" in policy:
        objective = _selection_objective(policy)
        if objective != str(policy.get("selection_objective", "cost")).strip().lower():
            repairs.append(f"{prefix}.selection_objective reset to {objective}")
        policy["selection_objective"] = objective
    if "noise_slack_model" in policy:
        model = str(policy.get("noise_slack_model", "worst_case")).strip().lower()
        if model not in {
            "worst_case",
            "tuneinsight",
            "tuneinsight_avgcase",
            "average_case",
            "avgcase",
            "off",
            "none",
            "ignore",
        }:
            repairs.append(f"{prefix}.noise_slack_model reset to worst_case")
            model = "worst_case"
        policy["noise_slack_model"] = model
    if "scale_floor_bits" in policy:
        candidates = _scale_floor_candidates_from_context(context)
        raw_floor = _int_hint(policy.get("scale_floor_bits"), int(ckks["Sw"]))
        normalized_floor = min(
            candidates,
            key=lambda value: (abs(int(value) - raw_floor), -int(value)),
        )
        if normalized_floor != raw_floor:
            repairs.append(f"{prefix}.scale_floor_bits normalized to {normalized_floor}")
        policy["scale_floor_bits"] = int(normalized_floor)
    if "prefer_component_budget_fit" in policy:
        policy["prefer_component_budget_fit"] = _bool_hint(
            policy.get("prefer_component_budget_fit"), False
        )
    if "force_bootstrap_anchors" in policy:
        policy["force_bootstrap_anchors"] = _bool_hint(
            policy.get("force_bootstrap_anchors"), False
        )
    if "bootstrap_anchor_selector" in policy:
        selector = str(policy.get("bootstrap_anchor_selector", "")).strip().lower()
        allowed_selectors = {
            "",
            "default",
            "component",
            "nonlinear_phase_boundaries",
            "nonlinear_phase_boundary",
            "phase_boundaries",
            "phase_boundary",
            "reference_bootstrap_locations",
            "reference_bootstrap_location",
            "reference_locations",
        }
        if selector not in allowed_selectors:
            repairs.append(f"{prefix}.bootstrap_anchor_selector reset to default")
            selector = ""
        policy["bootstrap_anchor_selector"] = selector
    for list_key in ("bootstrap_anchor_include_patterns", "bootstrap_anchor_exclude_patterns"):
        if list_key in policy:
            normalized_patterns = _string_list_hint(policy.get(list_key), limit=16)
            if not normalized_patterns and policy.get(list_key):
                repairs.append(f"{prefix}.{list_key} reset to []")
            policy[list_key] = normalized_patterns
    if "mcts_prior_order" in policy:
        policy["mcts_prior_order"] = _bool_hint(policy.get("mcts_prior_order"), True)
    for list_key in ("mcts_action_allowlist", "mcts_action_blocklist"):
        if list_key in policy:
            if isinstance(policy.get(list_key), list):
                policy[list_key] = [str(item) for item in policy[list_key][:16]]
            else:
                repairs.append(f"{prefix}.{list_key} reset to []")
                policy[list_key] = []
    if "force_bootstrap_nodes" in policy and not isinstance(policy.get("force_bootstrap_nodes"), list):
        repairs.append(f"{prefix}.force_bootstrap_nodes reset to []")
        policy["force_bootstrap_nodes"] = []
    clamp_int("max_scale_candidates", 3, 128)
    clamp_int("beam_width", 1, 12)
    clamp_int("state_cap_per_node", 1, 64)
    clamp_int("mcts_rollout_budget", 1, 256)
    clamp_int("mcts_max_repair_bootstraps", 0, 1024)
    clamp_int("mcts_action_cap", 1, 64)
    clamp_int("boundary_state_cap", 1, 16)
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


def _boundary_group_selector_is_well_formed(selector: Any) -> bool:
    if not isinstance(selector, dict):
        return False
    if "group_key" in selector and isinstance(selector.get("group_key"), dict):
        selector = selector["group_key"]
    return any(
        key in selector
        for key in (
            "in_lvl",
            "in_scl",
            "maino_v",
            "main_dag_size",
            "key",
        )
    )


def _selector_matches_any_boundary_group(selector: Any, context: dict[str, Any]) -> bool:
    if not _boundary_group_selector_is_well_formed(selector):
        return False
    contexts = [context]
    for task in context.get("sampled_budget_tasks", []) or []:
        if isinstance(task, dict) and isinstance(task.get("context"), dict):
            contexts.append(task["context"])
    saw_budget_groups = False
    for candidate_context in contexts:
        budgets = candidate_context.get("io_budgets", [])
        if not isinstance(budgets, list) or not budgets:
            continue
        saw_budget_groups = True
        groups = _budget_boundary_groups(
            [_io_budget_from_json(item) for item in budgets if isinstance(item, dict)]
        )
        if any(_boundary_group_selector_matches_key(selector, key) for key in groups):
            return True
    return not saw_budget_groups


def _boundary_group_selector_matches_key(selector: Any, group_key: tuple) -> bool:
    if not isinstance(selector, dict):
        return False
    if "group_key" in selector and isinstance(selector.get("group_key"), dict):
        selector = selector["group_key"]
    if "key" in selector:
        return str(selector.get("key")) == _boundary_group_key_string(group_key)
    fields = {
        "in_lvl": int(group_key[0]),
        "in_scl": int(group_key[1]),
        "maino_v": str(group_key[2]),
        "main_dag_size": int(group_key[3]),
    }
    for key, expected in fields.items():
        if key in selector and selector.get(key) is not None:
            if str(selector.get(key)) != str(expected):
                return False
    return True


def _boundary_group_key_string(group_key: tuple) -> str:
    return (
        f"in_lvl={int(group_key[0])};in_scl={int(group_key[1])};"
        f"maino_v={str(group_key[2])};main_dag_size={int(group_key[3])}"
    )


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
        elif op == "target_boundary_group":
            selector = patch.get("selector") or patch.get("group_key") or {}
            if isinstance(selector, dict):
                policy = {
                    key: value
                    for key, value in patch.items()
                    if key in _PATCHABLE_POLICY_KEYS
                    or key
                    in {
                        "boundary_scale_policy",
                        "boundary_state_cap",
                        "scale_lattice",
                        "bootstrap_penalty",
                        "max_scale_candidates",
                        "beam_width",
                        "state_cap_per_node",
                    }
                }
                result.setdefault("boundary_group_policies", []).append(
                    {"selector": selector, "policy": policy}
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
    "max_scale",
    "bootstrap_penalty",
    "rescale_penalty",
    "level_drop_penalty",
    "min_internal_level",
    "scale_penalty",
    "boundary_scale_penalty",
    "reserve_penalty",
    "min_transition_reserve",
    "min_decryptability_reserve",
    "noise_slack_model",
    "scale_floor_bits",
    "beam_width",
    "state_cap_per_node",
    "scale_lattice",
    "budget_aggressive",
    "selection_bootstrap_penalty",
    "mcts_rollout_budget",
    "mcts_exploration_weight",
    "mcts_max_repair_bootstraps",
    "mcts_action_cap",
    "enable_direct_budget_beam",
            "include_seed_repair_actions",
            "enable_sampled_latency_beam",
            "selection_objective",
            "prefer_component_budget_fit",
            "force_bootstrap_anchors",
            "force_bootstrap_nodes",
            "noise_slack_model",
            "boundary_scale_policy",
    "boundary_state_cap",
    "preferred_boundary_scale",
    "boundary_scale",
    "bootstrap_anchor_count",
    "bootstrap_anchor_level",
    "bootstrap_anchor_selector",
    "bootstrap_anchor_include_patterns",
    "bootstrap_anchor_exclude_patterns",
    "bootstrap_anchors",
    "component_bootstrap_budgets",
    "unit_bootstrap_budgets",
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
            "frontier_bootstrap": 0.0,
            "frontier_rescale": 0.0,
            "frontier_total_bootstrap": 0.0,
            "frontier_total_rescale": 0.0,
            "frontier_total_cost_usec": 0.0,
            "avg_total_bootstrap": 0.0,
            "avg_total_rescale": 0.0,
            "avg_total_cost_usec": 0.0,
            "min_bootstrap": 0.0,
            "min_rescale": 0.0,
            "min_cost_usec": 0.0,
            "max_bootstrap": 0.0,
            "max_rescale": 0.0,
            "max_cost_usec": 0.0,
        }
    avg_bootstraps = [float(item.get("avg_bootstrap", 0.0) or 0.0) for item in usable]
    avg_rescales = [float(item.get("avg_rescale", 0.0) or 0.0) for item in usable]
    min_bootstraps = [float(item.get("min_bootstrap", 0.0) or 0.0) for item in usable]
    min_rescales = [float(item.get("min_rescale", 0.0) or 0.0) for item in usable]
    min_costs = [float(item.get("min_cost_usec", 0.0) or 0.0) for item in usable]
    max_bootstraps = [float(item.get("max_bootstrap", 0.0) or 0.0) for item in usable]
    max_rescales = [float(item.get("max_rescale", 0.0) or 0.0) for item in usable]
    max_costs = [float(item.get("max_cost_usec", 0.0) or 0.0) for item in usable]
    avg_costs = [float(item.get("avg_cost_usec", 0.0) or 0.0) for item in usable]
    count = len(usable)
    return {
        "boundary_group_count": float(count),
        "avg_bootstrap": float(sum(avg_bootstraps) / count),
        "avg_rescale": float(sum(avg_rescales) / count),
        "avg_cost_usec": float(sum(avg_costs) / count),
        "frontier_bootstrap": float(sum(min_bootstraps) / count),
        "frontier_rescale": float(sum(min_rescales) / count),
        "frontier_cost_usec": float(sum(min_costs) / count),
        "frontier_total_bootstrap": float(sum(min_bootstraps)),
        "frontier_total_rescale": float(sum(min_rescales)),
        "frontier_total_cost_usec": float(sum(min_costs)),
        "avg_total_bootstrap": float(sum(avg_bootstraps)),
        "avg_total_rescale": float(sum(avg_rescales)),
        "avg_total_cost_usec": float(sum(avg_costs)),
        "min_bootstrap": float(min(min_bootstraps)),
        "min_rescale": float(min(min_rescales)),
        "min_cost_usec": float(min(min_costs)),
        "max_bootstrap": float(max(max_bootstraps)),
        "max_rescale": float(max(max_rescales)),
        "max_cost_usec": float(max(max_costs)),
    }


def _boundary_group_unsolved_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: Counter = Counter()
    examples: list[dict[str, Any]] = []
    for item in summaries:
        if not isinstance(item, dict):
            continue
        if item.get("complete") is True:
            continue
        reason = str(item.get("unsolved_reason", "unknown"))
        reasons[reason] += 1
        if len(examples) < 8:
            examples.append(
                {
                    "reason": reason,
                    "group_key": item.get("group_key", {}),
                    "requested_output_levels": item.get("requested_output_levels", []),
                    "solved_budgets": int(item.get("solved_budgets", 0) or 0),
                    "requested_budgets": int(item.get("requested_budgets", 0) or 0),
                    "candidate_solved_budgets": int(
                        item.get("candidate_solved_budgets", 0) or 0
                    ),
                }
            )
    return {"reasons": dict(reasons), "examples": examples}


def _boundary_group_top_cost_summary(
    summaries: list[dict[str, Any]], limit: int = 8
) -> list[dict[str, Any]]:
    ranked = [
        item
        for item in summaries
        if isinstance(item, dict) and int(item.get("solved_budgets", 0) or 0) > 0
    ]
    ranked.sort(
        key=lambda item: (
            _finite_float(item.get("min_cost_usec"), 0.0),
            _finite_float(item.get("max_cost_usec"), 0.0),
        ),
        reverse=True,
    )
    result = []
    for item in ranked[: max(0, int(limit))]:
        group_key = item.get("group_key", {})
        result.append(
            {
                "group_key": group_key,
                "key": _boundary_group_key_string(
                    (
                        _safe_int(group_key.get("in_lvl"), -1)
                        if isinstance(group_key, dict)
                        else -1,
                        _safe_int(group_key.get("in_scl"), -1)
                        if isinstance(group_key, dict)
                        else -1,
                        str(group_key.get("maino_v", ""))
                        if isinstance(group_key, dict)
                        else "",
                        _safe_int(group_key.get("main_dag_size"), 0)
                        if isinstance(group_key, dict)
                        else 0,
                    )
                ),
                "requested_output_levels": item.get("requested_output_levels", []),
                "min_cost_usec": _finite_float(item.get("min_cost_usec"), 0.0),
                "max_cost_usec": _finite_float(item.get("max_cost_usec"), 0.0),
                "min_bootstrap": _finite_float(item.get("min_bootstrap"), 0.0),
                "max_bootstrap": _finite_float(item.get("max_bootstrap"), 0.0),
                "min_rescale": _finite_float(item.get("min_rescale"), 0.0),
                "max_rescale": _finite_float(item.get("max_rescale"), 0.0),
                "selected_source_counts": dict(item.get("selected_source_counts", {}) or {}),
            }
        )
    return result


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


def _location_count_delta(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    limit: int = 16,
) -> dict[str, Any]:
    candidate_counts = Counter({str(key): int(value or 0) for key, value in dict(candidate or {}).items()})
    reference_counts = Counter({str(key): int(value or 0) for key, value in dict(reference or {}).items()})
    delta = candidate_counts.copy()
    delta.subtract(reference_counts)
    added = {
        key: count
        for key, count in delta.most_common(limit)
        if count > 0
    }
    removed = {
        key: -count
        for key, count in sorted(delta.items(), key=lambda item: (item[1], item[0]))[:limit]
        if count < 0
    }
    return {
        "candidate_total": int(sum(candidate_counts.values())),
        "reference_total": int(sum(reference_counts.values())),
        "delta_total": int(sum(candidate_counts.values()) - sum(reference_counts.values())),
        "added_top": added,
        "removed_top": removed,
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
    metadata = _node_comment_metadata(attrs)
    layer = metadata.get("layer", "none")
    op = metadata.get("op", str(attrs.get("op", "unknown")))
    scope = metadata.get("scope", "")
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
