from __future__ import annotations

import builtins
import json
import os
import sys
import types
from pathlib import Path

import pytest

from scripts.latency_estimator.latency_estimator import LatencyEstimator
import scripts.optimizer.orbit.openevolve_backend as oe_backend
from scripts.optimizer.orbit.openevolve_backend import (
    OpenEvolvePlacementWorker,
    PlacementBuilder,
    _aggregate_counts,
    build_context,
    build_compile_context,
    evaluate_compile_candidate_program,
    evaluate_candidate_program,
    run_compile_openevolve,
    serialize_assignment_record,
    solve_budget_batch,
    tdag_from_context,
    valid_transition,
)
from scripts.optimizer.orbit.qbp_manager import QBPManager
from scripts.optimizer.orbit.orbit_core import orbit_core
from scripts.params.params import Params
from scripts.tdag.tdag import Tdag


def _params(toy_cost_json: str, **kwargs) -> Params:
    return Params(
        toy_cost_json,
        "Orbit",
        "compile",
        Sw=40,
        CSw=40,
        threads=1,
        comp=False,
        part=False,
        placement_backend="openevolve",
        **kwargs,
    )


def _toy_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_toy")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "0",
        op="mul",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.query;op=linear",
    )
    graph.add_edge("arg0", "0", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"0"}
    return graph


def _mul_chain_pdag(params: Params, length: int = 8) -> Tdag:
    graph = Tdag(params, "oe_mul_chain")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    prev = "arg0"
    for idx in range(length):
        node = f"mul{idx}"
        graph.add_node(
            node,
            op="mul",
            weight=1,
            level=None,
            scale=None,
            op_descr={"single": 1, "double": 0},
            comment=f"scope=chain.{idx};op=mul",
        )
        graph.add_edge(prev, node, weight=1)
        prev = node
    graph.inputs = {"arg0"}
    graph.outputs = {prev}
    return graph


def _branch_merge_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_branch_merge")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    for name in ("left", "right"):
        graph.add_node(
            name,
            op="mul",
            weight=1,
            level=None,
            scale=None,
            op_descr={"single": 1, "double": 0},
            comment=f"scope={name};op=mul",
        )
        graph.add_edge("arg0", name, weight=1)
    graph.add_node("merge", op="add", weight=1, level=None, scale=None, op_descr={"single": 1}, comment="")
    graph.add_edge("left", "merge", weight=1)
    graph.add_edge("right", "merge", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"merge"}
    return graph


def _constant_bias_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_constant_bias")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "bias",
        op="constant",
        weight=1,
        level=None,
        scale=None,
        op_descr={},
        comment="scope=classifier.bias;op=bias",
    )
    graph.add_node(
        "add",
        op="add",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 1},
        comment="scope=classifier;op=linear_bias",
    )
    graph.add_edge("arg0", "add", weight=1)
    graph.add_edge("bias", "add", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"add"}
    return graph


def _constant_mul_pdag(params: Params) -> Tdag:
    graph = Tdag(params, "oe_constant_mul")
    graph.add_node("arg0", op="input", weight=1, level=None, scale=None, op_descr={}, comment="")
    graph.add_node(
        "weight",
        op="constant",
        weight=1,
        level=None,
        scale=None,
        op_descr={},
        comment="scope=dense.weight;op=weight",
    )
    graph.add_node(
        "mul",
        op="mul",
        weight=1,
        level=None,
        scale=None,
        op_descr={"single": 0, "double": 1},
        comment="scope=dense;op=linear",
    )
    graph.add_edge("arg0", "mul", weight=1)
    graph.add_edge("weight", "mul", weight=1)
    graph.inputs = {"arg0"}
    graph.outputs = {"mul"}
    return graph


def _relaxing_profile_path(tmp_path: Path) -> str:
    profile = {
        "schema_version": "orbit-resilience-constraints-v0",
        "constraints": [{"node": "add", "min_scale": 16, "ports": ["in", "out"]}],
    }
    profile_path = tmp_path / "relaxing_constraints.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return str(profile_path)


class FakeLLMModelConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeLLMConfig:
    def __init__(self):
        self.api_base = None
        self.models = []
        self.evaluator_models = []
        self.timeout = None
        self.retries = None
        self.retry_delay = None
        self.max_tokens = None
        self.temperature = None

    def update_model_params(self, args, overwrite=False):
        for model in self.models + self.evaluator_models:
            for key, value in args.items():
                if overwrite or getattr(model, key, None) is None:
                    setattr(model, key, value)


class FakeDatabaseConfig:
    def __init__(self):
        self.random_seed = None
        self.feature_dimensions = []
        self.log_prompts = None
        self.num_islands = 5


class FakeEvaluatorConfig:
    def __init__(self):
        self.timeout = None
        self.parallel_evaluations = None
        self.max_retries = None


class FakePromptConfig:
    def __init__(self):
        self.max_artifact_bytes = 20 * 1024


class FakeConfig:
    def __init__(self):
        self.random_seed = None
        self.checkpoint_interval = None
        self.llm = FakeLLMConfig()
        self.database = FakeDatabaseConfig()
        self.evaluator = FakeEvaluatorConfig()
        self.prompt = FakePromptConfig()


def _install_fake_openevolve(monkeypatch, fake_run_evolution, loaded_config=None):
    fake_openevolve = types.ModuleType("openevolve")
    fake_openevolve.__path__ = []
    fake_openevolve.run_evolution = fake_run_evolution

    fake_config_mod = types.ModuleType("openevolve.config")
    fake_config_mod.Config = FakeConfig
    fake_config_mod.LLMModelConfig = FakeLLMModelConfig
    fake_config_mod.load_config = lambda _path: loaded_config or FakeConfig()
    fake_openevolve.config = fake_config_mod

    monkeypatch.setitem(sys.modules, "openevolve", fake_openevolve)
    monkeypatch.setitem(sys.modules, "openevolve.config", fake_config_mod)


def test_context_roundtrip_and_conservative_placement(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    budgets = [{"in_lvl": -1, "in_scl": 40}]

    context = build_context(graph, budgets, params)
    restored = tdag_from_context(context)

    assert restored.name == graph.name
    assert restored.nodes["0"]["op"] == "mul"
    assert context["schema_version"] == "orbit-openevolve-placement-context-v2"
    assert context["tdag"]["topological_order"]
    assert context["graph_summary"]["max_scale"] == params.Sf + 2 * params.Sw
    assert context["budget_summary"]["count"] == 1
    assert context["constraints"]["local_scale_lower_bounds"]["0"]["in"] == 40
    assert context["latency_model"]["available_ops"]

    le = LatencyEstimator(params)
    io_to_assign, io_to_cost = solve_budget_batch(graph, budgets, le, params)

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True


def test_level_preserving_scheduler_limits_mul_chain_bootstraps(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=8)
    le = LatencyEstimator(params)

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "refresh_fanout_at_level_floor": True},
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    counts = _aggregate_counts([assign])
    assert assign.check_assign() is True
    assert counts["bootstrap"] < 8


def test_level_preserving_fallback_keeps_batch_complete_and_tracks_improvement(
    toy_cost_json: str,
):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    le = LatencyEstimator(params)
    diagnostics = {}

    _io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "allow_seed_fallback": True},
        diagnostics,
    )

    _default_assign, default_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
    )

    assert diagnostics["solved_budgets"] == 1
    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert diagnostics["candidate_improved_budgets"] == 0
    assert min(next(iter(io_to_cost.values())).values()) == min(
        next(iter(default_cost.values())).values()
    )


def test_seed_fallback_does_not_reward_invalid_candidate(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    seed_program = tmp_path / "seed.py"
    invalid_program = tmp_path / "invalid.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    seed_program.write_text(
        "def place(context):\n    return {'strategy': 'waterline_seed'}\n",
        encoding="utf-8",
    )
    invalid_program.write_text(
        "def place(context):\n"
        "    return {\n"
        "        'preferred_node_scales': {'0': 9999},\n"
        "        'allow_seed_fallback': True,\n"
        "    }\n",
        encoding="utf-8",
    )

    seed = evaluate_candidate_program(context_path, seed_program)
    invalid = evaluate_candidate_program(context_path, invalid_program)

    assert invalid["metrics"]["validity"] == 0.0
    assert invalid["metrics"]["effective_validity"] == 0.0
    assert invalid["metrics"]["fallback_selected_budgets"] == 0.0
    assert invalid["metrics"]["combined_score"] > 0.0
    assert invalid["metrics"]["combined_score"] < seed["metrics"]["combined_score"]


def test_level_preserving_scheduler_handles_branch_merge(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving"},
    )

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert assign.v_scl_in["merge"] >= params.scale_lower_bound("merge", graph.nodes["merge"], "in")


def test_level_preserving_scheduler_handles_bypass_budget(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    budget = {
        "in_lvl": -1,
        "in_scl": 40,
        "maino_v": "left",
        "main_dag_size": 3,
        "main_qbp_cost": {(12, 80): 10.0, (10, 40): 20.0},
    }

    io_to_assign, io_to_cost = solve_budget_batch(graph, [budget], le, params)

    assert io_to_assign
    assert io_to_cost
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert (assign.v_lvl_in["left"], assign.v_scl_in["left"]) in budget["main_qbp_cost"]


def test_relax_only_respects_explicit_constant_scale_for_additive_constants(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        resilience_profile=_relaxing_profile_path(tmp_path),
        resilience_constraint_policy="relax-only",
    )
    graph = _constant_bias_pdag(params)
    le = LatencyEstimator(params)

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 16}],
        le,
        params,
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert params.Sw == 16
    assert params.Csw == 40
    assert assign.check_assign() is True
    assert assign.v_scl_out["bias"] == params.Csw
    assert assign.v_scl_in["add"] >= params.Csw


def test_level_preserving_allows_relaxed_waterline_constant_mul(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        resilience_profile=_relaxing_profile_path(tmp_path),
        resilience_constraint_policy="relax-only",
    )
    graph = _constant_mul_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "level_preserving", "allow_seed_fallback": True},
        diagnostics,
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert params.Sw == 16
    assert params.Csw == 40
    assert build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)["graph_summary"][
        "max_scale"
    ] == params.Sf + 2 * params.Csw
    assert diagnostics["candidate_solved_budgets"] == 1
    assert assign.check_assign() is True
    assert assign.v_scl_in["mul"] == 80


def test_evaluate_candidate_program_returns_metrics_and_artifacts(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    program_path.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 40}}\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["combined_score"] > 0.0
    assert result["metrics"]["graph_nodes"] == 2.0
    assert "invalid_reasons" in result["artifacts"]
    assert "reference_avg_latency_usec" in result["artifacts"]
    assert "policy_summary" in result["artifacts"]


def test_builder_api_explicit_placement_record(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budget = {"in_lvl": -1, "in_scl": 40}
    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [budget],
        le,
        params,
        {"strategy": "level_preserving"},
    )
    assign = next(iter(next(iter(io_to_assign.values())).values()))
    record = serialize_assignment_record(graph.name, budget, assign)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(json.dumps(build_context(graph, [budget], params)), encoding="utf-8")
    program_path.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        f"RECORD = {record!r}\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).placement_records([RECORD])\n",
        encoding="utf-8",
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["fallback_selected_budgets"] == 0.0
    assert result["metrics"]["combined_score"] > 0.0


def test_open_helpers_expose_ilp_compatible_transition(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)

    assert valid_transition(context, 16, 40, 15, 40)
    assert not valid_transition(context, 1, 131, 16, 40)
    assert PlacementBuilder(context).level_preserving()["policy"]["strategy"] == "level_preserving"


def test_patch_vocabulary_and_portfolio_normalize(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    builder = PlacementBuilder(context)\n"
        "    builder.set_policy('strategy', 'latency_beam').prefer_node_level('0', 16).disable_seed_fallback()\n"
        "    return builder.portfolio(builder.latency_beam()['policy'], builder.level_preserving()['policy'])\n",
        encoding="utf-8",
    )

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "latency_beam"
    assert hints["allow_seed_fallback"] is False
    assert hints["preferred_node_levels"]["0"] == 16
    assert len(hints["portfolio"]) == 2
    assert oe_backend._static_validate_hints(context, hints)["valid"] is True


def test_latency_beam_policy_produces_valid_assignment(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    le = LatencyEstimator(params)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        {"strategy": "latency_beam", "beam_width": 4},
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    assert assign.check_assign() is True
    assert io_to_cost


def test_builder_depth_fanout_policy_uses_graph_summary(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).depth_fanout_aware()["policy"]

    assert policy["strategy"] == "level_preserving"
    assert policy["refresh_fanout_at_level_floor"] is True
    assert policy["min_internal_level"] == params.bts_lb + 1
    assert policy["preferred_node_levels"]["mul0"] == params.lvl_ub


def test_builder_low_scale_frontier_policy_is_compile_seed(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "level_preserving"
    assert hints["refresh_fanout_at_level_floor"] is True
    assert hints["max_scale_candidates"] == 10
    assert hints["scale_penalty"] == 0.0
    assert hints["allow_seed_fallback"] is False
    assert not hints.get("portfolio")


def test_discover_finalists_uses_best_and_checkpoint_scores(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_code = "def place(context):\n    return {'tag': 'result_best'}\n"
    high_code = "def place(context):\n    return {'tag': 'high_checkpoint'}\n"
    best_dir_code = "def place(context):\n    return {'tag': 'best_dir'}\n"
    low_code = "def place(context):\n    return {'tag': 'low_checkpoint'}\n"
    (output_dir / "best").mkdir(parents=True)
    (output_dir / "best" / "best_program.py").write_text(best_dir_code, encoding="utf-8")
    (output_dir / "best" / "best_program_info.json").write_text(
        json.dumps({"metrics": {"combined_score": 1.5}}),
        encoding="utf-8",
    )
    for name, code, score in (
        ("checkpoint_10", high_code, 1.9),
        ("checkpoint_20", low_code, 0.2),
    ):
        program_dir = output_dir / "checkpoints" / name / "programs"
        program_dir.mkdir(parents=True)
        (program_dir / "program.json").write_text(
            json.dumps({"code": code, "metrics": {"combined_score": score}}),
            encoding="utf-8",
        )

    finalists = oe_backend._discover_finalist_codes(output_dir, best_code, 3)

    assert finalists[0] == best_code
    assert high_code in finalists
    assert best_dir_code in finalists
    assert low_code not in finalists


def test_evaluator_validity_counts_duplicate_requested_budgets(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "candidate.py"
    budgets = [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 40}]
    context_path.write_text(json.dumps(build_context(graph, budgets, params)), encoding="utf-8")
    program_path.write_text("def place(context):\n    return {}\n", encoding="utf-8")

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert result["metrics"]["solved_budgets"] == 2.0
    assert result["metrics"]["budget_count"] == 2.0


def test_evaluator_rejects_invalid_policy_score(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    good_program = tmp_path / "good.py"
    bad_program = tmp_path / "bad.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    good_program.write_text("def place(context):\n    return {}\n", encoding="utf-8")
    bad_program.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 9999}}\n",
        encoding="utf-8",
    )

    good = evaluate_candidate_program(context_path, good_program)
    bad = evaluate_candidate_program(context_path, bad_program)

    assert good["metrics"]["validity"] == 1.0
    assert bad["metrics"]["validity"] == 0.0
    assert bad["metrics"]["combined_score"] < good["metrics"]["combined_score"]


def test_partition_fallback_selected_candidate_scores_below_valid(
    toy_cost_json: str, tmp_path: Path, monkeypatch
):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context_path = tmp_path / "context.json"
    program_path = tmp_path / "fallback.py"
    context_path.write_text(
        json.dumps(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)),
        encoding="utf-8",
    )
    program_path.write_text(
        "def place(context):\n    return {'strategy': 'level_preserving', 'allow_seed_fallback': True}\n",
        encoding="utf-8",
    )

    def fake_solve_budget_batch(_tdag, _budgets, _le, _params, _hints, diagnostics):
        diagnostics["solved_budgets"] = 1
        diagnostics["candidate_solved_budgets"] = 1
        diagnostics["fallback_selected_budgets"] = 1
        diagnostics["candidate_improved_budgets"] = 0
        diagnostics["costs"] = [10.0]
        diagnostics["candidate_costs"] = [30.0]
        diagnostics["assignments"] = []
        diagnostics["selected_source_counts"] = {"seed_fallback": 1}
        return {(-1, 40): {(16, 40): object()}}, {(-1, 40): {(16, 40): 10.0}}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)
    monkeypatch.setattr(
        oe_backend,
        "_reference_metrics",
        lambda *_args, **_kwargs: {
            "avg_latency_usec": 10.0,
            "bootstrap_count": 1,
            "rescale_count": 1,
            "solved_budgets": 1,
        },
    )

    result = evaluate_candidate_program(context_path, program_path)

    assert result["metrics"]["fallback_selected_budgets"] == 1.0
    assert result["metrics"]["candidate_only_avg_latency_usec"] == 30.0
    assert json.loads(result["artifacts"]["selected_source_counts"]) == {"seed_fallback": 1}
    assert result["metrics"]["combined_score"] < 1.0
    assert result["metrics"]["combined_score"] > 0.0


def test_polybert_sampled_budgets_keep_bypass_and_cost_extremes(toy_cost_json: str):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = "polybert-sampled"
    manager = QBPManager(params, None)
    budgets = []
    for idx in range(120):
        budget = {
            "in_lvl": idx % 4,
            "in_scl": 20 + idx,
            "out_lvl": (idx % params.lvl_ub) + 1,
            "main_dag_size": idx,
            "main_qbp_cost": {(1, 20): float(idx)},
        }
        if idx % 17 == 0:
            budget["maino_v"] = f"fork_{idx}"
        budgets.append(budget)

    sampled = manager._sample_openevolve_eval_budgets(budgets)

    assert 0 < len(sampled) <= 48
    assert any("maino_v" in budget for budget in sampled)
    assert max(min(budget["main_qbp_cost"].values()) for budget in sampled) == 119.0


def test_evaluator_uses_quality_as_partial_validity_tiebreaker(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    graph = _mul_chain_pdag(params, length=4)
    context_path = tmp_path / "context.json"
    seed_program = tmp_path / "seed.py"
    evolved_program = tmp_path / "evolved.py"
    budgets = [
        {"in_lvl": -1, "in_scl": 40},
        {"in_lvl": 1, "in_scl": 9999},
    ]
    context_path.write_text(json.dumps(build_context(graph, budgets, params)), encoding="utf-8")
    seed_program.write_text(
        "def place(context):\n    return {'strategy': 'waterline_seed'}\n",
        encoding="utf-8",
    )
    evolved_program.write_text(
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'refresh_fanout_at_level_floor': True}\n",
        encoding="utf-8",
    )

    seed = evaluate_candidate_program(context_path, seed_program)
    evolved = evaluate_candidate_program(context_path, evolved_program)

    assert seed["metrics"]["validity"] == evolved["metrics"]["validity"] == 0.5
    assert evolved["metrics"]["avg_latency_usec"] < seed["metrics"]["avg_latency_usec"]
    assert evolved["metrics"]["combined_score"] > seed["metrics"]["combined_score"]
    assert evolved["metrics"]["combined_score"] < 1.0


def test_compile_harness_scores_final_compile_latency(toy_cost_json: str, tmp_path: Path):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="toy",
    )
    graph = _branch_merge_pdag(params)
    context = build_compile_context(graph, params)
    context_path = tmp_path / "compile_context.json"
    seed_program = tmp_path / "seed.py"
    evolved_program = tmp_path / "evolved.py"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    seed_program.write_text("def place(context):\n    return {}\n", encoding="utf-8")
    evolved_program.write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving()\n",
        encoding="utf-8",
    )

    seed = evaluate_compile_candidate_program(context_path, seed_program)
    evolved = evaluate_compile_candidate_program(context_path, evolved_program)

    assert seed["metrics"]["validity"] == 1.0
    assert evolved["metrics"]["validity"] == 1.0
    assert "candidate_final_latency_usec" in evolved["artifacts"]
    assert evolved["metrics"]["final_latency_usec"] > 0.0


def test_generated_gemini_config_defaults(toy_cost_json: str, monkeypatch):
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None)
    params = _params(toy_cost_json, openevolve_iterations=50, openevolve_seed=7)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert config.random_seed == 7
    assert config.database.random_seed == 7
    assert config.llm.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.llm.models[0].name == "gemini-3.1-flash-lite"
    assert config.llm.models[0].api_base == config.llm.api_base
    assert config.llm.models[0].random_seed == 7
    assert config.llm.timeout == 180
    assert config.llm.retries == 1
    assert config.llm.retry_delay == 2
    assert config.llm.max_tokens == 2048
    assert config.llm.models[0].timeout == 180
    assert config.evaluator.timeout == 180
    assert config.evaluator.parallel_evaluations == 1
    assert config.checkpoint_interval == 5
    assert config.database.feature_dimensions == [
        "final_latency_usec",
        "boundary_quality",
        "bootstrap_count",
        "rescale_count",
        "fallback_selected_budgets",
        "profile_risk",
        "placement_runtime_sec",
    ]


def test_explicit_openevolve_config_preserves_model_and_updates_seed(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    loaded = FakeConfig()
    loaded.llm.models = [FakeLLMModelConfig(name="custom-model", api_base="https://custom.invalid/v1")]
    loaded.llm.evaluator_models = [
        FakeLLMModelConfig(name="custom-evaluator", api_base="https://custom.invalid/v1")
    ]
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None, loaded_config=loaded)
    params = _params(
        toy_cost_json,
        openevolve_config=str(tmp_path / "config.yaml"),
        openevolve_model="gemini-3.1-pro-preview",
        openevolve_seed=99,
    )
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert config is loaded
    assert config.random_seed == 99
    assert config.database.random_seed == 99
    assert config.llm.models[0].name == "custom-model"
    assert config.llm.evaluator_models[0].name == "custom-evaluator"
    assert config.llm.models[0].random_seed == 99


def test_openevolve_env_uses_openai_key(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with worker._openevolve_runtime_env():
        assert os.environ["OPENAI_API_KEY"] == "test-openai-key"


def test_openevolve_env_gemini_fallback_bridges_openai(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")

    with worker._openevolve_runtime_env():
        assert os.environ["OPENAI_API_KEY"] == "test-gemini-key"

    assert "OPENAI_API_KEY" not in os.environ


def test_openevolve_env_missing_key_has_clear_error(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=2)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError) as excinfo:
        with worker._openevolve_runtime_env():
            pass

    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "GEMINI_API_KEY" in msg
    assert "AIza" not in msg


def test_mocked_openevolve_runtime_runs_once_per_budget_batch(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    class FakeResult:
        best_code = (
            "def place(context):\n"
            "    return {'strategy': 'waterline_seed'}\n"
        )

    def fake_run_evolution(**kwargs):
        assert os.environ["OPENAI_API_KEY"] == "test-gemini-key"
        calls.append(kwargs)
        return FakeResult()

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gurobipy" or name.startswith("gurobipy.") or name == "pulp":
            raise AssertionError("OpenEvolve placement imported an ILP solver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    params = _params(
        toy_cost_json,
        openevolve_iterations=3,
        openevolve_output_dir=str(tmp_path),
    )
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    default_assign, _default_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 41}],
        LatencyEstimator(params),
        params,
    )

    io_to_assign, io_to_cost = worker.get_qbp(
        graph,
        [{"in_lvl": -1, "in_scl": 40}, {"in_lvl": -1, "in_scl": 41}],
    )

    assert len(calls) == 1
    assert calls[0]["iterations"] == 3
    assert calls[0]["config"].llm.models[0].name == "gemini-3.1-flash-lite"
    assert calls[0]["config"].llm.models[0].api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert calls[0]["config"].database.random_seed == 42
    assert Path(calls[0]["initial_program"]).is_file()
    assert Path(calls[0]["evaluator"]).is_file()
    assert "OPENAI_API_KEY" not in os.environ
    assert io_to_assign
    assert io_to_cost
    assert io_to_cost != _default_cost


def test_mocked_compile_harness_runs_once_for_compile(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    calls = []

    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving()\n"
        )

    def fake_run_evolution(**kwargs):
        calls.append(kwargs)
        return FakeResult()

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )
    graph = _branch_merge_pdag(params)

    assign, timings = orbit_core(graph, LatencyEstimator(params), params)

    assert assign is not None
    assert assign.check_assign() is True
    assert len(calls) == 1
    assert calls[0]["iterations"] == 2
    assert Path(calls[0]["initial_program"]).is_file()
    assert Path(calls[0]["evaluator"]).is_file()
    assert "OpenEvolve Compile Harness Time" in timings


def test_compile_harness_fail_open_returns_initial_seed(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    def fake_run_evolution(**_kwargs):
        raise TimeoutError("llm timed out")

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints["refresh_fanout_at_level_floor"] is True
    assert (tmp_path / "compile_oe_toy" / "openevolve_recovery.json").is_file()


def test_compile_harness_fail_open_recovers_checkpoint_candidate(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    code = (
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving(level_drop_penalty=321.0)\n"
    )

    def fake_run_evolution(**kwargs):
        program_dir = Path(kwargs["output_dir"]) / "checkpoints" / "checkpoint_1" / "programs"
        program_dir.mkdir(parents=True)
        (program_dir / "candidate.json").write_text(
            json.dumps({"code": code, "metrics": {"combined_score": 2.0}}),
            encoding="utf-8",
        )
        raise TimeoutError("llm timed out")

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 321.0 else 100.0,
            "bootstrap_count": 1,
            "rescale_count": 1,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="toy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 321.0


def test_compile_harness_reruns_best_candidate_on_full_bundle(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    class FakeResult:
        best_code = (
            "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
            "def place(context):\n"
            "    return PlacementBuilder(context).level_preserving(level_drop_penalty=123.0)\n"
        )

    def fake_run_evolution(**_kwargs):
        return FakeResult()

    eval_suites = []

    def fake_evaluate_compile_hints(context, hints, *, suppress_output):
        eval_suites.append((context.get("harness", {}).get("eval_suite"), hints.get("level_drop_penalty")))
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 100.0 if hints else 120.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {},
            "bootstrap_locations": {},
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints["level_drop_penalty"] == 123.0
    assert ("polybert-sampled", None) in eval_suites
    assert ("polybert-full", None) in eval_suites
    assert ("polybert-full", 123.0) in eval_suites
    assert (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").is_file()


def test_zero_iteration_openevolve_path_does_not_import_gurobipy(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_iterations=0)
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gurobipy" or name.startswith("gurobipy.") or name == "pulp":
            raise AssertionError("OpenEvolve placement imported an ILP solver")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    io_to_assign, io_to_cost = worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    assert io_to_assign
    assert io_to_cost
