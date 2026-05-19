from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import types
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.assignment import Assign
from scripts.latency_estimator.latency_estimator import LatencyEstimator
import scripts.optimizer.orbit.openevolve_backend as oe_backend
from scripts.optimizer.orbit.openevolve_backend import (
    OpenEvolvePlacementWorker,
    PlacementBuilder,
    PlacementConstraints,
    PlacementMCTS,
    _aggregate_counts,
    build_context,
    build_compile_context,
    candidate_actions,
    evaluate_compile_candidate_program,
    evaluate_candidate_program,
    run_compile_openevolve,
    serialize_assignment_record,
    solve_budget_batch,
    tdag_from_context,
    transition_slack,
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
    assert context["constraints"]["ilp_semantics"]["solver_free"] is True
    assert "decryptable" in context["constraints"]["ilp_semantics"]["constraints"]
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
    assert invalid["metrics"]["effective_validity"] == 1.0
    assert invalid["metrics"]["fallback_selected_budgets"] == 1.0
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
    assert not valid_transition(context, 16, params.max_scale() + 1, 15, 40)
    slack = transition_slack(context, 16, 40, 15, 40)
    assert slack["valid"] is True
    assert slack["uses_bootstrap"] is False
    assert slack["transition_reserve_bits"] >= 0
    invalid = transition_slack(context, 1, 131, 16, 40)
    assert invalid["valid"] is False
    assert "input_not_decryptable" in invalid["violations"]
    above_max = transition_slack(context, 16, params.max_scale() + 1, 15, 40)
    assert above_max["valid"] is False
    assert "input_scale_above_max" in above_max["violations"]
    assert PlacementBuilder(context).level_preserving()["policy"]["strategy"] == "level_preserving"
    constraints = PlacementConstraints(context)
    assert constraints.decryptability_bound(16) == params.Sf * (16 - params.lvl_lb + 2) - 7
    assert constraints.boundary_output_scale_bound(16) == params.Sf * (16 + 1) - 7
    assert constraints.valid_transition(16, 40, 15, 40)
    assert constraints.transition_slack(16, 40, 15, 40)["valid"] is True
    assert constraints.node_scale_lower_bound("0", "out") == params.Sw
    assert constraints.edge_scale_lower_bound("arg0", "0") == params.Sw
    assert constraints.mul_input_scale([40]) == 80
    states = constraints.legal_output_states(16, 40, include_bootstrap=False)
    assert states
    assert all(not state["uses_bootstrap"] for state in states)
    assert states[0]["level"] <= params.lvl_ub
    assert PlacementBuilder(context).constraints.legal_output_states(16, 40)


def test_assignment_validation_rejects_non_decryptable_candidate_record(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    assign = Assign(graph)
    too_large_scale = params.max_scale() + 1
    assign.v_lvl_in["arg0"] = params.lvl_ub
    assign.v_scl_in["arg0"] = 40
    assign.v_lvl_out["arg0"] = params.lvl_ub
    assign.v_scl_out["arg0"] = 40
    assign.e_lvl_out[("arg0", "0")] = params.lvl_ub
    assign.e_scl_out[("arg0", "0")] = 40
    assign.v_lvl_out["0"] = params.lvl_ub
    assign.v_scl_out["0"] = too_large_scale

    with pytest.raises(ValueError, match="decryptability bounds"):
        assign.check_assign()


def test_none_min_internal_level_matches_ilp_lower_bound(toy_cost_json: str):
    params = _params(toy_cost_json)
    options = oe_backend._policy_options({"strategy": "level_preserving", "min_internal_level": None}, params)

    assert options["min_internal_level"] is None
    assert oe_backend._effective_min_internal_level(options, params) == params.lvl_lb


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


def test_builder_noise_guarded_policy_preserves_headroom(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).noise_guarded_refresh()["policy"]

    assert policy["strategy"] == "level_preserving"
    assert policy["refresh_fanout_at_level_floor"] is True
    assert policy["allow_seed_fallback"] is False
    assert policy["min_internal_level"] == 12
    assert policy["max_scale_candidates"] == 12
    assert policy["reserve_penalty"] > 0
    assert policy["min_transition_reserve"] > 0


def test_profile_layer_refresh_targets_reference_bert_layernorm_comments(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    graph.nodes["left"]["comment"] = (
        "scope=bert.encoder.layer.0.attention.output.LayerNorm;op=layer_norm"
    )
    graph.nodes["right"]["comment"] = (
        "scope=bert.encoder.layer.0.output.LayerNorm;op=layer_norm"
    )
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    policy = PlacementBuilder(context).profile_layer_refresh(include_attention=True)["policy"]

    assert policy["preferred_node_levels"]["left"] == params.lvl_ub
    assert policy["preferred_node_levels"]["right"] == params.lvl_ub
    assert policy["preferred_node_scales"]["left"] == params.Sf
    assert policy["preferred_node_scales"]["right"] == params.Sf


def test_builder_compile_seed_includes_bounded_reliable_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "level_preserving"
    assert hints["refresh_fanout_at_level_floor"] is True
    assert hints["max_scale_candidates"] == 10
    assert hints["min_internal_level"] is None
    assert hints["level_drop_penalty"] == 20_000_000.0
    assert hints["allow_seed_fallback"] is False
    assert [policy["strategy"] for policy in hints["portfolio"]] == [
        "latency_beam",
        "latency_beam",
        "level_preserving",
        "latency_beam",
        "level_preserving",
        "level_preserving",
        "level_preserving",
        "level_preserving",
        "level_preserving",
    ]
    assert hints["portfolio"][0]["budget_aggressive"] is True
    assert hints["portfolio"][1]["selection_bootstrap_penalty"] == 500_000_000.0
    assert hints["portfolio"][4]["bootstrap_penalty"] == 650_000_000.0
    assert hints["portfolio"][5]["bootstrap_penalty"] == 650_000_000.0
    assert hints["portfolio"][6]["min_internal_level"] == 6
    assert hints["portfolio"][7]["min_internal_level"] == 8
    assert hints["portfolio"][8]["min_internal_level"] == 12


def test_initial_compile_seed_exposes_active_bootstrap_mcts_knobs(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(
        toy_cost_json,
        openevolve_search_mode="bootstrap-mcts",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    program_path = tmp_path / "initial.py"
    source = oe_backend._initial_compile_program_source("bootstrap-mcts")
    assert "builder.budget_fulfillment_beam" not in source
    program_path.write_text(source, encoding="utf-8")

    hints = oe_backend._load_candidate_hints(program_path, context)
    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")

    assert hints["strategy"] == "bootstrap_mcts"
    assert hints["mcts_rollout_budget"] == 24
    assert hints["mcts_exploration_weight"] == 1.15
    raw_budget_beams = [
        action for action in hints["mcts_actions"] if action.get("name") == "budget_fulfillment_beam"
    ]
    assert raw_budget_beams
    assert raw_budget_beams[0]["prior"] == 0.34
    assert raw_budget_beams[0]["policy"]["beam_width"] == 4
    capped_names = [
        action["name"]
        for action in sampled["mcts_actions"][: sampled["mcts_action_cap"]]
    ]
    assert "budget_fulfillment_beam" in capped_names


def test_sampled_compile_eval_keeps_bounded_candidate_portfolio(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")
    full = oe_backend._compile_hints_for_eval_suite(hints, "polybert-full")

    assert sampled["strategy"] == hints["strategy"]
    assert len(sampled["portfolio"]) == 2
    assert sampled["allow_seed_fallback"] is True
    assert sampled["max_scale_candidates"] <= 16
    assert sampled["beam_width"] <= 4
    assert sampled["allow_bootstrap"] is False
    assert sampled["portfolio"][0]["strategy"] == "latency_beam"
    assert sampled["portfolio"][0]["budget_aggressive"] is True
    assert sampled["portfolio"][0]["allow_seed_fallback"] is True
    assert sampled["portfolio"][0]["state_cap_per_node"] <= 16
    assert sampled["portfolio"][0]["beam_width"] <= 6
    assert len(full["portfolio"]) == 9


def test_full_bundle_finalist_replay_bounds_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    finalist = oe_backend._finalist_hints_for_full_bundle(hints)

    assert finalist["strategy"] == "level_preserving"
    assert "portfolio" not in finalist
    assert finalist["allow_seed_fallback"] is False
    assert finalist["max_scale_candidates"] <= 16


def test_compile_fail_open_uses_validated_zero_iteration_portfolio(
    toy_cost_json: str, tmp_path: Path
):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    initial_hints = oe_backend._load_candidate_hints(program_path, context)

    fail_open = oe_backend._bounded_fail_open_hints(initial_hints)

    assert fail_open["fail_open_reason"] == "zero_iteration_seed_portfolio"
    assert fail_open["strategy"] == "level_preserving"
    assert isinstance(fail_open.get("portfolio"), list)
    assert {policy.get("min_internal_level") for policy in fail_open["portfolio"]} >= {6, 8, 12}


def test_target_bootstrap_score_rewards_absolute_progress_without_reference():
    high = oe_backend._target_bootstrap_score(9, 795, None)
    lower = oe_backend._target_bootstrap_score(9, 72, None)

    assert high > 0.0
    assert lower > high
    assert oe_backend._target_bootstrap_score(9, 9, None) == 1.0


def test_assignment_count_summary_does_not_sum_qbp_alternatives(monkeypatch):
    assignments = [object(), object(), object()]
    counts = iter(
        [
            {"bootstrap": 10, "rescale": 3},
            {"bootstrap": 20, "rescale": 5},
            {"bootstrap": 30, "rescale": 7},
        ]
    )
    monkeypatch.setattr(oe_backend, "_maintenance_counts", lambda _assign: next(counts))

    summary = oe_backend._assignment_count_summary(assignments)

    assert summary["aggregate_bootstrap"] == 60.0
    assert summary["avg_bootstrap"] == 20.0
    assert summary["min_bootstrap"] == 10.0
    assert summary["max_bootstrap"] == 30.0


def test_bootstrap_mcts_api_builds_low_bootstrap_actions(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)

    hints = PlacementMCTS(context).low_bootstrap_seed(target_bootstraps=9, action_cap=6)
    sampled = oe_backend._compile_hints_for_eval_suite(hints, "polybert-sampled")

    assert hints["strategy"] == "bootstrap_mcts"
    assert hints["target_bootstrap_count"] == 9
    assert len(hints["mcts_actions"]) >= 4
    assert hints["mcts_actions"][0]["name"] == "strict_no_bootstrap"
    assert any(
        action["name"] == "budget_fulfillment_beam"
        and action["policy"]["strategy"] == "latency_beam"
        and action["policy"]["direct_budget_policy"] is True
        for action in hints["mcts_actions"]
    )
    assert any(
        action["name"] == "waterline_budget_repair"
        and action["policy"]["strategy"] == "waterline_seed"
        and action["policy"]["direct_budget_policy"] is True
        for action in hints["mcts_actions"]
    )
    assert any(action["policy"].get("forbid_bootstrap") for action in hints["mcts_actions"])
    assert any(action["policy"]["allow_bootstrap"] for action in hints["mcts_actions"])
    assert sampled["mcts_rollout_budget"] <= 2
    assert sampled["mcts_action_cap"] <= 2
    assert sampled["mcts_max_repair_bootstraps"] <= 4
    capped_action_names = [
        action["name"]
        for action in sampled["mcts_actions"][: sampled["mcts_action_cap"]]
    ]
    assert "budget_fulfillment_beam" in capped_action_names
    sampled_latency_actions = [
        action
        for action in sampled["mcts_actions"]
        if action.get("policy", {}).get("strategy") == "latency_beam"
    ]
    assert [action["name"] for action in sampled_latency_actions] == ["budget_fulfillment_beam"]
    assert sampled_latency_actions[0]["policy"]["beam_width"] <= 2
    assert sampled_latency_actions[0]["policy"]["state_cap_per_node"] <= 4
    assert sampled_latency_actions[0]["policy"]["max_scale_candidates"] <= 8


def test_compile_evaluator_caps_bootstrap_mcts_sampled_rollouts(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_budget_aggressive=True,
        openevolve_target_bootstrap_count=9,
    )
    graph = _branch_merge_pdag(params)
    context_path = tmp_path / "compile_context.json"
    program_path = tmp_path / "candidate.py"
    context_path.write_text(json.dumps(build_compile_context(graph, params)), encoding="utf-8")
    program_path.write_text(
        "def place(context):\n"
        "    return {\n"
        "        'strategy': 'bootstrap_mcts',\n"
        "        'target_bootstrap_count': 9,\n"
        "        'mcts_rollout_budget': 48,\n"
        "        'mcts_action_cap': 32,\n"
        "        'budget_aggressive': True,\n"
        "    }\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_evaluate_compile_hints(_context, hints, suppress_output=False):
        captured.update(hints)
        assert suppress_output is True
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 10.0,
            "bootstrap_count": 3,
            "rescale_count": 2,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.1,
            "fallback_selected_budgets": 0,
            "boundary_quality": 1.0,
            "reserve_summary": {},
            "selected_output_state": {},
            "bottleneck_summary": [],
            "bootstrap_locations": [],
            "rescale_locations": [],
            "log_tail": "",
            "diagnostics": {
                "requested_budgets": 1,
                "candidate_solved_budgets": 1,
                "solved_budgets": 1,
            },
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    result = evaluate_compile_candidate_program(context_path, program_path)

    assert result["metrics"]["validity"] == 1.0
    assert captured["strategy"] == "bootstrap_mcts"
    assert captured["mcts_rollout_budget"] <= 4
    assert captured["mcts_action_cap"] <= 4
    assert captured["mcts_max_repair_bootstraps"] == 32
    assert captured["boundary_state_cap"] == 1


def test_bootstrap_mcts_rejects_rollouts_over_repair_cap(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    graph = _toy_pdag(params)

    class FakeAssign:
        def check_assign(self):
            return True

    monkeypatch.setattr(oe_backend, "build_conservative_assign", lambda *_args, **_kwargs: FakeAssign())
    monkeypatch.setattr(oe_backend, "_policy_assignment_score", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(
        oe_backend,
        "_aggregate_counts",
        lambda _assignments: {"bootstrap": 5, "rescale": 0},
    )

    with pytest.raises(oe_backend.PlacementError, match="bootstrap repair cap"):
        oe_backend._build_bootstrap_mcts_assign(
            graph,
            params,
            {"in_lvl": -1, "in_scl": 40},
            {
                "strategy": "bootstrap_mcts",
                "mcts_rollout_budget": 1,
                "mcts_action_cap": 1,
                "mcts_max_repair_bootstraps": 4,
            },
            None,
        )


def test_mcts_selection_expands_unvisited_actions_first():
    actions = [
        oe_backend.MCTSAction("visited", {"strategy": "level_preserving"}),
        oe_backend.MCTSAction("unvisited", {"strategy": "level_preserving"}),
    ]
    stats = [
        oe_backend._MCTSNodeStats(visits=3, reward_sum=3.0),
        oe_backend._MCTSNodeStats(),
    ]

    assert oe_backend._select_mcts_action(actions, stats, step=4, exploration_weight=1.4) == 1


def test_bootstrap_mcts_seed_solves_toy_without_more_bootstraps_than_legacy(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    budgets = [{"in_lvl": -1, "in_scl": 40}]
    legacy_diag = {}
    mcts_diag = {}

    solve_budget_batch(graph, budgets, le, params, oe_backend._zero_iteration_portfolio_hints(), legacy_diag)
    solve_budget_batch(graph, budgets, le, params, oe_backend._bootstrap_mcts_seed_policy(params), mcts_diag)

    legacy_counts = _aggregate_counts(legacy_diag["assignments"])
    mcts_counts = _aggregate_counts(mcts_diag["assignments"])
    assert mcts_diag["solved_budgets"] == 1
    assert mcts_counts["bootstrap"] <= legacy_counts["bootstrap"]


def test_bootstrap_mcts_batch_keeps_complete_boundary_group(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [
        {"in_lvl": -1, "in_scl": 40, "out_lvl": level}
        for level in range(1, params.lvl_ub + 1)
    ]
    captured_groups = []

    def fake_group_attempts(pdag, params_arg, group_budgets, le_arg, hints, diagnostics):
        captured_groups.append([budget["out_lvl"] for budget in group_budgets])
        attempts = {}
        for idx, budget in enumerate(group_budgets):
            attempt = oe_backend._solve_one_budget_attempt(
                pdag,
                params_arg,
                budget,
                le_arg,
                "candidate:boundary_mcts:test",
                oe_backend._default_policy_hints(),
            )
            attempts[idx] = [attempt]
        return attempts

    monkeypatch.setattr(oe_backend, "_boundary_mcts_group_attempts", fake_group_attempts)
    diagnostics = {}

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert captured_groups == [list(range(1, params.lvl_ub + 1))]
    assert diagnostics["requested_budgets"] == params.lvl_ub
    assert diagnostics["candidate_solved_budgets"] == params.lvl_ub
    assert diagnostics["fallback_selected_budgets"] == 0
    assert io_to_assign
    assert io_to_cost


def test_bootstrap_mcts_final_compile_fail_opens_to_seed(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, openevolve_iterations=1)
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [{"in_lvl": -1, "in_scl": 40, "out_lvl": 1}]
    diagnostics = {}
    fallback_called = False

    def fake_boundary(*_args, **_kwargs):
        return {}, {}

    def fake_fast_seed(pdag, budget_list, le_arg, params_arg, diagnostics_arg):
        nonlocal fallback_called
        fallback_called = True
        return oe_backend.solve_budget_batch(
            pdag,
            budget_list,
            le_arg,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            diagnostics_arg,
        )

    monkeypatch.setattr(oe_backend, "_solve_budget_batch_boundary_mcts", fake_boundary)
    monkeypatch.setattr(oe_backend, "_solve_budget_batch_fast_seed", fake_fast_seed)

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert fallback_called
    assert diagnostics["fail_open_seed_replay"] is True
    assert io_to_assign
    assert io_to_cost


def test_bootstrap_mcts_can_opt_into_direct_budget_beam_before_actions(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": params.Sw}],
        le,
        params,
        {
            **oe_backend._bootstrap_mcts_seed_policy(params),
            "enable_direct_budget_beam": True,
            "allow_seed_fallback": False,
        },
        diagnostics,
    )

    assert diagnostics["candidate_solved_budgets"] == 1
    assert diagnostics["fallback_selected_budgets"] == 0
    assert diagnostics["selected_source_counts"] == {"candidate:direct_budget_beam": 1}


def test_bootstrap_mcts_skips_direct_budget_beam_by_default(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    diagnostics = {}

    solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": params.Sw}],
        le,
        params,
        oe_backend._bootstrap_mcts_seed_policy(params),
        diagnostics,
    )

    assert "candidate:direct_budget_beam" not in diagnostics["selected_source_counts"]


def test_sampled_seed_fallback_skips_latency_beam(toy_cost_json: str):
    params = _params(toy_cost_json)
    params.openevolve_evaluating_candidate = True
    params.openevolve_eval_suite = "polybert-sampled"

    sources = [source for source, _policy in oe_backend._seed_fallback_attempts(params)]

    assert "seed_fallback_latency_beam" not in sources
    assert sources == ["seed_fallback", "seed_fallback_relaxed", "seed_fallback_waterline"]


def test_compile_harness_retries_bypass_replay_without_bypass(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json, bpsdepth=15)
    params.part = True
    params.openevolve_eval_suite = "polybert-full"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)
    calls = []

    def fake_solve_partition(tdag, qbp_manager, prev_cost, le, params_arg):
        calls.append(params_arg.bpsdepth)
        if params_arg.bpsdepth is not None:
            raise AssertionError("Main PDAG #toy QBP not found in manager. Please add it first.")
        return oe_backend.solve_budget_batch(
            tdag,
            [{"in_lvl": -1, "in_scl": params_arg.Sw}],
            le,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            {},
        )

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(context, {}, suppress_output=True)

    assert result["valid"] is True
    assert calls == [15, None]
    assert "retrying with bypass disabled" in result["log_tail"]


def test_sampled_compile_harness_scores_partial_budget_progress(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_eval_suite = "polybert-sampled"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)

    def fake_solve_partition(tdag, qbp_manager, prev_cost, le, params_arg):
        diagnostics = {}
        oe_backend.solve_budget_batch(
            tdag,
            [{"in_lvl": -1, "in_scl": params_arg.Sw}],
            le,
            params_arg,
            oe_backend._low_scale_frontier_policy(),
            diagnostics,
        )
        qbp_manager.openevolve_diagnostics.append(diagnostics)
        raise oe_backend.PlacementError("compile replay produced no valid final partitioning")

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(context, {}, suppress_output=True)

    assert result["valid"] is True
    assert result["sampled_progress_only"] is True
    assert result["validity"] == 1.0
    assert result["diagnostics"]["sampled_progress_only"] is True
    assert result["bootstrap_count"] >= 0


def test_sampled_compile_harness_uses_cached_budget_tasks(
    toy_cost_json: str,
    monkeypatch,
):
    params = _params(toy_cost_json)
    params.openevolve_eval_suite = "polybert-sampled"
    graph = _toy_pdag(params)
    context = build_compile_context(graph, params)
    context["sampled_budget_tasks"] = [
        {
            "kind": "normal",
            "context": build_context(
                graph,
                [{"in_lvl": -1, "in_scl": params.Sw}],
                params,
            ),
        }
    ]

    def fake_solve_partition(*_args, **_kwargs):
        raise AssertionError("cached sampled eval must not replay solve_partition")

    monkeypatch.setattr(
        "scripts.optimizer.orbit.iterative_partition.solve_partition",
        fake_solve_partition,
    )

    result = oe_backend._evaluate_compile_hints(
        context,
        oe_backend._bootstrap_mcts_seed_policy(params),
        suppress_output=True,
    )

    assert result["valid"] is True
    assert result["sampled_progress_only"] is True
    assert result["diagnostics"]["sampled_direct_budget_eval"] is False
    assert result["diagnostics"]["sampled_qbp_group_eval"] is True
    assert result["diagnostics"]["sampled_task_count"] == 1
    assert result["diagnostics"]["solved_budgets"] == 1


def test_bootstrap_mcts_sampled_budget_sampler_preserves_boundary_groups(
    toy_cost_json: str,
):
    params = _params(
        toy_cost_json,
        openevolve_harness="compile",
        openevolve_search_mode="bootstrap-mcts",
        openevolve_eval_suite="polybert-sampled",
        openevolve_max_unit_samples=4,
    )
    params.openevolve_evaluating_candidate = True
    manager = QBPManager(params, LatencyEstimator(params))
    budgets = [
        {"in_lvl": -1, "in_scl": params.Sw, "out_lvl": out_lvl}
        for out_lvl in range(1, params.lvl_ub + 1)
    ]

    sampled = manager._sample_openevolve_eval_budgets(_toy_pdag(params), budgets)

    expected_levels = {1, max(1, params.bts_lb + 1), max(1, params.lvl_ub // 2), params.lvl_ub}
    assert len(sampled) == len(expected_levels)
    assert {int(item["out_lvl"]) for item in sampled} == expected_levels
    assert {
        (int(item["in_lvl"]), int(item["in_scl"]), str(item.get("maino_v", "")))
        for item in sampled
    } == {(-1, params.Sw, "")}


def test_bootstrap_mcts_returns_complete_qbp_boundary_group(toy_cost_json: str):
    params = _params(toy_cost_json, openevolve_target_bootstrap_count=9)
    params.openevolve_evaluating_candidate = True
    graph = _toy_pdag(params)
    le = LatencyEstimator(params)
    budgets = [
        {"in_lvl": -1, "in_scl": params.Sw, "out_lvl": out_lvl}
        for out_lvl in range(1, params.lvl_ub + 1)
    ]
    diagnostics = {}

    io_to_assign, io_to_cost = solve_budget_batch(
        graph,
        budgets,
        le,
        params,
        {**oe_backend._bootstrap_mcts_seed_policy(params), "enable_direct_budget_beam": True},
        diagnostics,
    )

    assert io_to_assign
    assert io_to_cost
    out_levels = {
        out_lvl
        for out_to_cost in io_to_cost.values()
        for out_lvl, _out_scl in out_to_cost
    }
    assert out_levels == set(range(1, params.lvl_ub + 1))
    assert diagnostics["requested_boundary_groups"] == 1
    assert diagnostics["solved_boundary_groups"] == 1
    assert diagnostics["candidate_solved_boundary_groups"] == 1
    assert diagnostics["boundary_group_summaries"][0]["complete"] is True


def test_boundary_scale_candidates_obey_output_level_bound(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    policy = oe_backend._with_default_policy(
        {
            "boundary_scale_policy": "sf",
            "max_scale_candidates": 32,
        }
    )

    candidates = oe_backend._boundary_scale_candidates(
        graph,
        params,
        {"in_lvl": -1, "in_scl": 40, "out_lvl": 1},
        policy,
    )

    assert candidates
    assert all(scale <= params.boundary_output_scale_bound(1) for scale in candidates)
    assert all(scale >= params.scale_lower_bound("0", graph.nodes["0"], "out") for scale in candidates)


def test_sampled_invalid_best_skips_expensive_full_bundle(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "combined_score": 0.0925,
                    "validity": 0.0,
                    "effective_validity": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    assert oe_backend._sampled_best_invalid_reason(output_dir) == "sampled_best_solved_no_budgets"


def test_sampled_fallback_only_best_skips_expensive_full_bundle(tmp_path: Path):
    output_dir = tmp_path / "openevolve_output"
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "best_program_info.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "combined_score": 0.39,
                    "validity": 0.40,
                    "effective_validity": 0.40,
                    "candidate_qbp_coverage": 0.0,
                    "candidate_solved_boundary_groups": 0,
                    "fallback_selected_budgets": 34,
                    "fallback_selected_groups": 11,
                }
            }
        ),
        encoding="utf-8",
    )

    assert (
        oe_backend._sampled_best_invalid_reason(output_dir)
        == "sampled_best_has_no_direct_qbp_coverage"
    )


def test_full_bundle_finalist_variants_include_bounded_portfolio(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    context = build_context(_mul_chain_pdag(params, length=4), [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "initial.py"
    program_path.write_text(oe_backend._initial_compile_program_source(), encoding="utf-8")
    hints = oe_backend._load_candidate_hints(program_path, context)

    variants = oe_backend._finalist_hint_variants(hints)

    assert [label for label, _policy in variants][0] == "primary"
    assert len(variants) == 6
    assert any(label.startswith("portfolio_") for label, _policy in variants[1:])
    assert all("portfolio" not in policy for _label, policy in variants)


def test_layer_nonlinear_units_are_extracted_from_comments(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = Tdag(params, "unit_graph")
    graph.add_node("arg0", op="input", weight=1, op_descr={}, comment="")
    graph.add_node(
        "softmax",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.attention.self.qk_softmax.softmax_mul;op=qk_softmax_mul",
    )
    graph.add_node(
        "act",
        op="mul",
        weight=1,
        op_descr={"single": 0, "double": 1},
        comment="scope=bert.encoder.layer.0.intermediate.gelu;op=quadratic_activation_square",
    )
    graph.add_edge("arg0", "softmax")
    graph.add_edge("softmax", "act")
    graph.inputs = {"arg0"}
    graph.outputs = {"act"}

    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)

    unit_ids = {unit["id"] for unit in context["placement_units"]}
    assert "layer:layer.0" in unit_ids
    assert "nonlinear:layer.0:attention_softmax" in unit_ids
    assert "nonlinear:layer.0:activation" in unit_ids
    assert context["unit_op_histogram"]["mul"] == 4


def test_unit_policy_api_and_lenient_repairs(toy_cost_json: str, tmp_path: Path):
    params = _params(toy_cost_json)
    graph = _toy_pdag(params)
    context = build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params)
    program_path = tmp_path / "candidate.py"
    program_path.write_text(
        """
def place(context):
    return {
        "global_policy": {"strategy": "not-real", "max_scale_candidates": 100},
        "unit_policies": [
            {
                "selector": {"layer": "layer.0"},
                "policy": {"min_internal_level": 999, "beam_width": 99},
            },
            {"selector": {"unit_id": "missing"}, "policy": {"max_scale_candidates": 8}},
        ],
        "patches": [{"op": "prefer_node_scale", "node": "missing", "scale": 999}],
    }
""",
        encoding="utf-8",
    )

    hints = oe_backend._load_candidate_hints(program_path, context)

    assert hints["strategy"] == "level_preserving"
    assert hints["max_scale_candidates"] == 32
    assert len(hints["unit_policies"]) == 1
    assert hints["unit_policies"][0]["policy"]["min_internal_level"] == params.lvl_ub
    assert hints["unit_policies"][0]["policy"]["beam_width"] == 8
    assert hints["preferred_node_scales"] == {}
    assert oe_backend._repair_count(hints) >= 4


def test_unit_policy_shape_keeps_flat_policy_compatibility(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_context(_toy_pdag(params), [{"in_lvl": -1, "in_scl": 40}], params)

    flat = oe_backend._normalize_candidate_hints({"max_scale_candidates": 7}, context)
    structured = oe_backend._normalize_candidate_hints(
        {
            "global_policy": {"max_scale_candidates": 9},
            "unit_policies": [
                {
                    "selector": {"layer": "layer.0"},
                    "policy": {"max_scale_candidates": 5},
                }
            ],
        },
        context,
    )

    assert flat["max_scale_candidates"] == 7
    assert structured["max_scale_candidates"] == 9
    assert structured["unit_policies"][0]["policy"]["max_scale_candidates"] == 5


def test_sampled_compile_eval_rejects_expensive_policy(toy_cost_json: str):
    params = _params(toy_cost_json)
    context = build_compile_context(_mul_chain_pdag(params, length=4), params)
    context["harness"]["eval_suite"] = "polybert-sampled"

    assert oe_backend._sampled_policy_static_reasons(
        context,
        {"strategy": "waterline_seed"},
    ) == []

    context["harness"]["leniency"] = "strict"
    reasons = oe_backend._sampled_policy_static_reasons(
        context,
        {"strategy": "waterline_seed"},
    )

    assert "sampled policy rejects strategy 'waterline_seed'" in reasons


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
    params.openevolve_search_mode = "legacy"
    manager = QBPManager(params, None)
    budgets = []
    for group_idx in range(20):
        for out_lvl in range(1, params.lvl_ub + 1):
            budget = {
                "in_lvl": group_idx % 4,
                "in_scl": 20 + group_idx,
                "out_lvl": out_lvl,
                "main_dag_size": group_idx,
                "main_qbp_cost": {(1, 20): float(group_idx)},
            }
            if group_idx in {0, 19}:
                budget["maino_v"] = f"fork_{group_idx}"
            budgets.append(budget)

    sampled = manager._sample_openevolve_eval_budgets(budgets)

    assert 0 < len(sampled) <= params.openevolve_max_unit_samples
    assert any("maino_v" in budget for budget in sampled)
    assert max(min(budget["main_qbp_cost"].values()) for budget in sampled) == 19.0
    by_group = defaultdict(set)
    for budget in sampled:
        by_group[(budget["in_lvl"], budget["in_scl"], budget.get("maino_v", ""))].add(
            budget["out_lvl"]
        )
    assert all(levels == set(range(1, params.lvl_ub + 1)) for levels in by_group.values())


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


def test_compile_invalid_metrics_cover_configured_feature_dimensions(
    toy_cost_json: str,
    tmp_path: Path,
):
    params = _params(toy_cost_json, openevolve_eval_suite="toy")
    graph = _toy_pdag(params)
    context_path = tmp_path / "compile_context.json"
    bad_program = tmp_path / "bad.py"
    context_path.write_text(json.dumps(build_compile_context(graph, params)), encoding="utf-8")
    bad_program.write_text(
        "def place(context):\n"
        "    return {'preferred_node_scales': {'0': 9999}}\n",
        encoding="utf-8",
    )

    result = evaluate_compile_candidate_program(context_path, bad_program)

    for key in (
        "final_latency_usec",
        "boundary_quality",
        "bootstrap_count",
        "rescale_count",
        "fallback_selected_budgets",
        "profile_risk",
        "placement_runtime_sec",
        "estimated_precision_bits",
        "output_margin_bits",
    ):
        assert key in result["metrics"]


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
        "estimated_precision_bits",
        "output_margin_bits",
    ]


def test_generated_multi_provider_config_uses_per_model_keys(toy_cost_json: str, monkeypatch):
    _install_fake_openevolve(monkeypatch, lambda **_kwargs: None)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    params = _params(
        toy_cost_json,
        openevolve_iterations=100,
        openevolve_provider="gemini",
        openevolve_model="gemini-3.1-flash-lite",
        openevolve_primary_weight=0.7,
        openevolve_secondary_provider="openai",
        openevolve_secondary_model="gpt-5.5",
        openevolve_secondary_weight=0.3,
    )
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))

    config = worker._openevolve_config_arg()

    assert [model.name for model in config.llm.models] == [
        "gemini-3.1-flash-lite",
        "gpt-5.5",
    ]
    assert config.llm.models[0].api_base == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert config.llm.models[1].api_base == "https://api.openai.com/v1"
    assert config.llm.models[0].api_key == "test-gemini-key"
    assert config.llm.models[1].api_key == "test-openai-key"
    assert config.llm.models[0].weight == pytest.approx(0.7)
    assert config.llm.models[1].weight == pytest.approx(0.3)


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
    assert hints["max_scale_candidates"] == 10
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
        openevolve_search_mode="legacy",
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints["level_drop_penalty"] == 123.0
    assert ("polybert-sampled", 20_000_000.0) in eval_suites
    assert ("polybert-full", 20_000_000.0) in eval_suites
    assert ("polybert-full", 123.0) in eval_suites
    assert (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").is_file()


def test_full_bundle_finalist_prefers_seed_over_bootstrap_regression(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    params = _params(
        toy_cost_json,
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        noise_estimator="off",
    )
    context = build_compile_context(_toy_pdag(params), params)
    initial_hints = {
        "strategy": "level_preserving",
        "level_drop_penalty": 111.0,
        "allow_seed_fallback": False,
    }
    candidate_code = (
        "def place(context):\n"
        "    return {'strategy': 'level_preserving', 'level_drop_penalty': 999.0, "
        "'allow_seed_fallback': False}\n"
    )

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_seed = hints.get("level_drop_penalty") == 111.0
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if not is_seed else 100.0,
            "bootstrap_count": 60 if not is_seed else 44,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "fallback_selected_groups": 0,
            "candidate_qbp_coverage": 1.0,
            "selected_output_state": {},
            "reserve_summary": {},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)

    selected = oe_backend._run_full_bundle_finalists(
        tmp_path,
        tmp_path / "openevolve_output",
        context,
        candidate_code,
        params,
        initial_hints,
    )

    summary = json.loads((tmp_path / "finalists" / "full_bundle_summary.json").read_text())
    assert selected["level_drop_penalty"] == 111.0
    assert summary["finalist_gate"]["seed_bootstrap_count"] == 44
    assert any(
        item.get("finalist_gate", {}).get("bootstrap_regression") is True
        for item in summary["candidates"]
    )


def test_compile_harness_reuses_existing_output_without_llm(
    toy_cost_json: str,
    tmp_path: Path,
    monkeypatch,
):
    root = tmp_path / "compile_oe_toy"
    best_dir = root / "openevolve_output" / "best"
    best_dir.mkdir(parents=True)
    best_dir.joinpath("best_program.py").write_text(
        "from scripts.optimizer.orbit.openevolve_backend import PlacementBuilder\n"
        "def place(context):\n"
        "    return PlacementBuilder(context).level_preserving(level_drop_penalty=456.0)\n",
        encoding="utf-8",
    )

    def fake_run_evolution(**_kwargs):
        raise AssertionError("reuse mode must not call OpenEvolve")

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 456.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=50,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_search_mode="legacy",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
        openevolve_reuse_output=True,
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 456.0
    assert (root / "finalists" / "full_bundle_summary.json").is_file()
    assert (root / "finalists" / "full_bundle_progress.json").is_file()


def test_noise_estimator_invalid_finalist_falls_back_to_initial_seed(
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

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 123.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    def fake_noise(_context, result, _params):
        return {
            "valid": False,
            "fallback": False,
            "estimated_precision_bits": 1.0,
            "output_margin_bits": -1.0,
            "unsupported_ops": [],
        }

    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    monkeypatch.setattr(oe_backend, "estimate_compile_result_noise", fake_noise)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_search_mode="legacy",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["strategy"] == "level_preserving"
    assert hints.get("level_drop_penalty") != 123.0
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["candidates"][0]["noise_estimator"]["valid"] is False


def test_noise_estimator_off_allows_fast_finalist(
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

    _install_fake_openevolve(monkeypatch, lambda **_kwargs: FakeResult())
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(
        oe_backend,
        "_evaluate_compile_hints",
        lambda _context, hints, *, suppress_output: {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if hints.get("level_drop_penalty") == 123.0 else 100.0,
            "bootstrap_count": 4,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        },
    )
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_output_dir=str(tmp_path),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 123.0


def test_full_bundle_finalist_gate_ignores_legacy_min_bootstrap_count(
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

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_fast_low_bootstrap = hints.get("level_drop_penalty") == 123.0
        is_balanced_seed = hints.get("max_scale_candidates") == 8
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if is_fast_low_bootstrap else 90.0 if is_balanced_seed else 120.0,
            "bootstrap_count": 6 if is_fast_low_bootstrap else 10,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"final_latency_usec": 100.0, "min_bootstrap_count": 10}),
        encoding="utf-8",
    )
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
        openevolve_reference_json=str(reference),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints["level_drop_penalty"] == 123.0
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["finalist_gate"]["forced_bootstrap_floor"] is None
    assert summary["selected_index"] == 0


def test_full_bundle_finalist_gate_honors_explicit_forced_bootstrap_floor(
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

    def fake_evaluate_compile_hints(_context, hints, *, suppress_output):
        is_fast_low_bootstrap = hints.get("level_drop_penalty") == 123.0
        is_balanced_seed = hints.get("max_scale_candidates") == 8
        return {
            "valid": True,
            "validity": 1.0,
            "final_latency_usec": 50.0 if is_fast_low_bootstrap else 90.0 if is_balanced_seed else 120.0,
            "bootstrap_count": 6 if is_fast_low_bootstrap else 10,
            "rescale_count": 2,
            "boundary_quality": 1.0,
            "profile_risk": 0.0,
            "placement_runtime_sec": 0.01,
            "fallback_selected_budgets": 0,
            "selected_output_state": {"out_scl": 40},
            "bootstrap_locations": {},
            "rescale_locations": {},
            "bottleneck_summary": [],
            "diagnostics": {},
            "log_tail": "",
        }

    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps({"final_latency_usec": 100.0, "force_min_bootstrap_count": 10}),
        encoding="utf-8",
    )
    _install_fake_openevolve(monkeypatch, fake_run_evolution)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr(oe_backend, "_evaluate_compile_hints", fake_evaluate_compile_hints)
    params = _params(
        toy_cost_json,
        openevolve_iterations=2,
        openevolve_harness="compile",
        openevolve_eval_suite="polybert-sampled",
        openevolve_finalists=1,
        openevolve_search_mode="legacy",
        openevolve_output_dir=str(tmp_path),
        openevolve_reference_json=str(reference),
        noise_estimator="off",
    )

    hints = run_compile_openevolve(_toy_pdag(params), LatencyEstimator(params), params)

    assert hints.get("level_drop_penalty") != 123.0
    assert hints["strategy"] == "level_preserving"
    assert hints["max_scale_candidates"] == 10
    summary = json.loads(
        (tmp_path / "compile_oe_toy" / "finalists" / "full_bundle_summary.json").read_text()
    )
    assert summary["finalist_gate"]["forced_bootstrap_floor"] == 10
    assert any(
        candidate.get("finalist_gate", {}).get("bootstrap_reject")
        for candidate in summary["candidates"]
        if candidate.get("label") == "candidate_0"
    )


def test_plaintext_quality_gate_rejects_near_constant_reference():
    reject, reason = oe_backend._finalist_plaintext_quality_reject(
        {
            "harness": {
                "reference_json": {
                    "plaintext_sanity": {
                        "records": 10,
                        "near_constant_first2_logits": True,
                        "first2_margin_min": 0.0001,
                    },
                    "min_plaintext_logit_margin": 0.001,
                }
            }
        }
    )
    assert reject is True
    assert reason == "near_constant_first2_logits"


def test_profile_noise_metadata_gate_requires_estimator_profile():
    reject, reason = oe_backend._profile_noise_metadata_reject(
        {"resilience": {"enabled": True, "ckks_noise_model": None}}
    )
    assert reject is True
    assert reason == "missing_ckks_noise_model"
    reject, reason = oe_backend._profile_noise_metadata_reject(
        {
            "resilience": {
                "enabled": True,
                "ckks_noise_model": {"model": "tuneinsight-lattigo-v6", "fallback": False},
            }
        }
    )
    assert reject is False
    assert reason is None


def test_balanced_noise_margin_is_not_tied_to_reference_bootstrap_count(
    toy_cost_json: str,
    tmp_path: Path,
):
    params = _params(toy_cost_json, openevolve_reference_json=str(tmp_path / "reference.json"))
    (tmp_path / "reference.json").write_text(
        json.dumps({"min_bootstrap_count": 11, "force_min_bootstrap_count": 11}),
        encoding="utf-8",
    )
    context = build_compile_context(_toy_pdag(params), params)

    policy = PlacementBuilder(context).balanced_noise_margin()["policy"]

    assert policy["max_scale_candidates"] == 8


def test_ilp_style_reserve_metrics_are_exposed_to_openevolve(toy_cost_json: str):
    params = _params(toy_cost_json)
    graph = _branch_merge_pdag(params)
    le = LatencyEstimator(params)
    io_to_assign, _io_to_cost = solve_budget_batch(
        graph,
        [{"in_lvl": -1, "in_scl": 40}],
        le,
        params,
        PlacementBuilder(build_context(graph, [{"in_lvl": -1, "in_scl": 40}], params))
        .noise_guarded_refresh()
        ["policy"],
    )

    assign = next(iter(next(iter(io_to_assign.values())).values()))
    reserve = oe_backend._assignment_reserve_summary(assign, graph, params)

    assert reserve["min_decryptability_reserve_bits"] is not None
    assert reserve["min_transition_reserve_bits"] is not None
    assert reserve["min_decryptability_reserve_bits"] >= 0
    assert reserve["min_transition_reserve_bits"] >= 0
    assert oe_backend._reserve_quality_score(reserve) >= 0.0


def test_estimator_backed_profile_rejects_trace_noise_warnings(toy_cost_json: str):
    params = _params(toy_cost_json)
    noise = {
        "valid": True,
        "warning_ops": ["trace_precision_below_margin"],
    }

    reject, reason = oe_backend._finalist_noise_warning_reject(
        {
            "resilience": {
                "enabled": True,
                "ckks_noise_model": {"model": "tuneinsight-lattigo-v6", "fallback": False},
            }
        },
        noise,
        params,
    )
    assert reject is True
    assert reason == "trace_precision_below_margin"

    reject, reason = oe_backend._finalist_noise_warning_reject(
        {"resilience": {"enabled": False}},
        noise,
        params,
    )
    assert reject is False
    assert reason is None


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


def test_zero_iteration_worker_uses_safety_portfolio_in_legacy_mode(toy_cost_json: str, monkeypatch):
    params = _params(
        toy_cost_json,
        openevolve_iterations=0,
        openevolve_search_mode="legacy",
    )
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    captured = {}

    def fake_solve_budget_batch(_pdag, _budgets, _le, _params, hints, _diagnostics):
        captured.update(hints)
        return {}, {}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)

    worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    portfolio = captured.get("portfolio")
    assert isinstance(portfolio, list)
    assert {policy.get("min_internal_level") for policy in portfolio} >= {6, 8, 12}
    assert all(policy.get("allow_seed_fallback") is False for policy in portfolio)


def test_zero_iteration_worker_uses_bootstrap_mcts_seed_by_default(toy_cost_json: str, monkeypatch):
    params = _params(toy_cost_json, openevolve_iterations=0)
    graph = _toy_pdag(params)
    worker = OpenEvolvePlacementWorker(params, LatencyEstimator(params))
    captured = {}

    def fake_solve_budget_batch(_pdag, _budgets, _le, _params, hints, _diagnostics):
        captured.update(hints)
        return {}, {}

    monkeypatch.setattr(oe_backend, "solve_budget_batch", fake_solve_budget_batch)

    worker.get_qbp(graph, [{"in_lvl": -1, "in_scl": 40}])

    assert captured["strategy"] == "bootstrap_mcts"
    assert captured["mcts_rollout_budget"] >= 2
    assert captured["mcts_action_cap"] <= 2
    assert captured["boundary_state_cap"] == 1
    assert captured["selection_bootstrap_penalty"] > 0


def test_qbp_manager_openevolve_backend_does_not_import_ilp_solvers(
    toy_cost_json: str,
):
    script = f"""
import sys
from scripts.latency_estimator.latency_estimator import LatencyEstimator
from scripts.optimizer.orbit.qbp_manager import QBPManager
from scripts.params.params import Params

params = Params(
    {toy_cost_json!r},
    "Orbit",
    "compile",
    Sw=40,
    CSw=40,
    threads=1,
    comp=False,
    part=False,
    placement_backend="openevolve",
    openevolve_iterations=0,
)
manager = QBPManager(params, LatencyEstimator(params))
assert manager.ilp_worker.__class__.__name__ == "OpenEvolvePlacementWorker"
assert "scripts.optimizer.orbit.ilp_worker" not in sys.modules
assert "scripts.optimizer.orbit.ilp_core" not in sys.modules
assert "pulp" not in sys.modules
assert not any(name == "gurobipy" or name.startswith("gurobipy.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
