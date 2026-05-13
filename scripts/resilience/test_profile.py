import json
import tempfile
from pathlib import Path

from scripts.params.params import Params
from scripts.resilience import ResilienceProfile
from scripts.tdag.check_tdag import check_tdag
from scripts.tdag.tdag import Tdag


def _write_json(data: dict) -> Path:
    handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    with handle:
        json.dump(data, handle)
    return Path(handle.name)


def test_orbit_constraints_match_node_and_comment():
    path = _write_json(
        {
            "constraints": [
                {
                    "match": {"node": "17"},
                    "min_scale": 30,
                    "ports": ["in", "out"],
                },
                {
                    "match": {"comment_contains": "encoder.layer.1"},
                    "min_scale": 34,
                },
            ]
        }
    )
    profile = ResilienceProfile.load(path)

    assert profile.scale_lower_bound("17", {"comment": ""}, 40, "out") == 30
    assert profile.scale_lower_bound(
        "18",
        {"comment": "// encoder.layer.1"},
        40,
        "in",
    ) == 34
    assert profile.scale_lower_bound("19", {"comment": ""}, 40, "out") == 40


def test_ckks_profiler_summary_converts_to_comment_matcher():
    path = _write_json(
        {
            "schema_version": "0.5.0",
            "model": {"name": "bert-base-uncased"},
            "layers": [
                {
                    "target_id": "edge:encoder.layer.0:out",
                    "name": "encoder.layer.0",
                    "module_names": ["encoder.layer.0"],
                    "ckks_hint": {"min_log2_scale": 27},
                    "tolerance_profile": {
                        "tau_abs": 0.01,
                        "fragility_class": "resilient",
                    },
                }
            ],
        }
    )
    profile = ResilienceProfile.load(path)

    assert profile.schema_version == "0.5.0"
    assert profile.model_name == "bert-base-uncased"
    assert profile.scale_lower_bound(
        "42",
        {"comment": "// block encoder.layer.0 output"},
        40,
        "out",
    ) == 27


def test_orbit_constraints_match_rotom_scope_metadata():
    path = _write_json(
        {
            "schema_version": "orbit-resilience-constraints-v0",
            "constraints": [
                {
                    "target_id": "edge:bert.encoder.layer.0.attention.self.query:out",
                    "match": {
                        "comment_regex": (
                            r"(?:^|[;\s])scope="
                            r"bert\.encoder\.layer\.0\.attention\.self\.query"
                            r"(?:$|[;\s])"
                        )
                    },
                    "min_scale": 29,
                    "ports": ["in", "out"],
                }
            ],
        }
    )
    profile = ResilienceProfile.load(path)

    assert (
        profile.scale_lower_bound(
            "101",
            {
                "comment": (
                    "scope=bert.encoder.layer.0.attention.self.query;"
                    "op=linear;layer=0"
                )
            },
            40,
            "out",
        )
        == 29
    )


def test_check_tdag_uses_resilience_local_scale_bound():
    path = _write_json(
        {
            "constraints": [
                {
                    "match": {"node": "x"},
                    "min_scale": 28,
                    "ports": ["out"],
                }
            ]
        }
    )
    params = Params(
        "cost_models/profiled_LATTIGONEW_CPU64k_3_16.json",
        "Orbit",
        mode="compile",
        Sw=40,
        resilience_profile=str(path),
    )
    tdag = Tdag(params, "local_scale_test")
    tdag.add_node("x", op="input", level=1, scale=28, weight=1, op_descr={}, comment="")
    tdag.inputs.add("x")
    tdag.outputs.add("x")

    assert check_tdag(tdag)


def test_lattigo_cost_model_uses_polynomial_degree_key():
    params = Params(
        "cost_models/profiled_LATTIGONEW_CPU64k_3_16.json",
        "Orbit",
        mode="compile",
        Sw=40,
    )

    assert params.poly_deg == 131072
    assert params.max_slot == 65536


if __name__ == "__main__":
    test_orbit_constraints_match_node_and_comment()
    test_ckks_profiler_summary_converts_to_comment_matcher()
    test_orbit_constraints_match_rotom_scope_metadata()
    test_check_tdag_uses_resilience_local_scale_bound()
    test_lattigo_cost_model_uses_polynomial_degree_key()
    print("PASS resilience profile tests")
