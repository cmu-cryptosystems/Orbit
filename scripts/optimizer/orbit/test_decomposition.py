from scripts.optimizer.orbit.iterative_partition import (
    BoundaryState,
    _error_bucket,
    _prune_boundary_states,
)
from scripts.params.params import Params


def _params(**kwargs):
    return Params(
        "cost_models/profiled_LATTIGONEW_CPU64k_3_16.json",
        "Orbit",
        mode="compile",
        Sw=40,
        resilience_mode="error-state",
        resilience_decomposition="bounded-dp",
        **kwargs,
    )


def test_bounded_dp_state_pruning_keeps_lower_cost_and_error():
    params = _params(resilience_max_boundary_states=8, resilience_error_buckets=8)
    states = [
        BoundaryState(level=12, scale=40, error_abs=1e-5, cost=10.0),
        BoundaryState(level=12, scale=40, error_abs=1e-4, cost=11.0),
        BoundaryState(level=11, scale=40, error_abs=1e-6, cost=12.0),
    ]

    kept, pruned = _prune_boundary_states(states, params)

    assert pruned >= 1
    assert any(state.level == 12 and state.error_abs == 1e-5 for state in kept)
    assert not any(state.level == 12 and state.error_abs == 1e-4 for state in kept)


def test_bounded_dp_pruning_caps_boundary_states():
    params = _params(resilience_max_boundary_states=2, resilience_error_buckets=8)
    states = [
        BoundaryState(level=level, scale=40, error_abs=1e-6 * level, cost=float(level))
        for level in range(1, 6)
    ]

    kept, _ = _prune_boundary_states(states, params)

    assert len(kept) == 2
    assert [state.level for state in kept] == [1, 2]


def test_error_buckets_are_ordered_and_capped():
    params = _params(resilience_error_buckets=4)

    low = _error_bucket(1e-9, params)
    high = _error_bucket(params.resilience_error_model["max_error_abs"] * 2, params)

    assert low <= high
    assert high == 3


def test_params_enable_bounded_resilience_decomposition_only_for_error_state():
    params = _params()
    assert params.use_bounded_resilience_decomposition() is False

    params.resilience_profile = type("Profile", (), {"constraints": [object()]})()
    assert params.use_bounded_resilience_decomposition() is True

    params.resilience_mode = "waterline"
    assert params.use_bounded_resilience_decomposition() is False


if __name__ == "__main__":
    test_bounded_dp_state_pruning_keeps_lower_cost_and_error()
    test_bounded_dp_pruning_caps_boundary_states()
    test_error_buckets_are_ordered_and_capped()
    test_params_enable_bounded_resilience_decomposition_only_for_error_state()
    print("PASS bounded decomposition tests")
