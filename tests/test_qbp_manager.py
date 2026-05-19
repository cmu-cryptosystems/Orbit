from scripts.optimizer.orbit.qbp_manager import _main_qbp_input_keys


class _FakeQBP:
    def __init__(self, costs):
        self._costs = costs

    def get_i_costs(self, key):
        return self._costs.get(key)

    def get_all_costs(self):
        return self._costs


def test_main_qbp_input_keys_expands_wildcard_level_by_scale():
    qbp = _FakeQBP(
        {
            (1, 40): {(3, 51): 1.0},
            (2, 40): {(3, 51): 2.0},
            (3, 50): {(3, 51): 3.0},
        }
    )

    assert _main_qbp_input_keys(qbp, -1, 40) == [(1, 40), (2, 40)]


def test_main_qbp_input_keys_keeps_concrete_level():
    qbp = _FakeQBP({(2, 40): {(3, 51): 2.0}})

    assert _main_qbp_input_keys(qbp, 2, 40) == [(2, 40)]
    assert _main_qbp_input_keys(qbp, 3, 40) == []
