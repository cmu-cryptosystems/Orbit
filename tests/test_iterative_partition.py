from types import SimpleNamespace

from scripts.optimizer.orbit.iterative_partition import _register_partition_result


class _FakeQBPManager:
    def __init__(self):
        self.pdag_name_to_qbp = {}
        self.calls = []

    def add_qbp_existing(self, pdag, io_to_cost, io_to_assign):
        self.calls.append((pdag.name, io_to_cost, io_to_assign))
        self.pdag_name_to_qbp[pdag.name] = (object(), {})


def test_register_partition_result_for_enclosing_bypass_once():
    manager = _FakeQBPManager()
    pdag = SimpleNamespace(name="main_subgraph")
    io_to_assign = {(-1, 40): {(1, 40): "assign"}}
    io_to_cost = {(-1, 40): {(1, 40): 12.0}}

    _register_partition_result(manager, pdag, (io_to_assign, io_to_cost))
    _register_partition_result(manager, pdag, (io_to_assign, io_to_cost))

    assert manager.calls == [("main_subgraph", io_to_cost, io_to_assign)]
    assert "main_subgraph" in manager.pdag_name_to_qbp


def test_register_partition_result_ignores_missing_result():
    manager = _FakeQBPManager()
    pdag = SimpleNamespace(name="main_subgraph")

    _register_partition_result(manager, pdag, None)

    assert manager.calls == []
