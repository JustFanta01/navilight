"""Protocol-level regression tests for Navilight distributed path-vector routing."""
from __future__ import annotations

import sys
import types

try:
    import pyvista  # noqa: F401
except ImportError:
    # Protocol tests do not instantiate the viewer.
    sys.modules["pyvista"] = types.SimpleNamespace()

import navilight


def build_distributed():
    geometry, controller, communication, manager = navilight.build_application(num_floors=2)
    strategy = manager._strategies["distributed-path-vector"]
    strategy.recompute()
    return geometry, controller, manager, strategy


def block_second_floor_stair_access(controller, manager):
    for edge in (("J1_E", "SE1"), ("J1_W", "SW1")):
        assert controller.toggle_edge(*edge)
        manager.notify_edge_changed(*edge)


def test_bootstrap_converges_to_oracle():
    _, _, _, strategy = build_distributed()
    assert strategy.diagnostics.global_summary() == {
        "wrong_nodes": 0.0,
        "max_error": 0.0,
        "unsafe_routes": 0.0,
    }
    path, cost = strategy.get_path("R_F")
    assert path[-1] in {"EXIT_E", "EXIT_W"}
    assert cost < float("inf")


def test_isolated_upper_floor_withdraws_internal_guidance():
    _, controller, manager, strategy = build_distributed()
    block_second_floor_stair_access(controller, manager)
    strategy.settle_until_quiet()
    isolated_nodes = [
        "J1_W", "W1_WC", "J1_C", "W1_CE", "J1_E", "J1_N", "L_1",
        "R_F", "R_G", "R_H", "R_I", "R_J",
    ]
    for node in isolated_nodes:
        assert strategy.get_value(node) == float("inf")
        assert strategy.get_next(node) is None
    # The stair landing itself still has a valid downward route; access to it is blocked.
    assert strategy.get_next("SW1") == "SW0"
    assert strategy.get_next("SE1") == "SE0"
    assert strategy.diagnostics.global_summary()["wrong_nodes"] == 0.0


def test_route_recovers_after_unblocking_one_stair_access():
    _, controller, manager, strategy = build_distributed()
    block_second_floor_stair_access(controller, manager)
    strategy.settle_until_quiet()
    assert controller.toggle_edge("J1_W", "SW1")
    manager.notify_edge_changed("J1_W", "SW1")
    strategy.settle_until_quiet()
    path, _ = strategy.get_path("R_F")
    assert path == ["R_F", "J1_W", "SW1", "SW0", "EXIT_W"]
    assert strategy.diagnostics.global_summary()["wrong_nodes"] == 0.0


def test_stale_edge_observation_cannot_reopen_newer_blocked_link():
    _, controller, _, strategy = build_distributed()
    assert controller.toggle_edge("J1_W", "SW1")
    blocked, version = controller.edge_event("J1_W", "SW1")
    assert blocked is True
    strategy.engine.observe_incident_edge_change("J1_W", "SW1", blocked, version)
    strategy.engine.observe_incident_edge_change("J1_W", "SW1", False, version - 1)
    device = strategy.engine.node_to_device["J1_W"]
    assert strategy.engine.states[device].links["SW1"].blocked is True
