"""Qt-based automated and interactive recording scenario for Navilight.

Dependencies:
    pip install pyvistaqt pyqt5

Place this file next to the canonical ``navilight.py`` implementation and run:
    python recorded_demo_qt.py --interval-ms 300 --ticks-per-update 1 --start-room R_A

Why Qt:
    The standard ``pyvista.Plotter.add_timer_event`` path uses the VTK
    interactor loop for both simulation callbacks and camera/key input.
    ``BackgroundPlotter.add_callback`` schedules the scenario through Qt while
    the render window remains responsive to keyboard and mouse interaction.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from pyvistaqt import BackgroundPlotter

import navilight


RouteSignature = Tuple[bool, Optional[str], float, Tuple[str, ...]]


@dataclass(frozen=True)
class PhysicalEvent:
    tick: int
    u: str
    v: str
    blocked: bool
    description: str


class QtRecordedScenario:
    """Drive deterministic edge failures while preserving camera interaction."""

    def __init__(
        self,
        viewer: navilight.InteractiveBuildingViewer,
        controller: navilight.MovementGraphController,
        strategy: navilight.DistributedPathVectorStrategy,
        *,
        ticks_per_update: int,
    ) -> None:
        self.viewer = viewer
        self.controller = controller
        self.strategy = strategy
        self.ticks_per_update = ticks_per_update
        self.frame = 0
        self.paused = False
        self.message = "Stable routing field: both upper-floor stair accesses available"
        self.events = {
            event.tick: event
            for event in (
                PhysicalEvent(16, "R_E", "J0_E", True, ""),
                PhysicalEvent(40, "R_A", "J0_W", True, ""),
                PhysicalEvent(82, "J0_E", "W0_CE", True, ""),
                PhysicalEvent(112, "J0_W", "SW0", True, ""),
                
                # PhysicalEvent(116, "J1_E", "SE1", True, "Block east stair access on upper floor"),
                # PhysicalEvent(140, "J1_W", "SW1", True, "Block west stair access: upper floor isolated"),
                # PhysicalEvent(182, "J1_W", "SW1", False, "Restore west stair access: routes recover"),
                # PhysicalEvent(212, "J1_E", "SE1", False, "Restore east stair access"),
            )
        }

    def _route_signature(self) -> Dict[str, RouteSignature]:
        return {
            state.controlled_node: (
                state.route.reachable,
                state.route.exit_id,
                state.route.cost,
                state.route.path,
            )
            for state in self.strategy.engine.states.values()
        }

    def _select_edge(self, u: str, v: str) -> None:
        wanted = navilight.canonical_edge(u, v)
        for index, edge in enumerate(self.viewer.edge_list):
            if navilight.canonical_edge(*edge) == wanted:
                self.viewer.selected_edge_index = index
                self.viewer._highlight_selected_edge(u, v)
                return

    def _apply_event(self, event: PhysicalEvent) -> bool:
        changed = self.controller.set_edge_blocked(event.u, event.v, event.blocked)
        self._select_edge(event.u, event.v)
        if changed:
            self.strategy.on_edge_status_changed(event.u, event.v)
        self.message = event.description
        print("[event]", event.description)
        return changed

    def _banner(self) -> str:
        start_room = self.viewer.current_start
        route = self.strategy.engine.route_for_node(start_room)
        if route.reachable:
            next_hop = route.path[1] if len(route.path) > 1 else "-"
            route_text = "cost={:.1f}, next={}".format(route.cost, next_hop)
        else:
            route_text = "WITHDRAWN / no known exit-reaching path"
        run_state = "PAUSED" if self.paused else "RUNNING"
        return (
            "AUTOMATED PATH-VECTOR SCENARIO | {} | frame={}\n"
            "{}\n"
            "Start room {}: {}\n"
            "Mouse: rotate/zoom/pan | SPACE: pause | D: input debug"
        ).format(run_state, self.frame, self.message, start_room, route_text)

    def _update_banner(self) -> None:
        self.viewer.plotter.remove_actor("automated_scenario", render=False)
        self.viewer.plotter.add_text(
            self._banner(),
            position="lower_right",
            font_size=10,
            name="automated_scenario",
        )

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        print("[input] space ->", "paused" if self.paused else "running")
        self._update_banner()
        self.viewer.plotter.render()

    def debug_input(self) -> None:
        print("[input] d received by Qt/PyVista interactor at frame", self.frame)

    def callback(self) -> None:
        """Qt periodic callback. Dynamic actors redraw only on visible changes."""
        if self.paused:
            return

        before = self._route_signature()
        event = self.events.get(self.frame)
        edge_changed = False

        if event is not None:
            edge_changed = self._apply_event(event)
        elif self.strategy.engine.pending_work() > 0:
            self.strategy.tick(self.ticks_per_update)

        after = self._route_signature()
        route_changed = before != after

        if edge_changed or route_changed:
            self._update_banner()
            self.viewer.refresh_after_incremental_update()

        self.frame += 1


def _execute_qt_app(plotter: BackgroundPlotter) -> None:
    """Run the Qt event loop across PyQt/PySide API variants."""
    app = plotter.app
    exec_method = getattr(app, "exec", None)
    if exec_method is None:
        exec_method = getattr(app, "exec_", None)
    if exec_method is None:
        raise RuntimeError("Unable to locate Qt application event loop entry point.")
    exec_method()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Qt Navilight recording scenario.")
    parser.add_argument("--interval-ms", type=int, default=300, help="Scenario callback interval in milliseconds.")
    parser.add_argument("--ticks-per-update", type=int, default=1, help="Path-vector ticks per callback while messages are pending.")
    parser.add_argument("--frames", type=int, default=132, help="Number of scenario callback frames.")
    parser.add_argument(
        "--start-room",
        default="R_A",
        help="Room node to highlight and track in the scenario banner (one-floor demo: R_A, R_B, R_C, R_D or R_E).",
    )
    args = parser.parse_args()

    geometry, controller, communication, manager = navilight.build_application(num_floors=1)
    manager.next()
    strategy = manager.current()
    if not isinstance(strategy, navilight.DistributedPathVectorStrategy):
        raise RuntimeError("Expected distributed-path-vector strategy.")

    viewer = navilight.InteractiveBuildingViewer(geometry, controller, manager, communication)

    # The viewer creates a standard blocking pv.Plotter by default. Replace it
    # before building any scene actor with the Qt-backed interactive plotter.
    viewer.plotter.close()
    viewer.plotter = BackgroundPlotter(
        show=True,
        window_size=(1500, 920),
        toolbar=False,
        menu_bar=False,
        editor=False,
        auto_update=False,
        title="Navilight - Automated Path-Vector Scenario",
    )
    viewer.plotter.set_background("white")
    viewer.plotter.enable_trackball_style()

    # A Qt timer may redraw while the user is rotating or zooming the camera.
    # Never restore a saved camera position from those periodic callbacks:
    # dynamic actors already use reset_camera=False, so interaction remains live.
    viewer.preserve_camera_on_refresh = False

    # Geometry selection is intentionally disabled in recording mode. Camera
    # interaction and the built-in button widgets remain available.
    viewer.build_scene(enable_picking=False)
    valid_start_rooms = sorted(viewer.start_nodes)
    if args.start_room not in valid_start_rooms:
        parser.error(
            "invalid --start-room {!r}; valid choices are: {}".format(
                args.start_room,
                ", ".join(valid_start_rooms),
            )
        )
    viewer.current_start = args.start_room
    viewer._highlight_selected_node(viewer.current_start)
    viewer.refresh_path_only()

    scenario = QtRecordedScenario(
        viewer,
        controller,
        strategy,
        ticks_per_update=args.ticks_per_update,
    )
    scenario._update_banner()
    viewer.plotter.add_key_event("space", scenario.toggle_pause)
    viewer.plotter.add_key_event("d", scenario.debug_input)
    viewer.plotter.add_callback(
        scenario.callback,
        interval=args.interval_ms,
        count=args.frames,
    )
    viewer.plotter.render()

    print("Qt recording demo started.")
    print("  Start room:", viewer.current_start)
    print("  Left drag: rotate | Right drag: zoom | Middle drag: pan")
    print("  Space: pause/resume | D: confirm keyboard input in terminal")
    _execute_qt_app(viewer.plotter)


if __name__ == "__main__":
    main()
