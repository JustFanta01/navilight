"""Qt automated recording scenarios for Navilight.

The standard PyVista timer shares the VTK interactor loop with camera and key
input.  This helper uses ``pyvistaqt.BackgroundPlotter`` so the deterministic
scenario can advance while the 3D window remains interactive.

Run next to ``navilight.py``:

    python recorded_demo_qt.py --preset slow-one-floor
    python recorded_demo_qt.py --preset normal-dynamic
    python recorded_demo_qt.py --preset fast-two-floor

The presets are meant for recordings:

- ``slow-one-floor`` shows visible path-vector gossip propagation, one protocol
  tick per visual update.
- ``normal-dynamic`` shows a two-floor evacuation path adapting to blocked
  staircase/exit access while a valid alternative remains available.
- ``fast-two-floor`` is the stress/limit case for second-floor isolation
  detection: both upper-floor stair accesses are blocked, so internal guidance
  withdraws until one access recovers.

Why Qt:
    The standard ``pyvista.Plotter.add_timer_event`` path uses the VTK
    interactor loop for both simulation callbacks and camera/key input.
    ``BackgroundPlotter.add_callback`` schedules the scenario through Qt while
    the render window remains responsive to keyboard and mouse interaction.

"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Tuple

from pyvistaqt import BackgroundPlotter

import navilight


RouteSignature = Tuple[bool, Optional[str], float, Tuple[str, ...]]


# ============================================================
# SCENARIO MODEL
# ============================================================

@dataclass(frozen=True)
class PhysicalEvent:
    """Scheduled physical movement-edge event for the recording scenario."""

    tick: int
    u: str
    v: str
    blocked: bool
    description: str


@dataclass(frozen=True)
class ScenarioPreset:
    """Complete configuration for one deterministic recording preset."""

    name: str
    title: str
    floors: int
    start_room: str
    interval_ms: int
    ticks_per_update: int
    frames: int
    start_delay_frames: int
    initial_message: str
    events: Tuple[PhysicalEvent, ...]


PRESETS: Dict[str, ScenarioPreset] = {
    "slow-one-floor": ScenarioPreset(
        name="slow-one-floor",
        title="Slow one-floor gossip propagation",
        floors=1,
        start_room="R_A",
        interval_ms=450,
        ticks_per_update=1,
        frames=160,
        start_delay_frames=0,
        initial_message="One-floor baseline: slow path-vector propagation is visible tick by tick",
        events=(
            PhysicalEvent(16, "R_E", "J0_E", True, "Block Room E access: local route withdraws from that branch"),
            PhysicalEvent(42, "R_A", "J0_W", True, "Block Room A access: selected start becomes unreachable"),
            PhysicalEvent(78, "R_A", "J0_W", False, "Restore Room A access: route information propagates back"),
            PhysicalEvent(112, "J0_E", "W0_CE", True, "Block east corridor segment: routes shift toward the west exit"),
            PhysicalEvent(138, "J0_E", "W0_CE", False, "Restore east corridor segment: shortest routes recover"),
        ),
    ),
    "normal-dynamic": ScenarioPreset(
        name="normal-dynamic",
        title="Normal two-floor dynamic evacuation",
        floors=2,
        start_room="R_J",
        interval_ms=240,
        ticks_per_update=8,
        frames=230,
        start_delay_frames=0,
        initial_message="Two-floor baseline: upper east room initially exits through the east stair",
        events=(
            PhysicalEvent(
                18,
                "R_D",
                "J0_E",
                True,
                "Fire 1 starts near first-floor Room D: local east-side room access is blocked",
            ),
            PhysicalEvent(
                36,
                "J0_E",
                "W0_CE",
                True,
                "Fire 1 spreads into the first-floor east corridor: east stair can no longer cross to the west corridor",
            ),
            PhysicalEvent(
                58,
                "SE0",
                "EXIT_E",
                True,
                "East exit is closed by smoke: upper-floor traffic must abandon the east-stair exit path",
            ),
            PhysicalEvent(
                82,
                "J1_E",
                "W1_CE",
                True,
                "Fire 2 affects the upper east corridor: direct east-to-centre movement is blocked",
            ),
            PhysicalEvent(
                104,
                "J1_C",
                "J1_N",
                True,
                "Fire 2 reaches the upper central connector: guidance avoids the central area through the north bypass",
            ),
            PhysicalEvent(
                128,
                "R_I",
                "J1_E",
                True,
                "Upper east-side smoke spreads near Room I: local branch is withdrawn while the bypass remains valid",
            ),
            PhysicalEvent(
                154,
                "R_E",
                "J0_E",
                True,
                "Fire 1 spreads near first-floor Room E: more local guidance is withdrawn on the east side",
            ),
            PhysicalEvent(
                184,
                "J1_C",
                "W1_CE",
                True,
                "Upper central corridor is fully unusable: routes keep using the north-west detour to the west stair",
            ),
        ),
    ),
    "fast-two-floor": ScenarioPreset(
        name="fast-two-floor",
        title="Fast two-floor isolation limit test",
        floors=2,
        start_room="R_F",
        interval_ms=220,
        ticks_per_update=12,
        frames=170,
        start_delay_frames=0,
        initial_message="Two-floor limit test: fast gossip, both stairs initially reachable",
        events=(
            PhysicalEvent(18, "J1_E", "SE1", True, "Block upper east stair access: west stair remains valid"),
            PhysicalEvent(50, "J1_W", "SW1", True, "Block upper west stair access too: second floor becomes isolated"),
            PhysicalEvent(96, "J1_W", "SW1", False, "Restore west stair access: second-floor guidance recovers quickly"),
            PhysicalEvent(130, "J1_E", "SE1", False, "Restore east stair access: both staircase options are back"),
        ),
    ),
}


# ============================================================
# QT SCENARIO DRIVER
# ============================================================

class QtRecordedScenario:
    """Drive deterministic edge events while preserving camera interaction."""

    def __init__(
        self,
        viewer: navilight.InteractiveBuildingViewer,
        controller: navilight.MovementGraphController,
        strategy: navilight.DistributedPathVectorStrategy,
        preset: ScenarioPreset,
        *,
        initial_paused: bool = False,
    ) -> None:
        self.viewer = viewer
        self.controller = controller
        self.strategy = strategy
        self.preset = preset
        self.frame = 0
        self.paused = initial_paused
        self.message = preset.initial_message
        self.events = {preset.start_delay_frames + event.tick: event for event in preset.events}

    def _route_signature(self) -> Dict[str, RouteSignature]:
        """Capture local route states to detect visible protocol changes."""
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
        """Highlight the edge currently affected by the scenario."""
        wanted = navilight.canonical_edge(u, v)
        for index, edge in enumerate(self.viewer.edge_list):
            if navilight.canonical_edge(*edge) == wanted:
                self.viewer.selected_edge_index = index
                self.viewer._highlight_selected_edge(u, v)
                return

    def _apply_event(self, event: PhysicalEvent) -> bool:
        """Apply one scheduled blockage/recovery event to the movement graph."""
        changed = self.controller.set_edge_blocked(event.u, event.v, event.blocked)
        self._select_edge(event.u, event.v)
        if changed:
            self.strategy.on_edge_status_changed(event.u, event.v)
        self.message = event.description
        print("[event]", event.description)
        return changed

    def _next_event_text(self) -> str:
        """Describe the next scheduled event relative to the current frame."""
        future_ticks = [tick for tick in self.events if tick > self.frame]
        if not future_ticks:
            return "no more scheduled physical events"
        tick = min(future_ticks)
        event = self.events[tick]
        wait = max(0, tick - self.frame)
        action = "block" if event.blocked else "restore"
        return f"next event in {wait} frame(s): {action} {event.u}-{event.v}"

    def _banner(self) -> str:
        """Build the lower-right recording status overlay."""
        start_room = self.viewer.current_start
        route = self.strategy.engine.route_for_node(start_room)
        if route.reachable:
            next_hop = route.path[1] if len(route.path) > 1 else "-"
            route_text = "cost={:.1f}, next={}".format(route.cost, next_hop)
        else:
            route_text = "WITHDRAWN / no known exit-reaching path"

        run_state = "PAUSED" if self.paused else "RUNNING"
        pending = self.strategy.engine.pending_work()
        return (
            "{} | {} | frame={} | pending={}\n"
            "{}\n"
            "{}\n"
            "Start room {}: {}\n"
            "Gossip: {} tick(s)/update, {} ms/update | Mouse: rotate/zoom/pan | SPACE: pause/start"
        ).format(
            self.preset.title,
            run_state,
            self.frame,
            pending,
            self.message,
            self._next_event_text(),
            start_room,
            route_text,
            self.preset.ticks_per_update,
            self.preset.interval_ms,
        )

    def _update_banner(self) -> None:
        """Refresh the recording status overlay without rebuilding the scene."""
        self.viewer.plotter.remove_actor("automated_scenario", render=False)
        self.viewer.plotter.add_text(
            self._banner(),
            position="lower_right",
            font_size=10,
            name="automated_scenario",
        )

    def toggle_pause(self) -> None:
        """Pause or resume the automated Qt scenario."""
        self.paused = not self.paused
        print("[input] space ->", "paused" if self.paused else "running")
        self._update_banner()
        self.viewer.plotter.render()

    def debug_input(self) -> None:
        """Confirm that keyboard events reach the Qt/PyVista interactor."""
        print("[input] d received by Qt/PyVista interactor at frame", self.frame)

    def callback(self) -> None:
        """Qt periodic callback. Redraw only when the displayed state changes."""
        if self.paused:
            return
        if self.frame >= self.preset.start_delay_frames + self.preset.frames:
            return

        before = self._route_signature()
        event = self.events.get(self.frame)
        edge_changed = False

        if event is not None:
            edge_changed = self._apply_event(event)
        elif self.strategy.engine.pending_work() > 0:
            self.strategy.tick(self.preset.ticks_per_update)

        after = self._route_signature()
        route_changed = before != after

        if edge_changed or route_changed:
            self._update_banner()
            self.viewer.refresh_after_incremental_update()

        self.frame += 1


# ============================================================
# CLI AND APPLICATION BOOTSTRAP
# ============================================================

def _execute_qt_app(plotter: BackgroundPlotter) -> None:
    """Run the Qt event loop across PyQt/PySide API variants."""
    app = plotter.app
    exec_method = getattr(app, "exec", None)
    if exec_method is None:
        exec_method = getattr(app, "exec_", None)
    if exec_method is None:
        raise RuntimeError("Unable to locate Qt application event loop entry point.")
    exec_method()


def _override_preset(args: argparse.Namespace) -> ScenarioPreset:
    """Start from a preset and apply explicit command-line overrides."""
    preset = PRESETS[args.preset]
    overrides = {
        "floors": args.floors,
        "start_room": args.start_room,
        "interval_ms": args.interval_ms,
        "ticks_per_update": args.ticks_per_update,
        "frames": args.frames,
        "start_delay_frames": args.start_delay_frames,
    }
    clean_overrides = {key: value for key, value in overrides.items() if value is not None}
    return replace(preset, **clean_overrides)


def _print_preset_summary(preset: ScenarioPreset) -> None:
    """Print a compact textual summary before the Qt window starts."""
    print("Qt recording demo started.")
    print("  Preset:", preset.name)
    print("  Floors:", preset.floors)
    print("  Start room:", preset.start_room)
    print("  Interval:", preset.interval_ms, "ms")
    print("  Gossip ticks/update:", preset.ticks_per_update)
    print("  Scenario frames:", preset.frames)
    print("  Start delay frames:", preset.start_delay_frames)
    print("  Total callback frames:", preset.frames + preset.start_delay_frames)
    print("  Events:")
    for event in preset.events:
        action = "block" if event.blocked else "restore"
        scheduled_tick = preset.start_delay_frames + event.tick
        print(f"    frame {scheduled_tick:>3}: {action} {event.u}-{event.v} | {event.description}")
    print("  Left drag: rotate | Right drag: zoom | Middle drag: pan")
    print("  Space: pause/resume/start | D: confirm keyboard input in terminal")


def _build_parser() -> argparse.ArgumentParser:
    """Create the scenario CLI."""
    parser = argparse.ArgumentParser(description="Interactive Qt Navilight recording scenarios.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="normal-dynamic",
        help="Recording preset. Defaults to the normal two-floor dynamic evacuation scenario.",
    )
    parser.add_argument("--floors", type=int, help="Override number of demo floors from the selected preset.")
    parser.add_argument("--interval-ms", type=int, help="Override callback interval in milliseconds.")
    parser.add_argument("--ticks-per-update", type=int, help="Override path-vector ticks per callback while messages are pending.")
    parser.add_argument("--frames", type=int, help="Override number of scenario callback frames after the camera setup delay.")
    parser.add_argument(
        "--start-delay-frames",
        type=int,
        help="Add idle callback frames before the first physical event, so the camera can be adjusted.",
    )
    parser.add_argument(
        "--manual-start",
        action="store_true",
        help="Start paused at frame 0. Set the camera, then press Space to begin the scenario.",
    )
    parser.add_argument("--start-room", help="Override highlighted room, for example R_A, R_E, R_F or R_J.")
    return parser


def main() -> None:
    """Parse CLI options, build the Qt viewer and run the selected scenario."""
    args = _build_parser().parse_args()
    preset = _override_preset(args)

    geometry, controller, communication, manager = navilight.build_application(num_floors=preset.floors)
    manager.next()  # centralized-bellman-oracle -> distributed-path-vector
    strategy = manager.current()
    if not isinstance(strategy, navilight.DistributedPathVectorStrategy):
        raise RuntimeError("Expected distributed-path-vector strategy.")

    viewer = navilight.InteractiveBuildingViewer(geometry, controller, manager, communication)

    # The viewer creates a standard blocking pv.Plotter by default. Replace it
    # before building scene actors with the Qt-backed interactive plotter.
    viewer.plotter.close()
    viewer.plotter = BackgroundPlotter(
        show=True,
        window_size=(1500, 920),
        toolbar=False,
        menu_bar=False,
        editor=False,
        auto_update=False,
        title=f"Navilight - {preset.title}",
    )
    viewer.plotter.set_background("white")
    viewer.plotter.enable_trackball_style()

    # Timer callbacks may redraw while the user is rotating or zooming. Dynamic
    # actors already use reset_camera=False, so do not restore a cached camera
    # pose from periodic updates.
    viewer.preserve_camera_on_refresh = False

    # Geometry selection is disabled in recording mode. Camera interaction and
    # the built-in button widgets remain available.
    viewer.build_scene(enable_picking=False)
    valid_start_rooms = sorted(viewer.start_nodes)
    if preset.start_room not in valid_start_rooms:
        raise SystemExit(
            "invalid --start-room {!r}; valid choices for {} floor(s): {}".format(
                preset.start_room,
                preset.floors,
                ", ".join(valid_start_rooms),
            )
        )
    viewer.current_start = preset.start_room
    viewer._highlight_selected_node(viewer.current_start)
    viewer.refresh_path_only()

    scenario = QtRecordedScenario(viewer, controller, strategy, preset, initial_paused=args.manual_start)
    scenario._update_banner()
    viewer.plotter.add_key_event("space", scenario.toggle_pause)
    viewer.plotter.add_key_event("d", scenario.debug_input)
    # Do not pass a finite ``count``. The callback itself stops advancing after
    # the configured duration, while the Qt timer remains alive if the user
    # pauses the scenario to adjust the camera.
    viewer.plotter.add_callback(
        scenario.callback,
        interval=preset.interval_ms,
    )
    viewer.plotter.render()

    _print_preset_summary(preset)
    _execute_qt_app(viewer.plotter)


if __name__ == "__main__":
    main()
