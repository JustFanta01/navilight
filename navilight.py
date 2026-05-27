r"""
============================================================
NAVILIGHT — DISTRIBUTED ADAPTIVE EVACUATION GUIDANCE POC
============================================================

Navilight models guidance devices installed at discrete routing waypoints in a
building.  Each device selects the direction it displays using only:

- the state of movement links incident to its controlled waypoint;
- route advertisements received from devices at adjacent waypoints.

The distributed strategy is a local path-vector protocol.  It deliberately
avoids a replicated global value table inside each device.  A centralized
shortest-path computation is retained only as an observer-side diagnostic
oracle for the UI; it is never used to decide actuator arrows.

Model layers:

- Movement graph G_M = (V, A): physical routes between routing waypoints.
- Communication graph G_C = (D, L): radio/data connectivity between devices.
- Route agent at each controlled waypoint: local route selection and gossip.
- Viewer: interaction, rendering and optional oracle comparison.

A corridor light should be represented by adding a routing waypoint at its
physical installation location and associating a guidance device to that
waypoint.  A movement node is therefore a routing state, not necessarily a
semantic room or junction.
============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

try:
    import pyvista as pv
except ImportError as exc:
    raise SystemExit("Install with: pip install pyvista networkx numpy") from exc


Vec3 = Tuple[float, float, float]
EdgeId = Tuple[str, str]
INF = float("inf")


def canonical_edge(u: str, v: str) -> EdgeId:
    return tuple(sorted((u, v)))


def finite_text(value: float, digits: int = 1) -> str:
    return "inf" if np.isinf(value) else f"{value:.{digits}f}"


# ============================================================
# PHYSICAL MODEL
# ============================================================

@dataclass
class Space:
    name: str
    kind: str
    center: Vec3
    size: Vec3
    color: str = "lightgray"
    opacity: float = 0.3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuidanceDevice:
    """Physical indicator/controller associated with one routing waypoint.

    ``controlled_node`` is explicit.  There is no nearest-node approximation:
    if a light is mounted halfway along a corridor, the movement graph must
    contain a waypoint at that position.
    """

    name: str
    controlled_node: str
    position: Vec3
    communication_radius: float = 15.0
    display: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class BuildingGeometry:
    """Physical building geometry, movement graph and installed devices."""

    def __init__(self) -> None:
        self.spaces: Dict[str, Space] = {}
        self.devices: Dict[str, GuidanceDevice] = {}
        self.movement_graph = nx.Graph()

    def add_space(
        self,
        name: str,
        kind: str,
        center: Vec3,
        size: Vec3,
        color: str,
        opacity: float,
        **metadata: Any,
    ) -> None:
        self.spaces[name] = Space(name, kind, center, size, color, opacity, metadata)

    def add_movement_node(
        self,
        node_id: str,
        kind: str,
        position: Vec3,
        label: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        self.movement_graph.add_node(
            node_id,
            kind=kind,
            position=np.array(position, dtype=float),
            label=label or node_id,
            value=INF,
            next=None,
            **attrs,
        )

    def add_movement_edge(self, u: str, v: str, weight: Optional[float] = None, **attrs: Any) -> None:
        p0 = self.movement_graph.nodes[u]["position"]
        p1 = self.movement_graph.nodes[v]["position"]
        base_weight = float(np.linalg.norm(p1 - p0)) if weight is None else float(weight)
        self.movement_graph.add_edge(
            u,
            v,
            base_weight=base_weight,
            weight=base_weight,
            blocked=False,
            version=0,
            **attrs,
        )

    def add_device(
        self,
        name: str,
        controlled_node: str,
        *,
        position: Optional[Vec3] = None,
        communication_radius: float = 15.0,
        display: bool = True,
        **metadata: Any,
    ) -> None:
        if controlled_node not in self.movement_graph:
            raise KeyError(f"Unknown controlled movement node: {controlled_node}")
        if position is None:
            p = self.movement_graph.nodes[controlled_node]["position"] + np.array([0.0, 0.0, 0.9])
            position = tuple(float(x) for x in p)
        self.devices[name] = GuidanceDevice(
            name=name,
            controlled_node=controlled_node,
            position=position,
            communication_radius=communication_radius,
            display=display,
            metadata=metadata,
        )

    def deploy_device_at_every_routing_node(self, communication_radius: float = 15.0) -> None:
        """Install one route agent/indicator at each routing state.

        This is a proof-of-concept deployment assumption: every decision point
        represented in G_M is instrumented.  Devices do not need global state;
        they only control their own node and communicate with nearby devices.
        """
        self.devices.clear()
        for node, attrs in self.movement_graph.nodes(data=True):
            self.add_device(
                f"ACT_{node}",
                node,
                communication_radius=communication_radius,
                level=attrs.get("level"),
                kind=attrs.get("kind"),
            )

    @classmethod
    def demo_building(cls, num_floors: int = 2) -> "BuildingGeometry":
        if num_floors < 1:
            raise ValueError("num_floors must be >= 1")

        b = cls()
        floor_height = 6.0
        base_z = 1.5
        zs = [base_z + i * floor_height for i in range(num_floors)]

        for level, z in enumerate(zs):
            b.add_space(f"Floor_{level}", "floor", (0, 0, z - 1.5), (44, 36, 0.3), "silver", 1.0, level=level)
            b.add_space(f"MainCorridor_{level}", "corridor", (0, 0, z), (30, 4, 3), "lightgreen", 0.23, level=level)
            b.add_space(f"NorthCorridor_{level}", "corridor", (0, 10, z), (30, 4, 3), "palegreen", 0.20, level=level)
            b.add_space(f"ConnectorCorridor_{level}", "corridor", (0, 5, z), (4, 6, 3), "mediumseagreen", 0.18, level=level)
            rooms = [("A", (-11, 5, z)), ("B", (-11, -5, z)), ("C", (0, 15, z)), ("D", (11, -5, z)), ("E", (11, 5, z))]
            for i, (_, pos) in enumerate(rooms):
                letter = chr(ord("A") + level * 5 + i)
                b.add_space(f"Room_{letter}", "room", pos, (7, 6, 3), "lightblue", 0.35, level=level)
            b.add_space(f"Lab_{level}", "lab", (0, -8, z), (15, 12, 3), "lightblue", 0.35, level=level)
            b.add_space(f"EastStairVol_{level}", "stair", (19, 0, z), (4, 5, 3), "plum", 0.23, level=level)
            b.add_space(f"WestStairVol_{level}", "stair", (-19, 0, z), (4, 5, 3), "plum", 0.23, level=level)

            letters = [chr(ord("A") + level * 5 + i) for i in range(5)]
            b.add_movement_node(f"R_{letters[0]}", "room", (-11, 5, z), label=f"Room {letters[0]}", level=level)
            b.add_movement_node(f"R_{letters[1]}", "room", (-11, -5, z), label=f"Room {letters[1]}", level=level)
            b.add_movement_node(f"R_{letters[2]}", "room", (0, 15, z), label=f"Room {letters[2]}", level=level)
            b.add_movement_node(f"R_{letters[3]}", "room", (11, -5, z), label=f"Room {letters[3]}", level=level)
            b.add_movement_node(f"R_{letters[4]}", "room", (11, 5, z), label=f"Room {letters[4]}", level=level)
            b.add_movement_node(f"L_{level}", "lab", (0, -8, z), label=f"Lab {level}", level=level)
            b.add_movement_node(f"J{level}_W", "junction", (-11, 0, z), label=f"J{level}W", level=level)
            b.add_movement_node(f"J{level}_C", "junction", (0, 0, z), label=f"J{level}C", level=level)
            b.add_movement_node(f"J{level}_E", "junction", (11, 0, z), label=f"J{level}E", level=level)
            b.add_movement_node(f"J{level}_N", "junction", (0, 10, z), label=f"J{level}N", level=level)
            b.add_movement_node(f"J{level}_CN1", "junction", (-11, 10, z), label=f"J{level}CN1", level=level)
            b.add_movement_node(f"J{level}_CN2", "junction", (11, 10, z), label=f"J{level}CN2", level=level)
            b.add_movement_node(f"W{level}_WC", "waypoint", (-5.5, 0, z), label=f"W{level}WC", level=level)
            b.add_movement_node(f"W{level}_CE", "waypoint", (5.5, 0, z), label=f"W{level}CE", level=level)
            b.add_movement_node(f"SE{level}", "stair", (19, 0, z), label=f"East stair {level}", level=level)
            b.add_movement_node(f"SW{level}", "stair", (-19, 0, z), label=f"West stair {level}", level=level)
            if level == 0:
                b.add_movement_node("EXIT_E", "exit", (23, 0, z), label="East exit", level=level)
                b.add_movement_node("EXIT_W", "exit", (-23, 0, z), label="West exit", level=level)

        for level in range(num_floors):
            letters = [chr(ord("A") + level * 5 + i) for i in range(5)]
            b.add_movement_edge(f"R_{letters[0]}", f"J{level}_W")
            b.add_movement_edge(f"R_{letters[1]}", f"J{level}_W")
            b.add_movement_edge(f"R_{letters[2]}", f"J{level}_N")
            b.add_movement_edge(f"R_{letters[3]}", f"J{level}_E")
            b.add_movement_edge(f"R_{letters[4]}", f"J{level}_E")
            b.add_movement_edge(f"R_{letters[1]}", f"L_{level}")
            b.add_movement_edge(f"R_{letters[3]}", f"L_{level}")
            # Explicit waypoints model corridor-mounted direction indicators.
            b.add_movement_edge(f"J{level}_W", f"W{level}_WC")
            b.add_movement_edge(f"W{level}_WC", f"J{level}_C")
            b.add_movement_edge(f"J{level}_C", f"W{level}_CE")
            b.add_movement_edge(f"W{level}_CE", f"J{level}_E")
            b.add_movement_edge(f"J{level}_C", f"J{level}_N")
            b.add_movement_edge(f"J{level}_W", f"SW{level}")
            b.add_movement_edge(f"J{level}_E", f"SE{level}")
            b.add_movement_edge(f"J{level}_CN1", f"J{level}_N", weight=13.0)
            b.add_movement_edge(f"J{level}_CN2", f"J{level}_N", weight=13.0)
            b.add_movement_edge(f"J{level}_CN1", f"R_{letters[0]}", weight=13.0)
            b.add_movement_edge(f"J{level}_CN2", f"R_{letters[4]}", weight=13.0)

        b.add_movement_edge("SE0", "EXIT_E")
        b.add_movement_edge("SW0", "EXIT_W")
        for level in range(num_floors - 1):
            b.add_movement_edge(f"SE{level}", f"SE{level + 1}", weight=8.0)
            b.add_movement_edge(f"SW{level}", f"SW{level + 1}", weight=8.0)

        b.deploy_device_at_every_routing_node(communication_radius=15.0)
        return b


class MovementGraphController:
    """Mutation boundary for the physical movement topology."""

    def __init__(self, movement_graph: nx.Graph) -> None:
        self.G = movement_graph

    def set_edge_blocked(self, u: str, v: str, blocked: bool = True) -> bool:
        edge = self.G[u][v]
        if bool(edge.get("blocked", False)) == blocked:
            return False
        edge["blocked"] = blocked
        edge["version"] = int(edge.get("version", 0)) + 1
        return True

    def toggle_edge(self, u: str, v: str) -> bool:
        return self.set_edge_blocked(u, v, not bool(self.G[u][v].get("blocked", False)))

    def reset_edges(self) -> List[EdgeId]:
        changed: List[EdgeId] = []
        for u, v in self.G.edges:
            if self.set_edge_blocked(u, v, False):
                changed.append(canonical_edge(u, v))
        return changed

    def edge_event(self, u: str, v: str) -> Tuple[bool, int]:
        edge = self.G[u][v]
        return bool(edge.get("blocked", False)), int(edge.get("version", 0))

    def shortest_path_to_nearest_exit(self, start: str) -> Tuple[List[str], float]:
        usable = nx.Graph()
        usable.add_nodes_from(self.G.nodes)
        for u, v, attrs in self.G.edges(data=True):
            if not attrs.get("blocked", False):
                usable.add_edge(u, v, weight=float(attrs["base_weight"]))
        exits = [n for n, attrs in self.G.nodes(data=True) if attrs["kind"] == "exit"]
        best_path: Optional[List[str]] = None
        best_cost = INF
        for exit_node in exits:
            try:
                path = nx.shortest_path(usable, start, exit_node, weight="weight")
                cost = float(nx.path_weight(usable, path, weight="weight"))
                if cost < best_cost:
                    best_path, best_cost = path, cost
            except nx.NetworkXNoPath:
                pass
        if best_path is None:
            raise nx.NetworkXNoPath(f"No path from {start} to any exit.")
        return best_path, best_cost


# ============================================================
# COMMUNICATION MODEL
# ============================================================

class CommunicationEngine:
    """Radio/data connectivity between physical guidance devices."""

    def __init__(self, devices: Dict[str, GuidanceDevice]) -> None:
        self.devices = devices
        self.communication_graph = nx.Graph()
        self.node_to_device: Dict[str, str] = {}
        self.rebuild()

    def rebuild(self) -> None:
        graph = nx.Graph()
        self.node_to_device.clear()
        for name, device in self.devices.items():
            if device.controlled_node in self.node_to_device:
                raise RuntimeError(f"Multiple devices control node {device.controlled_node}")
            self.node_to_device[device.controlled_node] = name
            graph.add_node(name, position=np.array(device.position), controlled_node=device.controlled_node)
        names = list(self.devices)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                a, b = self.devices[left], self.devices[right]
                distance = float(np.linalg.norm(np.array(a.position) - np.array(b.position)))
                if distance <= min(a.communication_radius, b.communication_radius):
                    graph.add_edge(left, right, distance=distance)
        self.communication_graph = graph

    def validate_connectivity(self) -> None:
        if not self.communication_graph.nodes:
            raise RuntimeError("No guidance devices deployed.")
        if not nx.is_connected(self.communication_graph):
            components = [sorted(c) for c in nx.connected_components(self.communication_graph)]
            raise RuntimeError(f"Communication graph is disconnected: {components}")

    def validate_movement_adjacency_links(self, movement_graph: nx.Graph) -> None:
        """Every physical one-hop routing candidate must have a data link."""
        missing: List[EdgeId] = []
        for u, v in movement_graph.edges:
            du = self.node_to_device.get(u)
            dv = self.node_to_device.get(v)
            if du is None or dv is None or not self.communication_graph.has_edge(du, dv):
                missing.append(canonical_edge(u, v))
        if missing:
            raise RuntimeError(
                "Path-vector deployment requires directly communicating devices "
                f"at both ends of each movement edge. Missing: {missing}"
            )


# ============================================================
# ROUTING STRATEGY INTERFACE AND CENTRAL ORACLE
# ============================================================

class RoutingStrategy(ABC):
    def __init__(self, movement_graph: nx.Graph) -> None:
        self.G = movement_graph

    @abstractmethod
    def recompute(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_value(self, node: str) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_next(self, node: str) -> Optional[str]:
        raise NotImplementedError

    @abstractmethod
    def get_path(self, start: str) -> Tuple[List[str], float]:
        raise NotImplementedError

    def has_global_policy(self) -> bool:
        return True

    def on_edge_status_changed(self, u: str, v: str) -> None:
        self.recompute()

    def on_graph_reset(self, changed_edges: Iterable[EdgeId]) -> None:
        self.recompute()

    def device_policy_arrows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return []

    def debug_device_line(self, device_name: str, device: GuidanceDevice) -> str:
        return ""


class CentralizedBellmanStrategy(RoutingStrategy):
    r"""Synchronous Bellman relaxation retained as a reference strategy.

    V(x) = 0 for exits, otherwise
    V(x) = \min_{y \in N(x)} (c(x,y) + V(y)).

    The reference state is stored inside this strategy.  It does not mutate
    the movement graph fields used by another strategy.
    """

    def __init__(self, movement_graph: nx.Graph, steps: int = 80) -> None:
        super().__init__(movement_graph)
        self.steps = steps
        self.values: Dict[str, float] = {}
        self.next_hops: Dict[str, Optional[str]] = {}

    def recompute(self) -> None:
        self.values = {
            node: 0.0 if attrs["kind"] == "exit" else INF
            for node, attrs in self.G.nodes(data=True)
        }
        self.next_hops = {node: None for node in self.G.nodes}
        for _ in range(self.steps):
            new_values: Dict[str, float] = {}
            new_next: Dict[str, Optional[str]] = {}
            for x, attrs in self.G.nodes(data=True):
                if attrs["kind"] == "exit":
                    new_values[x], new_next[x] = 0.0, None
                    continue
                candidates = [
                    (float(self.G[x][y]["base_weight"]) + self.values[y], y)
                    for y in self.G.neighbors(x)
                    if not self.G[x][y].get("blocked", False)
                ]
                if candidates:
                    best_value, best_next = min(candidates)
                    new_values[x] = best_value
                    new_next[x] = None if np.isinf(best_value) else best_next
                else:
                    new_values[x], new_next[x] = INF, None
            self.values, self.next_hops = new_values, new_next

    def get_value(self, node: str) -> float:
        return self.values.get(node, INF)

    def get_next(self, node: str) -> Optional[str]:
        return self.next_hops.get(node)

    def get_path(self, start: str, max_hops: int = 200) -> Tuple[List[str], float]:
        path, cost, current = [start], 0.0, start
        visited = {start}
        for _ in range(max_hops):
            if self.G.nodes[current]["kind"] == "exit":
                return path, cost
            nxt = self.get_next(current)
            if nxt is None or self.G[current][nxt].get("blocked", False):
                raise nx.NetworkXNoPath(f"No centralized route from {start} to an exit.")
            if nxt in visited:
                raise RuntimeError("Cycle in centralized Bellman policy.")
            cost += float(self.G[current][nxt]["base_weight"])
            current = nxt
            visited.add(current)
            path.append(current)
        raise RuntimeError("Path reconstruction exceeded max_hops.")


# ============================================================
# LOCAL PATH-VECTOR ROUTING PROTOCOL
# ============================================================

@dataclass(frozen=True)
class LocalLink:
    neighbor_node: str
    cost: float
    blocked: bool
    version: int


@dataclass(frozen=True)
class RouteAdvertisement:
    """One device's selected route, or its explicit withdrawal."""

    sender_device: str
    sender_node: str
    generation: int
    reachable: bool
    exit_id: Optional[str]
    cost: float
    path: Tuple[str, ...]

    @classmethod
    def withdrawal(cls, sender_device: str, sender_node: str, generation: int) -> "RouteAdvertisement":
        return cls(sender_device, sender_node, generation, False, None, INF, tuple())


@dataclass
class RouteAgentState:
    device_name: str
    controlled_node: str
    is_exit: bool
    links: Dict[str, LocalLink]
    route: RouteAdvertisement
    received_routes: Dict[str, RouteAdvertisement] = field(default_factory=dict)
    inbox: Deque[RouteAdvertisement] = field(default_factory=deque)
    changed: bool = True


class DistributedPathVectorEngine:
    r"""Local, asynchronous, loop-resistant routing protocol.

    Each agent controls one routing waypoint x.  It knows only incident
    physical links and advertisements received from neighbouring waypoint
    agents.  For a traversable local link (x, y), a candidate is:

        c(x,y) + A_y.cost

    where A_y is y's latest advertised route.  A candidate is rejected when x
    already occurs in A_y.path, which prevents route loops and count-to-infinity
    reuse after a disconnection.

    No agent stores the complete building value field or calls the diagnostic
    shortest-path oracle to select its displayed direction.
    """

    def __init__(
        self,
        movement_graph: nx.Graph,
        communication: CommunicationEngine,
        devices: Dict[str, GuidanceDevice],
    ) -> None:
        self.G = movement_graph  # Deployment/config source and UI world only.
        self.communication = communication
        self.C = communication.communication_graph
        self.devices = devices
        self.node_to_device = dict(communication.node_to_device)
        self.states: Dict[str, RouteAgentState] = {}
        self.tick_count = 0
        self.last_messages_sent = 0
        self.last_route_changes = 0
        self.initialize()

    def initialize(self) -> None:
        self.states.clear()
        for device_name, device in self.devices.items():
            node = device.controlled_node
            links = {
                neighbor: LocalLink(
                    neighbor_node=neighbor,
                    cost=float(self.G[node][neighbor]["base_weight"]),
                    blocked=bool(self.G[node][neighbor].get("blocked", False)),
                    version=int(self.G[node][neighbor].get("version", 0)),
                )
                for neighbor in self.G.neighbors(node)
            }
            is_exit = self.G.nodes[node]["kind"] == "exit"
            route = (
                RouteAdvertisement(device_name, node, 0, True, node, 0.0, (node,))
                if is_exit
                else RouteAdvertisement.withdrawal(device_name, node, 0)
            )
            self.states[device_name] = RouteAgentState(device_name, node, is_exit, links, route)
        self._broadcast_changed_routes()

    def _targets(self, device_name: str) -> List[str]:
        return list(self.C.neighbors(device_name))

    def _broadcast_changed_routes(self) -> int:
        sent = 0
        for state in self.states.values():
            if not state.changed:
                continue
            for target in self._targets(state.device_name):
                self.states[target].inbox.append(state.route)
                sent += 1
            state.changed = False
        self.last_messages_sent = sent
        return sent

    def _process_inbox(self, state: RouteAgentState) -> None:
        while state.inbox:
            advertisement = state.inbox.popleft()
            sender_node = advertisement.sender_node
            # Extra radio neighbours are allowed, but only a physically adjacent
            # waypoint can be used as a first routing hop.
            if sender_node not in state.links:
                continue
            previous = state.received_routes.get(sender_node)
            if previous is None or advertisement.generation > previous.generation:
                state.received_routes[sender_node] = advertisement

    def _select_route(self, state: RouteAgentState) -> RouteAdvertisement:
        if state.is_exit:
            return state.route
        candidates: List[Tuple[float, str, RouteAdvertisement]] = []
        for neighbor, link in state.links.items():
            if link.blocked:
                continue
            advertisement = state.received_routes.get(neighbor)
            if advertisement is None or not advertisement.reachable:
                continue
            if state.controlled_node in advertisement.path:
                continue
            if len(set(advertisement.path)) != len(advertisement.path):
                continue
            candidates.append((link.cost + advertisement.cost, neighbor, advertisement))
        next_generation = state.route.generation + 1
        if not candidates:
            candidate = RouteAdvertisement.withdrawal(state.device_name, state.controlled_node, next_generation)
        else:
            cost, _, downstream = min(candidates, key=lambda item: (item[0], item[1]))
            candidate = RouteAdvertisement(
                sender_device=state.device_name,
                sender_node=state.controlled_node,
                generation=next_generation,
                reachable=True,
                exit_id=downstream.exit_id,
                cost=cost,
                path=(state.controlled_node,) + downstream.path,
            )
        current_signature = (state.route.reachable, state.route.exit_id, state.route.cost, state.route.path)
        candidate_signature = (candidate.reachable, candidate.exit_id, candidate.cost, candidate.path)
        return state.route if current_signature == candidate_signature else candidate

    def observe_incident_edge_change(self, u: str, v: str, blocked: bool, version: int) -> None:
        """Deliver a physical observation only to the two endpoint agents."""
        for node, neighbor in ((u, v), (v, u)):
            device_name = self.node_to_device[node]
            state = self.states[device_name]
            current = state.links.get(neighbor)
            if current is None:
                raise RuntimeError(f"Device at {node} has no incident local link to {neighbor}")
            if version > current.version:
                state.links[neighbor] = LocalLink(neighbor, current.cost, blocked, version)

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.tick_count += 1
            for state in self.states.values():
                self._process_inbox(state)
            changes = 0
            for state in self.states.values():
                selected = self._select_route(state)
                if selected is not state.route:
                    state.route = selected
                    state.changed = True
                    changes += 1
            self.last_route_changes = changes
            self._broadcast_changed_routes()

    def pending_work(self) -> int:
        return sum(len(state.inbox) + int(state.changed) for state in self.states.values())

    def run_until_quiet(self, max_ticks: int = 500) -> None:
        for _ in range(max_ticks):
            before = self.pending_work()
            self.tick(1)
            if before == 0 and self.pending_work() == 0 and self.last_route_changes == 0:
                return
        raise RuntimeError("Path-vector routing did not settle within max_ticks.")

    def route_for_node(self, node: str) -> RouteAdvertisement:
        return self.states[self.node_to_device[node]].route

    def next_for_node(self, node: str) -> Optional[str]:
        route = self.route_for_node(node)
        return route.path[1] if route.reachable and len(route.path) >= 2 else None

    def route_is_structurally_safe(self, node: str) -> bool:
        """Check the locally selected path-vector proof, not a global search."""
        route = self.route_for_node(node)
        state = self.states[self.node_to_device[node]]
        if state.is_exit:
            return route.reachable and route.path == (node,)
        return (
            route.reachable
            and bool(route.exit_id)
            and len(route.path) >= 2
            and route.path[0] == node
            and route.path[-1] == route.exit_id
            and len(set(route.path)) == len(route.path)
        )


class PathVectorDiagnostics:
    """Observer-side validation only; it never changes distributed decisions."""

    def __init__(self, engine: DistributedPathVectorEngine, controller: MovementGraphController) -> None:
        self.engine = engine
        self.G = engine.G
        self.controller = controller

    def exact_cost(self, node: str) -> float:
        try:
            _, cost = self.controller.shortest_path_to_nearest_exit(node)
            return cost
        except nx.NetworkXNoPath:
            return INF

    def global_summary(self) -> Dict[str, float]:
        wrong, unsafe, max_error = 0, 0, 0.0
        for node in self.G.nodes:
            distributed = self.engine.route_for_node(node).cost
            exact = self.exact_cost(node)
            if np.isinf(distributed) and np.isinf(exact):
                error = 0.0
            elif np.isinf(distributed) != np.isinf(exact):
                error = INF
            else:
                error = abs(distributed - exact)
            if np.isinf(error) or error > 1e-6:
                wrong += 1
            if np.isinf(error):
                max_error = INF
            elif not np.isinf(max_error):
                max_error = max(max_error, error)
            if self.G.nodes[node]["kind"] != "exit" and not np.isinf(distributed) and not self.engine.route_is_structurally_safe(node):
                unsafe += 1
        return {"wrong_nodes": float(wrong), "max_error": max_error, "unsafe_routes": float(unsafe)}

    def device_table_rows(self) -> List[Dict[str, str]]:
        rows: List[Dict[str, str]] = []
        for device_name, state in self.engine.states.items():
            route = state.route
            rows.append({
                "device": device_name,
                "node": state.controlled_node,
                "V": finite_text(route.cost),
                "next": self.engine.next_for_node(state.controlled_node) or "-",
                "exit": route.exit_id or "-",
                "safe": "OK" if self.engine.route_is_structurally_safe(state.controlled_node) else "NO",
                "gen": str(route.generation),
                "inbox": str(len(state.inbox)),
            })
        return rows


class DistributedPathVectorStrategy(RoutingStrategy):
    """Viewer-facing wrapper around local path-vector route agents."""

    def __init__(
        self,
        movement_graph: nx.Graph,
        communication: CommunicationEngine,
        devices: Dict[str, GuidanceDevice],
        controller: MovementGraphController,
        *,
        bootstrap_ticks: int = 500,
        ticks_per_event: int = 1,
    ) -> None:
        super().__init__(movement_graph)
        self.devices = devices
        self.controller = controller
        self.bootstrap_ticks = bootstrap_ticks
        self.ticks_per_event = ticks_per_event
        self.engine = DistributedPathVectorEngine(movement_graph, communication, devices)
        self.diagnostics = PathVectorDiagnostics(self.engine, controller)

    def has_global_policy(self) -> bool:
        return False

    def recompute(self) -> None:
        self.engine.run_until_quiet(self.bootstrap_ticks)
        self._publish_for_visualization()

    def on_edge_status_changed(self, u: str, v: str) -> None:
        blocked, version = self.controller.edge_event(u, v)
        self.engine.observe_incident_edge_change(u, v, blocked, version)
        self.engine.tick(self.ticks_per_event)
        self._publish_for_visualization()

    def on_graph_reset(self, changed_edges: Iterable[EdgeId]) -> None:
        for u, v in changed_edges:
            blocked, version = self.controller.edge_event(u, v)
            self.engine.observe_incident_edge_change(u, v, blocked, version)
        self.engine.run_until_quiet(self.bootstrap_ticks)
        self._publish_for_visualization()

    def tick(self, n: int = 1) -> None:
        self.engine.tick(n)
        self._publish_for_visualization()

    def settle_until_quiet(self, max_ticks: int = 500) -> None:
        self.engine.run_until_quiet(max_ticks)
        self._publish_for_visualization()

    def _publish_for_visualization(self) -> None:
        # The viewer reads this strategy through get_value()/get_next().
        # Do not publish into shared graph attributes: another strategy may be active.
        return

    def get_value(self, node: str) -> float:
        return self.engine.route_for_node(node).cost

    def get_next(self, node: str) -> Optional[str]:
        return self.engine.next_for_node(node)

    def get_path(self, start: str) -> Tuple[List[str], float]:
        route = self.engine.route_for_node(start)
        if not route.reachable:
            raise nx.NetworkXNoPath(f"No distributed route from {start} to an exit.")
        return list(route.path), float(route.cost)

    def device_policy_arrows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return uniformly sized local actuator directions.

        The arrow represents the selected first-hop direction, not the metric
        length of the physical edge. Every displayed actuator therefore uses
        the same normalized vector and one fixed visual scale.
        """
        arrows: List[Tuple[np.ndarray, np.ndarray]] = []
        for device in self.devices.values():
            if not device.display:
                continue
            node = device.controlled_node
            nxt = self.engine.next_for_node(node)
            if nxt is None or not self.engine.route_is_structurally_safe(node):
                continue
            start = np.array(device.position, dtype=float)
            direction = self.G.nodes[nxt]["position"] - start
            norm = float(np.linalg.norm(direction))
            if norm > 1e-9:
                arrows.append((start, direction / norm))
        return arrows

    def debug_device_line(self, device_name: str, device: GuidanceDevice) -> str:
        route = self.engine.states[device_name].route
        nxt = self.engine.next_for_node(device.controlled_node) or "-"
        return f"  {device.controlled_node}: V={finite_text(route.cost)}, next={nxt}, exit={route.exit_id or '-'}"

    def debug_summary(self) -> str:
        summary = self.diagnostics.global_summary()
        return (
            f"path-vector ticks={self.engine.tick_count} | pending={self.engine.pending_work()} | "
            f"changes={self.engine.last_route_changes} | msgs={self.engine.last_messages_sent} | "
            f"wrong_nodes={int(summary['wrong_nodes'])} | max_err={finite_text(summary['max_error'], 2)} | "
            f"unsafe={int(summary['unsafe_routes'])}"
        )

    def device_table_rows(self) -> List[Dict[str, str]]:
        return self.diagnostics.device_table_rows()


class StrategyManager:
    def __init__(self) -> None:
        self._strategies: Dict[str, RoutingStrategy] = {}
        self._current: Optional[str] = None

    def register(self, name: str, strategy: RoutingStrategy) -> None:
        self._strategies[name] = strategy
        if self._current is None:
            self._current = name

    def names(self) -> List[str]:
        return list(self._strategies)

    def current_name(self) -> str:
        if self._current is None:
            raise RuntimeError("No routing strategy registered.")
        return self._current

    def current(self) -> RoutingStrategy:
        return self._strategies[self.current_name()]

    def next(self) -> None:
        names = self.names()
        index = names.index(self.current_name())
        self._current = names[(index + 1) % len(names)]

    def notify_edge_changed(self, u: str, v: str) -> None:
        for strategy in self._strategies.values():
            strategy.on_edge_status_changed(u, v)

    def notify_graph_reset(self, changed_edges: Iterable[EdgeId]) -> None:
        changed = list(changed_edges)
        for strategy in self._strategies.values():
            strategy.on_graph_reset(changed)


# ============================================================
# INTERACTIVE VIEWER
# ============================================================

class InteractiveBuildingViewer:
    """PyVista viewer; distributed decisions remain inside the strategy."""

    def __init__(
        self,
        geometry: BuildingGeometry,
        controller: MovementGraphController,
        strategy_manager: StrategyManager,
        communication: CommunicationEngine,
    ) -> None:
        self.geometry = geometry
        self.controller = controller
        self.strategies = strategy_manager
        self.communication = communication
        self.G = geometry.movement_graph
        self.C = communication.communication_graph
        self.plotter = pv.Plotter(window_size=(1500, 920))
        self.plotter.set_background("white")
        self.start_nodes = [n for n, attrs in self.G.nodes(data=True) if attrs["kind"] == "room"]
        self.current_start = self.start_nodes[0]
        self.edge_list = list(self.G.edges())
        self.selected_edge_index = 0
        self.pick_mode = "node"
        self.gossip_ticks_per_click = 1
        self.gossip_settle_ticks = 500
        self.edge_actors: Dict[EdgeId, Any] = {}
        self.path_actor = None
        self.heatmap_actor = None
        self.selected_node_actor = None
        self.selected_edge_actor = None
        self.policy_arrow_actors: List[Any] = []
        self.device_arrow_actors: List[Any] = []
        self.current_path: Optional[List[str]] = None
        self.field_cost: Optional[float] = None
        self.exact_cost: Optional[float] = None
        self.error: Optional[str] = None
        # Manual mode protects an adjusted view across explicit redraws.
        # Timer-driven Qt demos disable this: restoring a cached camera pose
        # from each callback would fight trackball mouse interaction.
        self.preserve_camera_on_refresh = True
        self._scene_camera_initialized = False

    def strategy(self) -> RoutingStrategy:
        return self.strategies.current()

    def _capture_camera_state(self) -> Dict[str, Any]:
        """Capture the manually adjusted recording view before actor updates."""
        return {
            "camera_position": [
                tuple(point) for point in self.plotter.camera_position
            ],
            "parallel_scale": float(self.plotter.camera.parallel_scale),
            "view_angle": float(self.plotter.camera.view_angle),
        }

    def _restore_camera_state(self, state: Dict[str, Any]) -> None:
        """Restore camera pose and zoom after dynamic actor replacement."""
        self.plotter.camera_position = state["camera_position"]
        self.plotter.camera.parallel_scale = state["parallel_scale"]
        self.plotter.camera.view_angle = state["view_angle"]
        # Dynamic actor bounds can change after route withdrawal/recovery.
        # Update clipping only; never refit/reset the manually selected view.
        self.plotter.reset_camera_clipping_range()

    def fit_initial_camera(self) -> None:
        """Fit the completed scene once, without taking over later interaction."""
        self.plotter.view_isometric(render=False)
        self.plotter.reset_camera(render=False)
        self.plotter.reset_camera_clipping_range()
        self._scene_camera_initialized = True

    def _dynamic_camera_state(self) -> Optional[Dict[str, Any]]:
        """Capture a stable manual view only when redraws are user-driven."""
        if self.preserve_camera_on_refresh and self._scene_camera_initialized:
            return self._capture_camera_state()
        return None

    def _finish_dynamic_refresh(self, camera_state: Optional[Dict[str, Any]]) -> None:
        if not self._scene_camera_initialized:
            # All static and first dynamic actors are now available.
            self.fit_initial_camera()
        elif camera_state is not None:
            self._restore_camera_state(camera_state)
        else:
            # Qt timer mode: never write camera_position while the user moves it.
            self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def _selected_edge(self) -> Tuple[str, str]:
        return self.edge_list[self.selected_edge_index]

    def toggle_selected_edge(self) -> None:
        u, v = self._selected_edge()
        if self.controller.toggle_edge(u, v):
            self.strategies.notify_edge_changed(u, v)
        self.refresh_after_incremental_update()

    def reset_edges(self) -> None:
        changed = self.controller.reset_edges()
        self.strategies.notify_graph_reset(changed)
        self.refresh_after_incremental_update()

    def tick_distributed_once(self) -> None:
        strategy = self.strategy()
        if hasattr(strategy, "tick"):
            strategy.tick(self.gossip_ticks_per_click)
        self.refresh_after_incremental_update()

    def settle_distributed(self) -> None:
        strategy = self.strategy()
        if hasattr(strategy, "settle_until_quiet"):
            strategy.settle_until_quiet(self.gossip_settle_ticks)
        self.refresh_after_incremental_update()

    def next_strategy(self) -> None:
        self.strategies.next()
        self.refresh_full()

    def _add_button(self, callback, position, label: str, size: int = 30):
        widget = None
        def pressed(state: bool) -> None:
            if state:
                callback()
                if widget is not None:
                    widget.GetRepresentation().SetState(0)
        widget = self.plotter.add_checkbox_button_widget(
            pressed, value=False, position=position, size=size,
            color_on="lightblue", color_off="white", border_size=1,
        )
        self.plotter.add_text(label, position=(position[0] + size + 8, position[1] + 5), font_size=9)
        return widget

    def _add_ui(self) -> None:
        actions = [
            (self.reset_edges, "Reset edges"),
            (self.next_strategy, "Next strategy"),
            (self.tick_distributed_once, "Tick distributed"),
            (self.settle_distributed, "Settle distributed"),
            (self.print_device_table, "Print device table"),
            (self.print_path_info, "Print path"),
            (lambda: self._set_pick_mode("node"), "Pick: node"),
            (lambda: self._set_pick_mode("edge"), "Pick: edge"),
        ]
        y = 10 + len(actions) * 40
        for callback, label in actions:
            self._add_button(callback, (10, y), label)
            y -= 40

    def _set_pick_mode(self, mode: str) -> None:
        self.pick_mode = mode
        self._update_text()
        self.plotter.render()

    def _on_point_picked(self, point, *args) -> None:
        if point is None:
            return
        p = np.array(point, dtype=float)
        if self.pick_mode == "node":
            candidates = [(float(np.linalg.norm(attrs["position"] - p)), n) for n, attrs in self.G.nodes(data=True) if attrs["kind"] == "room"]
            distance, node = min(candidates)
            if distance <= 3.0:
                self.current_start = node
                self._highlight_selected_node(node)
                self.refresh_path_only()
        else:
            best_edge: Optional[Tuple[str, str]] = None
            best_distance = INF
            for u, v in self.G.edges:
                a, b = self.G.nodes[u]["position"], self.G.nodes[v]["position"]
                segment = b - a
                t = np.clip(float(np.dot(p - a, segment) / max(np.dot(segment, segment), 1e-12)), 0.0, 1.0)
                distance = float(np.linalg.norm(p - (a + t * segment)))
                if distance < best_distance:
                    best_distance, best_edge = distance, (u, v)
            if best_edge is not None and best_distance <= 2.5:
                canonical = canonical_edge(*best_edge)
                self.selected_edge_index = [canonical_edge(*e) for e in self.edge_list].index(canonical)
                self.toggle_selected_edge()
                self._highlight_selected_edge(*best_edge)

    def _add_static_scene(self) -> None:
        for space in self.geometry.spaces.values():
            x, y, z = space.center
            sx, sy, sz = space.size
            mesh = pv.Box(bounds=(x - sx/2, x + sx/2, y - sy/2, y + sy/2, z - sz/2, z + sz/2))
            self.plotter.add_mesh(mesh, color=space.color, opacity=space.opacity, show_edges=True, pickable=False)
        points = np.array([attrs["position"] for _, attrs in self.G.nodes(data=True)])
        labels = [n for n in self.G.nodes]
        self.plotter.add_mesh(pv.PolyData(points), color="black", point_size=13, render_points_as_spheres=True, pickable=True)
        self.plotter.add_point_labels(points, labels, font_size=8, point_size=0, shape_opacity=0.12)
        for u, v in self.G.edges:
            actor = self.plotter.add_mesh(pv.Line(self.G.nodes[u]["position"], self.G.nodes[v]["position"]), color="gray", line_width=4, pickable=True)
            self.edge_actors[canonical_edge(u, v)] = actor
        display_devices = [d for d in self.geometry.devices.values() if d.display]
        device_points = np.array([d.position for d in display_devices], dtype=float)
        self.plotter.add_mesh(pv.PolyData(device_points), color="orange", point_size=13, render_points_as_spheres=True, pickable=False)
        for left, right in self.C.edges:
            p0 = self.C.nodes[left]["position"]
            p1 = self.C.nodes[right]["position"]
            self.plotter.add_mesh(pv.Line(p0, p1), color="deepskyblue", line_width=1, opacity=0.12, pickable=False)

    def _highlight_selected_node(self, node: str) -> None:
        if self.selected_node_actor is not None:
            self.plotter.remove_actor(self.selected_node_actor)
        self.selected_node_actor = self.plotter.add_mesh(
            pv.Sphere(radius=0.7, center=self.G.nodes[node]["position"]),
            color="yellow",
            pickable=False,
            reset_camera=False,
        )

    def _highlight_selected_edge(self, u: str, v: str) -> None:
        if self.selected_edge_actor is not None:
            self.plotter.remove_actor(self.selected_edge_actor)
        self.selected_edge_actor = self.plotter.add_mesh(
            pv.Line(self.G.nodes[u]["position"], self.G.nodes[v]["position"]),
            color="orange",
            line_width=10,
            pickable=False,
            reset_camera=False,
        )

    def _draw_dynamic(self) -> None:
        if self.path_actor is not None:
            self.plotter.remove_actor(self.path_actor)
            self.path_actor = None
        if self.current_path and len(self.current_path) >= 2:
            points = np.array([self.G.nodes[n]["position"] for n in self.current_path])
            self.path_actor = self.plotter.add_mesh(
                pv.lines_from_points(points),
                color="lime",
                line_width=9,
                pickable=False,
                reset_camera=False,
            )
        if self.heatmap_actor is not None:
            self.plotter.remove_actor(self.heatmap_actor)
        values = [100.0 if np.isinf(self.strategy().get_value(n)) else self.strategy().get_value(n) for n in self.G.nodes]
        pdata = pv.PolyData(np.array([self.G.nodes[n]["position"] for n in self.G.nodes]))
        pdata["value"] = np.array(values)
        self.heatmap_actor = self.plotter.add_mesh(
            pdata,
            scalars="value",
            point_size=25,
            render_points_as_spheres=True,
            cmap="coolwarm",
            scalar_bar_args={"title": "Route cost"},
            pickable=False,
            reset_camera=False,
        )
        for actor in self.policy_arrow_actors + self.device_arrow_actors:
            self.plotter.remove_actor(actor)
        self.policy_arrow_actors.clear()
        self.device_arrow_actors.clear()
        if self.strategy().has_global_policy():
            for node in self.G.nodes:
                nxt = self.strategy().get_next(node)
                if nxt is not None:
                    direction = self.G.nodes[nxt]["position"] - self.G.nodes[node]["position"]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-9:
                        self.policy_arrow_actors.append(
                            self.plotter.add_mesh(
                                pv.Arrow(
                                    start=self.G.nodes[node]["position"],
                                    direction=direction / norm,
                                    scale=1.1,
                                ),
                                color="royalblue",
                                pickable=False,
                                reset_camera=False,
                            )
                        )
        else:
            for start, direction in self.strategy().device_policy_arrows():
                self.device_arrow_actors.append(
                    self.plotter.add_mesh(
                        pv.Arrow(start=start, direction=direction, scale=1.1),
                        color="darkorange",
                        pickable=False,
                        reset_camera=False,
                    )
                )
        path_edges = {canonical_edge(self.current_path[i], self.current_path[i+1]) for i in range(len(self.current_path or []) - 1)}
        for edge, actor in self.edge_actors.items():
            actor.prop.line_width = 4
            if self.G[edge[0]][edge[1]].get("blocked", False):
                actor.prop.color = (1.0, 0.1, 0.1)
            elif edge in path_edges:
                actor.prop.color = (0.0, 0.85, 0.0)
            else:
                actor.prop.color = (0.55, 0.55, 0.55)

    def _recompute_path(self) -> None:
        self.current_path, self.field_cost, self.exact_cost, self.error = None, None, None, None
        try:
            self.current_path, self.field_cost = self.strategy().get_path(self.current_start)
        except Exception as exc:
            self.error = str(exc)
        try:
            _, self.exact_cost = self.controller.shortest_path_to_nearest_exit(self.current_start)
        except nx.NetworkXNoPath:
            self.exact_cost = INF

    def _table_text(self, max_rows: int = 18) -> str:
        strategy = self.strategy()
        if not hasattr(strategy, "device_table_rows"):
            return "Device table available in path-vector mode"
        rows = strategy.device_table_rows()
        lines = ["Local path-vector device state", f"{'Node':<9} {'V':>6} {'Next':<9} {'Exit':<7} {'Safe':<4} {'Gen':>4}", "-" * 47]
        for row in rows[:max_rows]:
            lines.append(f"{row['node']:<9} {row['V']:>6} {row['next']:<9} {row['exit']:<7} {row['safe']:<4} {row['gen']:>4}")
        if len(rows) > max_rows:
            lines.append(f"... {len(rows) - max_rows} additional devices; use Print device table")
        return "\n".join(lines)

    def _update_text(self) -> None:
        self.plotter.remove_actor("status", render=False)
        self.plotter.remove_actor("table", render=False)
        self.plotter.remove_actor("help", render=False)
        value = self.strategy().get_value(self.current_start)
        lines = [
            f"Start: {self.current_start} | Strategy: {self.strategies.current_name()} | Pick: {self.pick_mode}",
            f"Local/distributed value: {finite_text(value, 2)} | Exact diagnostic cost: {finite_text(self.exact_cost if self.exact_cost is not None else INF, 2)}",
        ]
        if hasattr(self.strategy(), "debug_summary"):
            lines.append(self.strategy().debug_summary())
        lines.append(f"Status: {self.error}" if self.error else "Path: " + " -> ".join(self.current_path or []))
        self.plotter.add_text("\n".join(lines), position="upper_left", font_size=9, name="status")
        self.plotter.add_text(self._table_text(), position="upper_right", font_size=8, name="table")
        self.plotter.add_text("Pick an edge to block/unblock | Tick shows propagation | Orange arrows are local device decisions", position="lower_left", font_size=9, name="help")

    def refresh_full(self) -> None:
        self.strategy().recompute()
        self.refresh_after_incremental_update()

    def refresh_after_incremental_update(self) -> None:
        camera_state = self._dynamic_camera_state()
        self._recompute_path()
        self._draw_dynamic()
        self._update_text()
        self._finish_dynamic_refresh(camera_state)

    def refresh_path_only(self) -> None:
        camera_state = self._dynamic_camera_state()
        self._recompute_path()
        self._draw_dynamic()
        self._update_text()
        self._finish_dynamic_refresh(camera_state)

    def print_device_table(self) -> None:
        strategy = self.strategy()
        if not hasattr(strategy, "device_table_rows"):
            print("Device table available in distributed path-vector mode.")
            return
        rows = strategy.device_table_rows()
        print(f"\n{'Node':<10} {'Cost':>8} {'Next':<10} {'Exit':<10} {'Safe':<5} {'Gen':>5} {'Inbox':>6}")
        print("-" * 62)
        for row in rows:
            print(f"{row['node']:<10} {row['V']:>8} {row['next']:<10} {row['exit']:<10} {row['safe']:<5} {row['gen']:>5} {row['inbox']:>6}")
        print()

    def print_path_info(self) -> None:
        self._recompute_path()
        if self.error:
            print(f"\nNo displayed route for {self.current_start}: {self.error}\n")
        else:
            print(f"\nDisplayed route: {' -> '.join(self.current_path or [])} | cost={self.field_cost:.2f} | exact={finite_text(self.exact_cost if self.exact_cost is not None else INF, 2)}\n")

    def build_scene(self, *, enable_picking: bool = True) -> None:
        """Build the 3D scene.

        Interactive picking is enabled for the manual simulator, but can be
        disabled for automated screen-recording scenarios.  The picker binds
        mouse interaction that otherwise belongs to camera navigation.
        """
        self._add_static_scene()
        self._add_ui()
        self.plotter.show_grid()
        if enable_picking:
            self.plotter.enable_surface_point_picking(
                callback=self._on_point_picked,
                show_point=True,
                clear_on_no_selection=True,
                font_size=0,
                left_clicking=False,
            )
        self.refresh_full()
        self._highlight_selected_node(self.current_start)
        self._highlight_selected_edge(*self._selected_edge())

    def show(self) -> None:
        self.build_scene(enable_picking=True)
        self.plotter.show()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def build_application(num_floors: int = 2) -> Tuple[BuildingGeometry, MovementGraphController, CommunicationEngine, StrategyManager]:
    geometry = BuildingGeometry.demo_building(num_floors=num_floors)
    controller = MovementGraphController(geometry.movement_graph)
    communication = CommunicationEngine(geometry.devices)
    communication.validate_connectivity()
    communication.validate_movement_adjacency_links(geometry.movement_graph)
    manager = StrategyManager()
    manager.register("centralized-bellman-oracle", CentralizedBellmanStrategy(geometry.movement_graph))
    distributed = DistributedPathVectorStrategy(
        geometry.movement_graph,
        communication,
        geometry.devices,
        controller,
        bootstrap_ticks=500,
        ticks_per_event=1,
    )
    manager.register("distributed-path-vector", distributed)
    return geometry, controller, communication, manager


def main() -> None:
    geometry, controller, communication, manager = build_application(num_floors=2)
    # Select distributed mode at startup; centralized mode remains an optional oracle view.
    manager.next()
    viewer = InteractiveBuildingViewer(geometry, controller, manager, communication)
    viewer.show()


if __name__ == "__main__":
    main()
