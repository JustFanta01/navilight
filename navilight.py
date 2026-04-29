"""
============================================================
SMART BUILDING NAVIGATION SIMULATION — STRATEGY REFACTOR
============================================================

This simulation models evacuation routing in a building using three
separate layers:

1. MOVEMENT GRAPH  G_M = (V, A)
   - Nodes = physical positions: rooms, junctions, stairs, exits
   - Edges = walkable paths with traversal costs
   - Defines the physical walkability of the building

2. COMMUNICATION GRAPH  G_C = (S, L)
   - Nodes = sensors/controllers
   - Edges = communication links between nearby devices
   - Defines how routing information can be exchanged

3. ROUTING STRATEGIES
   - A RoutingStrategy exposes the same interface regardless of algorithm
   - The viewer only talks to StrategyManager.current()
   - New algorithms, such as D* Lite, can be registered without changing
     the viewer

------------------------------------------------------------

CURRENT STRATEGIES
------------------------------------------------------------

CENTRALIZED DIFFUSION:
    - Synchronous Bellman value diffusion on the movement graph
    - Has full knowledge of G_M
    - Produces a global value field V(x) and policy pi(x)

DISTRIBUTED DIFFUSION:
    - Each sensor keeps local estimates V_s(x)
    - Sensors update observed movement nodes locally
    - Sensors exchange estimates through G_C
    - Produces a distributed approximation of the global value field

------------------------------------------------------------

MATHEMATICAL IDEA
------------------------------------------------------------

The desired value field is the shortest distance to the nearest exit:

$$
V(x) = \min_{e \in E} d(x,e)
$$

The Bellman fixed point is:

$$
V(x)=
\begin{cases}
0, & x \in E \\
\min\limits_{y \in \mathcal{N}(x)}
\left(c(x,y)+V(y)\right), & x \notin E
\end{cases}
$$

The induced routing policy is:

$$
\pi(x)=
\arg\min_{y \in \mathcal{N}(x)}
\left(c(x,y)+V(y)\right)
$$

------------------------------------------------------------

KEY ARCHITECTURAL IDEA
------------------------------------------------------------

Movement graph      = physics
Communication graph = information flow
Routing strategy    = algorithm
Viewer              = visualization and interaction only

The viewer is algorithm-agnostic.
It does not know whether the current algorithm is centralized,
distributed, D* Lite, or something else.

============================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    import pyvista as pv
except ImportError as e:
    raise SystemExit("Install with: pip install pyvista networkx numpy") from e


Vec3 = Tuple[float, float, float]


# ============================================================
# DATA MODEL
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
class Sensor:
    name: str
    position: Vec3
    range_radius: float = 9.0
    observed_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# GEOMETRY + MOVEMENT GRAPH
# ============================================================

class BuildingGeometry:
    """
    Owns the physical/semantic building model.

    This class does not know any routing algorithm.

    It owns:
    - spaces: visualization volumes such as rooms/corridors/stairs
    - movement_graph: physical walkability graph
    - sensors: devices that observe nodes and communicate externally

    Sensor coverage is same-floor by default:

    $$
    O_s = \{x \in V : \|p_s - p_x\| \le r_s
    \text{ and } level(s)=level(x)\}
    $$

    Each sensor also gets an anchor node:

    $$
    a_s = \arg\min_{x : level(x)=level(s)} \|p_s-p_x\|
    $$

    Sensor arrows are drawn from the sensor position toward the routing
    policy of its anchor node, not toward an arbitrary observed node.
    """

    def __init__(self) -> None:
        self.spaces: Dict[str, Space] = {}
        self.sensors: Dict[str, Sensor] = {}
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
            value=float("inf"),
            next=None,
            **attrs,
        )

    def add_movement_edge(
        self,
        u: str,
        v: str,
        weight: Optional[float] = None,
        **attrs: Any,
    ) -> None:
        p0 = self.movement_graph.nodes[u]["position"]
        p1 = self.movement_graph.nodes[v]["position"]
        dist = float(np.linalg.norm(p1 - p0))
        w = dist if weight is None else weight

        self.movement_graph.add_edge(
            u,
            v,
            weight=w,
            base_weight=w,
            blocked=False,
            **attrs,
        )

    def add_sensor(
        self,
        name: str,
        position: Vec3,
        range_radius: float = 9.0,
        **metadata: Any,
    ) -> None:
        self.sensors[name] = Sensor(
            name=name,
            position=position,
            range_radius=range_radius,
            metadata=metadata,
        )

    def compute_sensor_coverage(self, radius_scale: float = 1.0) -> None:
        """
        Compute observed nodes and anchor node for every sensor.

        Observed nodes:

        $$
        O_s =
        \{x \in V :
        \|p_s-p_x\| \le r_s \cdot \alpha
        \land level(s)=level(x)\}
        $$

        Anchor node:

        $$
        a_s =
        \arg\min_{x : level(x)=level(s)}
        \|p_s-p_x\|
        $$

        The anchor is used for drawing sensor arrows:

        $$
        \text{sensor arrow}(s) \sim \pi(a_s)
        $$

        This avoids visually unstable behavior where a sensor arrow points
        toward the globally best observed node instead of the place where
        the sensor is physically installed.
        """
        for s in self.sensors.values():
            s.observed_nodes.clear()

            p_s = np.array(s.position, dtype=float)
            sensor_level = s.metadata.get("level")

            best_anchor = None
            best_anchor_dist = float("inf")

            for n, attrs in self.movement_graph.nodes(data=True):
                node_level = attrs.get("level")

                if sensor_level is not None and node_level != sensor_level:
                    continue

                p_n = attrs["position"]
                d = float(np.linalg.norm(p_s - p_n))

                if d <= s.range_radius * radius_scale:
                    s.observed_nodes.append(n)

                if d < best_anchor_dist:
                    best_anchor_dist = d
                    best_anchor = n

            s.metadata["anchor_node"] = best_anchor
            s.metadata["anchor_dist"] = best_anchor_dist

    @classmethod
    def demo_building(cls, num_floors: int = 2) -> "BuildingGeometry":
        """
        Build a parameterized multi-floor demo building.

        Change only num_floors to scale the example:

        - num_floors=1 tests purely horizontal routing
        - num_floors=2 tests vertical stairs
        - larger values test multi-floor propagation
        """
        if num_floors < 1:
            raise ValueError("num_floors must be >= 1")

        b = cls()

        floor_height = 6.0
        base_z = 1.5
        zs = [base_z + i * floor_height for i in range(num_floors)]

        floor_size = (44, 36, 0.3)

        for level, z in enumerate(zs):
            b.add_space(
                f"Floor_{level}",
                "floor",
                center=(0, 0, z - 1.5),
                size=floor_size,
                color="silver",
                opacity=1.0,
                level=level,
            )

            b.add_space(
                f"MainCorridor_{level}",
                "corridor",
                center=(0, 0, z),
                size=(30, 4, 3),
                color="lightgreen",
                opacity=0.23,
                level=level,
            )

            b.add_space(
                f"NorthCorridor_{level}",
                "corridor",
                center=(0, 10, z),
                size=(30, 4, 3),
                color="palegreen",
                opacity=0.20,
                level=level,
            )

            b.add_space(
                f"ConnectorCorridor_{level}",
                "corridor",
                center=(0, 5, z),
                size=(4, 6, 3),
                color="mediumseagreen",
                opacity=0.18,
                level=level,
            )

            rooms = [
                ("A", (-11, 5, z)),
                ("B", (11, 5, z)),
                ("C", (0, 15, z)),
                ("D", (11, -5, z)),
                ("E", (-11, -5, z)),
            ]

            for suffix, pos in rooms:
                global_letter = chr(ord("A") + level * 5 + ord(suffix) - ord("A"))
                b.add_space(
                    f"Room_{global_letter}",
                    "room",
                    center=pos,
                    size=(7, 6, 3),
                    color="lightblue",
                    opacity=0.35,
                    level=level,
                )

            b.add_space(
                f"Lab_{level}",
                "lab",
                center=(0, -8, z),
                size=(15, 12, 3),
                color="lightblue",
                opacity=0.35,
                level=level,
            )

            b.add_space(
                f"EastStairVol_{level}",
                "stair",
                center=(19, 0, z),
                size=(4, 5, 3),
                color="plum",
                opacity=0.23,
                level=level,
            )

            b.add_space(
                f"WestStairVol_{level}",
                "stair",
                center=(-19, 0, z),
                size=(4, 5, 3),
                color="plum",
                opacity=0.23,
                level=level,
            )

        # Movement nodes
        for level, z in enumerate(zs):
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

            b.add_movement_node(f"J{level}_CN1", "junction", (-11, 10, z), label=f"J{level}_CN1", level=level)
            b.add_movement_node(f"J{level}_CN2", "junction", (11, 10, z), label=f"J{level}_CN2", level=level)

            b.add_movement_node(f"SE{level}", "stair", (19, 0, z), label=f"East stair {level}", level=level)
            b.add_movement_node(f"SW{level}", "stair", (-19, 0, z), label=f"West stair {level}", level=level)

            if level == 0:
                b.add_movement_node("EXIT_E", "exit", (23, 0, z), label="East exit", level=level)
                b.add_movement_node("EXIT_W", "exit", (-23, 0, z), label="West exit", level=level)

        # In-floor paths
        for level in range(len(zs)):
            letters = [chr(ord("A") + level * 5 + i) for i in range(5)]

            b.add_movement_edge(f"R_{letters[0]}", f"J{level}_W")
            b.add_movement_edge(f"R_{letters[1]}", f"J{level}_W")
            b.add_movement_edge(f"R_{letters[2]}", f"J{level}_N")
            b.add_movement_edge(f"R_{letters[3]}", f"J{level}_E")
            b.add_movement_edge(f"R_{letters[4]}", f"J{level}_E")

            b.add_movement_edge(f"R_{letters[1]}", f"L_{level}")
            b.add_movement_edge(f"R_{letters[3]}", f"L_{level}")

            b.add_movement_edge(f"J{level}_W", f"J{level}_C")
            b.add_movement_edge(f"J{level}_C", f"J{level}_E")
            b.add_movement_edge(f"J{level}_C", f"J{level}_N")
            b.add_movement_edge(f"J{level}_W", f"SW{level}")
            b.add_movement_edge(f"J{level}_E", f"SE{level}")

            # Alternative longer lateral path through north corridor
            b.add_movement_edge(f"J{level}_CN1", f"J{level}_N", weight=13.0)
            b.add_movement_edge(f"J{level}_CN2", f"J{level}_N", weight=13.0)
            b.add_movement_edge(f"J{level}_CN1", f"R_{letters[0]}", weight=13.0)
            b.add_movement_edge(f"J{level}_CN2", f"R_{letters[4]}", weight=13.0)

        # Ground exits
        b.add_movement_edge("SE0", "EXIT_E")
        b.add_movement_edge("SW0", "EXIT_W")

        # Vertical stair connections
        for level in range(num_floors - 1):
            b.add_movement_edge(f"SE{level}", f"SE{level+1}", weight=8.0)
            b.add_movement_edge(f"SW{level}", f"SW{level+1}", weight=8.0)

        # Sensors are not movement nodes.
        for level, z in enumerate(zs):
            b.add_sensor(f"CTRL_{level}_1", (-19, -2, z + 1), range_radius=15.0, level=level)
            b.add_sensor(f"SENS_W_{level}", (-11, 0, z + 1), range_radius=15.0, level=level)
            b.add_sensor(f"SENS_C_{level}", (0, 0, z + 1), range_radius=15.0, level=level)
            b.add_sensor(f"SENS_E_{level}", (11, 0, z + 1), range_radius=15.0, level=level)
            b.add_sensor(f"SENS_N_{level}", (0, 10, z + 1), range_radius=15.0, level=level)
            b.add_sensor(f"SENS_L_{level}", (0, -8, z + 1), range_radius=15.0, level=level)

        b.compute_sensor_coverage()
        return b


# ============================================================
# GRAPH UTILS
# ============================================================

class MovementGraphController:
    """
    Owns graph mutation operations.

    Routing strategies compute policies, but they do not own user actions
    such as blocking/unblocking edges. This controller centralizes mutation
    of edge state so all strategies see the same movement graph.
    """

    def __init__(self, movement_graph: nx.Graph) -> None:
        self.G = movement_graph

    def set_edge_blocked(self, u: str, v: str, blocked: bool = True) -> None:
        self.G[u][v]["blocked"] = blocked
        self.G[u][v]["weight"] = 1e9 if blocked else self.G[u][v]["base_weight"]

    def toggle_edge(self, u: str, v: str) -> None:
        self.set_edge_blocked(u, v, not self.G[u][v]["blocked"])

    def reset_edges(self) -> None:
        for u, v in self.G.edges:
            self.set_edge_blocked(u, v, False)

    def shortest_path_to_nearest_exit(self, start: str) -> Tuple[List[str], float]:
        exits = [n for n, a in self.G.nodes(data=True) if a["kind"] == "exit"]

        best_path = None
        best_cost = float("inf")

        for ex in exits:
            try:
                path = nx.shortest_path(self.G, start, ex, weight="weight")
                cost = float(nx.path_weight(self.G, path, weight="weight"))
                if cost < best_cost:
                    best_path = path
                    best_cost = cost
            except nx.NetworkXNoPath:
                pass

        if best_path is None or best_cost >= 1e9:
            raise nx.NetworkXNoPath(f"No path from {start} to any exit.")

        return best_path, best_cost

    def validate_reachability(self) -> None:
        """
        Ensure every movement node can reach at least one exit.

        Required condition:

        $$
        \forall x \in V, \exists e \in E : x \leadsto e
        $$

        This check is structural. Run it when topology/blocked edges change,
        not at every path query.
        """
        exits = [n for n, a in self.G.nodes(data=True) if a["kind"] == "exit"]
        unreachable = []

        for n in self.G.nodes:
            reachable = False
            for ex in exits:
                try:
                    nx.shortest_path(self.G, n, ex, weight="weight")
                    reachable = True
                    break
                except nx.NetworkXNoPath:
                    continue

            if not reachable:
                unreachable.append(n)

        if unreachable:
            print("\nERROR: Unreachable movement nodes detected:\n")
            for n in unreachable:
                print(f"  - {n}")
            print("\nFix: unblock paths or add movement edges to at least one exit.\n")
            raise RuntimeError("Movement graph contains nodes that cannot reach an exit.")


# ============================================================
# ROUTING STRATEGY INTERFACE
# ============================================================

class RoutingStrategy(ABC):
    """
    Abstract routing algorithm interface.

    The viewer only depends on this abstraction.

    A strategy must provide:
    - recompute(): update internal value field/policy
    - get_value(node): expose routing value V(node)
    - get_next(node): expose policy pi(node), if available
    - get_path(start): return a path under the strategy
    - sensor_policy_arrows(): optional sensor-level arrows

    A future D* Lite strategy can implement this interface without changing
    the viewer.
    """

    def __init__(self, movement_graph: nx.Graph) -> None:
        self.G = movement_graph

    @abstractmethod
    def recompute(self) -> None:
        pass

    @abstractmethod
    def get_value(self, node: str) -> float:
        pass

    @abstractmethod
    def get_next(self, node: str) -> Optional[str]:
        pass

    @abstractmethod
    def get_path(self, start: str) -> Tuple[List[str], float]:
        pass

    def has_global_policy(self) -> bool:
        return True

    def sensor_policy_arrows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return []

    def debug_sensor_line(self, sensor_name: str, sensor: Sensor) -> str:
        return ""


class CentralizedDiffusionStrategy(RoutingStrategy):
    """
    Centralized Bellman diffusion strategy.

    Boundary condition:

    $$
    V_0(x)=
    \begin{cases}
    0, & x \in E \\
    +\infty, & x \notin E
    \end{cases}
    $$

    Synchronous Bellman update:

    $$
    V_{k+1}(x)=
    \begin{cases}
    0, & x \in E \\
    \min\limits_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V_k(y)\right), & x \notin E
    \end{cases}
    $$

    The policy is:

    $$
    \pi(x)=
    \arg\min_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V(y)\right)
    $$

    Transient values may be wrong before convergence. Information propagates
    approximately one graph edge per iteration.
    """

    def __init__(self, movement_graph: nx.Graph, controller: MovementGraphController, steps: int = 40) -> None:
        super().__init__(movement_graph)
        self.controller = controller
        self.steps = steps

    def initialize_values(self) -> None:
        for _, attrs in self.G.nodes(data=True):
            attrs["value"] = 0.0 if attrs["kind"] == "exit" else float("inf")
            attrs["next"] = None

    def diffusion_step(self) -> None:
        new_values = {}
        new_next = {}

        for x, attrs in self.G.nodes(data=True): # for each movement node
            if attrs["kind"] == "exit":
                new_values[x] = 0.0
                new_next[x] = None
                continue

            best_value = float("inf")
            best_neighbor = None

            for y in self.G.neighbors(x):
                if self.G[x][y]["blocked"]:
                    continue

                candidate = self.G.nodes[y]["value"] + self.G[x][y]["weight"]

                if candidate < best_value:
                    best_value = candidate
                    best_neighbor = y

            new_values[x] = best_value
            new_next[x] = best_neighbor

        for n in self.G.nodes:
            self.G.nodes[n]["value"] = new_values[n]
            self.G.nodes[n]["next"] = new_next[n]

    def recompute(self) -> None:
        # Reset is required because this min-only relaxation cannot correct
        # old optimistic values after edge costs increase.
        self.initialize_values()
        
        for _ in range(self.steps):
            self.diffusion_step()

    def get_value(self, node: str) -> float:
        return float(self.G.nodes[node]["value"])

    def get_next(self, node: str) -> Optional[str]:
        return self.G.nodes[node].get("next")

    def get_path(self, start: str, max_hops: int = 100) -> Tuple[List[str], float]:
        path = [start]
        visited = {start}
        current = start
        cost = 0.0

        for _ in range(max_hops):
            if self.G.nodes[current]["kind"] == "exit":
                return path, cost

            nxt = self.get_next(current)

            if nxt is None:
                raise nx.NetworkXNoPath(f"No route from {start} to any exit.")

            if self.G[current][nxt]["blocked"]:
                raise nx.NetworkXNoPath(f"Route from {start} became blocked.")

            cost += float(self.G[current][nxt]["base_weight"])
            current = nxt

            if current in visited:
                raise RuntimeError("Cycle detected in value-field policy.")

            visited.add(current)
            path.append(current)

        raise RuntimeError("Path reconstruction exceeded max_hops.")


class DistributedDiffusionEngine:
    """
    Internal distributed diffusion engine.

    Each sensor $s$ maintains:

    $$
    V_s(x)
    $$

    Local physical update, only if $x \in O_s$:

    $$
    V_s^{k+1}(x)
    \leftarrow
    \min_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V_s^k(y)\right)
    $$

    Communication update:

    $$
    V_s^{k+1}(x)
    \leftarrow
    \min
    \left(
      V_s^{k+1}(x),
      \min_{q \in \mathcal{N}_C(s)} V_q^k(x)
    \right)
    $$

    Important assumption:

    $$
    \bigcup_{s \in S} O_s = V
    $$

    If a movement node is observed by no sensor, no distributed process can
    locally update its value.
    """

    def __init__(self, movement_graph: nx.Graph, communication_graph: nx.Graph, sensors: Dict[str, Sensor]) -> None:
        self.G = movement_graph
        self.C = communication_graph
        self.sensors = sensors
        self.sensor_values: Dict[str, Dict[str, float]] = {}
        self.sensor_next: Dict[str, Dict[str, Optional[str]]] = {}
        self._initialize()

    def _initialize(self) -> None:
        for s in self.sensors:
            self.sensor_values[s] = {}
            self.sensor_next[s] = {}

            for n, attrs in self.G.nodes(data=True):
                self.sensor_values[s][n] = 0.0 if attrs["kind"] == "exit" else float("inf")
                self.sensor_next[s][n] = None

    def validate_sensor_coverage(self) -> None:
        uncovered_nodes = []

        for n in self.G.nodes:
            if not any(n in s.observed_nodes for s in self.sensors.values()):
                uncovered_nodes.append(n)

        if uncovered_nodes:
            print("\nERROR: Incomplete sensor coverage\n")
            print("The following movement nodes are NOT observed by any sensor:\n")
            for n in uncovered_nodes:
                print(f"  - {n}")
            print("\nFix: add sensors or increase sensor range so every node is observed.\n")
            raise RuntimeError("Distributed routing cannot proceed: incomplete sensor coverage.")

    def step(self) -> None:
        new_values = {}
        new_next = {}

        for s_name, sensor in self.sensors.items():                         # for each sensor
            new_values[s_name] = {}
            new_next[s_name] = {}

            observed = set(sensor.observed_nodes)

            for x in self.G.nodes:                                          # for each movement node
                current_val = self.sensor_values[s_name][x]
                best_val = current_val
                best_next = self.sensor_next[s_name][x]

                if x in observed:                                           # if *observed* movement node
                    for y in self.G.neighbors(x):                           # for each adjacent movement node
                        if self.G[x][y]["blocked"]:
                            continue

                        candidate = self.sensor_values[s_name][y] + self.G[x][y]["weight"]

                        if candidate < best_val:
                            best_val = candidate
                            best_next = y

                for s2 in self.C.neighbors(s_name):                         # for each neighboring sensor
                    candidate = self.sensor_values[s2][x]

                    if candidate < best_val:
                        best_val = candidate
                        best_next = self.sensor_next[s2][x]

                new_values[s_name][x] = best_val
                new_next[s_name][x] = best_next

        self.sensor_values = new_values
        self.sensor_next = new_next

    def run(self, steps: int = 40) -> None:
        # Reset is required because this min-only relaxation cannot correct
        # old optimistic values after edge costs increase.
        self._initialize()
        for _ in range(steps):
            self.step()

    def extract_node_field(self) -> Dict[str, float]:
        node_values = {}

        for n in self.G.nodes:
            best = float("inf")

            for s_name, sensor in self.sensors.items():
                if n in sensor.observed_nodes:
                    best = min(best, self.sensor_values[s_name][n])

            node_values[n] = best

        return node_values

    def get_local_policy(self, sensor_name: str) -> Dict[str, Optional[str]]:
        return self.sensor_next[sensor_name]


class DistributedDiffusionStrategy(RoutingStrategy):
    """
    Distributed routing strategy.

    This wraps DistributedDiffusionEngine behind the same RoutingStrategy
    interface used by centralized algorithms.

    It exposes:
    - distributed node value field for heatmap
    - local sensor policy arrows
    - exact path fallback for visual path display

    The exact path fallback is only for viewer convenience. The distributed
    algorithm itself produces local sensor policies, not a single global
    next-pointer field.
    """

    def __init__(
        self,
        movement_graph: nx.Graph,
        communication_graph: nx.Graph,
        sensors: Dict[str, Sensor],
        controller: MovementGraphController,
        steps: int = 40,
    ) -> None:
        super().__init__(movement_graph)
        self.engine = DistributedDiffusionEngine(movement_graph, communication_graph, sensors)
        self.sensors = sensors
        self.controller = controller
        self.steps = steps

    def has_global_policy(self) -> bool:
        return False

    def validate_sensor_coverage(self) -> None:
        self.engine.validate_sensor_coverage()

    def recompute(self) -> None:
        self.engine.run(steps=self.steps)

        node_vals = self.engine.extract_node_field()

        for n in self.G.nodes:
            self.G.nodes[n]["value"] = node_vals[n]
            self.G.nodes[n]["next"] = None

    def get_value(self, node: str) -> float:
        return float(self.G.nodes[node]["value"])

    def get_next(self, node: str) -> Optional[str]:
        return None

    def get_path(self, start: str) -> Tuple[List[str], float]:
        return self.controller.shortest_path_to_nearest_exit(start)

    def sensor_policy_arrows(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        arrows = []

        for s_name, s in self.sensors.items():
            anchor = s.metadata.get("anchor_node")
            if anchor is None:
                continue

            local_policy = self.engine.get_local_policy(s_name)
            nxt = local_policy.get(anchor)

            if nxt is None:
                continue

            if self.G[anchor][nxt]["blocked"]:
                continue

            p0 = np.array(s.position, dtype=float)
            p1 = self.G.nodes[nxt]["position"]

            direction = p1 - p0
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                arrows.append((p0, direction / norm))

        return arrows

    def debug_sensor_line(self, sensor_name: str, sensor: Sensor) -> str:
        anchor = sensor.metadata.get("anchor_node")
        obs_count = len(sensor.observed_nodes)
        nxt = self.engine.get_local_policy(sensor_name).get(anchor) if anchor else None
        return f"  {sensor_name}: anchor={anchor}, next={nxt}, obs={obs_count}, policy=local"


class StrategyManager:
    """
    Runtime registry for routing strategies.

    The viewer owns only this manager. It does not know which concrete
    algorithms exist.

    Add a new routing algorithm by:

    $$
    manager.register("dstar-lite", DStarLiteStrategy(...))
    $$

    without changing InteractiveBuildingViewer.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, RoutingStrategy] = {}
        self._current: Optional[str] = None

    def register(self, name: str, strategy: RoutingStrategy) -> None:
        self._strategies[name] = strategy
        if self._current is None:
            self._current = name

    def names(self) -> List[str]:
        return list(self._strategies.keys())

    def current_name(self) -> str:
        if self._current is None:
            raise RuntimeError("No routing strategy registered.")
        return self._current

    def current(self) -> RoutingStrategy:
        return self._strategies[self.current_name()]

    def set(self, name: str) -> None:
        if name not in self._strategies:
            raise KeyError(f"Unknown strategy: {name}")
        self._current = name

    def next(self) -> None:
        names = self.names()
        if not names:
            raise RuntimeError("No routing strategies registered.")

        idx = names.index(self.current_name())
        self._current = names[(idx + 1) % len(names)]


# ============================================================
# COMMUNICATION ENGINE
# ============================================================

class CommunicationEngine:
    """
    Builds the communication graph from sensor positions and ranges.

    This graph is independent from the movement graph:

    $$
    G_C = (S,L)
    $$

    where an edge exists when two sensors are within mutual communication
    range.
    """

    def __init__(self, sensors: Dict[str, Sensor]) -> None:
        self.sensors = sensors
        self.communication_graph = nx.Graph()
        self.rebuild()

    def rebuild(self) -> None:
        G = nx.Graph()

        for name, s in self.sensors.items():
            G.add_node(
                name,
                position=np.array(s.position, dtype=float),
                range_radius=s.range_radius,
            )

        names = list(self.sensors.keys())

        for i, a in enumerate(names):
            for b in names[i + 1:]:
                sa = self.sensors[a]
                sb = self.sensors[b]

                pa = np.array(sa.position, dtype=float)
                pb = np.array(sb.position, dtype=float)

                d = float(np.linalg.norm(pa - pb))
                allowed = min(sa.range_radius, sb.range_radius)

                if d <= allowed:
                    G.add_edge(a, b, distance=d)

        self.communication_graph = G

    def validate_connectivity(self) -> None:
        """
        Optional distributed assumption check.

        Strong connectivity is not meaningful for this undirected graph;
        ordinary connectivity is the relevant condition:

        $$
        \forall s,q \in S, \exists \text{ communication path } s \leadsto q
        $$

        If the graph is disconnected, distributed information may remain
        trapped in isolated sensor components.
        """
        if self.communication_graph.number_of_nodes() == 0:
            raise RuntimeError("Communication graph has no sensors.")

        if not nx.is_connected(self.communication_graph):
            components = list(nx.connected_components(self.communication_graph))
            print("\nWARNING: Communication graph is disconnected.")
            for i, comp in enumerate(components):
                print(f"  component {i}: {sorted(comp)}")


# ============================================================
# INTERACTIVE VISUALIZATION
# ============================================================

class InteractiveBuildingViewer:
    """
    3D PyVista viewer.

    The viewer is routing-algorithm agnostic.

    It depends on:
    - BuildingGeometry for static geometry
    - MovementGraphController for edge mutation
    - StrategyManager for routing computation
    - CommunicationEngine only for visualizing sensor links

    It does not know which concrete strategy is currently active.
    """

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

        self.plotter = pv.Plotter(window_size=(1450, 900))
        self.plotter.set_background("white")

        self.start_nodes = [n for n, a in self.G.nodes(data=True) if a["kind"] == "room"]
        if not self.start_nodes:
            raise RuntimeError("No room nodes available as start nodes.")

        self.start_index = 0
        self.current_start = self.start_nodes[self.start_index]

        self.edge_list = list(self.G.edges())
        self.selected_edge_index = 0

        self.edge_actors: Dict[Tuple[str, str], Any] = {}
        self.path_actor = None
        self.heatmap_actor = None
        self.policy_arrow_actors: List[Any] = []
        self.sensor_arrow_actors: List[Any] = []

        self.status_actor_name = "status_text"
        self.help_actor_name = "help_text"

        self.selected_node_actor = None
        self.selected_edge_actor = None

        self.pick_mode = "node"

        self.current_path: Optional[List[str]] = None
        self.field_cost: Optional[float] = None
        self.exact_cost: Optional[float] = None
        self.error: Optional[str] = None

    # ----------------------------
    # Strategy access
    # ----------------------------

    def strategy(self) -> RoutingStrategy:
        return self.strategies.current()

    # ----------------------------
    # UI actions
    # ----------------------------

    def next_start(self) -> None:
        self.start_index = (self.start_index + 1) % len(self.start_nodes)
        self.current_start = self.start_nodes[self.start_index]
        self._highlight_selected_node(self.current_start)
        self.refresh_path_only()

    def prev_start(self) -> None:
        self.start_index = (self.start_index - 1) % len(self.start_nodes)
        self.current_start = self.start_nodes[self.start_index]
        self._highlight_selected_node(self.current_start)
        self.refresh_path_only()

    def cycle_edge(self) -> None:
        self.selected_edge_index = (self.selected_edge_index + 1) % len(self.edge_list)
        u, v = self._selected_edge()
        self._highlight_selected_edge(u, v)
        self.redraw_selection_only()

    def toggle_selected_edge(self) -> None:
        u, v = self._selected_edge()
        self.controller.toggle_edge(u, v)
        self._highlight_selected_edge(u, v)
        self.refresh_full()

    def reset_edges(self) -> None:
        self.controller.reset_edges()
        self.refresh_full()

    def next_strategy(self) -> None:
        self.strategies.next()
        print(f"Strategy: {self.strategies.current_name()}")
        self.refresh_full()

    def _set_pick_node_mode(self) -> None:
        self.pick_mode = "node"
        self._update_text()

    def _set_pick_edge_mode(self) -> None:
        self.pick_mode = "edge"
        self._update_text()

    # ----------------------------
    # Checkbox UI
    # ----------------------------

    def _add_button(self, callback, position, label, size=32):
        """
        Checkbox behaving like a push button.

        PyVista installations do not always expose add_button_widget,
        while add_checkbox_button_widget is widely available.
        """
        widget = None

        def _cb(state):
            if state:
                callback()
                if widget is not None:
                    widget.GetRepresentation().SetState(0)

        widget = self.plotter.add_checkbox_button_widget(
            _cb,
            value=False,
            position=position,
            size=size,
            color_on="lightblue",
            color_off="white",
            border_size=1,
        )

        self.plotter.add_text(
            label,
            position=(position[0] + size + 10, position[1] + 5),
            font_size=10,
        )

        return widget
    
    def _add_ui(self) -> None:
        x0 = 10
        y0 = 10
        dy = 45
        num_buttons = 5

        y = y0 + num_buttons * dy

        # self._add_button(self.prev_start, (x0, y), "Prev Start")
        # y -= dy

        # self._add_button(self.next_start, (x0, y), "Next Start")
        # y -= dy

        # self._add_button(self.toggle_selected_edge, (x0, y), "Toggle Edge")
        # y -= dy

        self._add_button(self.reset_edges, (x0, y), "Reset Edges")
        y -= dy

        self._add_button(self.next_strategy, (x0, y), "Next Strategy")
        y -= dy

        # self._add_button(self.cycle_edge, (x0, y), "Next Edge")
        # y -= dy

        self._add_button(self.print_path_info, (x0, y), "Print Path")
        y -= dy

        self._add_button(self._set_pick_node_mode, (x0, y), "Pick: Node")
        y -= dy

        self._add_button(self._set_pick_edge_mode, (x0, y), "Pick: Edge")

    # ----------------------------
    # Picking
    # ----------------------------

    def _on_point_picked(self, point, *args):
        if point is None:
            return

        p = np.array(point, dtype=float)

        if self.pick_mode == "node":
            self._pick_node(p)
        elif self.pick_mode == "edge":
            self._pick_edge(p)

    def _pick_node(self, p: np.ndarray) -> None:
        best_node = None
        best_dist = float("inf")

        for n, attrs in self.G.nodes(data=True):
            d = float(np.linalg.norm(attrs["position"] - p))
            if d < best_dist:
                best_dist = d
                best_node = n

        if best_node is None or best_dist > 3.0:
            return

        if self.G.nodes[best_node]["kind"] != "room":
            return

        self.current_start = best_node

        if best_node in self.start_nodes:
            self.start_index = self.start_nodes.index(best_node)

        self._highlight_selected_node(best_node)
        self.refresh_path_only()

    def _point_to_segment_distance(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            return float(np.linalg.norm(p - a)), a

        t = float(np.dot(p - a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return float(np.linalg.norm(p - proj)), proj

    def _pick_edge(self, p: np.ndarray) -> None:
        best_edge = None
        best_dist = float("inf")

        for u, v in self.G.edges:
            a = self.G.nodes[u]["position"]
            b = self.G.nodes[v]["position"]

            dist, _ = self._point_to_segment_distance(p, a, b)

            if dist < best_dist:
                best_dist = dist
                best_edge = (u, v)

        if best_edge is None or best_dist > 2.5:
            return

        u, v = best_edge

        self.controller.toggle_edge(u, v)

        canonical = self._canonical_edge(u, v)
        canonical_edges = [self._canonical_edge(a, b) for a, b in self.edge_list]
        if canonical in canonical_edges:
            self.selected_edge_index = canonical_edges.index(canonical)

        self._highlight_selected_edge(u, v)
        self.refresh_full()

    # ----------------------------
    # Actor highlights
    # ----------------------------

    def _highlight_selected_node(self, node_id: str) -> None:
        if self.selected_node_actor is not None:
            self.plotter.remove_actor(self.selected_node_actor)
            self.selected_node_actor = None

        pos = self.G.nodes[node_id]["position"]
        sphere = pv.Sphere(radius=0.8, center=pos)

        self.selected_node_actor = self.plotter.add_mesh(
            sphere,
            color="yellow",
            smooth_shading=True,
            pickable=False,
        )

    def _highlight_selected_edge(self, u: str, v: str) -> None:
        if self.selected_edge_actor is not None:
            self.plotter.remove_actor(self.selected_edge_actor)
            self.selected_edge_actor = None

        p0 = self.G.nodes[u]["position"]
        p1 = self.G.nodes[v]["position"]

        self.selected_edge_actor = self.plotter.add_mesh(
            pv.Line(p0, p1),
            color="orange",
            line_width=12,
            pickable=False,
        )

    # ----------------------------
    # Static scene
    # ----------------------------

    def _canonical_edge(self, u: str, v: str) -> Tuple[str, str]:
        return tuple(sorted((u, v)))

    def _selected_edge(self) -> Tuple[str, str]:
        return self.edge_list[self.selected_edge_index]

    def _add_spaces(self) -> None:
        for s in self.geometry.spaces.values():
            x, y, z = s.center
            sx, sy, sz = s.size

            box = pv.Box(bounds=(
                x - sx / 2, x + sx / 2,
                y - sy / 2, y + sy / 2,
                z - sz / 2, z + sz / 2,
            ))

            self.plotter.add_mesh(
                box,
                color=s.color,
                opacity=s.opacity,
                show_edges=True,
                line_width=1,
                pickable=False,
            )

    def _add_movement_graph(self) -> None:
        points = []
        labels = []

        for n, attrs in self.G.nodes(data=True):
            points.append(attrs["position"])
            labels.append(n)

        pts = np.array(points)
        pdata = pv.PolyData(pts)

        self.plotter.add_mesh(
            pdata,
            color="black",
            point_size=15,
            render_points_as_spheres=True,
            pickable=True,
        )

        self.plotter.add_point_labels(
            pts,
            labels,
            font_size=9,
            point_size=0,
            shape_opacity=0.15,
        )

        for u, v in self.G.edges:
            p0 = self.G.nodes[u]["position"]
            p1 = self.G.nodes[v]["position"]
            line = pv.Line(p0, p1)

            actor = self.plotter.add_mesh(
                line,
                color="gray",
                line_width=5,
                pickable=True,
            )
            self.edge_actors[self._canonical_edge(u, v)] = actor

    def _add_sensors_and_communication_graph(self) -> None:
        sensor_points = []
        sensor_labels = []

        for name, s in self.geometry.sensors.items():
            sensor_points.append(s.position)
            sensor_labels.append(name)

        pts = np.array(sensor_points, dtype=float)
        pdata = pv.PolyData(pts)

        self.plotter.add_mesh(
            pdata,
            color="orange",
            point_size=18,
            render_points_as_spheres=True,
            pickable=False,
        )

        self.plotter.add_point_labels(
            pts,
            sensor_labels,
            font_size=8,
            point_size=0,
            shape_opacity=0.12,
        )

        for u, v in self.C.edges:
            p0 = self.C.nodes[u]["position"]
            p1 = self.C.nodes[v]["position"]

            self.plotter.add_mesh(
                pv.Line(p0, p1),
                color="deepskyblue",
                line_width=2,
                opacity=0.35,
                pickable=False,
            )

    def _add_stair_geometry(self) -> None:
        stair_pairs = []

        levels = sorted({attrs["level"] for _, attrs in self.G.nodes(data=True)})

        for level in levels[:-1]:
            stair_pairs.append((f"SE{level}", f"SE{level+1}"))
            stair_pairs.append((f"SW{level}", f"SW{level+1}"))

        for a, b in stair_pairs:
            p0 = self.G.nodes[a]["position"]
            p1 = self.G.nodes[b]["position"]

            tube = pv.Line(p0, p1).tube(radius=0.35)

            self.plotter.add_mesh(
                tube,
                color="purple",
                opacity=0.75,
                pickable=False,
            )

    # ----------------------------
    # Dynamic drawing
    # ----------------------------

    def _draw_policy_arrows(self) -> None:
        for actor in self.policy_arrow_actors:
            self.plotter.remove_actor(actor)

        self.policy_arrow_actors.clear()

        if not self.strategy().has_global_policy():
            return

        for n, attrs in self.G.nodes(data=True):
            nxt = self.strategy().get_next(n)

            if nxt is None:
                continue

            p0 = attrs["position"]
            p1 = self.G.nodes[nxt]["position"]

            direction = p1 - p0
            norm = np.linalg.norm(direction)

            if norm < 1e-9:
                continue

            actor = self.plotter.add_mesh(
                pv.Arrow(start=p0, direction=direction / norm, scale=1.3),
                color="royalblue",
                pickable=False,
            )
            self.policy_arrow_actors.append(actor)

    def _draw_sensor_policy_arrows(self) -> None:
        for actor in self.sensor_arrow_actors:
            self.plotter.remove_actor(actor)

        self.sensor_arrow_actors.clear()

        arrows = self.strategy().sensor_policy_arrows()

        # Centralized strategies usually do not define sensor arrows, so derive
        # them from the global policy using each sensor anchor.
        if not arrows and self.strategy().has_global_policy():
            for s in self.geometry.sensors.values():
                anchor = s.metadata.get("anchor_node")
                if anchor is None:
                    continue

                nxt = self.strategy().get_next(anchor)
                if nxt is None:
                    continue

                if self.G[anchor][nxt]["blocked"]:
                    continue

                p0 = np.array(s.position, dtype=float)
                p1 = self.G.nodes[nxt]["position"]

                direction = p1 - p0
                norm = np.linalg.norm(direction)

                if norm > 1e-9:
                    arrows.append((p0, direction / norm))

        for start, direction in arrows:
            actor = self.plotter.add_mesh(
                pv.Arrow(start=start, direction=direction, scale=1.1),
                color="darkorange",
                pickable=False,
            )
            self.sensor_arrow_actors.append(actor)

    def _draw_heatmap(self) -> None:
        points = []
        values = []

        for n, attrs in self.G.nodes(data=True):
            val = self.strategy().get_value(n)
            points.append(attrs["position"])
            values.append(100.0 if np.isinf(val) else val)

        pdata = pv.PolyData(np.array(points))
        pdata["value"] = np.array(values)

        if self.heatmap_actor is not None:
            self.plotter.remove_actor(self.heatmap_actor)

        self.heatmap_actor = self.plotter.add_mesh(
            pdata,
            scalars="value",
            point_size=30,
            render_points_as_spheres=True,
            cmap="coolwarm",
            show_scalar_bar=True,
            scalar_bar_args={"title": "Routing value"},
            pickable=False,
        )

    def _draw_path(self, path: Optional[List[str]]) -> None:
        if self.path_actor is not None:
            self.plotter.remove_actor(self.path_actor)
            self.path_actor = None

        if not path or len(path) < 2:
            return

        points = np.array([self.G.nodes[n]["position"] for n in path])
        poly = pv.lines_from_points(points)

        self.path_actor = self.plotter.add_mesh(
            poly,
            color="lime",
            line_width=10,
            pickable=False,
        )

    def _update_edge_colors(self, path: Optional[List[str]]) -> None:
        selected = self._canonical_edge(*self._selected_edge())

        path_edges = set()
        if path and len(path) >= 2:
            path_edges = {
                self._canonical_edge(path[i], path[i + 1])
                for i in range(len(path) - 1)
            }

        for edge, actor in self.edge_actors.items():
            u, v = edge
            blocked = self.G[u][v]["blocked"]

            actor.prop.line_width = 5

            if blocked:
                actor.prop.color = (1.0, 0.1, 0.1)
            elif edge in path_edges:
                actor.prop.color = (0.0, 0.9, 0.0)
            else:
                actor.prop.color = (0.55, 0.55, 0.55)

            if edge == selected:
                actor.prop.line_width = 9
                if not blocked and edge not in path_edges:
                    actor.prop.color = (1.0, 0.65, 0.0)

    # ----------------------------
    # Text/debug
    # ----------------------------

    def _sensor_debug_text(self, max_lines: int = 6) -> List[str]:
        lines = ["Sensor debug:"]

        for i, (s_name, s) in enumerate(self.geometry.sensors.items()):
            if i >= max_lines:
                remaining = len(self.geometry.sensors) - max_lines
                lines.append(f"  ... {remaining} more sensors")
                break

            custom_line = self.strategy().debug_sensor_line(s_name, s)
            if custom_line:
                lines.append(custom_line)
                continue

            anchor = s.metadata.get("anchor_node")
            obs_count = len(s.observed_nodes)
            nxt = self.strategy().get_next(anchor) if anchor else None

            lines.append(
                f"  {s_name}: anchor={anchor}, next={nxt}, obs={obs_count}, policy=global"
            )

        return lines

    def _status_text(
        self,
        path: Optional[List[str]],
        field_cost: Optional[float],
        exact_cost: Optional[float],
        error: Optional[str],
    ) -> str:
        u, v = self._selected_edge()
        blocked = self.G[u][v]["blocked"]

        value = self.strategy().get_value(self.current_start)
        value_text = "inf" if np.isinf(value) else f"{value:.2f}"

        lines = [
            f"Start: {self.current_start}",
            f"Pick mode: {self.pick_mode}",
            f"Selected movement edge: ({u}, {v}) | blocked={blocked}",
            f"Value at start: {value_text}",
            f"Communication nodes: {self.C.number_of_nodes()} | links: {self.C.number_of_edges()}",
            f"Strategy: {self.strategies.current_name()}",
        ]

        lines.extend(self._sensor_debug_text())

        if error:
            lines.append(f"Status: {error}")
        else:
            lines.append(f"Field-follow cost: {field_cost:.2f}")
            lines.append(f"Exact shortest cost: {exact_cost:.2f}")
            lines.append("Path: " + " -> ".join(path or []))

        return "\n".join(lines)

    def _help_text(self) -> str:
        return (
            "UI: buttons only | Pick Node selects start | Pick Edge toggles block | "
            "Next Strategy cycles algorithms"
        )

    def _update_text(self) -> None:
        self.plotter.remove_actor(self.status_actor_name, render=False)
        self.plotter.remove_actor(self.help_actor_name, render=False)

        self.plotter.add_text(
            self._status_text(
                self.current_path,
                self.field_cost,
                self.exact_cost,
                self.error,
            ),
            position="upper_left",
            font_size=10,
            name=self.status_actor_name,
        )

        self.plotter.add_text(
            self._help_text(),
            position="lower_left",
            font_size=10,
            name=self.help_actor_name,
        )

    # ----------------------------
    # Recompute/redraw
    # ----------------------------

    def recompute_routing(self) -> None:
        self.strategy().recompute()
        self._recompute_path()

    def _recompute_path(self) -> None:
        self.current_path = None
        self.field_cost = None
        self.exact_cost = None
        self.error = None

        try:
            self.current_path, self.field_cost = self.strategy().get_path(self.current_start)
            _, self.exact_cost = self.controller.shortest_path_to_nearest_exit(self.current_start)
        except Exception as exc:
            self.error = str(exc)

    def refresh_full(self) -> None:
        self.recompute_routing()
        self.redraw_full()

    def refresh_path_only(self) -> None:
        self._recompute_path()
        self._redraw_path_only()

    def redraw_full(self) -> None:
        self._draw_path(self.current_path)
        self._draw_heatmap()
        self._draw_policy_arrows()
        self._draw_sensor_policy_arrows()
        self._update_edge_colors(self.current_path)
        self._update_text()
        self.plotter.render()

    def _redraw_path_only(self) -> None:
        self._draw_path(self.current_path)
        self._update_edge_colors(self.current_path)
        self._update_text()
        self.plotter.render()

    def redraw_selection_only(self) -> None:
        self._update_edge_colors(self.current_path)
        self._update_text()
        self.plotter.render()

    # ----------------------------
    # Debug prints
    # ----------------------------

    def print_path_info(self) -> None:
        try:
            path, cost = self.strategy().get_path(self.current_start)
            exact_path, exact_cost = self.controller.shortest_path_to_nearest_exit(self.current_start)

            print("\n=== Routing Info ===")
            print(f"Strategy   : {self.strategies.current_name()}")
            print(f"Start      : {self.current_start}")
            print(f"Field path : {' -> '.join(path)}")
            print(f"Field cost : {cost:.2f}")
            print(f"Exact path : {' -> '.join(exact_path)}")
            print(f"Exact cost : {exact_cost:.2f}")
            print()
        except Exception as exc:
            print(f"\nRouting error: {exc}\n")

    # ----------------------------
    # Scene lifecycle
    # ----------------------------

    def build_scene(self) -> None:
        self._add_spaces()
        self._add_stair_geometry()
        self._add_movement_graph()
        self._add_sensors_and_communication_graph()
        self._add_ui()

        self.plotter.show_grid()

        # Surface point picking works better in a dense 3D scene.
        # Decorative geometry is marked pickable=False so picking targets
        # graph nodes/edges instead of room/corridor volumes.
        self.plotter.enable_surface_point_picking(
            callback=self._on_point_picked,
            show_point=True,
            clear_on_no_selection=True,
            font_size=0,
        )

        self.refresh_full()
        self._highlight_selected_node(self.current_start)
        u, v = self._selected_edge()
        self._highlight_selected_edge(u, v)

    def show(self) -> None:
        self.build_scene()
        self.plotter.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    NUM_FLOORS = 2

    geometry = BuildingGeometry.demo_building(num_floors=NUM_FLOORS)

    controller = MovementGraphController(geometry.movement_graph)
    communication = CommunicationEngine(geometry.sensors)

    # Structural checks are run at setup time.
    # Re-run reachability after topology changes if you decide to enforce
    # hard failures instead of allowing "no path" states during interaction.
    controller.validate_reachability()

    manager = StrategyManager()

    manager.register(
        "centralized-diffusion",
        CentralizedDiffusionStrategy(
            geometry.movement_graph,
            controller,
            steps=40,
        ),
    )

    distributed = DistributedDiffusionStrategy(
        geometry.movement_graph,
        communication.communication_graph,
        geometry.sensors,
        controller,
        steps=40,
    )
    distributed.validate_sensor_coverage()
    communication.validate_connectivity()

    manager.register("distributed-diffusion", distributed)

    viewer = InteractiveBuildingViewer(
        geometry=geometry,
        controller=controller,
        strategy_manager=manager,
        communication=communication,
    )

    viewer.show()


if __name__ == "__main__":
    main()
