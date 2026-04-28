"""
============================================================
SMART BUILDING NAVIGATION SIMULATION
============================================================

This simulation models evacuation routing in a building using:

1. MOVEMENT GRAPH (networkx.Graph)
   - Nodes = physical positions (rooms, junctions, stairs)
   - Edges = walkable paths
   - Used for computing distances and valid paths

2. COMMUNICATION GRAPH (networkx.Graph)
   - Nodes = sensors (cameras, controllers)
   - Edges = wireless communication links (range-based)
   - Used for distributed information exchange

3. SENSORS
   - Each sensor observes ZERO, ONE, or MULTIPLE movement nodes
   - Sensors can:
        - observe local state (if node in range)
        - relay information (even if observing nothing)

------------------------------------------------------------

ROUTING MODES
------------------------------------------------------------

CENTRALIZED:
    - Classic diffusion (Bellman-Ford) on movement graph
    - Gives exact shortest paths
    - Sensors simply "read" global solution

DISTRIBUTED:
    - Each sensor maintains local estimates of node values
    - Sensors exchange information via communication graph
    - Only observed nodes are updated locally
    - Converges to global solution (if graph connected)

------------------------------------------------------------

VISUALIZATION
------------------------------------------------------------

- Black nodes: movement graph
- Blue arrows: global routing policy (centralized)
- Orange arrows: sensor decisions
    - centralized mode: read global solution
    - distributed mode: computed locally

- Heatmap:
    - node values (distance-to-exit)

------------------------------------------------------------

KEY IDEA
------------------------------------------------------------

Movement graph = "physics"
Communication graph = "information flow"
Sensors = "distributed decision makers"

============================================================
"""

from __future__ import annotations

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
    Owns the physical/semantic model:
    - spaces: rooms, corridors, floors, stairs
    - movement_graph: where people can move
    - sensors: communication devices independent from movement nodes
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
        Assign observed_nodes automatically.

        Also assigns one anchor_node per sensor:
        - nearest movement node on the same floor
        - arrows are based on this anchor_node, not on arbitrary best observed node
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
    def demo_building(cls, num_floors: int = 3) -> "BuildingGeometry":
        b = cls()

        floor_height = 6.0
        base_z = 1.5
        zs = [base_z + i * floor_height for i in range(num_floors)]

        floor_size = (44, 32, 0.3)

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
                ("A", ( -11,   5, z)),
                ("B", (  11,   5, z)),
                ("C", (   0,  15, z)),
                ("D", (  11,  -5, z)),
                ("E", ( -11,  -5, z)),
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

            # center of rooms
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

        # In-floor alternative paths
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

        # optional top-floor bridge (kept behavior but generic)
        # top = num_floors - 1
        # b.add_movement_edge(f"SE{top}", f"SW{top}", weight=28.0)

        # Sensors are NOT movement nodes.
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
# ROUTING ENGINE
# ============================================================

class RoutingEngine:
    """
    Centralized routing engine on the movement graph.

    The movement graph is:

    $$
    G_M = (V, A)
    $$

    where:

    - $V$ is the set of movement nodes: rooms, junctions, stairs, exits
    - $A$ is the set of walkable edges
    - each edge $(x,y)$ has a traversal cost $c(x,y) > 0$

    The goal is to compute, for every node $x$, the shortest distance to
    the nearest exit.

    Let $E \subseteq V$ be the set of exit nodes.

    We define a value function:

    $$
    V(x) = \text{distance from node } x \text{ to the closest exit}
    $$

    Exit nodes are boundary conditions:

    $$
    V(x)=0 \quad \forall x \in E
    $$

    All other nodes are initialized to infinity:

    $$
    V(x)=+\infty \quad \forall x \notin E
    $$

    The diffusion update is the Bellman optimality update:

    $$
    V_{k+1}(x)=
    \begin{cases}
    0, & x \in E \\
    \min\limits_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V_k(y)\right),
    & x \notin E
    \end{cases}
    $$

    where $\mathcal{N}(x)$ is the set of neighbors of $x$.

    Blocked edges are excluded from the minimization.

    The routing policy is:

    $$
    \pi(x)=
    \arg\min\limits_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V(y)\right)
    $$

    The policy tells us which neighbor should be followed from node $x$.

    In centralized mode, this algorithm has access to the full movement graph,
    so after enough iterations it converges to the shortest-path distance field.
    """

    def __init__(self, movement_graph: nx.Graph) -> None:
        # TODO: Add assumption check: reachability, connectivity of movement graph.
        
        self.G = movement_graph

    def validate_reachability(self) -> None:
        """
        Ensure every node can reach at least one exit.

        Required condition:

        $$
        \forall x \in V, \exists \text{ path } x \to e, \quad e \in E
        $$

        If not, routing values will remain infinite.
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
            print("\n❌ ERROR: Unreachable nodes detected\n")

            for n in unreachable:
                print(f"  - {n}")

            print("\n💡 Fix: ensure connectivity to at least one exit.\n")

            raise RuntimeError("Centralized routing cannot proceed: unreachable nodes.")

    def set_edge_blocked(self, u: str, v: str, blocked: bool = True) -> None:
        self.G[u][v]["blocked"] = blocked
        self.G[u][v]["weight"] = 1e9 if blocked else self.G[u][v]["base_weight"]

    def toggle_edge(self, u: str, v: str) -> None:
        self.set_edge_blocked(u, v, not self.G[u][v]["blocked"])

    def reset_edges(self) -> None:
        for u, v in self.G.edges:
            self.set_edge_blocked(u, v, False)

    def initialize_values(self) -> None:
        for _, attrs in self.G.nodes(data=True):
            attrs["value"] = 0.0 if attrs["kind"] == "exit" else float("inf")
            attrs["next"] = None

    def diffusion_step(self) -> None:
        """
        Perform one synchronous Bellman diffusion step.

        For each node $x$, compute:

        $$
        V_{k+1}(x)=
        \min_{y \in \mathcal{N}(x)}
        \left(c(x,y)+V_k(y)\right)
        $$

        Exits remain fixed at value zero:

        $$
        V_{k+1}(x)=0 \quad \text{if } x \in E
        $$

        The update is synchronous:

        - all new values are computed from the old values
        - only after all nodes are processed, values are committed

        This avoids order-dependent behavior.
        """
        new_values = {}
        new_next = {}

        for x, attrs in self.G.nodes(data=True):
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

    def run_diffusion(self, steps: int = 30) -> None:
        self.initialize_values()
        for _ in range(steps):
            self.diffusion_step()

    def greedy_path_from_field(self, start: str, max_hops: int = 100) -> Tuple[List[str], float]:
        path = [start]
        visited = {start}
        current = start
        cost = 0.0

        for _ in range(max_hops):
            if self.G.nodes[current]["kind"] == "exit":
                return path, cost

            nxt = self.G.nodes[current]["next"]

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

class DistributedRoutingEngine:
    """
    Distributed routing engine.

    In centralized routing, one algorithm sees the whole movement graph.
    In distributed routing, each sensor stores its own local estimate of the
    value function.

    Each sensor $s$ maintains:

    $$
    V_s(x)
    $$

    which means:

    $$
    \text{sensor } s \text{'s estimate of the distance from node } x
    \text{ to the closest exit}
    $$

    Sensors are connected by a communication graph:

    $$
    G_C = (S, L)
    $$

    where:

    - $S$ is the set of sensors
    - $L$ is the set of communication links

    Each sensor can do two things:

    1. Local physical update, only for movement nodes it observes.
    2. Communication update, by receiving estimates from neighboring sensors.

    Let $O_s \subseteq V$ be the movement nodes observed by sensor $s$.

    If $x \in O_s$, sensor $s$ can apply the local Bellman update:

    $$
    V_s^{k+1}(x)
    =
    \min_{y \in \mathcal{N}(x)}
    \left(c(x,y)+V_s^k(y)\right)
    $$

    Communication lets sensor $s$ receive better estimates from neighboring
    sensors $q \in \mathcal{N}_C(s)$:

    $$
    V_s^{k+1}(x)
    =
    \min
    \left(
        V_s^{k+1}(x),
        \min_{q \in \mathcal{N}_C(s)} V_q^k(x)
    \right)
    $$

    Therefore, information about exits diffuses through the communication graph,
    while movement feasibility is still defined by the movement graph.

    Conceptually:

    - movement graph = physical walkability
    - communication graph = information propagation
    - sensor estimates = distributed approximation of the global value field
    """
    def __init__(
        self,
        movement_graph: nx.Graph,
        communication_graph: nx.Graph,
        sensors: Dict[str, Sensor],
    ) -> None:
        # TODO: Add assumption check: all movement nodes are checked by at least one sensor.
        # ~ validate_sensor_coverage

        self.G = movement_graph
        self.C = communication_graph
        self.sensors = sensors

        # sensor -> node -> value
        self.sensor_values: Dict[str, Dict[str, float]] = {}

        # sensor -> node -> next
        self.sensor_next: Dict[str, Dict[str, Optional[str]]] = {}

        self._initialize()

    def _initialize(self) -> None:
        for s in self.sensors:
            self.sensor_values[s] = {}
            self.sensor_next[s] = {}

            for n, attrs in self.G.nodes(data=True):
                if attrs["kind"] == "exit":
                    self.sensor_values[s][n] = 0.0
                else:
                    self.sensor_values[s][n] = float("inf")

                self.sensor_next[s][n] = None

    def step(self) -> None:
        """
        Perform one distributed routing step.

        Each sensor $s$ maintains a local value estimate:

        $$
        V_s(x)
        $$

        For every movement node $x$, the sensor first keeps its current value:

        $$
        V_s^{k+1}(x) \leftarrow V_s^k(x)
        $$

        If the sensor observes $x$, it performs a local Bellman update:

        $$
        V_s^{k+1}(x)
        \leftarrow
        \min_{y \in \mathcal{N}(x)}
        \left(c(x,y)+V_s^k(y)\right)
        $$

        Then it exchanges information with neighboring sensors in the communication
        graph:

        $$
        V_s^{k+1}(x)
        \leftarrow
        \min
        \left(
            V_s^{k+1}(x),
            \min_{q \in \mathcal{N}_C(s)} V_q^k(x)
        \right)
        $$

        Blocked movement edges are ignored in the local Bellman update.

        This is a distributed approximation of the centralized value diffusion.
        """
        new_values = {}
        new_next = {}

        for s_name, sensor in self.sensors.items():
            new_values[s_name] = {}
            new_next[s_name] = {}

            observed = set(sensor.observed_nodes)

            for x in self.G.nodes:
                current_val = self.sensor_values[s_name][x]
                best_val = current_val
                best_next = self.sensor_next[s_name][x]

                # if best_val == float("inf") and current_val == float("inf"):
                #     new_values[s_name][x] = current_val
                #     new_next[s_name][x] = best_next
                #     continue

                # --- local movement update ONLY if sensor sees node ---
                if x in observed:
                    for y in self.G.neighbors(x):
                        if self.G[x][y]["blocked"]:
                            continue

                        candidate = self.sensor_values[s_name][y] + self.G[x][y]["weight"]

                        if candidate < best_val:
                            best_val = candidate
                            best_next = y

                # --- communication update (always allowed) ---
                for s2 in self.C.neighbors(s_name):
                    candidate = self.sensor_values[s2][x]

                    if candidate < best_val:
                        best_val = candidate
                        best_next = self.sensor_next[s2][x]

                new_values[s_name][x] = best_val
                new_next[s_name][x] = best_next

        self.sensor_values = new_values
        self.sensor_next = new_next

    def validate_sensor_coverage(self) -> None:
        """
        Validate that every movement node is observed by at least one sensor.

        This is required for distributed routing to work correctly.

        Let:
        $$
        O_s \subseteq V
        $$
        be the set of nodes observed by sensor $s$.

        The required condition is:
        $$
        \bigcup_{s \in S} O_s = V
        $$

        If this condition is not satisfied, then some nodes cannot update their
        routing values because no sensor has visibility on them.

        In that case, the algorithm is invalid and we abort execution.
        """

        uncovered_nodes = []

        for n in self.G.nodes:
            observed = False

            for s in self.sensors.values():
                if n in s.observed_nodes:
                    observed = True
                    break

            if not observed:
                uncovered_nodes.append(n)

        if uncovered_nodes:
            print("\n❌ ERROR: Incomplete sensor coverage\n")
            print("The following movement nodes are NOT observed by any sensor:\n")

            for n in uncovered_nodes:
                print(f"  - {n}")

            print("\n💡 Fix: add sensors or increase sensor range so that every node is observed.\n")

            raise RuntimeError("Distributed routing cannot proceed: incomplete sensor coverage.")

    def run(self, steps: int = 30) -> None:
        # Questo è necessario perché quando blocchi/sblocchi archi, i valori devono poter anche aumentare.
        # Con l’algoritmo attuale solo-min, senza reset rimangono stime vecchie.
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

# ============================================================
# COMMUNICATION ENGINE
# ============================================================

class CommunicationEngine:
    """
    Builds a communication graph from sensor positions and communication range.

    This graph is independent from the movement graph:
    - nodes are sensors/controllers/lights
    - edges exist if devices are within mutual range
    """

    def __init__(self, sensors: Dict[str, Sensor]) -> None:
        # TODO: Add assumption check: strong connectivity of the sensors --> global reachability and info
        self.sensors = sensors
        self.communication_graph = nx.Graph()
        self.rebuild()

    def rebuild(self) -> None:
        G = nx.Graph()

        for name, s in self.sensors.items():
            G.add_node(
                name,
                position=np.array(s.position, dtype=float),
                range_radius=s.range_radius
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

    def sensor_policy_arrows(self, movement_graph: nx.Graph) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        One arrow per sensor using GLOBAL centralized policy.

        The sensor arrow is based on the sensor's anchor movement node,
        not on the best arbitrary observed node.
        """
        arrows = []

        for s in self.sensors.values():
            anchor = s.metadata.get("anchor_node")

            if anchor is None:
                continue

            nxt = movement_graph.nodes[anchor].get("next")

            if nxt is None:
                continue

            if movement_graph[anchor][nxt]["blocked"]:
                continue

            p0 = np.array(s.position, dtype=float)
            p1 = movement_graph.nodes[nxt]["position"]

            direction = p1 - p0
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                arrows.append((p0, direction / norm))

        return arrows
    
    def sensor_policy_arrows_local(self, distributed) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        One arrow per sensor using LOCAL distributed policy.

        The arrow is based on the sensor's anchor movement node.
        """
        arrows = []
        G = distributed.G

        for s_name, s in self.sensors.items():
            anchor = s.metadata.get("anchor_node")

            if anchor is None:
                continue

            local_policy = distributed.get_local_policy(s_name)
            nxt = local_policy.get(anchor)

            if nxt is None:
                continue

            if G[anchor][nxt]["blocked"]:
                continue

            p0 = np.array(s.position, dtype=float)
            p1 = G.nodes[nxt]["position"]

            direction = p1 - p0
            norm = np.linalg.norm(direction)

            if norm > 1e-9:
                arrows.append((p0, direction / norm))

        return arrows

# ============================================================
# INTERACTIVE VISUALIZATION
# ============================================================

class InteractiveBuildingViewer:
    def __init__(
        self,
        geometry: BuildingGeometry,
        routing: RoutingEngine,
        distributed_routing: DistributedRoutingEngine,
        communication: CommunicationEngine,
    ) -> None:
        self.geometry = geometry
        self.communication = communication
        self.routing = routing
        self.distributed = distributed_routing
        self.use_distributed = False
        

        self.G = geometry.movement_graph
        self.C = communication.communication_graph

        self.plotter = pv.Plotter(window_size=(1450, 900))
        self.plotter.set_background("white")

        self.start_nodes = [
            n for n, a in self.G.nodes(data=True)
            if a["kind"] == "room"
        ]

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
        self.pick_mode = "node"  # or "edge"

    def next_start(self) -> None:
        if not self.start_nodes:
            return

        self.start_index = (self.start_index + 1) % len(self.start_nodes)
        self.current_start = self.start_nodes[self.start_index]

        self.refresh_path_only()


    def prev_start(self) -> None:
        if not self.start_nodes:
            return

        self.start_index = (self.start_index - 1) % len(self.start_nodes)
        self.current_start = self.start_nodes[self.start_index]

        self.refresh_path_only()

    def _add_button(self, callback, position, label, size=32):
        """
        Checkbox behaving like a push-button + label
        """
        widget = None  # closure handle

        def _cb(state):
            if state:
                callback()
                # reset immediately → behave like button
                if widget is not None:
                    rep = widget.GetRepresentation()
                    rep.SetState(0)

        widget = self.plotter.add_checkbox_button_widget(
            _cb,
            value=False,
            position=position,
            size=size,
            color_on="lightblue",
            color_off="white",
            border_size=1,
        )

        # label
        self.plotter.add_text(
            label,
            position=(position[0] + size + 10, position[1] + 5),
            font_size=10,
        )

        return widget

    def _set_pick_node_mode(self):
        self.pick_mode = "node"

    def _set_pick_edge_mode(self):
        self.pick_mode = "edge"

    def _highlight_selected_node(self, node_id: str) -> None:
        # remove previous actor
        if self.selected_node_actor is not None:
            self.plotter.remove_actor(self.selected_node_actor)
            self.selected_node_actor = None

        pos = self.G.nodes[node_id]["position"]

        sphere = pv.Sphere(radius=0.8, center=pos)

        self.selected_node_actor = self.plotter.add_mesh(
            sphere,
            color="yellow",
            smooth_shading=True,
        )

    def _highlight_selected_edge(self, u, v):
        if self.selected_edge_actor is not None:
            self.plotter.remove_actor(self.selected_edge_actor)
            self.selected_edge_actor = None

        p0 = self.G.nodes[u]["position"]
        p1 = self.G.nodes[v]["position"]

        line = pv.Line(p0, p1)

        self.selected_edge_actor = self.plotter.add_mesh(
            line,
            color="orange",
            line_width=12,
        )

    def _on_point_picked(self, point, *args):
        if point is None:
            return

        p = np.array(point, dtype=float)

        if self.pick_mode == "node":
            self._pick_node(p)
        elif self.pick_mode == "edge":
            self._pick_edge(p)

    def _pick_node(self, p):
        best_node = None
        best_dist = float("inf")

        for n, attrs in self.G.nodes(data=True):
            d = np.linalg.norm(attrs["position"] - p)
            if d < best_dist:
                best_dist = d
                best_node = n

        # threshold (IMPORTANT)
        if best_dist > 3.0:
            return

        if self.G.nodes[best_node]["kind"] != "room":
            return

        self.current_start = best_node

        if best_node in self.start_nodes:
            self.start_index = self.start_nodes.index(best_node)

        self._highlight_selected_node(best_node)
        self.refresh_path_only()

    def _point_to_segment_distance(self, p, a, b):
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj), proj

    def _pick_edge(self, p):
        best_edge = None
        best_dist = float("inf")

        for u, v in self.G.edges:
            a = self.G.nodes[u]["position"]
            b = self.G.nodes[v]["position"]

            dist, proj = self._point_to_segment_distance(p, a, b)

            if dist < best_dist:
                best_dist = dist
                best_edge = (u, v)

        if best_edge is None:
            return

        # threshold to avoid accidental clicks
        if best_dist > 2.0:
            return

        u, v = best_edge

        # toggle block
        self.routing.toggle_edge(u, v)

        # update selection index too
        canonical = self._canonical_edge(u, v)
        if canonical in self.edge_list:
            self.selected_edge_index = self.edge_list.index(canonical)

        # recompute FULL (topology changed)
        self._highlight_selected_edge(u, v)
        self.refresh_full()

    def _add_ui(self) -> None:
        """
        Clean vertical control panel (no keyboard)
        """

        x0 = 10
        y0 = 10
        dy = 45
        num_buttons = 9

        y = y0 + num_buttons * dy

        # --- START NAVIGATION ---
        self._add_button(self.prev_start, (x0, y), "Prev Start")
        y -= dy

        self._add_button(self.next_start, (x0, y), "Next Start")
        y -= dy

        # --- EDGE CONTROL ---
        self._add_button(self.toggle_selected_edge, (x0, y), "Toggle Edge")
        y -= dy

        self._add_button(self.reset_edges, (x0, y), "Reset Edges")
        y -= dy

        # --- ROUTING MODE ---
        def _toggle_distributed_wrapper():
            self.toggle_distributed()

        self._add_button(_toggle_distributed_wrapper, (x0, y), "Toggle Distributed")
        y -= dy

        # --- GRAPH NAVIGATION ---
        self._add_button(self.cycle_edge, (x0, y), "Next Edge")
        y -= dy

        # --- DEBUG ---
        self._add_button(self.print_path_info, (x0, y), "Print Path")
        y -= dy

        # --- PICK MODE ---
        self._add_button(self._set_pick_node_mode, (x0, y), "Pick: Node")
        y -= dy

        self._add_button(self._set_pick_edge_mode, (x0, y), "Pick: Edge")
        y -= dy

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
            )

    def _add_movement_graph(self) -> None:
        self.node_positions = []
        self.node_ids = []
        
        points = []
        labels = []

        for n, attrs in self.G.nodes(data=True):
            pos = attrs["position"]
            points.append(pos)
            labels.append(n)

            self.node_positions.append(pos)
            self.node_ids.append(n)

        pts = np.array(points)
        pdata = pv.PolyData(pts)

        self.node_mesh = pdata  # ← IMPORTANT

        self.plotter.add_mesh(
            pdata,
            color="black",
            point_size=15,
            render_points_as_spheres=True,
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

            actor = self.plotter.add_mesh(line, color="gray", line_width=5)
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
            line = pv.Line(p0, p1)

            self.plotter.add_mesh(
                line,
                color="deepskyblue",
                line_width=2,
                opacity=0.35,
            )

    def _add_stair_geometry(self) -> None:
        stair_pairs = []

        levels = sorted({
            attrs["level"] for _, attrs in self.G.nodes(data=True)
        })

        for level in levels[:-1]:
            stair_pairs.append((f"SE{level}", f"SE{level+1}"))
            stair_pairs.append((f"SW{level}", f"SW{level+1}"))

        for a, b in stair_pairs:
            p0 = self.G.nodes[a]["position"]
            p1 = self.G.nodes[b]["position"]

            line = pv.Line(p0, p1)
            tube = line.tube(radius=0.35)

            self.plotter.add_mesh(
                tube,
                color="purple",
                opacity=0.75,
            )

    def _draw_policy_arrows(self) -> None:
        for actor in self.policy_arrow_actors:
            self.plotter.remove_actor(actor)

        self.policy_arrow_actors.clear()

        for n, attrs in self.G.nodes(data=True):
            nxt = attrs.get("next")

            if nxt is None:
                continue

            p0 = attrs["position"]
            p1 = self.G.nodes[nxt]["position"]

            direction = p1 - p0
            norm = np.linalg.norm(direction)

            if norm < 1e-9:
                continue

            arrow = pv.Arrow(
                start=p0,
                direction=direction / norm,
                scale=1.3,
            )

            actor = self.plotter.add_mesh(arrow, color="royalblue")
            self.policy_arrow_actors.append(actor)

    def _draw_sensor_policy_arrows(self) -> None:
        for actor in self.sensor_arrow_actors:
            self.plotter.remove_actor(actor)

        self.sensor_arrow_actors.clear()

        for start, direction in self.communication.sensor_policy_arrows(self.G):
            arrow = pv.Arrow(start=start, direction=direction, scale=1.1)
            actor = self.plotter.add_mesh(arrow, color="darkorange")
            self.sensor_arrow_actors.append(actor)

    def _draw_sensor_policy_arrows_local(self) -> None:
        for actor in self.sensor_arrow_actors:
            self.plotter.remove_actor(actor)

        self.sensor_arrow_actors.clear()

        arrows = self.communication.sensor_policy_arrows_local(self.distributed)

        for start, direction in arrows:
            arrow = pv.Arrow(start=start, direction=direction, scale=1.1)
            actor = self.plotter.add_mesh(arrow, color="darkorange")
            self.sensor_arrow_actors.append(actor)

    def _draw_heatmap(self) -> None:
        points = []
        values = []

        for _, attrs in self.G.nodes(data=True):
            val = attrs["value"]
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

    def _status_text(
        self,
        path: Optional[List[str]],
        field_cost: Optional[float],
        exact_cost: Optional[float],
        error: Optional[str],
    ) -> str:
        u, v = self._selected_edge()
        blocked = self.G[u][v]["blocked"]
        value = self.G.nodes[self.current_start]["value"]
        value_text = "inf" if np.isinf(value) else f"{value:.2f}"

        lines = [
            f"Start: {self.current_start}",
            f"Selected movement edge: ({u}, {v}) | blocked={blocked}",
            f"Value at start: {value_text}",
            f"Communication nodes: {self.C.number_of_nodes()} | links: {self.C.number_of_edges()}",
            f"Mode: {'DISTRIBUTED' if self.use_distributed else 'CENTRALIZED'}"
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
            "Keys: 1..6 select start | n next edge | b block/unblock | "
            "d distributed | r reset | p print path | h help"
        )
    def _sensor_debug_text(self, max_lines: int = 6) -> List[str]:
        lines = ["Sensor debug:"]

        for i, (s_name, s) in enumerate(self.geometry.sensors.items()):
            if i >= max_lines:
                remaining = len(self.geometry.sensors) - max_lines
                lines.append(f"  ... {remaining} more sensors")
                break

            anchor = s.metadata.get("anchor_node")
            obs_count = len(s.observed_nodes)

            if self.use_distributed:
                nxt = self.distributed.get_local_policy(s_name).get(anchor) if anchor else None
                mode = "local"
            else:
                nxt = self.G.nodes[anchor].get("next") if anchor else None
                mode = "global"

            lines.append(
                f"  {s_name}: anchor={anchor}, next={nxt}, obs={obs_count}, policy={mode}"
            )

        return lines

    def recompute_routing(self) -> None:
        if not self.use_distributed:
            self.routing.run_diffusion(steps=40)
        else:
            self.distributed.run(steps=40)

            node_vals = self.distributed.extract_node_field()

            for n in self.G.nodes:
                self.G.nodes[n]["value"] = node_vals[n]
                self.G.nodes[n]["next"] = None

        # AFTER global recompute → compute path once
        self._recompute_path()

    def _clear_policy_arrows(self) -> None:
        for actor in self.policy_arrow_actors:
            self.plotter.remove_actor(actor)
        self.policy_arrow_actors.clear()

    def redraw_full(self) -> None:
        self._draw_path(self.current_path)
        self._draw_heatmap()

        if not self.use_distributed:
            # global arrows
            self._draw_policy_arrows()
            self._draw_sensor_policy_arrows()
        else:
            # local arrows only
            self._clear_policy_arrows()
            self._draw_sensor_policy_arrows_local()

        self._update_edge_colors(self.current_path)
        self._update_text()
        self.plotter.render()

    def redraw_selection_only(self) -> None:
        # Only edge highlighting changes
        self._update_edge_colors(self.current_path)

        # Only status text needs update (selected edge info changed)
        self._update_text()

        self.plotter.render()

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

    def refresh_full(self) -> None:
        """
        Recompute EVERYTHING (used when graph changes)
        """
        self.recompute_routing()
        self.redraw_full()


    def _recompute_path(self) -> None:
        self.current_path = None
        self.field_cost = None
        self.exact_cost = None
        self.error = None

        try:
            if not self.use_distributed:
                self.current_path, self.field_cost = \
                    self.routing.greedy_path_from_field(self.current_start)
            else:
                # distributed: no global next → use exact path
                self.current_path, self.field_cost = \
                    self.routing.shortest_path_to_nearest_exit(self.current_start)

            _, self.exact_cost = \
                self.routing.shortest_path_to_nearest_exit(self.current_start)

        except Exception as exc:
            self.error = str(exc)

    def refresh_path_only(self) -> None:
        """
        Only recompute path from current start (cheap)
        """
        self._recompute_path()
        self.redraw_full()

    def select_start(self, index: int) -> None:
        self.current_start = self.start_nodes[index]
        self.refresh_path_only()

    def cycle_edge(self) -> None:
        self.selected_edge_index = (self.selected_edge_index + 1) % len(self.edge_list)
        self.redraw_selection_only()

    def toggle_selected_edge(self) -> None:
        u, v = self._selected_edge()
        self.routing.toggle_edge(u, v)
        self.refresh_full()

    def toggle_distributed(self) -> None:
        self.use_distributed = not self.use_distributed
        print(f"Distributed mode: {self.use_distributed}")
        self.refresh_full()

    def reset_edges(self) -> None:
        self.routing.reset_edges()
        self.refresh_full()

    def print_path_info(self) -> None:
        try:
            self.routing.run_diffusion(steps=40)
            path, cost = self.routing.greedy_path_from_field(self.current_start)
            exact_path, exact_cost = self.routing.shortest_path_to_nearest_exit(self.current_start)

            print("\n=== Routing Info ===")
            print(f"Start: {self.current_start}")
            print(f"Field path : {' -> '.join(path)}")
            print(f"Field cost : {cost:.2f}")
            print(f"Exact path : {' -> '.join(exact_path)}")
            print(f"Exact cost : {exact_cost:.2f}")
            print()
        except Exception as exc:
            print(f"\nRouting error: {exc}\n")

    def print_help(self) -> None:
        print("\n=== Controls ===")
        print("1..6 : select start node")
        print("n    : select next movement edge")
        print("d    : toggle distributed computing")
        print("b    : block/unblock selected movement edge")
        print("r    : reset all blocked edges")
        print("p    : print routing info")
        print("h    : print this help")
        print()

    def build_scene(self) -> None:
        self._add_spaces()
        self._add_stair_geometry()
        self._add_movement_graph()

        self._add_sensors_and_communication_graph()
        self._add_ui()
        self.plotter.show_grid()

        # self.plotter.enable_point_picking(
        #     callback=self._on_point_picked,
        #     use_mesh=True,
        #     show_message=False,
        #     pickable_window=False,
        # )
        self.plotter.enable_surface_point_picking(
            callback=self._on_point_picked,
            show_point=True,
            clear_on_no_selection=True,
            font_size=0
        )

        self.refresh_full()
        self._highlight_selected_node(self.current_start)

    def show(self) -> None:
        self.build_scene()
        self.print_help()
        self.plotter.show()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    geometry = BuildingGeometry.demo_building(num_floors=1)

    communication = CommunicationEngine(geometry.sensors)
    
    routing = RoutingEngine(geometry.movement_graph)
    distributed_routing = DistributedRoutingEngine(
        geometry.movement_graph,
        communication.communication_graph,
        geometry.sensors,
    )

    viewer = InteractiveBuildingViewer(
        geometry=geometry,
        routing=routing,
        distributed_routing=distributed_routing,
        communication=communication,
    )

    viewer.show()


if __name__ == "__main__":
    main()