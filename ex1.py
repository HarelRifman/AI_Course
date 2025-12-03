import utils
import search
from collections import deque, defaultdict
from math import inf, ceil
import heapq

id = ["217398338"]

############################################################
# STATE PACKING / UNPACKING
############################################################

def pack_state(game):
    walls = frozenset(game["Walls"])
    taps = tuple(sorted(game["Taps"].items()))
    plants = tuple(sorted(game["Plants"].items()))
    robots = tuple(sorted((rid, tuple(v)) for rid, v in game["Robots"].items()))
    size = game["Size"]
    return (walls, taps, plants, robots, size)

def unpack_state(state):
    walls, taps, plants, robots, size = state
    taps = dict(taps)
    plants = dict(plants)
    robots = {rid: tuple(v) for rid, v in robots}
    return walls, taps, plants, robots, size

############################################################
# BFS UTILITIES
############################################################

def bfs_from(start, walls, N, M):
    """Return dict: cell -> shortest-path distance from start (ignoring taps/plants)."""
    q = deque([start])
    dist = {start: 0}
    blocked = walls

    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < N and 0 <= nc < M and (nr, nc) not in blocked:
                if (nr, nc) not in dist:
                    dist[(nr, nc)] = d + 1
                    q.append((nr, nc))
    return dist


############################################################
# MINIMUM SPANNING TREE (Prim)
############################################################

def mst_cost(nodes, dist_lookup):
    """
    Compute MST over given nodes using Prim's algorithm.

    nodes: list of node identifiers.
    dist_lookup(u,v) returns edge weight between u,v.
    """
    if not nodes:
        return 0
    used = set()
    start = nodes[0]
    used.add(start)
    edges = []
    total = 0

    # push edges from start
    for v in nodes[1:]:
        w = dist_lookup(start, v)
        heapq.heappush(edges, (w, start, v))

    while edges and len(used) < len(nodes):
        w, u, v = heapq.heappop(edges)
        if v in used:
            continue
        used.add(v)
        total += w
        # add new edges from v
        for x in nodes:
            if x not in used:
                wx = dist_lookup(v, x)
                heapq.heappush(edges, (wx, v, x))

    return total


############################################################
# MAIN WATERING PROBLEM
############################################################

class WateringProblem(search.Problem):

    def __init__(self, initial):
        self.initial = pack_state(initial)

        walls, taps, plants, robots, size = unpack_state(self.initial)
        N, M = size
        self.walls = walls
        self.N, self.M = N, M

        self.taps = taps
        self.robot_caps = {rid: cap for rid, (_, _, _, cap) in robots.items()}

        # Precompute BFS FROM EACH PLANT
        self.bfs_from_plants = {}
        for (pr, pc), need in plants.items():
            self.bfs_from_plants[(pr, pc)] = bfs_from((pr, pc), walls, N, M)

        # Precompute BFS FROM EACH TAP
        self.bfs_from_taps = {}
        for tpos, amt in taps.items():
            self.bfs_from_taps[tpos] = bfs_from(tpos, walls, N, M)
    ############################################################
    # A* HEURISTIC (IDO + KFIR/OMER COMBINED)
    ############################################################
    def h_astar(self, node):
        """
        Admissible heuristic combining:
        - Kfir/Omer N-minimum distance mixture
        - Ido's MST lower bound (plants + robots)
        - Tap distance lower bounds
        - Water operation lower bounds
        """

        state = node.state
        walls, taps, plants, robots, size = unpack_state(state)

        # --- QUICK EXIT: goal reached
        total_need = sum(plants.values())
        if total_need == 0:
            return 0

        # --- Collect useful structures
        plant_positions = [(pr, pc) for (pr, pc), need in plants.items() if need > 0]
        robot_positions = {rid: (r, c) for rid, (r, c, load, cap) in robots.items()}
        robot_loads      = [load for (_, (_, _, load, _)) in robots.items()]
        plant_needs      = [need for (_, need) in plants.items() if need > 0]

        # --- Precomputed BFS maps
        bfsP = self.bfs_from_plants
        bfsT = self.bfs_from_taps

        ############################################################
        # 1) Kfir/Omer: N smallest combined distances
        #    N = total_need
        ############################################################
        n = total_need
        combo_costs = []

        # A) robot → plant distances
        for rid, (rr, rc) in robot_positions.items():
            for (pr, pc) in plant_positions:
                dist = bfsP[(pr, pc)].get((rr, rc), inf)
                if dist < inf:
                    combo_costs.append(dist)

        # B) plant → plant distances
        for i in range(len(plant_positions)):
            pi = plant_positions[i]
            for j in range(i + 1, len(plant_positions)):
                pj = plant_positions[j]
                dist = bfsP[pi].get(pj, inf)
                if dist < inf:
                    combo_costs.append(dist)

        combo_costs.sort()
        if len(combo_costs) < n:
            # insufficient connectivity → unreachable configuration
            return inf

        kfir_omer_term = sum(combo_costs[:n])

        ############################################################
        # 2) Pour/load lower bound:
        #    minimal operations needed:
        #    LOAD + POUR for each missing unit = 2 * total_need
        #    subtract what robots already carry
        ############################################################
        op_lower_bound = 2 * total_need - sum(robot_loads)

        ############################################################
        # 3) Tap → plant minimal distance:
        #    All plants must receive water from some tap.
        ############################################################
        min_plant_tap = inf
        for (pr, pc) in plant_positions:
            for tpos in taps.keys():
                d = bfsT[tpos].get((pr, pc), inf)
                if d < min_plant_tap:
                    min_plant_tap = d

        if min_plant_tap == inf:
            return inf

        ############################################################
        # 4) min(robot→tap, plant→tap) term
        ############################################################
        min_robot_tap = inf
        min_ptap = min_plant_tap  # already computed

        # robot → tap
        for rid, (rr, rc) in robot_positions.items():
            for tpos in taps.keys():
                d = bfsT[tpos].get((rr, rc), inf)
                if d < min_robot_tap:
                    min_robot_tap = d

        # take minimum of these
        rt_pt_tap_term = min(min_robot_tap, min_ptap)

        ############################################################
        # 5) Add plant→tap minimal again (as described)
        ############################################################
        extra_tap_term = min_plant_tap

        ############################################################
        # 6) MST over plants + robots (IDO's MST component)
        ############################################################

        # nodes: represent each plant as ("P", (pr,pc)), each robot as ("R", rid)
        nodes = []
        for (pr, pc) in plant_positions:
            nodes.append(("P", (pr, pc)))

        for rid in robot_positions:
            nodes.append(("R", rid))

        # Define distance lookup for MST:
        def dist_lookup(a, b):
            kind1, data1 = a
            kind2, data2 = b

            # plant - plant
            if kind1 == "P" and kind2 == "P":
                p1 = data1
                p2 = data2
                return bfsP[p1].get(p2, inf)

            # robot - robot: weight = 0
            if kind1 == "R" and kind2 == "R":
                return 0

            # robot - plant
            if kind1 == "R" and kind2 == "P":
                rid = data1
                pr, pc = data2
                rr, rc = robot_positions[rid]
                return bfsP[(pr, pc)].get((rr, rc), inf)

            if kind1 == "P" and kind2 == "R":
                rid = data2
                pr, pc = data1
                rr, rc = robot_positions[rid]
                return bfsP[(pr, pc)].get((rr, rc), inf)

            return inf

        mst_val = mst_cost(nodes, dist_lookup)
        if mst_val == inf:
            return inf

        ############################################################
        # Final combined heuristic:
        ############################################################
        h = (
            kfir_omer_term +
            op_lower_bound +
            min_plant_tap +
            rt_pt_tap_term +
            extra_tap_term +
            mst_val
        )

        return h
    ############################################################
    # GOAL TEST
    ############################################################
    def goal_test(self, state):
        walls, taps, plants, robots, size = unpack_state(state)
        return all(need == 0 for (_, need) in plants.items())

    ############################################################
    # SUCCESSOR FUNCTION
    ############################################################
    def successor(self, state):
        walls, taps, plants, robots, size = unpack_state(state)
        N, M = size

        total_need = sum(plants.values())
        total_water = sum(taps.values())

        # If impossible globally, no successors
        if total_need > total_water:
            return []

        successors = []
        occupied = {(r, c) for (r, (r, c, load, cap)) in robots.items()}
        tap_cells = set(taps.keys())
        plant_cells = {p for (p, need) in plants.items() if need > 0}

        # Movement directions
        moves = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }

        for rid, (r, c, load, cap) in robots.items():

            ########################################################
            # 1) POUR
            ########################################################
            if load > 0 and (r, c) in plant_cells:
                new_plants = dict(plants)
                new_plants[(r, c)] -= 1

                new_robots = dict(robots)
                new_robots[rid] = (r, c, load - 1, cap)

                successors.append((
                    f"POUR{{{rid}}}",
                    (
                        walls,
                        tuple(sorted(taps.items())),
                        tuple(sorted(new_plants.items())),
                        tuple(sorted(new_robots.items())),
                        size
                    )
                ))

            ########################################################
            # 2) LOAD
            ########################################################
            if (r, c) in tap_cells and load < cap and taps[(r, c)] > 0:
                new_taps = dict(taps)
                new_taps[(r, c)] -= 1

                new_robots = dict(robots)
                new_robots[rid] = (r, c, load + 1, cap)

                successors.append((
                    f"LOAD{{{rid}}}",
                    (
                        walls,
                        tuple(sorted(new_taps.items())),
                        tuple(sorted(plants.items())),
                        tuple(sorted(new_robots.items())),
                        size
                    )
                ))

            ########################################################
            # 3) MOVES (NO PRUNING – MUST maintain completeness)
            ########################################################
            for name, (dr, dc) in moves.items():
                nr, nc = r + dr, c + dc

                if not (0 <= nr < N and 0 <= nc < M):
                    continue
                if (nr, nc) in walls:
                    continue
                if (nr, nc) in occupied and (nr, nc) != (r, c):
                    continue

                new_robots = dict(robots)
                new_robots[rid] = (nr, nc, load, cap)

                successors.append((
                    f"{name}{{{rid}}}",
                    (
                        walls,
                        tuple(sorted(taps.items())),
                        tuple(sorted(plants.items())),
                        tuple(sorted(new_robots.items())),
                        size
                    )
                ))

        return successors


############################################################
# FACTORY
############################################################

def create_watering_problem(game):
    return WateringProblem(game)
