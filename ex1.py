import utils
import search
from collections import deque
from math import ceil, inf

id = ["217398338"] 


# ------------------------------
# Helpers for internal state format
# ------------------------------

def pack_state(game):
    walls = frozenset(game["Walls"])
    taps = tuple(sorted(game["Taps"].items()))
    plants = tuple(sorted(game["Plants"].items()))
    robots = tuple(sorted((rid, tuple(vals)) for rid, vals in game["Robots"].items()))
    size = game["Size"]
    return (walls, taps, plants, robots, size)


def unpack_state(state):
    walls, taps, plants, robots, size = state
    taps = dict(taps)
    plants = dict(plants)
    robots = {rid: tuple(v) for rid, v in robots}
    return walls, taps, plants, robots, size


# ------------------------------------------------------------
# BFS shortest distances (respecting walls)
# ------------------------------------------------------------

def bfs_from(start, walls, N, M):
    """Return dict (r,c) → shortest-path distance from start."""
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


# ------------------------------------------------------------
# WateringProblem
# ------------------------------------------------------------

class WateringProblem(search.Problem):

    def __init__(self, initial):
        self.initial = pack_state(initial)

        # Extract static map information for BFS precomputation
        walls, taps, plants, robots, size = unpack_state(self.initial)
        N, M = size

        self.N = N
        self.M = M
        self.walls = walls

        # Precompute BFS distances FROM:
        #  - each tap
        #  - each robot start location
        self.bfs_from_tap = {}
        for tpos in taps.keys():
            self.bfs_from_tap[tpos] = bfs_from(tpos, walls, N, M)

        self.bfs_from_robot = {}
        for rid, (r, c, load, cap) in robots.items():
            self.bfs_from_robot[rid] = bfs_from((r, c), walls, N, M)

        # store robot capacities
        self.robot_caps = {rid: cap for rid, (_, _, _, cap) in robots.items()}


    # --------------------------------------------------------
    # GOAL TEST
    # --------------------------------------------------------
    def goal_test(self, state):
        _, _, plants, _, _ = state
        return all(req == 0 for (_, req) in plants)


    # --------------------------------------------------------
    # SUCCESSOR (optimized + completeness-safe pruning)
    # --------------------------------------------------------
    def successor(self, state):
        walls, taps, plants, robots, size = unpack_state(state)
        N, M = size

        total_need = sum(plants.values())
        if total_need == 0:
            return []

        successors = []

        tap_cells = set(taps.keys())
        plant_cells = {p for p, need in plants.items() if need > 0}
        occupied = {(r, c) for rid, (r, c, _, _) in robots.items()}

        moves = {
            "UP":    (-1, 0),
            "DOWN":  (1, 0),
            "LEFT":  (0, -1),
            "RIGHT": (0, 1)
        }

        for rid, (r, c, load, cap) in robots.items():

            # POUR
            if (r, c) in plant_cells and load > 0:
                new_plants = dict(plants)
                new_robots = dict(robots)

                new_plants[(r, c)] -= 1
                new_robots[rid] = (r, c, load - 1, cap)

                successors.append((
                    f"POUR{{{rid}}}",
                    (walls,
                     tuple(sorted(taps.items())),
                     tuple(sorted(new_plants.items())),
                     tuple(sorted(new_robots.items())),
                     size)
                ))

            # LOAD
            if (r, c) in tap_cells and load < cap and taps[(r, c)] > 0 and total_need > 0:
                new_taps = dict(taps)
                new_robots = dict(robots)

                new_taps[(r, c)] -= 1
                new_robots[rid] = (r, c, load + 1, cap)

                successors.append((
                    f"LOAD{{{rid}}}",
                    (walls,
                     tuple(sorted(new_taps.items())),
                     tuple(sorted(plants.items())),
                     tuple(sorted(new_robots.items())),
                     size)
                ))

            # MOVEMENT with safe pruning
            if load > 0 and plant_cells:
                targets = plant_cells
            elif load == 0 and tap_cells:
                targets = tap_cells
            else:
                targets = plant_cells.union(tap_cells)

            # Best Manhattan target dist
            best_d = min(abs(r - tr) + abs(c - tc) for tr, tc in targets)

            for name, (dr, dc) in moves.items():
                nr, nc = r+dr, c+dc

                if not (0 <= nr < N and 0 <= nc < M):
                    continue
                if (nr, nc) in walls:
                    continue
                if (nr, nc) in occupied and (nr, nc) != (r, c):
                    continue

                # completeness-safe pruning:
                # allow moves that do not worsen Manhattan distance by >1
                d_new = min(abs(nr - tr) + abs(nc - tc) for tr, tc in targets)
                if d_new > best_d + 1:
                    continue

                new_robots = dict(robots)
                new_robots[rid] = (nr, nc, load, cap)

                successors.append((
                    f"{name}{{{rid}}}",
                    (walls,
                     tuple(sorted(taps.items())),
                     tuple(sorted(plants.items())),
                     tuple(sorted(new_robots.items())),
                     size)
                ))

        return successors


    # --------------------------------------------------------
    # A* HEURISTIC (Option A + RC2 + A2)
    # --------------------------------------------------------
    def h_astar(self, node):
        state = node.state
        walls, taps, plants, robots, size = unpack_state(state)

        total_need = sum(plants.values())
        if total_need == 0:
            return 0

        h = 0

        tap_positions = list(taps.keys())
        plant_list = [(p, need) for p, need in plants.items() if need > 0]

        # For each plant, compute minimum possible cost across all robots
        for (pr, pc), need in plant_list:
            best_lb = inf

            # For each tap, we need distance tap → plant
            for tpos in tap_positions:
                if (pr, pc) not in self.bfs_from_tap[tpos]:
                    continue  # unreachable plant (dead instance)
                d_tap_plant = self.bfs_from_tap[tpos][(pr, pc)]

                # For each robot, capacity matters
                for rid, (rr, rc, load, cap) in robots.items():

                    # shortest path robot → tap
                    robot_dist_map = self.bfs_from_robot[rid]
                    if tpos not in robot_dist_map:
                        continue  # unreachable tap

                    d_robot_tap = robot_dist_map[tpos]

                    # number of trips needed
                    trips = ceil(need / cap)

                    # minimal trip cost (lower bound):
                    # robot→tap + tap→plant
                    # LOAD/POUR cost at least = need
                    lb = (d_robot_tap + d_tap_plant) * trips + need

                    if lb < best_lb:
                        best_lb = lb

            h += best_lb

        return h


    # --------------------------------------------------------
    # GBFS HEURISTIC (fast + effective)
    # --------------------------------------------------------
    def h_gbfs(self, node):
        state = node.state
        walls, taps, plants, robots, size = unpack_state(state)

        total_need = sum(plants.values())
        if total_need == 0:
            return 0

        score = total_need * 15  # heavily prioritize reducing need

        tap_positions = list(taps.keys())
        plant_positions = [p for p, n in plants.items() if n > 0]

        best = inf

        for rid, (r, c, load, cap) in robots.items():
            dist_map = self.bfs_from_robot[rid]

            if load > 0:
                for p in plant_positions:
                    if p in dist_map:
                        best = min(best, dist_map[p])
            else:
                for t in tap_positions:
                    if t in dist_map:
                        best = min(best, dist_map[t])

        return score + (best if best < inf else 0)


# ------------------------------------------------------------
# Factory function
# ------------------------------------------------------------
def create_watering_problem(game):
    return WateringProblem(game)
