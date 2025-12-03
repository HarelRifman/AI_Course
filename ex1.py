import search
import random
import math
import itertools

ids = ["111111111", "222222222"]

class WateringProblem(search.Problem):
    """
    This class implements the Multi-Tap Plant Watering problem.
    """

    def __init__(self, initial):
        """
        Optimized Constructor.
        Pre-computes distances using a flattened matrix for O(1) lookup.
        """
        self.rows, self.cols = initial["Size"]
        self.walls = frozenset(initial["Walls"])
        self.num_cells = self.rows * self.cols
        
        # Static capacities: {robot_id: capacity}
        self.capacities = {r_id: val[3] for r_id, val in initial["Robots"].items()}
        
        # Flattening helper: (r, c) -> index
        self._to_idx = lambda r, c: r * self.cols + c
        
        # 1. Parse Initial State
        # Robots: (id, r, c, load) - Sorted by ID for canonical state
        sorted_robots = tuple(sorted(
            [(r_id, r[0], r[1], r[2]) for r_id, r in initial["Robots"].items()]
        ))
        
        # Plants: (r, c, need) - Sorted by coords
        sorted_plants = tuple(sorted(
            [(p[0], p[1], amt) for p, amt in initial["Plants"].items() if amt > 0]
        ))
        
        # Taps: (r, c, amount) - Sorted by coords
        sorted_taps = tuple(sorted(
            [(t[0], t[1], amt) for t, amt in initial["Taps"].items()]
        ))

        initial_state = (sorted_robots, sorted_plants, sorted_taps)
        search.Problem.__init__(self, initial_state)

        # 2. Pre-compute All-Pairs Shortest Paths (APSP)
        # We store this in a flat 2D array: dist_matrix[idx1][idx2]
        self.dist_matrix = self._compute_apsp()
        
        # Pre-cache tap indices for heuristic (corresponds to sorted_taps order)
        self.tap_indices = [self._to_idx(t[0], t[1]) for t in sorted_taps]

    def _compute_apsp(self):
        """
        Runs BFS from every valid cell to generate a complete distance matrix.
        Returns a list of lists (size N*M x N*M).
        """
        inf = float('inf')
        matrix = [[inf] * self.num_cells for _ in range(self.num_cells)]

        # Identify valid cells (not walls)
        valid_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in self.walls:
                    valid_cells.append((r, c))

        for start_r, start_c in valid_cells:
            start_idx = self._to_idx(start_r, start_c)
            matrix[start_idx][start_idx] = 0
            
            queue = [(start_r, start_c, 0)]
            visited = { (start_r, start_c) }
            
            idx = 0
            while idx < len(queue):
                r, c, dist = queue[idx]
                idx += 1
                curr_idx = self._to_idx(r, c)
                matrix[start_idx][curr_idx] = dist
                
                # Neighbors
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in self.walls and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc, dist + 1))
        return matrix

    def successor(self, state):
        """
        Generates successor states with Conditional Pruning.
        """
        robots, plants, taps = state
        
        # Quick lookup sets
        occupied = {(r[1], r[2]) for r in robots}
        plant_map = {(p[0], p[1]): i for i, p in enumerate(plants)}
        tap_map = {(t[0], t[1]): i for i, t in enumerate(taps)}
        
        successors = []
        
        for i, r in enumerate(robots):
            r_id, r_r, r_c, r_load = r
            
            # --- Conditional Pruning Logic ---
            # Check if this robot is "crowded" (any other robot is adjacent).
            # If NOT crowded, and we can perform a useful action (Pour/Load),
            # we do ONLY that action (prune moves).
            # If crowded, we allow moves to let the robot step aside.
            
            is_crowded = False
            for j, other in enumerate(robots):
                if i != j:
                    # Manhattan distance <= 1 means adjacent or same cell (impossible)
                    dist = abs(r_r - other[1]) + abs(r_c - other[2])
                    if dist <= 1:
                        is_crowded = True
                        break
            
            action_performed = False

            # 1. Try POUR
            if (r_r, r_c) in plant_map:
                p_idx = plant_map[(r_r, r_c)]
                p_r, p_c, p_need = plants[p_idx]
                if r_load > 0 and p_need > 0:
                    action_performed = True
                    new_robots = list(robots)
                    new_robots[i] = (r_id, r_r, r_c, r_load - 1)
                    
                    new_plants = list(plants)
                    if p_need - 1 == 0:
                        new_plants.pop(p_idx) 
                    else:
                        new_plants[p_idx] = (p_r, p_c, p_need - 1)
                    
                    successors.append(
                        (f"POUR{{{r_id}}}", (tuple(new_robots), tuple(new_plants), taps))
                    )
            
            # 2. Try LOAD
            elif (r_r, r_c) in tap_map: # Else-if: usually can't be on plant AND tap
                t_idx = tap_map[(r_r, r_c)]
                t_r, t_c, t_amt = taps[t_idx]
                r_cap = self.capacities[r_id]
                
                if t_amt > 0 and r_load < r_cap:
                    action_performed = True
                    new_robots = list(robots)
                    new_robots[i] = (r_id, r_r, r_c, r_load + 1)
                    
                    new_taps = list(taps)
                    if t_amt - 1 == 0:
                         new_taps[t_idx] = (t_r, t_c, 0)
                    else:
                         new_taps[t_idx] = (t_r, t_c, t_amt - 1)

                    successors.append(
                        (f"LOAD{{{r_id}}}", (tuple(new_robots), plants, tuple(new_taps)))
                    )

            # 3. Move Actions
            # If we acted and are NOT crowded, we skip moving (Pruning).
            # If we didn't act, OR we are crowded, we generate moves.
            if not (action_performed and not is_crowded):
                for name, dr, dc in [("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)]:
                    nr, nc = r_r + dr, r_c + dc
                    
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        if (nr, nc) not in self.walls:
                            if (nr, nc) not in occupied: 
                                new_robots = list(robots)
                                new_robots[i] = (r_id, nr, nc, r_load)
                                successors.append(
                                    (f"{name}{{{r_id}}}", (tuple(new_robots), plants, taps))
                                )

        return successors

    def goal_test(self, state):
        return len(state[1]) == 0

    def h_astar(self, node):
        """
        Admissible Heuristic: Resource Cost + MST Travel Cost + Tap Penalty
        """
        state = node.state
        robots, plants, taps = state
        
        if not plants:
            return 0
        
        # 1. Resource Costs
        total_need = 0
        for p in plants:
            total_need += p[2]
            
        total_carried = 0
        for r in robots:
            total_carried += r[3]
            
        h_pour = total_need
        deficit = max(0, total_need - total_carried)
        h_load = deficit
        
        # 2. MST (Travel Cost)
        # Nodes: 0 (Fleet), 1..N (Plants)
        
        plant_indices = [self._to_idx(p[0], p[1]) for p in plants]
        num_plants = len(plants)
        num_nodes = num_plants + 1
        
        min_dists = [float('inf')] * num_nodes
        visited = [False] * num_nodes
        
        # Init Fleet -> Plants distances
        # dist(Fleet, P) = min(dist(r, P) for r in robots)
        min_dists[0] = 0
        
        robot_indices = [self._to_idx(r[1], r[2]) for r in robots]
        
        for i in range(num_plants):
            p_idx = plant_indices[i]
            best_r_dist = float('inf')
            for r_idx in robot_indices:
                d = self.dist_matrix[r_idx][p_idx]
                if d < best_r_dist:
                    best_r_dist = d
            min_dists[i+1] = best_r_dist

        mst_weight = 0
        
        # Prim's Algorithm
        for _ in range(num_nodes):
            u = -1
            min_val = float('inf')
            
            # Simple linear scan is fast enough for small N
            for n in range(num_nodes):
                if not visited[n] and min_dists[n] < min_val:
                    min_val = min_dists[n]
                    u = n
            
            if u == -1: 
                break
                
            visited[u] = True
            mst_weight += min_val
            
            # Update neighbors (Only Plant->Plant)
            if u > 0:
                u_p_idx = plant_indices[u-1]
                for v in range(1, num_nodes):
                    if not visited[v]:
                        v_p_idx = plant_indices[v-1]
                        d = self.dist_matrix[u_p_idx][v_p_idx]
                        if d < min_dists[v]:
                            min_dists[v] = d

        # 3. Tap Travel Penalty
        h_tap = 0
        if deficit > 0:
            min_p_t = float('inf')
            # Check only active taps
            # Using pre-computed tap_indices + dynamic check
            for i, t in enumerate(taps):
                if t[2] > 0: # Tap has water
                    t_idx = self.tap_indices[i]
                    for p_idx in plant_indices:
                        d = self.dist_matrix[p_idx][t_idx]
                        if d < min_p_t:
                            min_p_t = d
            
            if min_p_t != float('inf'):
                h_tap = min_p_t
                    
        return h_pour + h_load + mst_weight + h_tap

    def h_gbfs(self, node):
        return self.h_astar(node)

def create_watering_problem(game):
    return WateringProblem(game)