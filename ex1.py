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
        Constructor.
        Parses the dictionary 'initial' into a hashable state representation.
        Pre-computes distances to save time during heuristic calculation.
        """
        # 1. Store Static Info (Walls, Grid Size, Capacities)
        self.rows, self.cols = initial["Size"]
        self.walls = frozenset(initial["Walls"])
        
        # Robot capacities are static, store them in a dict {id: cap}
        self.capacities = {r_id: val[3] for r_id, val in initial["Robots"].items()}
        
        # 2. Convert Dynamic Info to Hashable State
        # State format: ( (robots...), (plants...), (taps...) )
        # Robots: tuple of (id, x, y, load) - Sorted by ID
        # Plants: tuple of (x, y, need)     - Sorted by coords
        # Taps:   tuple of (x, y, amount)   - Sorted by coords
        
        sorted_robots = sorted(
            [(r_id, r[0], r[1], r[2]) for r_id, r in initial["Robots"].items()]
        )
        
        sorted_plants = sorted(
            [(loc[0], loc[1], amt) for loc, amt in initial["Plants"].items() if amt > 0]
        )
        
        sorted_taps = sorted(
            [(loc[0], loc[1], amt) for loc, amt in initial["Taps"].items()]
        )

        initial_state = (tuple(sorted_robots), tuple(sorted_plants), tuple(sorted_taps))
        
        search.Problem.__init__(self, initial_state)

        # 3. Pre-compute Distances (All-Pairs Shortest Paths from key points)
        # We need distances from every Tap and every Plant location to every other cell.
        # This is used heavily in the heuristic.
        self.dist_map = {} # Key: (r,c), Value: dict{(r2,c2): dist}
        
        # Collect all "interesting" locations (Starts of robots, Plants, Taps)
        # Actually, since robots move, we need general connectivity. 
        # But for the heuristic, we specifically need Dist(Plant, Plant), Dist(Robot, Plant), Dist(Plant, Tap).
        # We will run BFS from every Plant and Tap location.
        
        targets_to_map = set()
        for p in initial["Plants"]: targets_to_map.add(p)
        for t in initial["Taps"]: targets_to_map.add(t)
        # We also need distances from initial robot locations for the first H calc, 
        # but robots move, so we might need a map for *all* cells if we want to be perfect.
        # However, for the transcript heuristic, we usually measure from Robot -> Plant.
        # To be safe and fast given the grid size (usually small), let's map from all Plants and Taps.
        # When calculating H, if a robot is at an arbitrary cell, we might need BFS? 
        # No, that's too slow per node.
        # Optimization: Just map from all valid cells to all valid cells? 
        # Given 30s limit, we compute APSP (All Pairs Shortest Path) using BFS from every non-wall cell.
        # If grid is large (e.g. 10x10), 100 BFS runs is fine (takes ~0.5s).
        
        self.apsp = {} # (r1,c1) -> { (r2,c2): dist }
        
        non_wall_cells = []
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c) not in self.walls:
                    non_wall_cells.append((r,c))
        
        # Run BFS for every cell (APSP)
        for start_node in non_wall_cells:
            self.apsp[start_node] = self._bfs(start_node)

    def _bfs(self, start):
        """ Runs BFS from start to all reachable cells. Returns dict {cell: dist} """
        q = [(start, 0)]
        visited = {start: 0}
        idx = 0
        while idx < len(q):
            curr, dist = q[idx]
            idx += 1
            
            r, c = curr
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if (nr, nc) not in self.walls:
                        if (nr, nc) not in visited:
                            visited[(nr, nc)] = dist + 1
                            q.append(((nr, nc), dist + 1))
        return visited

    def successor(self, state):
        """
        Generates successor states.
        Format: [(action_string, next_state), ...]
        """
        robots, plants, taps = state
        
        # Convert lists for easier lookup
        # Occupied set for collision detection
        occupied = set()
        for r in robots: occupied.add((r[1], r[2]))
        for w in self.walls: occupied.add(w) # Walls are already in self.walls, but good for check
        
        # Maps for quick access
        plant_map = {(p[0], p[1]): i for i, p in enumerate(plants)}
        tap_map = {(t[0], t[1]): i for i, t in enumerate(taps)}
        
        successors = []
        
        # Iterate over each robot to generate its moves
        for i, r in enumerate(robots):
            r_id, r_r, r_c, r_load = r
            
            # 1. Move Actions (UP, DOWN, LEFT, RIGHT)
            moves = [("UP", -1, 0), ("DOWN", 1, 0), ("LEFT", 0, -1), ("RIGHT", 0, 1)]
            for name, dr, dc in moves:
                nr, nc = r_r + dr, r_c + dc
                
                # Check bounds
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    # Check collisions (Walls or other Robots)
                    # Note: We check against 'occupied'. 'occupied' contains current robot positions.
                    # A robot CANNOT move into a cell occupied by another robot.
                    if (nr, nc) not in self.walls:
                        is_blocked = False
                        for other_j, other_r in enumerate(robots):
                            if i != other_j:
                                if other_r[1] == nr and other_r[2] == nc:
                                    is_blocked = True
                                    break
                        
                        if not is_blocked:
                            # Valid Move
                            new_robots = list(robots)
                            new_robots[i] = (r_id, nr, nc, r_load)
                            
                            # Optimization: Sort robots to maintain canonical state? 
                            # Transcript says IDs must match. The order in 'robots' tuple is sorted by ID.
                            # So just replacing index i preserves order.
                            
                            successors.append(
                                (f"{name}{{{r_id}}}", (tuple(new_robots), plants, taps))
                            )

            # 2. LOAD Action
            # Preconditions: At tap location, tap has water, robot < capacity
            if (r_r, r_c) in tap_map:
                tap_idx = tap_map[(r_r, r_c)]
                t_r, t_c, t_amt = taps[tap_idx]
                r_cap = self.capacities[r_id]
                
                if t_amt > 0 and r_load < r_cap:
                    new_robots = list(robots)
                    new_robots[i] = (r_id, r_r, r_c, r_load + 1) # Load 1 unit
                    
                    new_taps = list(taps)
                    new_taps[tap_idx] = (t_r, t_c, t_amt - 1) # Reduce tap
                    
                    successors.append(
                        (f"LOAD{{{r_id}}}", (tuple(new_robots), plants, tuple(new_taps)))
                    )

            # 3. POUR Action
            # Preconditions: At plant location, robot has water, plant needs water
            if (r_r, r_c) in plant_map:
                p_idx = plant_map[(r_r, r_c)]
                p_r, p_c, p_need = plants[p_idx]
                
                if r_load > 0 and p_need > 0:
                    new_robots = list(robots)
                    new_robots[i] = (r_id, r_r, r_c, r_load - 1) # Pour 1 unit
                    
                    new_plants = list(plants)
                    if p_need - 1 == 0:
                        # Plant done, remove from tuple? 
                        # Or keep with 0? Keeping with 0 makes index matching easier but state larger.
                        # Ideally remove to reduce state size for hashing.
                        new_plants.pop(p_idx)
                    else:
                        new_plants[p_idx] = (p_r, p_c, p_need - 1)
                    
                    successors.append(
                        (f"POUR{{{r_id}}}", (tuple(new_robots), tuple(new_plants), taps))
                    )

        return successors

    def goal_test(self, state):
        """
        Goal is reached when all plants have 0 need.
        Since we remove plants with 0 need in successor, empty plants tuple = Goal.
        """
        _, plants, _ = state
        return len(plants) == 0

    def h_astar(self, node):
        """
        Admissible Heuristic.
        Based on the transcript ("Ido's Heuristic" + "Kfir's Heuristic"):
        1. Resource Cost: We definitely need 1 action to POUR for every missing unit.
        2. Load Cost: If total water needed > total water carried, we need (Need - Carried) LOAD actions.
        3. MST (Minimum Spanning Tree):
           Create a graph of all Active Plants + Virtual "Robot Fleet".
           Edges (P-P): Real grid distance.
           Edges (Fleet-P): Min grid distance from ANY robot to P.
           MST Weight approximates the movement cost to visit all plants.
        4. Tap Travel:
           If we need more water (Need > Carried), we must eventually visit a tap.
           Add min distance from (Cluster) to (Tap).
        """
        state = node.state
        robots, plants, taps = state
        
        if not plants:
            return 0
            
        # --- 1. Resource Accounting ---
        total_needed = sum(p[2] for p in plants)
        total_carried = sum(r[3] for r in robots)
        
        # Actions strictly required:
        # 1 POUR per needed unit
        h_pour = total_needed
        
        # 1 LOAD per deficit unit
        water_deficit = max(0, total_needed - total_carried)
        h_load = water_deficit
        
        # --- 2. MST (Movement Estimation) ---
        # Nodes: 0 (Fleet) to len(plants).
        # We use Prim's algorithm for small N.
        
        # Distances from Fleet (0) to Plants (1..N)
        # Weight is min dist from any robot to that plant
        # Distances between Plants (i..j)
        
        num_nodes = len(plants) + 1 # +1 for the Robot Fleet
        min_dists = [float('inf')] * num_nodes
        in_mst = [False] * num_nodes
        
        # Initialize distances from Fleet (Node 0)
        min_dists[0] = 0
        
        # Pre-calc distances from robots to plants to get Fleet->Plant edges
        # And Plant->Plant edges
        
        # We need a cache or fast lookup here. self.apsp is (r,c) -> dict
        
        # Current positions
        robot_locs = [(r[1], r[2]) for r in robots]
        plant_locs = [(p[0], p[1]) for p in plants]
        
        # Edge 0 -> i (Fleet to Plant i)
        for i in range(len(plants)):
            p_loc = plant_locs[i]
            # Find closest robot to this plant
            dist = float('inf')
            for r_loc in robot_locs:
                if r_loc in self.apsp:
                    d = self.apsp[r_loc].get(p_loc, float('inf'))
                    if d < dist: dist = d
            min_dists[i+1] = dist

        mst_weight = 0
        
        # Prim's Algorithm
        for _ in range(num_nodes):
            # Find min dist node not in MST
            u = -1
            min_val = float('inf')
            for n in range(num_nodes):
                if not in_mst[n] and min_dists[n] < min_val:
                    min_val = min_dists[n]
                    u = n
            
            if u == -1 or min_val == float('inf'):
                break # Disconnected graph? Should not happen if reachable.
            
            in_mst[u] = True
            mst_weight += min_val
            
            # Update neighbors (Only need to update Plant->Plant edges)
            # Fleet (0) has no outgoing edges to update essentially, 
            # because we initialized everything with Fleet->Plant.
            # So we only update if u is a Plant (u > 0).
            
            if u > 0:
                u_loc = plant_locs[u-1]
                # Try to relax edges to other plants v
                for v in range(1, num_nodes):
                    if not in_mst[v]:
                        v_loc = plant_locs[v-1]
                        # Dist u -> v
                        # Lookup in APSP
                        d = float('inf')
                        if u_loc in self.apsp:
                            d = self.apsp[u_loc].get(v_loc, float('inf'))
                        
                        if d < min_dists[v]:
                            min_dists[v] = d

        # --- 3. Tap Travel Cost ---
        # If we have a water deficit, we must go to a tap.
        # MST covers moving between plants and getting robots to plants.
        # It does NOT cover detours to taps.
        # If deficit > 0, we must add distance to nearest tap from the "Plant Cluster"
        # Admissible relaxation: Min dist from ANY plant to ANY tap.
        
        h_tap_travel = 0
        if water_deficit > 0:
            min_p_t = float('inf')
            # Check all plants to all taps
            for p_loc in plant_locs:
                if p_loc in self.apsp:
                    for t in taps:
                        t_loc = (t[0], t[1])
                        # Only consider taps with water? 
                        # Admissible: yes, but technically we might need to go to further one.
                        # Closest available tap is safe lower bound.
                        if t[2] > 0:
                            d = self.apsp[p_loc].get(t_loc, float('inf'))
                            if d < min_p_t: min_p_t = d
            
            if min_p_t != float('inf'):
                h_tap_travel = min_p_t

        return h_pour + h_load + mst_weight + h_tap_travel

    def h_gbfs(self, node):
        """
        Greedy heuristic.
        Speed is key.
        We relax the MST to just "sum of distances to targets" or similar.
        Or use h_astar but weighted.
        Given the transcript says "make it fast", we will skip the MST O(N^2) loop 
        and just look at the closest target.
        """
        state = node.state
        robots, plants, taps = state
        
        if not plants:
            return 0
            
        total_needed = sum(p[2] for p in plants)
        
        # Simple Greedy:
        # Distance from closest robot to closest plant + total need.
        
        min_dist = float('inf')
        robot_locs = [(r[1], r[2]) for r in robots]
        plant_locs = [(p[0], p[1]) for p in plants]
        
        for r_loc in robot_locs:
            if r_loc in self.apsp:
                for p_loc in plant_locs:
                    d = self.apsp[r_loc].get(p_loc, float('inf'))
                    if d < min_dist:
                        min_dist = d
        
        if min_dist == float('inf'): min_dist = 0
        
        return min_dist + total_needed


def create_watering_problem(game):
    return WateringProblem(game)