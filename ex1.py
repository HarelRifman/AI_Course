import search

ids = ["217398338"]

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
        self.dist_matrix = self._compute_apsp()
        
        # Pre-cache tap indices for heuristic
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
        
        occupied = {(r[1], r[2]) for r in robots}
        plant_map = {(p[0], p[1]): i for i, p in enumerate(plants)}
        tap_map = {(t[0], t[1]): i for i, t in enumerate(taps)}
        
        successors = []
        
        for i, r in enumerate(robots):
            r_id, r_r, r_c, r_load = r
            
            # Crowding check
            is_crowded = False
            for j, other in enumerate(robots):
                if i != j:
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
            elif (r_r, r_c) in tap_map:
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
        Admissible Heuristic: Actions + MST(Plants) + Minimum Entry Cost
        """
        state = node.state
        robots, plants, taps = state
        
        if not plants:
            return 0
        
        # --- 1. Resource Action Costs (Load/Pour) ---
        total_need = 0
        for p in plants:
            total_need += p[2]
            
        total_carried = 0
        for r in robots:
            total_carried += r[3]
            
        # Cost to POUR
        h_pour = total_need
        # Cost to LOAD (Deficit)
        deficit = max(0, total_need - total_carried)
        h_load = deficit
        
        # --- 2. Plant Traversal Cost (MST) ---
        # This calculates the minimum wire to connect all plants together.
        plant_indices = [self._to_idx(p[0], p[1]) for p in plants]
        num_plants = len(plants)
        mst_weight = 0
        
        # Prim's Algorithm for MST of Plants only
        if num_plants > 1:
            # Distance from tree to remaining nodes
            # Start with first plant as the tree
            min_dists = [self.dist_matrix[plant_indices[0]][plant_indices[i]] 
                         for i in range(num_plants)]
            visited = [False] * num_plants
            visited[0] = True
            
            # We need to add N-1 edges
            for _ in range(num_plants - 1):
                # Find closest unvisited node to the current tree
                u = -1
                min_val = float('inf')
                for i in range(num_plants):
                    if not visited[i] and min_dists[i] < min_val:
                        min_val = min_dists[i]
                        u = i
                
                if u == -1: break # Should not happen in connected component
                
                visited[u] = True
                mst_weight += min_val
                
                # Update distances from the new node u
                u_idx = plant_indices[u]
                for v in range(num_plants):
                    if not visited[v]:
                        v_idx = plant_indices[v]
                        d = self.dist_matrix[u_idx][v_idx]
                        if d < min_dists[v]:
                            min_dists[v] = d

        # --- 3. Entry Cost ---
        # We have a network of plants (MST). We need to enter it.
        # We can enter from a Robot -> Plant (if robot has water)
        # Or Robot -> Tap -> Plant (if robot is empty)
        # We take the global minimum cost to start watering.
        
        # Pre-calculate closest plant distance for every tap (needed for empty robots)
        # tap_to_closest_plant[t_idx] = distance
        tap_to_any_plant = []
        
        # Only check active taps (amt > 0)
        active_taps = [t for t in taps if t[2] > 0]
        if not active_taps and deficit > 0:
             return float('inf') # Dead end

        # If we have a deficit, we might need the taps.
        if deficit > 0 or not active_taps:
             # Optimization: If deficit > 0, we perform this check.
             # If deficit == 0, we can theoretically ignore empty robots, 
             # but to be safe/simple we calculate tap distances if taps exist.
             if active_taps:
                 for t in active_taps:
                     t_idx = self._to_idx(t[0], t[1])
                     min_d = float('inf')
                     for p_idx in plant_indices:
                         d = self.dist_matrix[t_idx][p_idx]
                         if d < min_d:
                             min_d = d
                     tap_to_any_plant.append((t_idx, min_d))

        min_entry_cost = float('inf')

        for r in robots:
            r_idx = self._to_idx(r[1], r[2])
            
            # Option A: Robot goes directly to a plant (valid if load > 0)
            if r[3] > 0:
                for p_idx in plant_indices:
                    d = self.dist_matrix[r_idx][p_idx]
                    if d < min_entry_cost:
                        min_entry_cost = d
            
            # Option B: Robot goes to Tap then Plant (valid always, necessary if empty)
            # Cost = Dist(R, Tap) + Dist(Tap, Closest_Plant_To_That_Tap)
            if active_taps:
                for t_idx, t_p_dist in tap_to_any_plant:
                    r_t_dist = self.dist_matrix[r_idx][t_idx]
                    total = r_t_dist + t_p_dist
                    if total < min_entry_cost:
                        min_entry_cost = total

        # If no path found (unreachable), min_entry_cost remains inf
        if min_entry_cost == float('inf'):
            return float('inf')

        return h_pour + h_load + mst_weight + min_entry_cost

    def h_gbfs(self, node):
        return self.h_astar(node)

def create_watering_problem(game):
    return WateringProblem(game)