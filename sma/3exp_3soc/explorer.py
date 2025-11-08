# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.


import sys

import networkx as nx

from map import Map
from vs.abstract_agent import AbstAgent
from vs.constants import VS

# Only to make errors more readable
class NoFrontier(Exception):
    pass

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, shared_env):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)

        # ---------------------------------------------------------------------------------
        self.return_path = None
        self.G = nx.Graph()         # Persistent graph of the known world
        self.frontier = set()       # Set of (x,y) tuples bordering the unknown
        self.exploration_path = []  # Current A* path to a frontier cell
        self.shared_env = shared_env
        # ---------------------------------------------------------------------------------

        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is a seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

        # Manually update graph and frontier for the starting position (0,0)
        self.update_graph_and_frontier((self.x, self.y), 1, self.check_walls_and_lim())

        # Zone-based exploration attributes
        self.zone_min_y = float('-inf')
        self.zone_max_y = float('inf')
        self.zone_penalty = 10000.0  # High cost for exploring outside the zone


        if "1" in self.NAME:  # Left Zone
            self.zone_max_y = -15
        elif "2" in self.NAME:  # Middle Zone
            self.zone_min_y = -14
            self.zone_max_y = 15
        elif "3" in self.NAME:  # Right Zone
            self.zone_min_y = 16


    def get_next_frontier_step(self):
        # If we have a path, follow it
        if self.exploration_path:
            target_x, target_y = self.exploration_path.pop(0)
            return target_x - self.x, target_y - self.y

        if not self.frontier:
            print(f"{self.NAME}: Frontier is empty. Exploration complete.")
            raise NoFrontier()

        def cost_function(cell):
            # Base cost is distance from agent
            dist = self.heuristic_euclidean((self.x, self.y), cell)
            cell_y = cell[1]
            penalty = 0.0

            # Calculate penalty based on distance from the allowed zone
            # If we start leaving our zone, the penalty increases
            if cell_y < self.zone_min_y:
                penalty = (self.zone_min_y - cell_y) * self.zone_penalty
            elif cell_y > self.zone_max_y:
                penalty = (cell_y - self.zone_max_y) * self.zone_penalty

            return dist + penalty

        closest_cell = min(
            self.frontier,
            key=cost_function
        )

        try:
            # A* can only path to nodes in self.G
            # The closest_cell is in the frontier (unknown), so it is not in G
            # We must find the neighbor of closest_cell that in in G
            # and path to that gateway node first.
            gateway_node = None
            for direction in range(8):
                dx, dy = Explorer.AC_INCR[direction]
                # Check neighbors of the frontier cell
                neighbor_coord = (closest_cell[0] - dx, closest_cell[1] - dy)
                if neighbor_coord in self.G:
                    gateway_node = neighbor_coord
                    break

            if gateway_node is None:
                sys.exit("A gateway should always exists")

            # Plan a path from current_pos to the gateway node
            path = nx.astar_path(self.G, (self.x, self.y), gateway_node, heuristic=self.heuristic_euclidean,
                                 weight='weight')
            # Add the frontier cell itself as the final step
            if path[-1] != closest_cell:
                path.append(closest_cell)
            # We have a new path. Store it
            # [1:] because we ignore the starting node
            self.exploration_path = path[1:]
            # Get the first step of this new path
            if self.exploration_path:
                target_x, target_y = self.exploration_path.pop(0)
                return target_x - self.x, target_y - self.y
            else:
                # Path was only 1 step (adjacent)
                # The target cell will be processed on the next call if it's still in frontier
                return 0, 0

        except nx.NetworkXNoPath as e:
            sys.exit(f"Shouldn't happen: {e}")

        except (nx.NodeNotFound, KeyError) as e:
            sys.exit(f"Shouldn't happen: {e}")

    def explore(self):
        try:
            dx, dy = self.get_next_frontier_step()
        except NoFrontier:
            self.set_state(VS.IDLE)
            self.shared_env.share_map()
            return

        # Moves the explorer agent to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        if result == VS.BUMPED:
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            self.exploration_path = []
            self.update_graph_and_frontier((self.x, self.y), self.map.get((self.x, self.y))[0],
                                           self.check_walls_and_lim())

        if result == VS.EXECUTED:
            # If dx and dy are 0, it means we are planning or waiting.
            if dx == 0 and dy == 0:
                return

            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[seq] = ((self.x, self.y), vs)
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            self.update_graph_and_frontier((self.x, self.y), difficulty, self.check_walls_and_lim())

        return

    @staticmethod
    def heuristic_euclidean(u, v):
        (x1, y1) = u
        (x2, y2) = v
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def update_graph_and_frontier(self, coord, difficulty, actions_res):
        # Add the new node to the graph
        if coord not in self.G:
            self.G.add_node(coord)

        self.frontier.discard(coord)
        # Check all 8 neighbors
        for direction in range(8):
            dx, dy = Explorer.AC_INCR[direction]
            neighbor_coord = (coord[0] + dx, coord[1] + dy)

            if actions_res[direction] == VS.CLEAR:
                # If this neighbor is not in our map, it's a new frontier cell
                if not self.map.in_map(neighbor_coord):
                    self.frontier.add(neighbor_coord)

                # If this neighbor is in our map creatw and edge
                elif neighbor_coord in self.G:
                    # Get neighbor's difficulty
                    neighbor_data = self.map.get(neighbor_coord)
                    # if neighbor_data:
                    neighbor_difficulty = neighbor_data[0]
                    if neighbor_difficulty < VS.OBST_WALL:
                        # Calculate cost: average difficulty * move cost
                        cost_multiplier = (difficulty + neighbor_difficulty) / 2
                        move_cost = self.COST_LINE if (dx == 0 or dy == 0) else self.COST_DIAG
                        weight = cost_multiplier * move_cost

                        self.G.add_edge(coord, neighbor_coord, weight=weight)

            elif actions_res[direction] == VS.WALL or actions_res[direction] == VS.END:
                # Neighbor is an obstacle. If it was on the frontier, remove it.
                self.frontier.discard(neighbor_coord)

    def plan_Astar_path(self):
        assert (self.return_path is None, "Path already planned")
        print(f"{self.NAME}: Planning A* path from ({self.x}, {self.y}) to (0,0)...")
        if (0, 0) not in self.G:
            print(f"{self.NAME}: Base (0,0) is not in the known graph! Cannot plan return path.")
            raise nx.NetworkXNoPath
        path = nx.astar_path(self.G, (self.x, self.y), (0, 0), heuristic=self.heuristic_euclidean, weight='weight')
        # Skip the current position
        self.return_path = path[1:]

    def execute_Astar_step(self):
        assert (self.return_path is not None, "Should not happen")
        target_x, target_y = self.return_path.pop(0)
        dx = target_x - self.x
        dy = target_y - self.y
        result = self.walk(dx, dy)

        if result == VS.BUMPED:
            sys.exit("SHOULDN'T HAPPEN: bad A* implementation")
        elif result == VS.EXECUTED:
            self.x += dx
            self.y += dy


    @property
    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        is_returning_to_base = not self.return_path is None

        # Should we return to base?
        cost_to_base = 0
        try:
            if self.x != 0 or self.y != 0:
                cost_to_base = nx.astar_path_length(self.G,(self.x, self.y), (0, 0),heuristic=self.heuristic_euclidean, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sys.exit("Shouldn't happen")

        # Just in case or calculations are incorrect
        SAFETY_MARGIN = 1.5
        if not is_returning_to_base:
            required_battery = cost_to_base * SAFETY_MARGIN

            # We return if there is no battery or we are not active
            if self.get_rtime() > required_battery and self.get_state() == VS.ACTIVE:
                self.explore()
                return True

        # --- RETURN TO BASE LOGIC----
        # We are already at the base
        if self.x == 0 and self.y == 0:
            if self.get_state() != VS.ENDED:
                self.shared_env.share_map()
                self.set_state(VS.ENDED)
            return False

        # We are not in the base and have no path to it
        if self.return_path is None:
            try:
                self.plan_Astar_path()
            except nx.NetworkXNoPath as e:
                sys.exit(f"A* error: {e}")
            except Exception as e:
                sys.exit(f"Should not happen: {e}")

        # Execute the next step in the A* path
        if self.return_path is not None:
            self.execute_Astar_step()
            print(f"{self.NAME} at position ({self.x}, {self.y}). rtime {self.get_rtime()}")
        else:
            sys.exit(f"Should not happen")

        return True