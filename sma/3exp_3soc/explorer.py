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
    def __init__(self, env, config_file, resc):
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

        # >>>>>>>>>
        # Zone-based exploration attributes
        self.zone_min_x = float('-inf')
        self.zone_max_x = float('inf')
        self.zone_penalty = 10000.0  # High cost for exploring outside the zone

        # Define zones based on agent name (relative to base 0,0)
        # Assumes a 94-wide map (approx 47 radius)
        # Zone 1 (Left): x <= -16
        # Zone 2 (Mid): -15 <= x <= 15
        # Zone 3 (Right): x >= 16

        if "1" in self.NAME:  # Left Zone
            self.zone_max_x = -15
        elif "2" in self.NAME:  # Middle Zone
            self.zone_min_x = -4
            self.zone_max_x = 15
        elif "3" in self.NAME:  # Right Zone
            self.zone_min_x = 16
        # <<<<<<<<<


    def get_next_frontier_step(self):
        # If we have a path, follow it
        if self.exploration_path:
            target_x, target_y = self.exploration_path.pop(0)
            return target_x - self.x, target_y - self.y

        # No path. We must plan a new one.
        if not self.frontier:
            # No more frontiers to explore
            print(f"{self.NAME}: Frontier is empty. Exploration complete.")
            raise NoFrontier()

        def cost_function(cell):
            # Base cost is distance from agent
            dist = self.heuristic_euclidean((self.x, self.y), cell)

            cell_x = cell[0]
            penalty = 0.0

            # Calculate penalty based on distance from the allowed zone
            # This creates a soft boundary, pulling agents to their zones
            if cell_x < self.zone_min_x:
                # Cell is to the left of the zone's left edge
                # The further left, the higher the penalty for Agents 2 and 3
                penalty = (self.zone_min_x - cell_x) * self.zone_penalty
            elif cell_x > self.zone_max_x:
                # Cell is to the right of the zone's right edge
                # The further right, the higher the penalty for Agents 1 and 2
                penalty = (cell_x - self.zone_max_x) * self.zone_penalty

            return dist + penalty

        closest_cell = min(
            self.frontier,
            key=cost_function
        )

        try:
            # A* can only path to nodes in self.G
            # The closest_cell is in the frontier (unknown), so it's not in G
            # We must find the neighbor of closest_cell that in in G (a "gateway")
            # and path to that gateway node first.
            gateway_node = None
            for direction in range(8):
                dx, dy = Explorer.AC_INCR[direction]
                # Check neighbors of the frontier cell
                neighbor_coord = (closest_cell[0] - dx, closest_cell[1] - dy)
                if neighbor_coord in self.G:
                    gateway_node = neighbor_coord
                    break  # Found a valid gataway

            if gateway_node is None:
                # print(f"{self.NAME}: Frontier {closest_cell} has no neighbor in G. Removing.")
                # self.frontier.remove(closest_cell)
                # return 0, 0
                sys.exit("A gateway should always exists")

            # Plan a path from current_pos to the gateway node
            path = nx.astar_path(self.G, (self.x, self.y), gateway_node, heuristic=self.heuristic_euclidean,
                                 weight='weight')

            # Add the frontier cell itself as the final step
            # This handles the case where we are already at the gateway_node
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

        except nx.NetworkXNoPath:
            sys.exit("Shouldn't happen")

        except (nx.NodeNotFound, KeyError) as e:
            sys.exit(f"Shouldn't happen: {e}")

    def explore(self):
        # get an random increment for x and y
        try:
            dx, dy = self.get_next_frontier_step()
        except NoFrontier:
            # Exploration is complete, just wait for time to run out
            self.set_state(VS.IDLE)  # Set state to idle
            return

        # Moves the explorer agent to another position
        rtime_bef = self.get_rtime()  ## get remaining batt time before the move
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()  ## get remaining batt time after the move

        # Test the result of the walk action
        # It should never bump, since always returns a valid position...
        # but for safety, let's test it anyway
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

            # We bumped, so the path is invalid. Clear it.
            self.exploration_path = []
            # Update graph/frontier with the wall info
            self.update_graph_and_frontier((self.x, self.y), self.map.get((self.x, self.y))[0],
                                           self.check_walls_and_lim())

        if result == VS.EXECUTED:
            # If dx and dy are 0, it means we are planning or waiting.
            if dx == 0 and dy == 0:
                return


            # update the agent's position relative to the origin of 
            # the coordinate system used by the agents
            self.x += dx
            self.y += dy          

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[seq] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

            # Update the graph and frontier with the new cell info
            self.update_graph_and_frontier((self.x, self.y), difficulty, self.check_walls_and_lim())
        #print(f"{self.NAME}:at ({self.x, self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

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

        # Remove this cell from the frontier (it is now "known")
        self.frontier.discard(coord)

        # Check all 8 neighbors
        for direction in range(8):
            dx, dy = Explorer.AC_INCR[direction]
            neighbor_coord = (coord[0] + dx, coord[1] + dy)

            if actions_res[direction] == VS.CLEAR:
                # Neighbor is accessible

                # If this neighbor is *not* in our map, it's a new frontier cell
                if not self.map.in_map(neighbor_coord):
                    self.frontier.add(neighbor_coord)

                # If this neighbor *is* in our map (and thus in G), create an edge
                elif neighbor_coord in self.G:
                    # Get neighbor's difficulty
                    neighbor_data = self.map.get(neighbor_coord)
                    if neighbor_data:  # Safety check
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
        print(f"{self.NAME}: Planning A* path from ({self.x}, {self.y}) to (0,0)...")
        G = self.G
        # Add nodes and edges from the map
        # Ensure the base (0,0) is in the graph, otherwise A* will fail
        if (0, 0) not in G:
            print(f"{self.NAME}: Base (0,0) is not in the known graph! Cannot plan return path.")
            raise nx.NetworkXNoPath

        path = nx.astar_path(G, (self.x, self.y), (0, 0), heuristic=self.heuristic_euclidean, weight='weight')

        # Store the path (excluding the first node, which is the current position)
        self.return_path = path[1:]

    def execute_Astar_step(self):
        if not self.return_path:
            return

        # Get the next waypoint from the A* path
        target_x, target_y = self.return_path.pop(0)  # Get the next step

        # Calculate the required move
        dx = target_x - self.x
        dy = target_y - self.y

        result = self.walk(dx, dy)

        if result == VS.BUMPED:
            sys.exit("SHOULDNT HAPPEN: bad A* implementation")


        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            # print(f"{self.NAME}: A* step to ({self.x}, {self.y}), rtime: {self.get_rtime()}")

    def deliberate(self) -> bool:
        """  The simulator calls this method at each cycle. 
        Must be implemented in every agent. The agent chooses the next action.
        """

        consumed_time = self.TLIM - self.get_rtime()
        
        # check if it is time to come back to the base      
        if consumed_time < 7000:
            # continue to explore only if active
            if self.get_state() == VS.ACTIVE:
                self.explore()
            return True

        # We use A* to return to base
        # Check if we have arrived at the base (0,0)
        if self.x == 0 and self.y == 0:
            # We are at the base, check if we were exploring or returning
            if self.get_state() == VS.ACTIVE:
                # We were still exploring and time run out
                # We are already at the base, so just shut down.
                print(f"{self.NAME}: Time up, already at base. rtime {self.get_rtime()}, invoking rescuer.")
                # self.resc.go_save_victims(self.map, self.victims)
                self.set_state(VS.ENDED)  # VS.ENDED exists
                return False

            # Handle the case where the agent was IDLE (frontier empty) and time ran out
            if self.get_state() == VS.IDLE:
                print(f"{self.NAME}: Frontier empty and time up. rtime {self.get_rtime()}, invoking rescuer.")
                # self.resc.go_save_victims(self.map, self.victims)
                self.set_state(VS.ENDED)
                return False

            if self.get_state() == VS.ENDED:
                # We have successfully returned using A* and are already done
                return False

        # Plan the A* path (only once)
        # If we are not at the base, and time is up, and we haven't planned a path yet
        # Also plan if we were IDLE (finished exploring) and need to return
        should_plant_Astar = self.return_path is None and (self.get_state() == VS.ACTIVE or self.get_state() == VS.IDLE)
        if should_plant_Astar:
            try:
                self.plan_Astar_path()
            except nx.NetworkXNoPath:
                print(f"{self.NAME}: A* CANNOT FIND A PATH BACK TO BASE (0,0)!")
                self.set_state(VS.DEAD)  # VS.DEAD exists
                return False  # Agent stops
            except Exception as e:
                print(f"{self.NAME}: A* planning failed: {e}")
                self.set_state(VS.DEAD)
                return False

        # Execute the next step in the A* path
        if self.return_path is not None:
            self.execute_Astar_step()

            # Check if we *just* arrived
            if self.x == 0 and self.y == 0:
                print(f"{self.NAME}: A* return complete. rtime {self.get_rtime()}, invoking rescuer.")
                self.resc.go_save_victims(self.map, self.victims)
                self.set_state(VS.ENDED)
                return False

        return True