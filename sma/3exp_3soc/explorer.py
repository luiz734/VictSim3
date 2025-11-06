# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.


import random
import sys

from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
import networkx as nx

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

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
        self.walk_stack = Stack()  # a stack to store the movements

        # ---------------------------------------------------------------------------------
        # We need a different stack to store coords instead of dx,dy
        self.dfs_stack = Stack()    # a stack for the DFS backtracking
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

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            direction = random.randint(0, 7)
            # Check if the corresponding position in walls_and_lim is CLEAR
            if obstacles[direction] == VS.CLEAR:
                return Explorer.AC_INCR[direction]

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

        # Find the closest frontier cell to the agent's current position
        closest_cell = min(
            self.frontier,
            key=lambda cell: self.heuristic_euclidean((self.x, self.y), cell)
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
            # This can happen if a frontier cell is "known" but unreachable
            # (e.g., seen diagonally, but blocked). We remove it and try again next cycle.
            # print(f"{self.NAME}: No path to frontier {closest_cell}. Removing from frontier.")
            # self.frontier.remove(closest_cell)
            # return 0, 0  # Stop for this cycle
            sys.exit("Shouldn't happen")
        except (nx.NodeNotFound, KeyError) as e:
            # This is a safety check. Can happen if G is out of sync or agent is not in G.
            print(f"{self.NAME}: A* error (NodeNotFound: {e}). Trying to recover.")
            if closest_cell in self.frontier:
                self.frontier.remove(closest_cell)  # Remove problematic cell
            # If agent position is not in G, it's a critical error, but we just wait
            return 0, 0

    def get_next_position_ONLINE_DFS(self):
        available_neighbors = []
        obstacles = self.check_walls_and_lim()
        for direction in range(8):
            dx, dy = Explorer.AC_INCR[direction]
            next_position = (self.x + dx, self.y + dy)
            if obstacles[direction] == VS.CLEAR and not self.map.in_map(next_position):
                available_neighbors.append((dx, dy))

        if len(available_neighbors) > 0:
            self.dfs_stack.push((self.x, self.y))
            # Without random they all go the same path
            return random.choice(available_neighbors)

        else:
            if self.dfs_stack.is_empty():
                return 0, 0
            target_x, target_y = self.dfs_stack.pop()
            return target_x - self.x, target_y - self.y

    def explore(self):
        # get an random increment for x and y
        try:
            dx, dy = self.get_next_frontier_step()
        except NoFrontier:
            # Exploration is complete, just wait for time to run out
            self.set_state(VS.IDLE)  # Set state to idle
            return
        # dx, dy = self.get_next_position()

        # Moves the explorer agent to another position
        rtime_bef = self.get_rtime()  ## get remaining batt time before the move
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()  ## get remaining batt time after the move

        # Test the result of the walk action
        # It should never bump, since get_next_position always returns a valid position...
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
            # puts the visited position in a stack. When the batt is low,
            # the explorer unstack each visited position to come back to the base
            # self.walk_stack.push((dx, dy))

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

        def heuristic_euclidean(u, v):
            (x1, y1) = u
            (x2, y2) = v
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


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
            # A* path led into a wall. This shouldn't happen if map is correct.
            # We must replan, or stop.
            print(f"{self.NAME}: BUMPED while following A* path! Stopping.")
            self.return_path = None  # Clear path
            self.set_state(VS.DEAD)
            sys.exit("SHOULDNT HAPPEN")
            return

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
        if consumed_time < self.get_rtime():
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
                self.resc.go_save_victims(self.map, self.victims)
                self.set_state(VS.ENDED)  # VS.ENDED exists
                return False

            # Handle the case where the agent was IDLE (frontier empty) and time ran out
            if self.get_state() == VS.IDLE:
                print(f"{self.NAME}: Frontier empty and time up. rtime {self.get_rtime()}, invoking rescuer.")
                self.resc.go_save_victims(self.map, self.victims)
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



