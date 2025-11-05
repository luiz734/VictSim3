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

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements

        # We need a different stack to store coords instead of dx,dy
        self.dfs_stack = Stack()  # a stack for the DFS backtracking
        self.return_path = None

        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is a seq number of the victim,(x,y) the position, <vs> the list of vital signals

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

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
        dx, dy = self.get_next_position_ONLINE_DFS()
        # dx, dy = self.get_next_position()

        # Moves the explorer agent to another position
        rtime_bef = self.get_rtime()   ## get remaining batt time before the move
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()   ## get remaining batt time after the move

        # Test the result of the walk action
        # It should never bump, since get_next_position always returns a valid position...
        # but for safety, let's test it anyway
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
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
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def plan_Astar_path(self):
        print(f"{self.NAME}: Planning A* path from ({self.x}, {self.y}) to (0,0)...")
        G = nx.Graph()

        # Add nodes and edges from the map
        for coord, map_data in self.map.map_data.items():
            (difficulty, _, actions_res) = map_data

            # Add node only if it's not a wall
            if difficulty >= VS.OBST_WALL:
                continue

            G.add_node(coord)

            # Check all 8 directions for neighbors *that are also in the map*
            for direction in range(8):
                if actions_res[direction] == VS.CLEAR:
                    dx, dy = Explorer.AC_INCR[direction]
                    neighbor_coord = (coord[0] + dx, coord[1] + dy)

                    if self.map.in_map(neighbor_coord):
                        # Get neighbor's data
                        neighbor_data = self.map.get(neighbor_coord)
                        if not neighbor_data:
                            continue

                        neighbor_difficulty = neighbor_data[0]

                        if neighbor_difficulty >= VS.OBST_WALL:
                            continue

                        # Calculate cost: average difficulty * move cost
                        cost_multiplier = (difficulty + neighbor_difficulty) / 2
                        move_cost = self.COST_LINE if (dx == 0 or dy == 0) else self.COST_DIAG
                        weight = cost_multiplier * move_cost

                        G.add_edge(coord, neighbor_coord, weight=weight)

        def heuristic_euclidean(u, v):
            (x1, y1) = u
            (x2, y2) = v
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


        path = nx.astar_path(G, (self.x, self.y), (0, 0), heuristic=heuristic_euclidean, weight='weight')

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
            # continue to explore
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

            if self.get_state() == VS.ENDED:
                # We have successfully returned using A* and are already done
                return False

        # Plan the A* path (only once)
        # If we are not at the base, and time is up, and we haven't planned a path yet
        should_plant_Astar = self.return_path is None and self.get_state() == VS.ACTIVE
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



