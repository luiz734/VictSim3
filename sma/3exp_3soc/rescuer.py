import sys
import networkx as nx

from map import Map
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from event_manager import EventManager, EventType


class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        """ Constructor for the rescuer agent
        @param env: a reference to the environment
        @param config_file: the absolute path to the rescuer's config file
        """
        super().__init__(env, config_file)

        # ---------------------------------------------------------------------------------
        self.G = nx.DiGraph()  # Graph of the known world (Directed to preserve entry costs)
        self.plan = []  # List of planned steps (dx, dy, save_victim)
        # ---------------------------------------------------------------------------------

        self.set_state(VS.IDLE)  # rescuer is idle initially
        self.x = 0  # current x position relative to the origin 0
        self.y = 0  # current y position relative to the origin 0
        self.map = None  # reference to the map
        self.victims = None  # reference to the victims

    def go_save_victims(self, map_data, victims):
        """ Receives the map and victims, builds the graph, and plans the rescue """
        self.map = map_data
        self.victims = victims

        print(f"\n\n*** R E S C U E R ***")
        self.map.draw()

        print(f"{self.NAME} found victims:")
        for seq, data in self.victims.items():
            print(f"{self.NAME} Victim {seq} at {data[0]}")

        # Build the graph fully since we have the map
        self.build_graph()

        # Generate the initial plan
        self.plan_rescue_path()

        print(f"{self.NAME} PLAN GENERATED ({len(self.plan)} steps)")
        self.set_state(VS.ACTIVE)

    def build_graph(self):
        """ Builds a directed graph from the provided map data matching Explorer style """
        # Add all nodes
        for coord in self.map.map_data.keys():
            self.G.add_node(coord)

        # Add edges based on available actions
        for coord, data in self.map.map_data.items():
            # data structure: (difficulty, victim_seq, actions_res)
            actions_res = data[2]

            for i, status in enumerate(actions_res):
                if status == VS.CLEAR:
                    # Determine neighbor coordinates
                    dx, dy = Rescuer.AC_INCR[i]
                    neighbor_coord = (coord[0] + dx, coord[1] + dy)

                    # Verify neighbor exists in map
                    if self.map.in_map(neighbor_coord):
                        neighbor_data = self.map.get(neighbor_coord)
                        difficulty = neighbor_data[0]

                        # Calculate weight based on movement type and destination difficulty
                        # If dx or dy is 0, it's a line move; otherwise diagonal
                        move_cost = self.COST_LINE if (dx == 0 or dy == 0) else self.COST_DIAG
                        weight = move_cost * difficulty

                        self.G.add_edge(coord, neighbor_coord, weight=weight)

    @staticmethod
    def heuristic_euclidean(u, v):
        (x1, y1) = u
        (x2, y2) = v
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def plan_rescue_path(self):
        """ Plans a full path visiting all victims using A* """
        current_pos = (0, 0)

        # Visit each victim (already in order)
        for seq, data in self.victims.items():
            victim_pos = data[0]

            if current_pos == victim_pos:
                continue

            try:
                # Returns list of nodes: [(x1,y1), (x2,y2), ...]
                path = nx.astar_path(self.G, current_pos, victim_pos, heuristic=self.heuristic_euclidean,
                                     weight='weight')

                # Convert absolute path to relative steps dx,dy
                for i in range(1, len(path)):
                    node = path[i]
                    prev = path[i - 1]
                    dx = node[0] - prev[0]
                    dy = node[1] - prev[1]

                    # Mark the very last step of this segment as the rescue step
                    is_last_step = (i == len(path) - 1)
                    self.plan.append((dx, dy, is_last_step))

                current_pos = victim_pos

            except nx.NetworkXNoPath:
                print(f"{self.NAME}: No path found from {current_pos} to {victim_pos}")

        # Plan return to base (initially)
        self.plan_return_to_base(current_pos)

    def plan_return_to_base(self, current_pos):
        """ Appends a path from current_pos to (0,0) to the plan """
        base_pos = (0, 0)
        if current_pos != base_pos:
            try:
                path_home = nx.astar_path(self.G, current_pos, base_pos, heuristic=self.heuristic_euclidean,
                                          weight='weight')
                for i in range(1, len(path_home)):
                    node = path_home[i]
                    prev = path_home[i - 1]
                    dx = node[0] - prev[0]
                    dy = node[1] - prev[1]
                    self.plan.append((dx, dy, False))
            except nx.NetworkXNoPath:
                print(f"{self.NAME}: No path found to return to base.")

    @property
    def deliberate(self) -> bool:
        """ The simulator calls this method at each cycle. """

        # Check if plan is empty
        if not self.plan:
            return False

        # Should we return to base?
        cost_to_base = 0
        try:
            if self.x != 0 or self.y != 0:
                cost_to_base = nx.astar_path_length(self.G, (self.x, self.y), (0, 0),
                                                    heuristic=self.heuristic_euclidean, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sys.exit("Shouldn't happen: No path to base")

        SAFETY_MARGIN = 1.5
        required_battery = cost_to_base * SAFETY_MARGIN

        # If battery is critical and we are not already at base, abort mission and return
        if self.get_rtime() <= required_battery and (self.x != 0 or self.y != 0):
            # If battery < required, FORCE return plan.
            print(f"{self.NAME}: Low battery! Aborting rescue, returning to base.")
            self.plan = []  # Clear current rescue mission
            self.plan_return_to_base((self.x, self.y))  # Force path to (0,0)

        # ------------------------------------------------------------------

        print(f"{self.NAME}: battery at ({self.x}, {self.y}) is {self.get_rtime()}")

        # Execute the next step in the plan
        if self.plan:
            dx, dy, there_is_vict = self.plan.pop(0)

            walked = self.walk(dx, dy)

            if walked == VS.EXECUTED:
                self.x += dx
                self.y += dy
                if there_is_vict:
                    rescued = self.first_aid()
                    if rescued:
                        print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                    else:
                        print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.y})")
            else:
                print(f"{self.NAME} Plan fail - walk error at ({self.x}, {self.y})")

        return True