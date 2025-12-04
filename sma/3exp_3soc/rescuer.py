import networkx as nx
import math
from vs.abstract_agent import AbstAgent
from vs.constants import VS


class Rescuer(AbstAgent):
    def __init__(self, env, config_file):
        super().__init__(env, config_file)
        self.map = None
        self.victims = None
        self.plan = []
        self.x = 0
        self.y = 0
        self.set_state(VS.IDLE)

    def go_save_victims(self, map_data, victims):
        self.map = map_data
        self.victims = victims

        print(f"\n\n*** R E S C U E R ***")
        self.map.draw()

        print(f"{self.NAME} found victims:")
        for seq, data in self.victims.items():
            print(f"{self.NAME} Victim {seq} at {data[0]}")

        self.__planner()

        print(f"{self.NAME} PLAN GENERATED ({len(self.plan)} steps)")
        self.set_state(VS.ACTIVE)

    def __build_graph(self):
        """Builds a directed graph from the explored map data."""
        G = nx.DiGraph()

        # Add all nodes
        for coord in self.map.map_data.keys():
            G.add_node(coord)

        # Add edges based on available actions
        for coord, data in self.map.map_data.items():
            # data structure: (difficulty, victim_seq, actions_res)
            actions_res = data[2]

            for i, status in enumerate(actions_res):
                if status == VS.CLEAR:
                    # Determine neighbor coordinates
                    dx, dy = Rescuer.AC_INCR[i]
                    neighbor = (coord[0] + dx, coord[1] + dy)

                    # Verify neighbor exists in map (it should if status is CLEAR, but safe check)
                    if self.map.in_map(neighbor):
                        neighbor_data = self.map.get(neighbor)
                        difficulty = neighbor_data[0]

                        # Calculate weight based on movement type and destination difficulty
                        # If dx or dy is 0, it's a line move; otherwise diagonal
                        base_cost = self.COST_LINE if (dx == 0 or dy == 0) else self.COST_DIAG
                        weight = base_cost * difficulty

                        G.add_edge(coord, neighbor, weight=weight)
        return G

    def __dist(self, a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def __planner(self):
        """Generates the rescue plan using A* via NetworkX."""
        G = self.__build_graph()
        current_pos = (0, 0)

        # 1. Visit each victim in the ordered sequence
        for seq, data in self.victims.items():
            victim_pos = data[0]

            if current_pos == victim_pos:
                continue

            try:
                # Returns list of nodes: [(x1,y1), (x2,y2), ...]
                path = nx.astar_path(G, current_pos, victim_pos, heuristic=self.__dist, weight='weight')

                # Convert absolute path to relative steps
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

        # 2. Return to base
        base_pos = (0, 0)
        if current_pos != base_pos:
            try:
                path_home = nx.astar_path(G, current_pos, base_pos, heuristic=self.__dist, weight='weight')
                for i in range(1, len(path_home)):
                    node = path_home[i]
                    prev = path_home[i - 1]
                    dx = node[0] - prev[0]
                    dy = node[1] - prev[1]
                    self.plan.append((dx, dy, False))
            except nx.NetworkXNoPath:
                print(f"{self.NAME}: No path found to return to base.")

    def deliberate(self) -> bool:
        if not self.plan:
            return False

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