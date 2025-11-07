import os
import sys
import threading

# import classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer
from vs.constants import VS

NUM_AGENTS = 3

class SharedEnvironmentData:
    def __init__(self):
        self.COST_DIAG = None
        self.AC_INCR = None
        self.COST_LINE = None
        self.unified_map_data = None
        self.unified_victims = {}
        self.explorers = []
        self.rescuers = []
        self.map_lock = threading.Lock()



        self.share_count = 0

    def add_explorer(self, e):
        self.explorers.append(e)
        self.COST_LINE = self.explorers[0].COST_LINE
        self.COST_DIAG = self.explorers[0].COST_DIAG
        self.AC_INCR = self.explorers[0].AC_INCR

    def add_rescuer(self, r):
        self.rescuers.append(r)

    def share_map(self):
        with self.map_lock:
            self.unified_map_data = {}
            self.unified_victims = {}

            for explorer in self.explorers:
                agent_map_data_items = list(explorer.map.map_data.items())
                for coord, data in agent_map_data_items:
                    if coord not in self.unified_map_data:
                        # If coordinate is new, add it
                        self.unified_map_data[coord] = data
                    else:
                        # If coordinate already exists, merge
                        existing_data = self.unified_map_data[coord]
                        # Merge logic: prioritize data that contains a victim ID.
                        if existing_data[1] == VS.NO_VICTIM and data[1] != VS.NO_VICTIM:
                            self.unified_map_data[coord] = data
                self.unified_victims.update(explorer.victims)

            self.share_count += 1
            if self.share_count == 3:
                self.combine_maps()
                self.create_clusters()

    def combine_maps(self):
        for explorer in self.explorers:
            explorer.map.map_data = self.unified_map_data

    def create_clusters(self):
        assert (self.share_count == 3)
        print("creating clusters")


def load_agents(env, config_base_folder, num_agents):
    shrev = SharedEnvironmentData()
    for i in range(1, num_agents + 1):
        # Each agent pair has its own configuration directory
        current_agent_config_folder = os.path.join(config_base_folder, f"config_ag_{i}")
        # Filepaths are relative to the agent's specific config folder
        rescuer_file = os.path.join(current_agent_config_folder, f"rescuer_{i}.txt")
        explorer_file = os.path.join(current_agent_config_folder, f"explorer_{i}.txt")

        try:
            if not os.path.isdir(current_agent_config_folder):
                raise FileNotFoundError(f"Configuration folder not found: {current_agent_config_folder}")

            # Instantiate Rescuer first
            resc = Rescuer(env, rescuer_file)
            # Explorer needs to know rescuer to send the map
            # that's why rescuer is instantiated before
            expl = Explorer(env, explorer_file, resc, shrev)

            shrev.add_explorer(expl)
            shrev.add_rescuer(resc)

        except FileNotFoundError as e:
            # Provide more context in the error message
            sys.exit(f"Error loading agent configuration: {e}")
        except Exception as e:
            sys.exit(f"An unexpected error occurred loading agent {i}: {e}")

    return shrev


def main(vict_folder, env_folder, config_base_folder):
    env = Env(vict_folder, env_folder)
    shared_env_data = load_agents(env, config_base_folder, NUM_AGENTS)
    # Run the environment simulator
    try:
        env.run()
    except KeyboardInterrupt:
        print("Canceled by user")


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")

    # dataset com sinais vitais das vitimas
    vict_folder = os.path.join("datasets/vict/", "10v")
    # vict_folder = os.path.join("datasets/vict/", "408v")

    # dataset do ambiente (paredes, posicao das vitimas)
    # env_folder = os.path.join("datasets/env/", "94x94_408v")
    env_folder = os.path.join("datasets/env/", "12x12_10v")

    # folder das configuracoes dos agentes
    # This is now the BASE folder containing config_ag_1, config_ag_2, etc.
    curr = os.getcwd()
    config_base_folder = os.path.join(curr, "sma", "3exp_3soc")

    main(vict_folder, env_folder, config_base_folder)

    print("------------------")
    print("---- FIM SMA -----")
    print("------------------")