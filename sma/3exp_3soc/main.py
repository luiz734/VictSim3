import os
import sys
import threading

import os
from vs.environment import Env
from agents_manager import AgentsManager

NUM_AGENTS = 3
DEBUG_SKIP_SIMULATION = False


def main(vict_folder, env_folder, config_base_folder):
    env = Env(vict_folder, env_folder)
    agents_manager = AgentsManager()
    agents_manager.load_agents(env, config_base_folder, NUM_AGENTS)

    if DEBUG_SKIP_SIMULATION:
        agents_manager.load_and_cluster_directly()
    else:
        try:
            env.run()
        except KeyboardInterrupt:
            print("Canceled by user")


RUN_PARAMS_10V = {"vict": "datasets/vict/10v",
                  "env": "datasets/env/12x12_10v"}
RUN_PARAMS_408V = {"vict": "datasets/vict/408v",
                   "env": "datasets/env/94x94_408v"}

if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")

    run_params = RUN_PARAMS_10V
    # run_params = RUN_PARAMS_408V

    # folder das configuracoes dos agentes
    # This is now the BASE folder containing config_ag_1, config_ag_2, etc.
    curr = os.getcwd()
    config_base_folder = os.path.join(curr, "sma", "3exp_3soc")

    main(run_params["vict"], run_params["env"], config_base_folder)

    print("------------------")
    print("---- FIM SMA -----")
    print("------------------")
