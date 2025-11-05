import os
import sys

# import classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

NUM_AGENTS = 3


def load_agents(env, config_base_folder, num_agents):
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
            Explorer(env, explorer_file, resc)

        except FileNotFoundError as e:
            # Provide more context in the error message
            sys.exit(f"Error loading agent configuration: {e}")
        except Exception as e:
            sys.exit(f"An unexpected error occurred loading agent {i}: {e}")


def main(vict_folder, env_folder, config_base_folder):
    env = Env(vict_folder, env_folder)
    load_agents(env, config_base_folder, NUM_AGENTS)
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
    vict_folder = os.path.join("..", "..", "datasets/vict/", "10v")

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = os.path.join("..", "..", "datasets/env/", "12x12_10v")

    # folder das configuracoes dos agentes
    # This is now the BASE folder containing config_ag_1, config_ag_2, etc.
    curr = os.getcwd()
    config_base_folder = curr

    main(vict_folder, env_folder, config_base_folder)

    print("------------------")
    print("---- FIM SMA -----")
    print("------------------")