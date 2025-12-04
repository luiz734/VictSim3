import os
import sys
import threading

import os

import numpy as np

from vs.environment import Env
from agents_manager import AgentsManager
from genetic import get_visit_order
from create_figures import save_benchmarks

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


def benchmak_strategies():
    # Create/Open file for writing
    output_file = "experiment_results.txt"

    print(f"Starting Benchmark... Results will be saved to {output_file}")

    # Write Header
    with open(output_file, "w") as f:
        f.write("File_Index,Strategy,Mean_Fitness,Std_Dev,Min_Fitness,Max_Fitness\n")

    strategies = ['RANDOM', 'HYBRID']
    n_runs = 50
    files = [1, 2, 3]

    for file_idx in files:
        for strategy in strategies:
            print(f"\nProcessing File {file_idx} | Strategy: {strategy}")
            fitness_results = []

            for i in range(n_runs):
                # Run algorithm silently (debug_mode=False)
                _, fitness = get_visit_order(file_idx, strategy=strategy, debug_mode=False)
                fitness_results.append(fitness)

                # Simple progress indicator
                print(f".", end="", flush=True)

            # Calculate Statistics
            fit_mean = np.mean(fitness_results)
            fit_std = np.std(fitness_results)
            fit_min = np.min(fitness_results)
            fit_max = np.max(fitness_results)

            print(f"\nStats -> Mean: {fit_mean:.2f} | Min: {fit_min:.2f}")

            # Append to file
            with open(output_file, "a") as f:
                f.write(f"{file_idx},{strategy},{fit_mean:.4f},{fit_std:.4f},{fit_min:.4f},{fit_max:.4f}\n")

    print("\nBenchmark Complete!")

if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")

    # benchmak_strategies()
    # save_benchmarks()
    # # for index in range(3):
    # #     order = get_visit_order(index + 1,  strategy='HYBRID', debug_mode=True)
    # #     print(order)
    # exit(0)

    # run_params = RUN_PARAMS_10V
    run_params = RUN_PARAMS_408V

    # folder das configuracoes dos agentes
    # This is now the BASE folder containing config_ag_1, config_ag_2, etc.
    curr = os.getcwd()
    config_base_folder = os.path.join(curr, "sma", "3exp_3soc")

    main(run_params["vict"], run_params["env"], config_base_folder)

    print("------------------")
    print("---- FIM SMA -----")
    print("------------------")
