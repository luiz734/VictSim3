import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import time
from deap import base, creator, tools

# --- Configuration ---
GRID_LIMIT = 50
CPU_CORES = 10
POPULATION_SIZE = 50
BLOB_RATIO = 0.1  # 10% of the population will be seeded
BLOB_LENGTH_PCT = 0.15  # Length of the optimized segment

# Global flag for interaction (only used in debug/single run)
stop_evolution = False


def calculate_distance(p1, p2):
    return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)


def eval_path(individual, ids_list, data_dict):
    distance = 0.0
    for i in range(len(individual) - 1):
        id_1 = ids_list[individual[i]]
        id_2 = ids_list[individual[i + 1]]
        p1 = data_dict[id_1]
        p2 = data_dict[id_2]
        distance += calculate_distance(p1, p2)
    return (distance,)


def create_blob_individual(ids_list, data_dict):
    num_victims = len(ids_list)
    indices = list(range(num_victims))
    unvisited = set(indices)

    segment_length = max(2, int(num_victims * BLOB_LENGTH_PCT))

    current_idx = random.choice(indices)
    path = [current_idx]
    unvisited.remove(current_idx)

    for _ in range(segment_length - 1):
        if not unvisited:
            break
        current_id = ids_list[current_idx]
        p1 = data_dict[current_id]

        nearest_idx = -1
        min_dist = float('inf')

        for candidate_idx in unvisited:
            candidate_id = ids_list[candidate_idx]
            p2 = data_dict[candidate_id]
            dist = calculate_distance(p1, p2)

            if dist < min_dist:
                min_dist = dist
                nearest_idx = candidate_idx

        current_idx = nearest_idx
        path.append(current_idx)
        unvisited.remove(current_idx)

    remaining = list(unvisited)
    random.shuffle(remaining)
    return creator.Individual(path + remaining)


def update_plot(df, best_order_ids, index, ax, generation, current_dist, strategy):
    ax.clear()
    ax.set_xlim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_ylim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_title(f'File {index} | {strategy} | Gen: {generation} | Dist: {current_dist:.2f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Plot Base
    ax.scatter(0, 0, marker='X', color='red', s=150, label='Base', zorder=5)

    # Plot Victims
    if 'tri' in df.columns:
        colors = {0: 'black', 1: 'red', 2: 'orange', 3: 'green'}
        c_list = [colors.get(r['tri'], 'blue') for _, r in df.iterrows()]
        ax.scatter(df['x'], df['y'], c=c_list, s=100, zorder=4)
    else:
        ax.scatter(df['x'], df['y'], c='blue', s=100, zorder=4)

    path_coords = []
    for vid in best_order_ids:
        row = df[df['id_vict'] == vid].iloc[0]
        path_coords.append((row['x'], row['y']))
        # ax.text(row['x'] + 1, row['y'] + 1, str(int(vid)), fontsize=10) # Hide text for speed in debug

    if path_coords:
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, color='gray', linestyle='--', alpha=0.6, marker='')


def get_visit_order(index, strategy='RANDOM', debug_mode=False):
    """
    Args:
        index (int): File index.
        strategy (str): 'RANDOM' or 'HYBRID'.
        debug_mode (bool): If True, shows plot. If False, runs silently.
    Returns:
        tuple: (best_order_ids, best_fitness_value)
    """
    global stop_evolution
    stop_evolution = False

    # 1. Load Data
    filename = f"cluster_{index}.txt"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        # Generate dummy data for testing if file missing
        # print(f"File {filename} not found. Using random dummy data.")
        # df = pd.DataFrame({'id_vict': range(20), 'x': np.random.uniform(-40,40,20), 'y': np.random.uniform(-40,40,20)})
        return [], 0.0

    victims = {int(row['id_vict']): {'x': row['x'], 'y': row['y']} for _, row in df.iterrows()}
    ids_list = list(victims.keys())
    num_victims = len(ids_list)

    # 2. Setup DEAP
    # Ensure classes exist
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_victims), num_victims)
    toolbox.register("individual_random", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("individual_blob", create_blob_individual, ids_list=ids_list, data_dict=victims)
    toolbox.register("evaluate", eval_path, ids_list=ids_list, data_dict=victims)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3. Execution
    # Note: We use a pool for each run. For massive benchmarks, creating pools
    # repeatedly has overhead, but it ensures clean state for each run.
    with multiprocessing.Pool(processes=CPU_CORES) as pool:
        toolbox.register("map", pool.map)

        if strategy == 'HYBRID':
            num_blob = int(POPULATION_SIZE * BLOB_RATIO)
            num_random = POPULATION_SIZE - num_blob
            pop = [toolbox.individual_blob() for _ in range(num_blob)] + \
                  [toolbox.individual_random() for _ in range(num_random)]
        else:
            pop = [toolbox.individual_random() for _ in range(POPULATION_SIZE)]

        hof = tools.HallOfFame(1)

        # Evaluate initial
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Visualization Setup
        if debug_mode:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))

        # Evolution Loop
        gen = 0
        MAX_GEN = 100

        while gen < MAX_GEN:
            offspring = toolbox.select(pop, len(pop))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.7:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if invalid_ind:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

            pop[:] = offspring
            hof.update(pop)
            gen += 1

            if debug_mode:
                best_indices = list(hof[0])
                best_ids = [ids_list[i] for i in best_indices]
                curr_fit = hof[0].fitness.values[0]
                update_plot(df, best_ids, index, ax, gen, curr_fit, strategy)
                plt.draw()
                plt.pause(0.001)

        best_indices = list(hof[0])
        best_order_ids = [ids_list[i] for i in best_indices]
        best_fitness = hof[0].fitness.values[0]

        if debug_mode:
            plt.close(fig)

        return best_order_ids, best_fitness