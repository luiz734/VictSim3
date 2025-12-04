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
    """
    Fitness = Total Distance + Weighted Latency Penalty

    Strategy:
    - We track 'current_time' (accumulated distance).
    - Every time we arrive at a victim, we penalize based on how long it took
      to get there multiplied by their urgency.
    """
    total_distance = 0.0
    current_time = 0.0
    weighted_penalty = 0.0

    # Priority Weights based on START Triage
    # 2 (Red):     Most Urgent -> High Penalty for delay
    # 1 (Yellow):  Urgent -> Medium Penalty
    # 0 (Green):   Not Urgent -> Low Penalty
    # 3 (Black):   Deceased -> No Penalty for delay
    urgency_weights = {
        2: 100.0,
        1: 10.0,
        0: 1.0,
        3: 0.0
    }

    # 1. Start at Base (0,0)
    prev_x, prev_y = 0.0, 0.0

    for i in range(len(individual)):
        # Get target victim
        victim_id = ids_list[individual[i]]
        victim_data = data_dict[victim_id]

        # Calculate distance from previous location (or base) to this victim
        dist = math.sqrt((prev_x - victim_data['x']) ** 2 + (prev_y - victim_data['y']) ** 2)

        # Update accumulators
        total_distance += dist
        current_time += dist

        # Apply Latency Penalty
        # "How long did it take to get here?" * "How important are they?"
        tri_class = victim_data['tri']
        weight = urgency_weights.get(tri_class, 0)
        weighted_penalty += (current_time * weight)

        # Update previous coordinates for next iteration
        prev_x, prev_y = victim_data['x'], victim_data['y']

    # OPTIONAL: Add return to base distance if required by rules,
    # but usually Triage optimization stops at the last victim.
    # If return to base is mandatory for total cost, uncomment below:
    # dist_home = math.sqrt((prev_x - 0)**2 + (prev_y - 0)**2)
    # total_distance += dist_home

    # Combine metrics.
    # The penalty will likely be much larger than distance, effectively driving the sorting.
    final_fitness = total_distance + weighted_penalty

    return (final_fitness,)


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
    ax.set_title(f'File {index} | {strategy} | Gen: {generation} | Fit: {current_dist:.0f}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.4)

    # Plot Base
    ax.scatter(0, 0, marker='X', color='blue', s=150, label='Base', zorder=5)

    # UPDATED: Plot Victims with correct START colors
    # 0=Green, 1=Yellow, 2=Red, 3=Black
    if 'tri' in df.columns:
        color_map = {0: 'green', 1: 'gold', 2: 'red', 3: 'black'}
        c_list = [color_map.get(r['tri'], 'gray') for _, r in df.iterrows()]
        ax.scatter(df['x'], df['y'], c=c_list, s=100, edgecolors='white', zorder=4)
    else:
        ax.scatter(df['x'], df['y'], c='gray', s=100, zorder=4)

    # Trace Path
    path_coords = []
    # Add Base as start point for visualization line
    path_coords.append((0, 0))

    for vid in best_order_ids:
        row = df[df['id_vict'] == vid].iloc[0]
        path_coords.append((row['x'], row['y']))
        # ax.text(row['x'] + 1, row['y'] + 1, str(int(vid)), fontsize=10)

    if path_coords:
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, color='gray', linestyle='--', alpha=0.6, marker='')

        # Arrow logic (optional, keeping your existing style)
        for i in range(len(path_coords) - 1):
            p1 = path_coords[i]
            p2 = path_coords[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            # Only draw arrows if points are far enough apart to see them
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                ax.arrow(p1[0], p1[1], dx, dy, head_width=1.5, head_length=1.5,
                         fc='gray', ec='gray', length_includes_head=True, alpha=0.5)


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
        return [], 0.0

    # UPDATED: Load 'tri' column along with x and y
    victims = {
        int(row['id_vict']): {
            'x': row['x'],
            'y': row['y'],
            'tri': int(row['tri'])
        }
        for _, row in df.iterrows()
    }
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