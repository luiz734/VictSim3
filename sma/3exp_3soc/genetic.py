import random
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deap import base, creator, tools

# --- Configuration ---
DEBUG = True
GRID_LIMIT = 50  # 94x94 grid -> approx -47 to +47. 50 provides margin.

# Global flag for interaction
stop_evolution = False


def on_key(event):
    """Callback to handle key presses during debug mode."""
    global stop_evolution
    if event.key == ' ':
        stop_evolution = True
        print("Stopping evolution for current file...")


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


def update_plot(df, best_order_ids, index, ax, generation):
    """Updates the existing plot with the current generation's best path."""
    ax.clear()

    # 1. Setup Grid and Limits
    ax.set_xlim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_ylim(-GRID_LIMIT, GRID_LIMIT)
    ax.set_title(f'Optimizing File {index} | Gen: {generation} | (Press SPACE to Finish)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, linestyle='--', alpha=0.4)

    # 2. Plot Base
    ax.scatter(0, 0, marker='X', color='red', s=150, label='Base', zorder=5)

    # 3. Plot Victims
    if 'tri' in df.columns:
        # Map triage classes to colors manually or via seaborn if desirable
        colors = {0: 'black', 1: 'red', 2: 'orange', 3: 'green'}
        c_list = [colors.get(r['tri'], 'blue') for _, r in df.iterrows()]
        ax.scatter(df['x'], df['y'], c=c_list, s=100, zorder=4)
    else:
        ax.scatter(df['x'], df['y'], c='blue', s=100, zorder=4)

    # 4. Trace Path
    path_coords = []
    for vid in best_order_ids:
        row = df[df['id_vict'] == vid].iloc[0]
        path_coords.append((row['x'], row['y']))
        ax.text(row['x'] + 1, row['y'] + 1, str(int(vid)), fontsize=10)

    if path_coords:
        # Unzip coordinates
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, color='gray', linestyle='--', alpha=0.6, marker='')

        # Add directional arrows for clarity
        for i in range(len(path_coords) - 1):
            p1 = path_coords[i]
            p2 = path_coords[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            ax.arrow(p1[0], p1[1], dx, dy, head_width=2, head_length=2, fc='gray', ec='gray', length_includes_head=True,
                     alpha=0.5)


def get_visit_order(index):
    global stop_evolution
    stop_evolution = False  # Reset flag for new file

    # 1. Load Data
    filename = f"cluster_{index}.txt"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"File {filename} not found.")
        return []

    victims = {int(row['id_vict']): {'x': row['x'], 'y': row['y']} for _, row in df.iterrows()}
    ids_list = list(victims.keys())
    num_victims = len(ids_list)

    # 2. Setup DEAP
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(num_victims), num_victims)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_path, ids_list=ids_list, data_dict=victims)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 3. Initialization
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    # Evaluate initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Setup Plotting for Debug
    if DEBUG:
        plt.ion()  # Interactive mode on
        fig, ax = plt.subplots(figsize=(8, 8))
        fig.canvas.mpl_connect('key_press_event', on_key)

    # 4. Evolution Loop
    gen = 0
    while True:
        # GA Steps
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
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        gen += 1

        # Debug Visualization
        if DEBUG:
            best_indices = list(hof[0])
            best_order_ids = [ids_list[i] for i in best_indices]

            update_plot(df, best_order_ids, index, ax, gen)
            plt.draw()
            plt.pause(0.05)  # Brief pause to render and catch events

            # Check for Spacebar press
            if stop_evolution:
                plt.close(fig)
                break
        else:
            # Standard stop condition if not debugging
            if gen >= 100:
                break

    # Final Result
    best_indices = list(hof[0])
    best_order_ids = [ids_list[i] for i in best_indices]

    # Save final static plot if needed
    if not DEBUG:
        # You can call a static plot function here if desired
        pass

    return best_order_ids