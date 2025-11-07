import os
import sys
import threading

import os
import sys
import threading
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
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
                try:
                    with open('debug_unified_victims.pkl', 'wb') as f:
                        pickle.dump(self.unified_victims, f)
                    print("DEBUG: 'debug_unified_victims.pkl' saved.")
                except Exception as e:
                    print(f"DEBUG: Failed to save victims pkl: {e}")
                self.combine_maps()
                self.create_clusters()

    def load_and_cluster_directly(self):
        print("--- DEBUG MODE: Skipping simulation ---")
        try:
            # 1. Load the pre-saved victim data
            with open('debug_unified_victims.pkl', 'rb') as f:
                self.unified_victims = pickle.load(f)
            print(f"Loaded {len(self.unified_victims)} victims from 'debug_unified_victims.pkl'")

            # 2. Call create_clusters
            self.share_count = 3
            self.create_clusters()

        except FileNotFoundError:
            print("ERROR: 'debug_unified_victims.pkl' not found.")
            print("You must run the simulation once (Etapa 1) to generate this file.")
        except Exception as e:
            print(f"ERROR loading debug data: {e}")


    def combine_maps(self):
        for explorer in self.explorers:
            explorer.map.map_data = self.unified_map_data

    def create_clusters(self):
        assert (self.share_count == 3)
        print("Creating clusters...")

        # 1. Load Classifier
        classifier = None
        try:
            with open('model_classifier.pkl', 'rb') as f:
                classifier = pickle.load(f)
            print("Model 'model_classifier.pkl' loaded.")
        except FileNotFoundError:
            print("ERROR: 'model_classifier.pkl' not found. Aborting clustering.")
            return
        except Exception as e:
            print(f"ERROR loading model: {e}. Aborting clustering.")
            return

        if not self.unified_victims:
            print("No unified victims to cluster.")
            return

        # 2. Process Victims and Prepare DataFrame
        processed_data = []
        for victim_seq, (coords, vital_signals) in self.unified_victims.items():
            # Extract the 10 features for the classifier
            vs_input = np.array(vital_signals[:10]).reshape(1, -1)

            tri = classifier.predict(vs_input)[0]

            x, y = coords

            # Calculate 'distance to base' feature
            dist_base = np.sqrt(x ** 2 + y ** 2)

            processed_data.append({
                'id_vict': victim_seq,
                'x': x,
                'y': y,
                'tri': tri,
                'dist_base': dist_base
            })

        if not processed_data:
            print("No data processed for clustering.")
            return

        df = pd.DataFrame(processed_data)

        # 3. Perform K-Means Clustering
        features_for_clustering = ['tri', 'dist_base']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features_for_clustering])

        k = 3

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df['cluster'] = kmeans.fit_predict(X_scaled)

        print(f"Clustering complete. {k} clusters generated.")

        # 4. Generate Plot
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df, x='x', y='y', hue='cluster', palette='deep', s=100, legend='full')
        plt.scatter(0, 0, marker='X', color='red', s=150, label='Base (0,0)', zorder=5)

        plt.title('Victim Clustering (k=3)')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        plot_filename = 'clusters_plot.png'
        plt.savefig(plot_filename)
        print(f"Clustering plot saved to '{plot_filename}'")
        plt.close()

        # 5. Save Cluster Files (Task Requirement)
        df['sobr'] = 0.0  # Placeholder as regressor is not used

        for cluster_num in range(k):
            cluster_df = df[df['cluster'] == cluster_num]

            if cluster_df.empty:
                continue

            filename = f"cluster_{cluster_num + 1}.txt"

            # Format: id_vict, x, y, sobr, tri
            output_df = cluster_df[['id_vict', 'x', 'y', 'sobr', 'tri']]

            output_df.to_csv(filename, index=False, header=['id_vict', 'x', 'y', 'sobr', 'tri'])

            print(f"Cluster file saved: {filename} ({len(cluster_df)} victims)")


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
    DEBUG_SKIP_SIMULATION = True
    # --------------------
    if DEBUG_SKIP_SIMULATION:
        shared_env_data.load_and_cluster_directly()
    else:
        try:
            env.run()
        except KeyboardInterrupt:
            print("Canceled by user")


if __name__ == '__main__':
    print("------------------")
    print("--- INICIO SMA ---")
    print("------------------")

    # dataset com sinais vitais das vitimas
    # vict_folder = os.path.join("datasets/vict/", "10v")
    vict_folder = os.path.join("datasets/vict/", "408v")

    # dataset do ambiente (paredes, posicao das vitimas)
    env_folder = os.path.join("datasets/env/", "94x94_408v")
    # env_folder = os.path.join("datasets/env/", "12x12_10v")

    # folder das configuracoes dos agentes
    # This is now the BASE folder containing config_ag_1, config_ag_2, etc.
    curr = os.getcwd()
    config_base_folder = os.path.join(curr, "sma", "3exp_3soc")

    main(vict_folder, env_folder, config_base_folder)

    print("------------------")
    print("---- FIM SMA -----")
    print("------------------")