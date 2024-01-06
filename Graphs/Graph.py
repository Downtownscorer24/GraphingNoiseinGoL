import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys

# Assuming the parent folder of the script folder is in the PYTHONPATH
sys.path.append('..')  # Adjust this path based on your folder structure
import NoiseMainModified as treatment

# Parameters for the simulation
num_configs = 10
grid_size = (60, 80)
alive_probability = 0.25
num_timesteps = 250
noise_levels = [0, 0.01, 0.1, 0.5, 0.9, 1.0]

# Load initial configurations
initial_configurations = np.load('initial_configurations.npy', allow_pickle=True)

# Use 'initial_configurations' in your simulations


# Create a folder to store the graphs
graph_folder_path = 'main'  # Adjust the path as needed
Path(graph_folder_path).mkdir(parents=True, exist_ok=True)

def run_simulation_and_create_graph(noise_level, initial_configs, num_timesteps, folder_path):
    avg_alive_percentages = np.zeros(num_timesteps)

    for config in initial_configs:
        cells = np.copy(config)
        alive_percentages = []

        for _ in range(num_timesteps):
            cells = treatment.update(cells, noise_level)  # Update the cells based on your modified NoiseMain.py
            alive_percent = np.sum(cells) / cells.size
            alive_percentages.append(alive_percent)

        avg_alive_percentages += np.array(alive_percentages)

    avg_alive_percentages /= len(initial_configs)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_timesteps), avg_alive_percentages, label=f"Noise Level: {noise_level}")
    plt.xlabel('Timestep')
    plt.ylabel('% of Cells Alive')
    plt.title(f'Average % of Cells Alive Over Time (Noise Level {noise_level}), main')
    plt.ylim(0, 1)  # Set the y-axis limits to cap at 100%
    plt.legend()
    plt.grid(True)

    # Save the plot
    plot_filename = f'noise_level_{noise_level}.png'
    plt.savefig(os.path.join(folder_path, plot_filename))
    plt.close()

# Running simulations and creating graphs for each noise level
for noise_level in noise_levels:
    run_simulation_and_create_graph(noise_level, initial_configurations, num_timesteps, graph_folder_path)

print(f"Graphs saved in {graph_folder_path}")
