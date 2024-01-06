import numpy as np

def generate_initial_configurations(num_configs, grid_size, alive_probability):
    return [np.random.choice([0, 1], size=grid_size, p=[1-alive_probability, alive_probability]) for _ in range(num_configs)]

# Parameters for the simulation
num_configs = 10
grid_size = (60, 80)
alive_probability = 0.25

# Generate initial configurations
initial_configurations = generate_initial_configurations(num_configs, grid_size, alive_probability)

# Save to a file (e.g., numpy binary file)
np.save('initial_configurations.npy', initial_configurations)
