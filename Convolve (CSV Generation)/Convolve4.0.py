import time
import numpy as np
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
import csv
import io
import multiprocessing
from scipy.signal import convolve2d

n_trials = 100
n_generations = 256
grid_size = 64

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations]

def binary_matrix_to_decimal(matrix):
    # Flatten the matrix into a 1D array
    flat_matrix = matrix.flatten()

    # Convert the 1D array to a string of binary digits
    binary_str = ''.join(map(str, flat_matrix))

    # Convert the binary string to a decimal number
    decimal = int(binary_str, 2)

    return decimal

# Additional functions needed for the update logic
def apply_noise(cells, noise_probability):
    noise_mask = np.random.rand(*cells.shape) < noise_probability
    noise_adjustment = np.where(noise_mask, np.random.choice([-1, 1], size=cells.shape), 0)
    return noise_adjustment

def calculate_neighbors(cells):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    neighbor_sum = convolve2d(cells, kernel, mode='same', boundary='wrap')
    return neighbor_sum

# Modified update function with regression model logic
def update(cells, noise_probability):
    # Apply Noise
    noise_adjustment = apply_noise(cells, noise_probability)

    # Calculate Noised Neighbor Count
    noised_neighbor_count = calculate_neighbors(cells) + noise_adjustment

    # Regression Model (noised neighbor count, neighbors' noised neighbor counts ---> guess)
    true_neighbor_count = -0.068 + 0.803 * noised_neighbor_count

    # Ensure true_neighbor_count is within valid range [0, 8] and round to nearest integer
    true_neighbor_count = np.round(np.clip(true_neighbor_count, 0, 8)).astype(int)

    # Game Logic with True Neighbor Count
    updated_cells = np.zeros_like(cells)
    birth = (true_neighbor_count == 3)
    survive = ((true_neighbor_count == 2) | (true_neighbor_count == 3)) & (cells == 1)
    updated_cells[birth | survive] = 1

    return updated_cells


def process_combination(params):
    combination, noise = params
    sums = []
    for _ in range(n_trials):
        # Create a 64x64 grid of zeros
        cells = np.zeros((grid_size, grid_size))

        # Place the 3x3 combination in the middle of the 64x64 grid
        start_row = start_col = (grid_size - 3) // 2
        cells[start_row:start_row + 3, start_col:start_col + 3] = combination

        for _ in range(n_generations):
            cells = update(cells, noise)
        sums.append(np.sum(cells))

    mean = np.mean(sums)
    std_dev = np.std(sums)

    # Add a small constant to the denominator to prevent division by zero
    epsilon = 1e-7
    cv = std_dev/(mean + epsilon)

    return {"combination": combination,"noise level": noise, "mean": mean, "std_dev": std_dev, "cv": cv}

def main():
    start_time = time.time()

    all_results = []

    num_cpus = multiprocessing.cpu_count()  # get number of VCPUs

    noise_values = np.arange(0, 1.01, 0.01)  # noise values from 0 to 1 in increments of 0.01
    for noise in noise_values:
        params = [(comb, noise) for comb in combinations]
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(process_combination, params))
        all_results.extend(results)

    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time} seconds")

    # Open (or create) a csv file in write mode
    with open("output1.csv", 'w', newline='') as file:
        writer = csv.writer(file)

        # Write headers
        writer.writerow(["Noise Level","Combination", "Mean", "Std Dev", "CV"])

        # Write data
        for result in all_results:
            combination_decimal = binary_matrix_to_decimal(result['combination'])
            writer.writerow([ result['noise level'], combination_decimal, result['mean'], result['std_dev'], result['cv']])

if __name__ == '__main__':
    main()
