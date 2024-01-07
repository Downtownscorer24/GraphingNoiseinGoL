import time
import numpy as np
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
import csv
import io
import multiprocessing

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

def update(cells, noise):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    alive = convolve(cells, kernel, mode='constant', cval=0)

    # "noise" modification
    is_noised = np.random.random(cells.shape) < noise
    noise_values = np.random.choice([-1, 1], size=cells.shape)
    noise_values *= is_noised
    alive = np.clip(alive + noise_values, 0, None)  # ensure alive neighbors can't be less than 0

    updated_cells = np.where(((cells == 1) & ((alive < 2) | (alive > 3))) |
                             ((cells == 0) & (alive != 3)), 0, 1)

    return updated_cells

def process_combination(params):
    combination, noise = params
    sums = []
    for _ in range(n_trials):
        # Create a 64x64 grid of zeros
        cells = np.zeros((grid_size, grid_size))

        # Place the 3x3 combination in the middle of the 59x59 grid
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
