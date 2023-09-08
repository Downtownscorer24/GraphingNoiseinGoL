import time
import numpy as np
import itertools
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
import csv
import multiprocessing

n_trials = 32
n_generations = 256
grid_size = 64

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations[:4]]

def binary_matrix_to_decimal(matrix):
    flat_matrix = matrix.flatten()
    binary_str = ''.join(map(str, flat_matrix))
    decimal = int(binary_str, 2)
    return decimal

def get_neighbor_counts(cells):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    return convolve(cells, kernel, mode='wrap')

def predict_future_state(cells, noise):
    alive = get_neighbor_counts(cells)

    is_noised = np.random.random(cells.shape) < noise
    noise_values = np.random.choice([-1, 1], size=cells.shape)
    noise_values *= is_noised

    alive = np.clip(alive + noise_values, 0, 8)
    future_state = np.where(((cells == 1) & ((alive == 2) | (alive == 3))) |
                            ((cells == 0) & (alive == 3)), 1, 0)
    return future_state

def update(cells, noise):
    neighbor_counts = get_neighbor_counts(cells)
    future_state = predict_future_state(cells, noise)
    future_neighbor_counts = get_neighbor_counts(future_state)
    alive_next_timestep = future_neighbor_counts + future_state

    next_state = np.where(
        (cells == 0) & (alive_next_timestep == 3), 1,
        np.where(
            (cells == 1) & (2 <= alive_next_timestep) & (alive_next_timestep <= 3), 1,
            np.where(
                (4 <= alive_next_timestep) & (alive_next_timestep <= 6), future_state,
                0
            )
        )
    )

    return next_state

def process_combination(params):
    combination, noise = params
    sums = []
    for _ in range(n_trials):
        cells = np.zeros((grid_size, grid_size))
        start_row = start_col = (grid_size - 3) // 2
        cells[start_row:start_row + 3, start_col:start_col + 3] = combination

        for _ in range(n_generations):
            cells = update(cells, noise)
        sums.append(np.sum(cells))

    mean = np.mean(sums)
    std_dev = np.std(sums)
    epsilon = 1e-7
    cv = std_dev/(mean + epsilon)

    return {"combination": combination,"noise level": noise, "mean": mean, "std_dev": std_dev, "cv": cv}

def main():
    start_time = time.time()
    all_results = []
    num_cpus = multiprocessing.cpu_count()
    noise_values = np.arange(0, 1.01, 0.01)

    for noise in noise_values:
        params = [(comb, noise) for comb in combinations]
        with ProcessPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(process_combination, params))
        all_results.extend(results)

    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time} seconds")

    with open("output1.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Noise Level","Combination", "Mean", "Std Dev", "CV"])

        for result in all_results:
            combination_decimal = binary_matrix_to_decimal(result['combination'])
            writer.writerow([ result['noise level'], combination_decimal, result['mean'], result['std_dev'], result['cv']])

if __name__ == '__main__':
    main()
