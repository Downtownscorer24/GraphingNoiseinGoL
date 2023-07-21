import time
import numpy as np
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve
import csv
import io

noise = 0
n_trials = 20
n_generations = 250

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations[:8]]  # Only first 16 combinations

def binary_matrix_to_decimal(matrix):
    # Flatten the matrix into a 1D array
    flat_matrix = matrix.flatten()

    # Convert the 1D array to a string of binary digits
    binary_str = ''.join(map(str, flat_matrix))

    # Convert the binary string to a decimal number
    decimal = int(binary_str, 2)

    return decimal
def update(cells):
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


def process_combination(combination):
    sums = []
    for _ in range(n_trials):
        # Create a 59x59 grid of zeros
        cells = np.zeros((59, 59))

        # Place the 3x3 combination in the middle of the 59x59 grid
        start_row = start_col = (59 - 3) // 2
        cells[start_row:start_row + 3, start_col:start_col + 3] = combination

        for _ in range(n_generations):
            cells = update(cells)
        sums.append(np.sum(cells))

    mean = np.mean(sums)
    std_dev = np.std(sums)

    return {"combination": combination, "mean": mean, "std_dev": std_dev}

def main():
    start_time = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_combination, combinations))

    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time} seconds")

    # Use StringIO object as file
    output = io.StringIO()

    writer = csv.writer(output)

    # Write headers
    writer.writerow(["Combination", "Mean", "Std Dev"])

    # Write data
    for result in results:
        combination_decimal = binary_matrix_to_decimal(result['combination'])
        writer.writerow([combination_decimal, result['mean'], result['std_dev']])

    # Get the CSV content
    csv_content = output.getvalue()

    # Close the StringIO object
    output.close()

    # Print the CSV content
    print(csv_content)

if __name__ == '__main__':
    main()
