import time
import numpy as np
import itertools
import random

noise = 0
n_trials = 20
n_generations = 250
grid_size = 3  # size of the grid

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations]


def update(cells):
    # Prepare neighbor grid
    padded_cells = np.pad(cells, pad_width=1, mode='constant', constant_values=0)
    neighbor_count = sum(np.roll(np.roll(padded_cells, i, 0), j, 1)
                         for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))

    # Apply noise
    noise_mask = np.random.random(cells.shape) < noise
    noise_adjustment = np.where(np.random.random(cells.shape) < 0.5, -1, 1)
    noise_adjustment *= noise_mask
    neighbor_count[1:-1, 1:-1] = np.clip(neighbor_count[1:-1, 1:-1] + noise_adjustment, 0, 8)

    # Apply game rules
    born = np.logical_and(cells == 0, neighbor_count[1:-1, 1:-1] == 3)
    survive = np.logical_and(cells == 1, np.logical_or(neighbor_count[1:-1, 1:-1] == 2, neighbor_count[1:-1, 1:-1] == 3))

    return np.where(np.logical_or(born, survive), 1, 0)

def main():
    all_sums = []

    for combination in combinations:
        sums = []
        for _ in range(n_trials):
            cells = np.zeros((grid_size, grid_size))
            start_row, start_col = grid_size // 2 - 1, grid_size // 2 - 1
            cells[start_row:start_row+3, start_col:start_col+3] = combination
            for _ in range(n_generations):
                cells = update(cells)
            sums.append(np.sum(cells))
        all_sums.append(np.mean(sums))

    mean_of_means = np.mean(all_sums)
    std_dev = np.std(all_sums)

    print(f"Mean of means: {mean_of_means}")
    print(f"Standard deviation: {std_dev}")


if __name__ == '__main__':
    main()
