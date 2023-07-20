import time
import pygame
import numpy as np
import itertools
import random

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)
COLOR_DEAD = (0, 0, 0)

noise = 0
n_trials = 20
n_generations = 250

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations]

def update(cells):
    n_rows, n_cols = cells.shape
    updated_cells = np.zeros_like(cells)

    for row, col in np.ndindex(cells.shape):
        alive = 0

        # Calculate neighbor indices considering cut-off edges
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbor_row = row + i
                neighbor_col = col + j

                # Check if neighbor indices are within the grid bounds
                if 0 <= neighbor_row < n_rows and 0 <= neighbor_col < n_cols:
                    alive += cells[neighbor_row, neighbor_col]

        # "noise" modification
        if random.random() < noise:
            if random.random() < 0.5:
                alive = max(0, alive - 1)  # ensure alive neighbors can't be less than 0
            else:
                alive += 1

        if cells[row, col] == 1:
            if alive < 2 or alive > 3 :
                updated_cells[row, col] = 0
            elif 2 <= alive <= 3 :
                updated_cells[row, col] = 1
        else:
            if alive == 3 :
                updated_cells[row, col] = 1

    return updated_cells


def main():
    start_time = time.time()

    for combination in combinations:
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
        print(f"Mean of sums: {mean}")
        print(f"Standard deviation of sums: {std_dev}")

    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()
