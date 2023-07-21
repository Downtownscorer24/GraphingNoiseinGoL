import time
import numpy as np
import itertools
import random
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import convolve

noise = 0
n_trials = 20
n_generations = 250

# Generate all possible 3x3 combinations
combinations = list(itertools.product([0, 1], repeat=9))
combinations = [np.array(comb).reshape((3, 3)) for comb in combinations[:16]]  # Only first 16 combinations

def binary_matrix_to_decimal(matrix):
    # Flatten the matrix into a 1D array
    flat_matrix = matrix.flatten()

    # Convert the 1D array to a string of binary digits
    binary_str = ''.join(map(str, flat_matrix))

    # Convert the binary string to a decimal number
    decimal = int(binary_str, 2)

    return decimal



def main():
    start_time = time.time()

    for comb in combinations:
        print(binary_matrix_to_decimal(comb))



    end_time = time.time()
    print(f"Time taken to run the function: {end_time - start_time} seconds")



if __name__ == '__main__':
    main()
