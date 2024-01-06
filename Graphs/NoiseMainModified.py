# Initial configuration: random cells alive; % dictated by random_percent_pop
# Noise factored in by noise_probability
# No treatment

import time
import pygame
import numpy as np
from scipy.signal import convolve2d

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

def update(cells, noise_probability):

    # Apply Noise
    noise_adjustment = apply_noise(cells, noise_probability)

    # Calculate Noised Neighbor Count
    noised_neighbor_count = calculate_neighbors(cells) + noise_adjustment


    # Game Logic with True Neighbor Count
    updated_cells = np.zeros_like(cells)
    birth = (noised_neighbor_count == 3)
    survive = ((noised_neighbor_count == 2) | (noised_neighbor_count == 3)) & (cells == 1)
    updated_cells[birth | survive] = 1

    return updated_cells
