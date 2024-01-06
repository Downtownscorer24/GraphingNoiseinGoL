# Initial configuration: random cells alive; % dictated by random_percent_pop
# Noise factored in by noise_probability
# regression model used to help cells guess what true neighbor count is

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

    # Calculate Sum of Neighbors' Noised Neighbor Counts
    neighbors_noised_neighbor_sum = calculate_neighbors(noised_neighbor_count)

    # Regression Model (noised neighbor count, neighbors' noised neighbor counts, noise level ---> guess)
    #true_neighbor_count = -0.0447713734307326 + 0.407165590149942 * noised_neighbor_count + 0.0830249042956387 * neighbors_noised_neighbor_sum - 0.419004103993928 * noise_probability

    # Regression Model (noised neighbor count, neighbors' noised neighbor counts ---> guess)
    true_neighbor_count = -0.236028491845496 + 0.406401047839722 * noised_neighbor_count + 0.078265356869915 * neighbors_noised_neighbor_sum

    # Ensure true_neighbor_count is within valid range [0, 8] and round to nearest integer
    true_neighbor_count = np.round(np.clip(true_neighbor_count, 0, 8)).astype(int)

    # Game Logic with True Neighbor Count
    updated_cells = np.zeros_like(cells)
    birth = (true_neighbor_count == 3)
    survive = ((true_neighbor_count == 2) | (true_neighbor_count == 3)) & (cells == 1)
    updated_cells[birth | survive] = 1

    return updated_cells


