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

    # Initialize true_neighbor_count with zeros
    true_neighbor_count = np.zeros_like(cells, dtype=float)

    # Apply the corrected models based on noised_neighbor_count
    true_neighbor_count[noised_neighbor_count == 0] = (
            -3.97457252811653E-02 + 2.63517962151024E-02 * neighbors_noised_neighbor_sum[noised_neighbor_count == 0]
    )

    true_neighbor_count[noised_neighbor_count == 1] = (
            -0.194932201455905 + 8.75308708573203E-02 * neighbors_noised_neighbor_sum[noised_neighbor_count == 1]
    )

    true_neighbor_count[noised_neighbor_count == 2] = (
            1.02854754427366 + 5.61051757908324E-02 * neighbors_noised_neighbor_sum[noised_neighbor_count == 2]
    )

    mask_3_5 = (noised_neighbor_count >= 3) & (noised_neighbor_count <= 5)
    true_neighbor_count[mask_3_5] = (
            -0.258180864139362 + 0.787647654473264 * noised_neighbor_count[mask_3_5] +
            3.29954507970766E-02 * neighbors_noised_neighbor_sum[mask_3_5]
    )

    mask_6_7 = (noised_neighbor_count >= 6) & (noised_neighbor_count <= 7)
    true_neighbor_count[mask_6_7] = (
            -1.79207022505006 + 1.07037822552138 * noised_neighbor_count[mask_6_7] +
            2.18569577000118E-02 * neighbors_noised_neighbor_sum[mask_6_7]
    )

    # Ensure true_neighbor_count is within valid range [0, 8] and round to nearest integer
    true_neighbor_count = np.round(np.clip(true_neighbor_count, 0, 8)).astype(int)

    # Game Logic with True Neighbor Count
    updated_cells = np.zeros_like(cells)
    birth = (true_neighbor_count == 3)
    survive = ((true_neighbor_count == 2) | (true_neighbor_count == 3)) & (cells == 1)
    updated_cells[birth | survive] = 1

    return updated_cells
