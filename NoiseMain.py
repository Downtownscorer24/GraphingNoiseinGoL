# Initial configuration: random cells alive; % dictated by random_percent_pop
# Noise factored in by noise_probability
# No treatment

import time
import pygame
import numpy as np
from scipy.signal import convolve2d


COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)
COLOR_DEAD = (0, 0, 0)

noise_probability = 0
random_percent_pop = 0

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

def draw_grid(screen, size):
    for x in range(0, screen.get_width(), size):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, screen.get_height()))
    for y in range(0, screen.get_height(), size):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (screen.get_width(), y))

def render(screen, cells, size):
    screen.fill(COLOR_BG)
    draw_grid(screen, size)
    alive_cells = np.argwhere(cells == 1)
    for row, col in alive_cells:
        pygame.draw.rect(screen, COLOR_ALIVE_NEXT, (col * size, row * size, size - 1, size - 1))



def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    total_cells = 60 * 80
    alive_cells = int(total_cells * random_percent_pop)
    cells_flat = np.concatenate([np.ones(alive_cells), np.zeros(total_cells - alive_cells)])
    np.random.shuffle(cells_flat)
    cells = cells_flat.reshape(60, 80)
    cells = np.random.choice([0, 1], size=(60, 80), p=[1 - random_percent_pop, random_percent_pop])

    render(screen, cells, 10)
    pygame.display.flip()
    pygame.display.update()

    running = False
    cell_deactivation = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    initial_config = np.copy(cells)
                elif event.key == pygame.K_d:
                    cell_deactivation = not cell_deactivation
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if cell_deactivation:
                    cells[pos[1] // 10, pos[0] // 10] = 0
                else:
                    cells[pos[1] // 10, pos[0] // 10] = 1
                render(screen, cells, 10)
                pygame.display.update()

        if running:
            cells = update(cells, noise_probability)
            render(screen, cells, 10)
            pygame.display.update()

        time.sleep(0.001)



if __name__ == '__main__':
    main()
