# This is a normal version of the GoL (normal rules, normal neighborhood sizes)
# The borders are unwrapped, so no cells from one edge interact with the opposite edge
# You can set how long the simulation runs for
# A graph will appear at the end detailing what percentage of the grid is alive across generations

import time
import pygame
import numpy as np
import matplotlib.pyplot as plt
import random

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)
COLOR_DEAD = (0, 0, 0)

death_prob = .99
survival_prob = .99
birth_prob = .99

def update(screen, cells, size, with_progress=False):
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

        color = COLOR_BG if cells[row, col] == 0 else COLOR_ALIVE_NEXT

        if cells[row, col] == 1:
            if alive < 2 or alive > 3 and random.random() < death_prob:
                updated_cells[row, col] = 0
                if with_progress:
                    color = COLOR_DIE_NEXT

            elif 2 <= alive <= 3 and random.random() < survival_prob:
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT
        else:
            if alive == 3 and random.random() < birth_prob:
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT


        pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))

    return updated_cells


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    cells = np.zeros((60, 80))
    screen.fill(COLOR_GRID)
    update(screen, cells, 10)

    pygame.display.flip()
    pygame.display.update()

    running = False
    cell_deactivation = False
    start_simulation = False
    generations = 0
    percent_alive = []

# Set how long the simulation runs for
    while generations < 250:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    start_simulation = True
                    update(screen, cells, 10)
                    pygame.display.update()
                elif event.key == pygame.K_d:
                    cell_deactivation = not cell_deactivation
            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if cell_deactivation:
                    cells[pos[1] // 10, pos[0] // 10] = 0
                else:
                    cells[pos[1] // 10, pos[0] // 10] = 1
                update(screen, cells, 10)
                pygame.display.update()

        screen.fill(COLOR_GRID)

        if running:
            if start_simulation:
                # Calculate the percentage for the initial configuration
                alive_count = np.sum(cells)
                total_cells = cells.size
                percent_alive.append(alive_count / total_cells * 100)
                start_simulation = False

            cells = update(screen, cells, 10, with_progress=True)
            pygame.display.update()

            # Calculate the percentage of cells alive
            alive_count = np.sum(cells)
            total_cells = cells.size
            percent_alive.append(alive_count / total_cells * 100)
            generations += 1

        time.sleep(0.001)

    # Plot the graph after the simulation
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, generations + 2), percent_alive)  # +2 to account for the initial configuration
    plt.title("Percentage of Cells Alive Across Generations")
    plt.xlabel("Generations")
    plt.ylabel("Percentage of Cells Alive (%)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

