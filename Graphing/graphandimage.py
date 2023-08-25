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
            if alive < 2 or alive > 3 :
                updated_cells[row, col] = 0
                if with_progress:
                    color = COLOR_DIE_NEXT

            elif 2 <= alive <= 3 :
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT
        else:
            if alive == 3 :
                updated_cells[row, col] = 1
                if with_progress:
                    color = COLOR_ALIVE_NEXT

        pygame.draw.rect(screen, color, (col * size, row * size, size - 1, size - 1))

    return updated_cells


def main():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    cells = np.zeros((60, 80))
    initial_config = None  # New variable to store initial configuration

    screen.fill(COLOR_GRID)
    update(screen, cells, 10)

    pygame.display.flip()
    pygame.display.update()

    running = False
    cell_deactivation = False
    start_simulation = False
    generations = 0
    percent_alive = []

    while generations < 250:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = not running
                    start_simulation = True
                    initial_config = np.copy(cells)  # Save initial configuration when simulation starts
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

    # Plot the graph and the initial configuration after the simulation
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 12))

    ax0.imshow(initial_config, cmap='binary')
    ax0.set_title("Initial Configuration")

    ax1.plot(range(1, generations + 2), percent_alive)  # +2 to account for the initial configuration
    ax1.set_title("Percentage of Cells Alive Across Generations")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Percentage of Cells Alive (%)")
    ax1.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the space between subplots
    plt.show()


if __name__ == '__main__':
    main()
