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

birth_prob = 0.99
survival_prob = 0.99
death_prob = 0.99

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
            if alive < 2 or alive > 3 and random.random() < death_prob :
                updated_cells[row, col] = 0
                if with_progress:
                    color = COLOR_DIE_NEXT

            elif 2 <= alive <= 3 and random.random() < survival_prob :
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
    initial_config = None

    cells[29,39] = 1
    cells[29,40] = 1
    cells[30,40] = 1
    cells[30,41] = 1
    cells[31,40] = 1

    screen.fill(COLOR_GRID)
    update(screen, cells, 10)

    pygame.display.flip()
    pygame.display.update()

    configuring = True
    cell_deactivation = False
    trials = 0
    generations = [0]*5
    percent_alive = [[] for _ in range(5)]


    while trials < 5:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and configuring:  # finalize initial configuration
                    initial_config = np.copy(cells)
                    configuring = False
                elif event.key == pygame.K_d:
                    cell_deactivation = not cell_deactivation
            if configuring and pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if cell_deactivation:
                    cells[pos[1] // 10, pos[0] // 10] = 0
                else:
                    cells[pos[1] // 10, pos[0] // 10] = 1
                update(screen, cells, 10)
                pygame.display.update()

        if not configuring:
            screen.fill(COLOR_GRID)
            cells = update(screen, cells, 10, with_progress=True)
            pygame.display.update()

            # Calculate the percentage of cells alive
            alive_count = np.sum(cells)
            total_cells = cells.size
            percent_alive[trials].append(alive_count / total_cells * 100)
            generations[trials] += 1

            if generations[trials] == 250:  # When one run is finished
                trials += 1
                cells = np.copy(initial_config)  # Reset cells to initial configuration for next trials

        time.sleep(0.001)

    # Plot the graph and the initial configuration after the simulation
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(10, 12))

    ax0.imshow(initial_config, cmap='binary')
    ax0.set_title("Initial Configuration")

    # Plot all the runs
    for run in range(len(percent_alive)):
        ax1.plot(range(1, len(percent_alive[run]) + 1), percent_alive[run], label=f"Run {run + 1}")

    ax1.set_title("Percentage of Cells Alive Across Generations")
    ax1.set_xlabel("Generations")
    ax1.set_ylabel("Percentage of Cells Alive (%)")
    # Add dummy lines for one_less and one_more to the legend
    birth_prob_legend = plt.Line2D((0, 1), (0, 0), color='white')  # this line is for the legend, not to be plotted
    survival_prob_legend = plt.Line2D((0, 1), (0, 0), color='white')  # this line is for the legend, not to be plotted
    death_prob_legend = plt.Line2D((0, 1), (0, 0), color='white')  # this line is for the legend, not to be plotted

    # Update the legend position
    ax1.legend([birth_prob_legend, survival_prob_legend, death_prob_legend] + [line for line in ax1.lines],
               [f'birth_prob: {birth_prob}', f'survival_prob: {survival_prob}', f'death_prob: {death_prob}'] + [f'Run {run + 1}' for run in
                                                                     range(len(percent_alive))], loc=(1.04, 1))

    ax1.grid(True)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Adjust the space between subplots
    plt.show()


if __name__ == '__main__':
    main()


