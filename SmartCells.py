# This version of the GoL attempts to combat noise
# Each cell looks to see the predicted state of its 3x3 grid in the next generation
# The cell determines if the 3x3 grid will be underpopulated, overpopulated, or normal
# If the cell sees that it will be overpopulated, it decides to die
# If the cell sees that it will be underpopulated, it decided to live on/become alive (although there are caveats
# to the latter)
# If the cell sees that it will be normal, its fate is determined by the fate it should have under standard GoL rules
# This mitigates noise since now the cell is taking in a different decision-making process

import time
import pygame
import numpy as np
from scipy.ndimage import convolve

COLOR_BG = (10, 10, 10)
COLOR_GRID = (40, 40, 40)
COLOR_DIE_NEXT = (170, 170, 170)
COLOR_ALIVE_NEXT = (255, 255, 255)
COLOR_DEAD = (0, 0, 0)
COLOR_TEXT = (255, 255, 255)

MAX_GENERATIONS = 256
REPEAT_TIMES = 5
noise = 0.5


# --------------------------- [Helper Functions] ---------------------------

def get_neighbor_counts(cells):
    """
    Compute the neighbor counts for each cell in the grid with wrap-around behavior.
    """
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    return convolve(cells, kernel, mode='wrap')

def predict_future_state(cells):
    """
    Predict the future state of the grid based on the standard GoL rules.
    """
    alive = get_neighbor_counts(cells)

    # Apply noise to the neighbor counts
    is_noised = np.random.random(cells.shape) < noise
    noise_values = np.random.choice([-1, 1], size=cells.shape)
    noise_values *= is_noised

    alive = np.clip(alive + noise_values, 0, 8)

    # the cell's future state is calculated
    future_state = np.where(((cells == 1) & ((alive == 2) | (alive == 3))) |
                            ((cells == 0) & (alive == 3)), 1, 0)
    return future_state

# ---------------------- [update_extended Function] ----------------------

def update(cells):
    """
    Optimized update function using the extended rules and noise.
    """
    # Calculate the current neighbor counts for the entire grid
    neighbor_counts = get_neighbor_counts(cells)

    # Predict the future state of each cell based on the standard GoL rules
    future_state = predict_future_state(cells)

    # Compute the future neighbor counts based on the predicted future state
    future_neighbor_counts = get_neighbor_counts(future_state)

    # Calculate predicted number of alive cells in the 3x3 neighborhood in the next timestep for the entire grid
    alive_next_timestep = future_neighbor_counts + future_state

    # Apply the new rules based on the predicted number of alive cells
    next_state = np.where(
        # Rules for dead cells
        (cells == 0) & (alive_next_timestep == 3), 1,
        np.where(
            # Rules for alive cells when 0 <= alive_next_timestep <= 3
            (cells == 1) & (2 <= alive_next_timestep) & (alive_next_timestep <= 3), 1,
            np.where(
                # Rules when 4 <= alive_next_timestep <= 6 (for both alive and dead cells)
                (4 <= alive_next_timestep) & (alive_next_timestep <= 6), future_state,
                # Rules for when alive_next_timestep is 7 or 8
                0
            )
        )
    )

    return next_state


def draw_grid(screen, size):
    for x in range(0, screen.get_width(), size):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, screen.get_height()))
    for y in range(0, screen.get_height(), size):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (screen.get_width(), y))


def render(screen, cells, size, generation):
    screen.fill(COLOR_BG)
    draw_grid(screen, size)
    alive_cells = np.argwhere(cells == 1)
    for row, col in alive_cells:
        pygame.draw.rect(screen, COLOR_ALIVE_NEXT, (col * size, row * size, size - 1, size - 1))

    # Render generation counter
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation}", True, COLOR_TEXT)
    screen.blit(text, (10, 10))


def random_3x3():
    """Return a random 3x3 grid with 0s and 1s."""
    return np.random.randint(2, size=(3,3))

def place_at_center(larger_grid, smaller_grid):
    """Place smaller grid at the center of the larger grid."""
    center_x = larger_grid.shape[0] // 2
    center_y = larger_grid.shape[1] // 2
    start_x = center_x - smaller_grid.shape[0] // 2
    start_y = center_y - smaller_grid.shape[1] // 2

    for i in range(smaller_grid.shape[0]):
        for j in range(smaller_grid.shape[1]):
            larger_grid[start_x + i][start_y + j] = smaller_grid[i][j]
    return larger_grid


def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((800, 600))

    cells = np.zeros((60, 80))

    # Generate a random 3x3 grid
    small_grid = random_3x3()

    # Place the 3x3 grid at the center of the cells
    cells = place_at_center(cells, small_grid)

    initial_config = np.copy(cells)
    generation = 0
    repeat_count = 0
    alive_tallies = []

    render(screen, cells, 10, generation)
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
                render(screen, cells, 10, generation)
                pygame.display.update()

        if running:
            if generation >= MAX_GENERATIONS:
                alive_tallies.append(np.sum(cells))
                repeat_count += 1

                if repeat_count >= REPEAT_TIMES:
                    running = False
                    repeat_count = 0
                    mean_alive = np.mean(alive_tallies)
                    print(f"Alive tallies at the end of each trial: {alive_tallies}")
                    print(f"Mean of alive cells after 250 generations over {REPEAT_TIMES} trials: {mean_alive}")

                generation = 0
                cells = np.copy(initial_config)
                continue

            cells = update(cells)
            generation += 1
            render(screen, cells, 10, generation)
            pygame.display.update()

        time.sleep(0.01)


if __name__ == '__main__':
    main()

