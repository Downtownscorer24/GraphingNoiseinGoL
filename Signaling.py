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
REPEAT_TIMES = 16
noise = 0.5



def update_with_signal(cells, noise):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    alive = convolve(cells, kernel, mode='constant', cval=0)

    # Make a copy of the alive matrix for comparing post-noise changes
    original_alive = alive.copy()

    # Introduce noise
    is_noised = np.random.random(cells.shape) < noise
    noise_values = np.random.choice([-1, 1], size=cells.shape)
    noise_values *= is_noised
    alive = np.clip(alive + noise_values, 0, None)  # ensure alive neighbors can't be less than 0

    # Identify cells that will die due to noise
    will_die = np.where(((cells == 1) & ((alive < 2) | (alive > 3))) &
                        (original_alive == 3))

    # For each cell that's about to die, distribute its current neighbor count to its neighbors
    for cell in zip(*will_die):
        # Get the coordinates of the neighbors
        neighbors = [(cell[0] + i, cell[1] + j) for i in [-1, 0, 1] for j in [-1, 0, 1] if (i, j) != (0, 0)]

        # Convert neighbors into a 1D array for random.choice
        neighbors_arr = np.array(neighbors).reshape(-1, 2)

        # Randomly choose 4 neighbors to give the distress signal
        indices = np.random.choice(neighbors_arr.shape[0], 4, replace=False)
        chosen_neighbors = neighbors_arr[indices]

        for neighbor in chosen_neighbors:
            # Ensure the neighbor coordinates are within bounds
            if 0 <= neighbor[0] < cells.shape[0] and 0 <= neighbor[1] < cells.shape[1]:
                alive[tuple(neighbor)] += 1

    # Compute the updated cells based on the modified neighbor counts
    updated_cells = np.where(((cells == 1) & ((alive < 2) | (alive > 3))) |
                             ((cells == 0) & (alive != 3)), 0, 1)

    return updated_cells


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

            cells = update_with_signal(cells, noise)
            generation += 1
            render(screen, cells, 10, generation)
            pygame.display.update()

        time.sleep(0.01)


if __name__ == '__main__':
    main()

