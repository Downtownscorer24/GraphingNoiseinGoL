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

# Increase the width of the screen to accommodate both grids.
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 600

FONT_SIZE = 15  # Reduced font size for the neighbor count matrix.

MAX_GENERATIONS = 100
REPEAT_TIMES = 5
noise = 1.00

def update(cells, noise):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])

    alive = convolve(cells, kernel, mode='constant', cval=0)

    is_noised = np.random.random(cells.shape) < noise
    noise_values = np.random.choice([-1, 1], size=cells.shape)
    noise_values *= is_noised
    alive = np.clip(alive + noise_values, 0, None)  # ensure alive neighbors can't be less than 0

    updated_cells = np.where(((cells == 1) & ((alive < 2) | (alive > 3))) |
                             ((cells == 0) & (alive != 3)), 0, 1)

    return updated_cells

def draw_grid(screen, size):
    for x in range(0, screen.get_width(), size):
        pygame.draw.line(screen, COLOR_GRID, (x, 0), (x, screen.get_height()))
    for y in range(0, screen.get_height(), size):
        pygame.draw.line(screen, COLOR_GRID, (0, y), (screen.get_width(), y))


def render(screen, cells, size, generation, offset_x=0):
    """
    Renders the game grid on the screen.
    """
    screen.fill(COLOR_BG, (offset_x, 0, 800, 600))  # Fill only the half specified by offset_x.
    draw_grid(screen, size)
    alive_cells = np.argwhere(cells == 1)
    for row, col in alive_cells:
        pygame.draw.rect(screen, COLOR_ALIVE_NEXT, (offset_x + col * size, row * size, size - 1, size - 1))

    # Render generation counter
    font = pygame.font.Font(None, 36)
    text = font.render(f"Generation: {generation}", True, COLOR_TEXT)
    screen.blit(text, (offset_x + 10, 10))


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


def generate_neighbor_count_matrix(cells, noise=0):
    """Generate a matrix representing the count of alive neighbors for each cell."""
    m, n = cells.shape
    neighbor_count = np.zeros((m, n), dtype=int)

    for x in range(m):
        for y in range(n):
            alive_neighbors = sum([
                cells[(x + i) % m][(y + j) % n]
                for i in range(-1, 2)
                for j in range(-1, 2)
                if i != 0 or j != 0
            ])
            neighbor_count[x][y] = alive_neighbors

    # Apply noise
    noise_matrix = np.random.randint(-noise, noise + 1, (m, n))
    neighbor_count = np.clip(neighbor_count + noise_matrix, 0, 8)

    return neighbor_count

def render_neighbor_matrix(screen, matrix, cells, cell_size, offset_x=0):
    m, n = matrix.shape

    font = pygame.font.Font(None, 15)  # Reduced font size to prevent overlap

    for x in range(m):
        for y in range(n):
            if cells[x][y]:  # If the cell in the main grid is alive
                pygame.draw.rect(screen, COLOR_ALIVE_NEXT, (y * cell_size + offset_x, x * cell_size, cell_size, cell_size))
                text = font.render(str(matrix[x][y]), True, COLOR_DEAD)  # Display the count in black
                screen.blit(text, (y * cell_size + offset_x + cell_size // 4, x * cell_size))
            else:
                pygame.draw.rect(screen, (matrix[x][y] * 30, matrix[x][y] * 30, matrix[x][y] * 30), (y * cell_size + offset_x, x * cell_size, cell_size, cell_size))
                text = font.render(str(matrix[x][y]), True, COLOR_TEXT)  # Display the count in white
                screen.blit(text, (y * cell_size + offset_x + cell_size // 4, x * cell_size))

def main():
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

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

    history = []

    neighbor_matrix = generate_neighbor_count_matrix(cells, noise)  # Initialize the neighbor matrix once here.

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

                # Advance generation using arrow keys
                elif event.key == pygame.K_RIGHT:  # move forward one generation
                    # Push the current state to history before updating
                    history.append((np.copy(cells), neighbor_matrix.copy(), generation))
                    if len(history) > MAX_GENERATIONS:  # If we exceed max generations, remove oldest.
                        history.pop(0)
                    cells = update(cells, noise)
                    generation += 1
                    neighbor_matrix = generate_neighbor_count_matrix(cells, noise)  # Update the neighbor matrix here.

                elif event.key == pygame.K_LEFT and generation > 0:  # move back one generation
                    # Pop the previous state from history
                    cells, neighbor_matrix, generation = history.pop()

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                if cell_deactivation:
                    cells[pos[1] // 10, pos[0] // 10] = 0
                else:
                    cells[pos[1] // 10, pos[0] // 10] = 1
                render(screen, cells, 10, generation)
                pygame.display.update()

        # Only regenerate the neighbor count matrix when a new generation is processed.
        if event.type == pygame.KEYDOWN and (event.key == pygame.K_RIGHT or event.key == pygame.K_LEFT):
            neighbor_matrix = generate_neighbor_count_matrix(cells, noise)

        # After handling events, render both grids:
        render(screen, cells, 10, generation)
        # In the main function's loop
        render_neighbor_matrix(screen, neighbor_matrix, cells, 10, 800)

        pygame.display.update()

        time.sleep(0.1)


if __name__ == '__main__':
    main()
