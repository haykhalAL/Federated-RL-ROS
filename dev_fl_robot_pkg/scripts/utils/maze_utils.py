import math


def generate_grid_centers(maze_size, cell_size):
    """
    Generate the center coordinate of every grid cell.

    Example:
        maze_size = 6
        cell_size = 1

    Returns:
        [
            (-2.5, -2.5),
            (-1.5, -2.5),
            ...
            (2.5, 2.5)
        ]
    """

    half = maze_size / 2.0

    rows = int(math.floor(maze_size / cell_size))

    centers = []

    for row in range(rows):

        for col in range(rows):

            x = -half + (col + 0.5) * cell_size
            y = -half + (row + 0.5) * cell_size

            centers.append((x, y))

    return centers