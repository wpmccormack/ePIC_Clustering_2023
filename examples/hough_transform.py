import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def normalize_points(points):
    """
    Normalize points to between 0 and 1
    """
    min_x = points[:, 0].min()
    min_y = points[:, 1].min()
    max_x = points[:, 0].max()
    max_y = points[:, 1].max()

    points[:, 0] = (points[:, 0] - min_x) / (max_x - min_x)
    points[:, 1] = (points[:, 1] - min_y) / (max_y - min_y)

    return points

def hough_transform(points, num_rhos, num_thetas):
    max_distance = np.sqrt(np.sum(np.square(points.max(axis=0) - points.min(axis=0))))
    d_rho = 2 * max_distance / num_rhos
    d_theta = np.pi / num_thetas

    thetas = np.linspace(-np.pi / 2, np.pi / 2, num_thetas)
    rhos = np.linspace(-max_distance, max_distance, num_rhos)
    accumulator = np.zeros((num_rhos, num_thetas), dtype=int)

    for x, y in points:
        for i, theta in enumerate(thetas):
            rho = x * np.cos(theta) + y * np.sin(theta)
            j = int((rho + max_distance) / d_rho) if (0 <= (rho + max_distance) / d_rho < num_rhos) else 0
            accumulator[j, i] += 1

    return accumulator, rhos, thetas
