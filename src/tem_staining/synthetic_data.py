"""
Functions for generatic synthetic data of ferritin rings.
"""

import cv2
import numpy as np
from scipy.spatial import Delaunay


def generate_ferritin_rings(
    image_width: int = 920,
    min_ring_spacing: int = 0,
    areal_density: float = 0.2,
    background_value: int = 80,
    ring_value: int = 180,
    inner_value: int = 100,
    gaussian_noise_sigma: int = 10,
    pixel_size: float = 3.187074353033632e-10,
    outer_radius: float = 5.5e-9,
    inner_radius: float = 4e-9,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate a synthetic image with ferritin rings. Ferritin rings centers are placed
    with a Poisson disk dart throwing algorithm. Currently, the algorithm does not have
    a termination conditions, therefore it might hang indefinitely if the areal density
    or the minimum ring spacing is too high (areal_density>0.4 for min_ring_spacing=0).

    Parameters
    ----------
    image_width : int, optional
        The width of the square image generated
    min_ring_spacing : int, optional
        The minimum distance between the outer edges of the ferritin rings
    areal_density : float, optional
        The areal density of the ferritin rings
    background_value : int, optional
        The value of the background pixels
    ring_value : int, optional
        The value of the pixels in the outer part of the ferritin rings
    inner_value : int, optional
        The value of the pixels in the inner part of the ferritin rings
    gaussian_noise_sigma : int, optional
        The standard deviation of the Gaussian noise added to the image. A sigma of 0
        is equal to no noise.
    pixel_size : float, optional
        The physical size of the pixels in the image, in m
    outer_radius : float, optional
        The outer radius of the ferritin rings, in m
    inner_radius : float, optional
        The inner radius of the ferritin rings, in m
    random_state: int | None, optional
        Seed for the random state used by the algorithm, provided for reproducibility

    Returns
    -------
    image = np.ndarray
        The synthetic image with ferritin rings
    point = list
        A list of the center coordinates of the ferritin rings
    """
    # Initialize image and random state
    image = np.full((image_width, image_width), background_value, dtype=np.uint8)
    rng = np.random.default_rng(seed=random_state)

    # Convert physical sizes to pixel sizes
    outer_radius = outer_radius / pixel_size
    inner_radius = inner_radius / pixel_size

    # Calculate the number of points to generate and minimum distance between them
    n_points = round(areal_density * image.size / (np.pi * outer_radius**2))
    min_distance = 2 * outer_radius + min_ring_spacing

    # Generate the points with a Poisson disk dart throwing algorithm
    points = _poisson_disk_dart_throw(image, min_distance, n_points, random_state)

    # Draw the ferritin rings
    image = _draw_ferritin_rings(
        image,
        points,
        ring_value,
        inner_value,
        outer_radius,
        inner_radius,
    )

    # Gaussian noise
    noise = rng.normal(0, gaussian_noise_sigma, image.shape)
    image = image + noise
    image = np.clip(image, 0, 255)

    return image, points


def _poisson_disk_dart_throw(
    image: np.ndarray,
    min_distance: float,
    num_points: float,
    random_state: int | None = None,
) -> list:
    """Generate the center coordinates for the rings"""

    def reject_point(new_point, points, min_distance):
        """Check if a new point is overlapping with existing points"""
        for point in points:
            if np.linalg.norm(np.array(new_point) - np.array(point)) < min_distance:
                return True
        return False

    def propose_point(image, random_generator):
        """Propose a new point"""
        new_point = (
            random_generator.integers(0, image.shape[1]),
            random_generator.integers(0, image.shape[0]),
        )
        return new_point

    centers = []
    rng = np.random.default_rng(seed=random_state)
    for _ in range(num_points):
        while True:
            new_center = propose_point(image, rng)
            if not reject_point(new_center, centers, min_distance):
                centers.append(new_center)
                break

    return centers


def _draw_ferritin_rings(
    image: np.ndarray,
    centers: list[tuple],
    ring_value: int,
    inner_value: int,
    outer_radius: float,
    inner_radius: float,
) -> np.ndarray:
    """Add the simulated ferritin rings to the image"""

    for center in centers:
        cv2.circle(image, center, round(outer_radius), ring_value, -1)
        cv2.circle(image, center, round(inner_radius), inner_value, -1)

    return image


def calculate_point_distances(points: np.ndarray) -> np.ndarray:
    """Calculate the distances between points and their nearest neighbors
    in a Delaunay triangulation. Neighbors are defined as the vertices of the
    Delaunay simplices that share an edge with the point. Note that the distances
    are returned with no particular order and are intended to be used for statistical
    analysis.

    Parameters
    ----------
    points : np.ndarray
        The points to calculate the distances between

    Returns
    -------
    distances : np.ndarray
        The distances between points and their nearest neighbors
    """

    tri = Delaunay(points)

    distances = []
    (indptr, indices) = tri.vertex_neighbor_vertices
    for i, point in enumerate(tri.points):
        for neighbor in indices[indptr[i] : indptr[i + 1]]:
            distances.append(np.linalg.norm(point - tri.points[neighbor]))

    return np.array(distances)
