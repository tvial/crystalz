"""
This method counts the number of atoms that contain the center of each voxel
"""

from typing import List, Tuple
import itertools as it

import numpy as np


def get_voxels(
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray],
    vectors: np.ndarray,
    resolution: int,
    x_max: float,
    y_max: float,
    z_max: float
) -> np.ndarray:
    """
    Counts the number of atoms containing the center of each voxel

    Parameters
    ----------
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Kinds + atom centers + their radii
    vectors: np.ndarray
        Matrix formed by the vectors of the unit cell
    resolution: int
        Number of voxels in each direction
    x_max: float
        X coordinate of the last voxel
    y_max: float
        Y coordinate of the last voxel
    z_max: float
        Z coordinate of the last voxel

    Returns
    -------
    np.ndarray
        An array of voxels
    """
    voxels = np.zeros((resolution, resolution, resolution))

    # The atoms must be repeated once, because some of the neigbouring cells may overlap the
    # unit one, if the radii are big. By repeating only once, we assume no atom spans more
    # than the dimension of a unit cell
    # The radii are squared beforehand to avoid doing it inside the loop
    centers, radii = repeat_once(atoms, vectors)
    squared_radii: np.ndarray = radii**2

    transfer_matrix = np.linalg.inv(vectors).T

    x_range = np.linspace(0, x_max, num=resolution, endpoint=False) + x_max / (2*resolution)
    y_range = np.linspace(0, y_max, num=resolution, endpoint=False) + x_max / (2*resolution)
    z_range = np.linspace(0, z_max, num=resolution, endpoint=False) + x_max / (2*resolution)

    M = np.zeros(3)
    for i, x in enumerate(x_range):
        M[0] = x
        for j, y in enumerate(y_range):
            M[1] = y
            for k, z in enumerate(z_range):
                M[2] = z
                # M is the center of the voxel in the [x_max, y_max, z_max]-scaled cube
                # ("real-world" coordinates)
                # (k, j, i) are the indices of the voxel in the array

                # M_lattice is M in lattice coordinates
                M_lattice = transfer_matrix @ M

                # M_cell is M in real-world coordinates "modulo" the unit cell
                M_cell = vectors.T @ (M_lattice - np.floor(M_lattice))

                # Since M_cell is in the unit cell, we can look it up
                voxels[k, j, i] = lookup_voxel(M_cell, centers, squared_radii)

    return voxels


def repeat_once(
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray],
    vectors: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augments the list of atoms by one copy in each direction (27 copies in total)

    Parameters
    ----------
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray]
        Kinds + atom centers + their radii
    vectors: np.ndarray
        Matrix formed by the vectors of the unit cell

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]:
        The atoms with their copies (kinds are discarded)
    """
    _, centers, radii = atoms
    v1, v2, v3 = vectors
    repetition = range(-1, 2)
    centers_augmented, radii_augmented = [], []
    for k1, k2, k3 in it.product(repetition, repetition, repetition):
        for center, radius in zip(centers, radii):
            centers_augmented.append(center + k1 * v1 + k2 * v2 + k3 * v3)
            radii_augmented.append(radius)
    return np.array(centers_augmented), np.array(radii_augmented)


def lookup_voxel(M_cell: np.ndarray, centers: np.ndarray, squared_radii:np.ndarray) -> float:
    """
    Returns the voxel value for a particular point in the unit cell.  Since the atoms are
    "augmented", a point that sits near the boundary of the cube may get contributions for
    atoms in the neighbouring cells

    Parameters
    ----------
    M_cell: np.ndarray
        Coordinates of the point to look up
    centers: np.ndarray
        Centers of the atoms
    squared_radii: np.ndarray
        Squared radii of the atoms

    Returns
    -------
    float
        The number of atoms in which M_cell resides
    """
    squared_distances = ((M_cell - centers)**2).sum(axis=1)
    return (squared_distances <= squared_radii).sum()
