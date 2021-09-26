"""
I/O functions for XYZ files (currently, only able to read such files)
"""

from typing import Iterable, Tuple

import numpy as np


VDW_RADII = {
    'Al': 1.84,
    'In': 1.93,
    'Ga': 1.87,
    'O': 1.52,
}
"""
VDW radii of the atoms encountered in the dataset.
Taken from https://docs.mdanalysis.org/stable/_modules/MDAnalysis/topology/tables.html
"""


def read_xyz(xyz_file: Iterable) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """
    Reads an XYZ file and returns arrays describing the structure of the compound

    Parameters
    ----------
    xyz_file: Iterable
        An iterable (typically a file descriptor) with XYZ raw data

    Returns
    -------
    Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        The first element is a tuple of 3 arrays:
            - an 1-D array for the atom kinds (elements)
            - a Nx3 2-D array for the atom centers
            - an 1-D for their radii
        The second element is a 3x3 matrix with the lattice vectors (one row per vector)
    """
    kinds, centers, radii = [], [], []
    vectors = []

    for line in xyz_file:
        line = line.strip()
        if line.startswith('lattice_vector'):
            vectors.append(np.array(list(map(float, line.split()[1:]))))
        elif line.startswith('atom'):
            # pylint: disable=invalid-name
            _, x, y, z, kind = line.split()
            kinds.append(kind)
            centers.append([float(x), float(y), float(z)])
            radii.append(VDW_RADII[kind])

    return (np.array(kinds), np.array(centers), np.array(radii)), np.array(vectors)
