"""
Dummy preprocessing method that yields random voxels
"""

from typing import Tuple

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
    Returns random voxels

    Parameters
    ----------
    atoms: Tuple[np.ndarray, np.ndarray, np.ndarray]
        This parameter is ignored
    vectors: np.ndarray
        This parameter is ignored
    resolution: int
        Number of voxels in each direction
    x_max: float
        This parameter is ignored
    y_max: float
        This parameter is ignored
    z_max: float
        This parameter is ignored

    Returns
    -------
    np.ndarray
        A random array of voxels
    """
    return np.random.random((resolution, resolution, resolution))
