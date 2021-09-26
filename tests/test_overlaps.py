import unittest

import numpy as np

from crystalz.preprocessing.overlaps import *


class TestOverlaps(unittest.TestCase):
    def test_augmentation_of_atoms(self):
        centers = np.array([
            [0, 0, 0],
            [.1, 0, -.2],
        ])
        radii = np.array([1, 2])
        atoms = 'Kinds are ignored', centers, radii
        vectors = np.array([
            [1.1, 0, 0],
            [0, 1.2, 0],
            [0, 0, 1.3]
        ])

        # We should explicitly list the new coordinates but it's tiresome,
        # and with the orthogonal vectors provided it's not too hard to
        # check that the code for expected_centers is correct
        # (we're not checking the non-orthogonal case though)
        expected_centers = []
        for k1 in (-1, 0, 1):
            for k2 in (-1, 0, 1):
                for k3 in (-1, 0, 1):
                    for x, y, z in centers:
                        expected_centers.append([x + 1.1*k1, y + 1.2*k2, z + 1.3*k3])
        expected_radii = [1, 2] * 27

        centers_augmented, radii_augmented = repeat_once(atoms, vectors)

        self.assertIsInstance(centers_augmented, np.ndarray)
        self.assertIsInstance(radii_augmented, np.ndarray)

        np.testing.assert_allclose(expected_radii, radii_augmented)
        np.testing.assert_allclose(expected_centers, centers_augmented)
