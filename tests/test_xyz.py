import io
import unittest

import numpy as np

from crystalz.io.xyz import read_xyz


class TestXYZ(unittest.TestCase):
    def test_reading_a_xyz_file_returns_structures(self):
        xyz_contents = '''
            # This is a fake XYZ file describing the unit cell of an fictitious crystal

            lattice_vector 1 -2 3
            lattice_vector 4 -5 6
            lattice_vector 7 -8 9
            
            atom 1.1 1.2 1.3 Al
            atom 2.1 2.2 2.3 In
            atom 3.1 3.2 3.3 In
            atom 4.1 4.2 4.3 Ga
            atom 5.1 5.2 5.3 O
            atom 6.1 6.2 6.3 O
        '''
        (kinds, centers, radii), vectors = read_xyz(io.StringIO(xyz_contents))

        self.assertIsInstance(kinds, np.ndarray)
        self.assertIsInstance(centers, np.ndarray)
        self.assertIsInstance(centers[0], np.ndarray)
        self.assertIsInstance(radii, np.ndarray)
        self.assertIsInstance(vectors, np.ndarray)
        self.assertIsInstance(vectors[0], np.ndarray)

        np.testing.assert_equal(['Al', 'In', 'In', 'Ga', 'O', 'O'], kinds)
        np.testing.assert_allclose(centers, [
            [1.1, 1.2, 1.3],
            [2.1, 2.2, 2.3],
            [3.1, 3.2, 3.3],
            [4.1, 4.2, 4.3],
            [5.1, 5.2, 5.3],
            [6.1, 6.2, 6.3]
        ])
        np.testing.assert_allclose(radii, [1.84, 1.93, 1.93, 1.87, 1.52, 1.52])
        np.testing.assert_allclose(vectors, [[1, -2, 3], [4, -5, 6], [7, -8, 9]])
