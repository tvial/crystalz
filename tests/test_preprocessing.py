import io
import unittest

import numpy as np

from crystalz.io.xyz import read_xyz
from crystalz.preprocessing import METHODS


class TestPreprocessing(unittest.TestCase):
    def test_all_processing_methods_accept_the_same_arguments(self):
        xyz_contents = '''
            # This is a fake XYZ file describing the unit cell of an fictitious crystal
            # We keep it small for the tests to run fast

            lattice_vector 1 -2 3
            lattice_vector 3 -5 6
            lattice_vector 7 -7 9
            
            atom 1.1 1.2 1.3 Al
            atom 2.1 2.2 2.3 In
        '''
        atoms, vectors = read_xyz(io.StringIO(xyz_contents))
        
        resolution = 32
        x_max, y_max, z_max = 1, 1, 1

        failing_methods = []
        for method_name, method_module in METHODS.items():
            try:
                method_module.get_voxels(
                    atoms,
                    vectors,
                    resolution,
                    x_max,
                    y_max,
                    z_max
                )
            except Exception as e:
                failing_methods.append((method_name, e))

        self.assertTrue(len(failing_methods) == 0, f'Some methods failed: {failing_methods}')
