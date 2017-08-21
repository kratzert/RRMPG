# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

import unittest

import numpy as np

from rrmpg.utils.metrics import nse, rmse

class TestNSE(unittest.TestCase):
    """Test of the NSE function."""
    
    def test_nse_for_equal_arrays(self):
        self.assertEqual(nse([1,2,3], [1,2,3]), 1)
        
    def test_nse_equal_and_constant_arrays(self):
        self.assertEqual(nse([2,2,2], [2,2,2]), 1)
        
    def test_nse_non_equal_arrays_constant_obs(self):
        self.assertEqual(nse(obs=[2,2,2], sim=[1,2,3]), -1*np.inf)
        
    def test_nse_simulation_equals_obs_mean(self):
        self.assertEqual(nse(obs=[1,2,3], sim=[2,2,2]), 0)
        
        
class TestRMSE(unittest.TestCase):
    """Test of the RMSE function."""
    
    def test_rmse_for_equal_arrays(self):
        self.assertEqual(rmse([1,2,3], [1,2,3]), 0)
        
    def test_rmse_for_nonequal_arrays(self):
        self.assertEqual(rmse([1,1,1], [3,3,3]), 2)
        