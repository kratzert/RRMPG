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
import pandas as pd

from rrmpg.utils.metrics import nse, rmse, mse
from rrmpg.utils.array_checks import check_for_negatives, validate_array_input

class TestNSE(unittest.TestCase):
    """Test of the NSE function."""
    
    def test_nse_for_equal_arrays(self):
        self.assertEqual(nse(obs=[1,2,3], sim=[1,2,3]), 1)
        
    def test_nse_constant_obs(self):
        with self.assertRaises(RuntimeError) as context:
            _ = nse(obs=[2,2,2], sim=[1,2,3])
        msg = ["The Nash-Sutcliffe-Efficiency coefficient is not defined ",
               "for the case, that all values in the observations are equal.",
               " Maybe you should use the Mean-Squared-Error instead."]
        expr = ("".join(msg) in str(context.exception)) 
        self.assertTrue(expr)
        
    def test_nse_simulation_equals_obs_mean(self):
        self.assertEqual(nse(obs=[1,2,3], sim=[2,2,2]), 0)
        
        
class TestRMSE(unittest.TestCase):
    """Test of the RMSE function."""
    
    def test_rmse_for_equal_arrays(self):
        self.assertEqual(rmse(obs=[1,2,3], sim=[1,2,3]), 0)
        
    def test_rmse_for_nonequal_arrays(self):
        self.assertEqual(rmse(obs=[1,1,1], sim=[3,3,3]), 2)


class TestMSE(unittest.TestCase):
    """Test of the MSE function."""
    
    def test_mse_for_equal_arrays(self):
        self.assertEqual(mse(obs=[1,2,3], sim=[1,2,3]), 0)
        
    def test_mse_for_nonequal_arrays(self):
        self.assertEqual(mse(obs=[1,1,1], sim=[3,3,3]), 4)        
        

class TestCheckForNegatives(unittest.TestCase):
    """Test the function that checks for negatives."""
    
    def test_func_without_negatives(self):
        arr = np.array([1,2,3,4,5], dtype=np.float64)
        self.assertFalse(check_for_negatives(arr))
        
    def test_func_with_negatives(self):
        arr = np.array([1,2,-3,4,5], dtype=np.float64)
        self.assertTrue(check_for_negatives(arr))     
        
        
class TestValidateArrayInput(unittest.TestCase):
    """Test the array validation function."""
    
    def test_func_with_pandas_series(self):
        vals = [1, 2, 3, 4]
        data = pd.Series(data=vals, dtype=np.float64)
        arr = validate_array_input(data, np.float64, 'arr')
        self.assertSequenceEqual(arr.tolist(), 
                                 np.array(vals, np.float64).tolist())
        
    def test_func_with_list(self):
        vals = [1., 2., 3., 4.]
        arr = validate_array_input(vals, np.float64, 'arr')
        self.assertSequenceEqual(vals, arr.tolist())
        
    def test_func_with_non_numerical_input(self):
        with self.assertRaises(ValueError) as context:
            _ = validate_array_input(['a', 'b', 1], np.float64, 'arr')
        msg = ["The data in the parameter array '{}'".format('arr'),
                " must be purely numerical."]
        self.assertTrue("".join(msg) in str(context.exception))
        
    def test_func_with_incorrect_datatype(self):
        with self.assertRaises(TypeError) as context:
            _ = validate_array_input((1, 2, 3), np.float64, 'arr')
        msg = ["The array {} must be either a list, ".format('arr'),
               "numpy.ndarray or pandas.Series"]
        self.assertTrue("".join(msg) in str(context.exception))     
              