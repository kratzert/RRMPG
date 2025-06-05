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
import os

import numpy as np
import pandas as pd

from rrmpg.models import ABCModel, HBVEdu, GR4J, Cemaneige, CemaneigeGR4J, CemaneigeHystGR4J, CemaneigeHystGR4JIce
from rrmpg.models.basemodel import BaseModel

class TestBaseModelFunctions(unittest.TestCase):
    """Test all functions implemented in the BaseModel.
    
    These will not be tested for all models, but one as an example for one of 
    the models.
    """
    
    def setUp(self):
        self.model = ABCModel()
        self.param_names = ['a', 'b', 'c']
        self.default_bounds =  {'a': (0, 1),
                                'b': (0, 1),
                                'c': (0, 1)}
        self.dtype = np.dtype([('a', np.float64),
                               ('b', np.float64),
                               ('c', np.float64)])
        unittest.TestCase.setUp(self)
        
    def test_get_parameter_names(self):
        self.assertEqual(self.model.get_parameter_names(), self.param_names)
        
    def test_get_params(self):
        params = self.model.get_params()
        for param in self.param_names:
            msg = "Failed, because '{}' not in '{}'".format(param, params.keys)
            self.assertIn(param, params, msg)
            
    def test_get_default_bounds(self):
        bounds = self.model.get_default_bounds()
        self.assertDictEqual(bounds, self.default_bounds)
        
    def test_get_dtype(self):
        self.assertEqual(self.dtype, self.model.get_dtype())
        
    def test_random_params_in_default_bounds(self):
        params = self.model.get_random_params()
        bnds = self.default_bounds
        
        for p in self.param_names:
            msg = ["Failed for param: '{}', which has a ".format(p),
                   "a value of {}, but lower bounds ".format(params[p][0]),
                   "is {} and upper bound {}.".format(bnds[p][0], bnds[p][1])]
            self.assertTrue(bnds[p][0] <= params[p][0] <= bnds[p][1], 
                            "".join(msg))
    
    def test_get_multiple_random_param_sets(self):
        num = 24
        params = self.model.get_random_params(num=num)
        self.assertEqual(num, params.size)
              
    def test_set_params(self):
        rand_params = self.model.get_random_params()
        # convert rand_params array to dict:
        params = {}
        for p in self.param_names:
            params[p] = rand_params[p][0]
        self.model.set_params(params)
        self.assertDictEqual(params, self.model.get_params()) 
        
        
class TestABCModel(unittest.TestCase):
    """Test ABC-Model specific functions."""
    
    def setUp(self):
        self.model = ABCModel()
        unittest.TestCase.setUp(self)
        
    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))
        
    def test_simulate_zero_rain(self):
        qsim = self.model.simulate(np.zeros(100))
        self.assertEqual(np.sum(qsim), 0)
        
    def test_simulate_negative_rain(self):
        with self.assertRaises(ValueError) as context:
            self.model.simulate([-1,1,1])
        expr = ("In the precipitation array are negative values." in
                str(context.exception)) 
        self.assertTrue(expr)


class TestHBVEdu(unittest.TestCase):
    """Test HBVEdu specific functions."""
    
    def setUp(self):
        # parameter set see https://github.com/kratzert/RRMPG/issues/10
        params = {'T_t': 0,
                  'DD': 4.25,
                  'FC': 177.1,
                  'Beta': 2.35,
                  'C': 0.02,
                  'PWP': 105.89,
                  'K_0': 0.05,
                  'K_1': 0.03,
                  'K_2': 0.02,
                  'K_p': 0.05,
                  'L': 4.87}
        self.model = HBVEdu(params=params)
        
    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))    
        
    def test_simulate_zero_rain(self):
        qsim = self.model.simulate(temp=np.random.uniform(-15,25,100),
                                   prec=np.zeros(100),
                                   month=np.random.randint(1,12,100),
                                   PE_m=np.random.uniform(0,4,12),
                                   T_m=np.random.uniform(-5,15,12))
        self.assertEqual(np.sum(qsim), 0)
        
    def test_simulate_negative_rain(self):
        with self.assertRaises(ValueError) as context:
            self.model.simulate(temp=np.random.uniform(-15,25,100),
                                prec=np.arange(-1,99),
                                month=np.random.randint(1,12,100),
                                PE_m=np.random.uniform(0,4,12),
                                T_m=np.random.uniform(-5,15,12))
        expr = ("In the precipitation array are negative values." in
                str(context.exception)) 
        self.assertTrue(expr)
        
    def test_simulated_against_validation_data(self):
        test_dir = os.path.dirname(__file__)
        daily_file = os.path.join(test_dir, 'data', 'hbv_daily_inputs.txt')
        daily_inputs = pd.read_csv(daily_file, sep='\t',
                                   names=['date', 'month', 'temp', 'prec'])
        monthly_file = os.path.join(test_dir, 'data', 'hbv_monthly_inputs.txt')
        monthly_inputs = pd.read_csv(monthly_file, sep=' ', 
                                     names=['temp', 'not_needed', 'evap'])
        
        qsim_matlab_file = os.path.join(test_dir, 'data', 'hbv_qsim.csv')
        qsim_matlab = pd.read_csv(qsim_matlab_file, header=None, 
                                  names=['qsim'])
        # fix parameters from provided MATLAB code from HBV paper
        area = 410
        soil_init = 100
        s1_init = 3
        s2_init = 10
        
        qsim = self.model.simulate(temp=daily_inputs.temp, 
                                   prec=daily_inputs.prec, 
                                   month=daily_inputs.month, 
                                   PE_m=monthly_inputs.evap, 
                                   T_m=monthly_inputs.temp, 
                                   snow_init=0, 
                                   soil_init=soil_init, 
                                   s1_init=s1_init, 
                                   s2_init=s2_init, 
                                   return_storage=False)
        
        # rescale qsim from mm/d to m³/s
        qsim = (qsim * area * 1000) / (24*60*60)
       
        self.assertTrue(np.allclose(qsim.flatten(), qsim_matlab.qsim))
        
    
class TestGR4J(unittest.TestCase):
    """Test the GR4J Model.
    
    This model is validated against the Excel implementation provided by the 
    model authors.
    """
    
    def setUp(self):
        # parameters are taken from the excel sheet
        params = {'x1': np.exp(5.76865628090826), 
                  'x2': np.sinh(1.61742503661094), 
                  'x3': np.exp(4.24316129943456), 
                  'x4': np.exp(-0.117506799276908)+0.5}
        self.model = GR4J(params=params)
        
    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))  
        
    def test_simulate_zero_rain(self):
        qsim = self.model.simulate(prec=np.zeros(100),
                                   etp=np.random.uniform(0,3,100),
                                   s_init=0, r_init=0)
        self.assertEqual(np.sum(qsim), 0)
        
    def test_simulate_compare_against_excel(self):
        # intial states are taken from excel
        s_init = 0.6
        r_init = 0.7
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(test_dir, 'data', 'gr4j_example_data.csv')
        data = pd.read_csv(data_file, sep=',')
        qsim = self.model.simulate(data.prec, data.etp, s_init=s_init, 
                                   r_init=r_init, return_storage=False)
        self.assertTrue(np.allclose(qsim.flatten(), data.qsim_excel))
        
class TestCemaneige(unittest.TestCase):
    """Test the Cemaneige snow routine.
    
    This model is validated against the Excel implementation provided by the 
    model authors.
    """
    
    def setUp(self):
        # parameters are taken from the excel sheet
        params = {'CTG': 0.25, 'Kf': 3.74}
        self.model = Cemaneige(params=params)
        
    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))  
        
    def test_simulate_compare_against_excel(self):
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(test_dir, 'data', 
                                 'cemaneige_validation_data.csv')
        df = pd.read_csv(data_file, sep=';')
        qsim = self.model.simulate(df.precipitation, df.mean_temp, df.min_temp, 
                                   df.max_temp, met_station_height=495, 
                                   altitudes=[550, 620, 700, 785, 920])
        self.assertTrue(np.allclose(qsim.flatten(), 
                                    df.liquid_outflow.to_numpy()))
        
class TestCemaneigeGR4J(unittest.TestCase):
    """Test the Cemaneige + GR4J couple model.
    
    This model is validated against the Excel implementation provided by the 
    model authors.
    """
    
    def setUp(self):
        # parameters are taken from the excel sheet
        params = {'CTG': 0.25, 
                  'Kf': 3.74,
                  'x1': np.exp(5.25483021675164),
                  'x2': np.sinh(1.58209470624126),
                  'x3': np.exp(4.3853181982412),
                  'x4': np.exp(0.954786342674327)+0.5}
        self.model = CemaneigeGR4J(params=params)
        
    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))  
        
    def test_simulate_compare_against_excel(self):
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(test_dir, 'data', 
                                 'cemaneigegr4j_validation_data.csv')
        df = pd.read_csv(data_file, sep=';', index_col=0)
        qsim = self.model.simulate(df.precipitation, df.mean_temp, df.min_temp, 
                                   df.max_temp, df.pe, met_station_height=495, 
                                   altitudes=[550, 620, 700, 785, 920],
                                   s_init=0.6, r_init=0.7)
        self.assertTrue(np.allclose(qsim.flatten(), 
                                    df.qsim.to_numpy()))

class TestCemaneigeHystGR4J(unittest.TestCase):
    """Test the CemaneigeHysteresis + GR4J couple model.

    XX
    """

    def setUp(self):
        # parameters are taken from the excel sheet
        params = {
            "Thacc": 18.6,
            "Rsp": 0.22,  # for CemaneigeHystGR4J
            "CTG": 0.78,
            "Kf": 4.02,
            "x1": 546,
            "x2": 0.53,
            "x3": 276,
            "x4": 1.32,
        }
        self.model = CemaneigeHystGR4J(params=params)

    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))

    def test_simulate_compare_against_excel(self):
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(
            test_dir, "data", "cemaneigehystgr4j_validation_data.csv"
        )
        df = pd.read_csv(data_file, index_col=0)
        qsim = self.model.simulate(
            df.precipitation,
            df.mean_temp,
            df.min_temp,
            df.max_temp,
            df.pe,
            met_station_height=700,
            altitudes=[550, 620, 700, 785, 920],
            s_init=0.5,
            r_init=0.4,
        )
        self.assertTrue(np.allclose(qsim.flatten(), df.qsim.to_numpy()))

class TestCemaneigeHystGR4JIce(unittest.TestCase):
    """Test the CemaneigeHysteresis + Ice model + GR4J couple model.

    XX
    """

    def setUp(self):
        # parameters are taken from the excel sheet
        params = {
            "Thacc": 18.6,
            "Rsp": 0.22,  # for CemaneigeHystGR4J
            "CTG": 0.78,
            "Kf": 4.02,
            "x1": 546,
            "x2": 0.53,
            "x3": 276,
            "x4": 1.32,
            "DDF": 5,  # Degree Day Factor for Ice
        }
        self.model = CemaneigeHystGR4JIce(params=params)

    def test_model_subclass_of_basemodel(self):
        self.assertTrue(issubclass(self.model.__class__, BaseModel))

    def test_simulate_compare_against_excel(self):
        test_dir = os.path.dirname(__file__)
        data_file = os.path.join(
            test_dir, "data", "cemaneigehystgr4jice_validation_data.csv"
        )
        df = pd.read_csv(data_file, index_col=0)
        frac_ice = np.array([0.02, 0.04, 0.25, 0.51, 0.71])
        qsim = self.model.simulate(
            df.precipitation,
            df.mean_temp,
            df.min_temp,
            df.max_temp,
            df.pe,
            frac_ice,
            met_station_height=700,
            altitudes=[550, 620, 700, 785, 920],
            s_init=0.5,
            r_init=0.4,
            sca_init=0.2,
        )
        self.assertTrue(np.allclose(qsim.flatten(), df.qsim.to_numpy()))