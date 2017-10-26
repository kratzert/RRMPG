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

from rrmpg.models import ABCModel
from rrmpg.tools.monte_carlo import monte_carlo

class TestMonteCarlo(unittest.TestCase):
    """Test the monte carlo implementation."""
    
    def setUp(self):
        self.model = ABCModel()
        self.rain = np.random.random(100)
        unittest.TestCase.setUp(self)
        
    def test_runs_for_correct_number(self):
        num = 24
        results = monte_carlo(self.model, num, prec=self.rain)
        self.assertEqual(results['qsim'].shape[1], num)
        