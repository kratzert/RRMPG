# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""Implementation of the Cemaneige snow acounting model."""

import numpy as np

from numba import njit, prange
from scipy import optimize

from .basemodel import BaseModel
from ..utils.array_checks import validate_array_input, check_for_negatives
from ..utils.metrics import mse

class CemaNeige(BaseModel):
    """Implementation of the Cemaneige snow acounting model.
    
    This class implements the Cemaneige snow acounting model, as presented 
    in [1]. This model should only be used with daily data.
    
    If no model parameters are passed upon initialization, generates random
    parameter set.
    
    [1] Audrey Valery, Vazken Andreassian, Charles Perrin. "'As simple as 
    possible but not simpler': What is useful in a temperature-based snow-
    accounting routine? Part 2 - Sensitivity analysis of the Cemaneige snow
    accounting routine in 380 Catchments." Journal of Hydrology 517 (2014) 
    1176-1187.
    
    Args:
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.
            
    Raises:
        ValueError: If a dictionary of model parameters is passed but one of
            the parameters is missing.
            
    """
    
    # List of model parameters
    _param_list = ['CTG', 'Kf']    

    # Dictonary with the default parameter bounds
    _default_bounds = {'CTG': (100, 1200),
                       'Kf': (-5, 3)}
    
    # Custom numpy datatype needed for the numba function
    _dtype = np.dtype([('CTG', np.float64),
                       ('Kf', np.float64)])
    
    def __init__(self, params=None):
        """Initialize a Cemaneige snow acounting model object.
        
        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.
                
        Raises:
            ValueError: If a dictionary of model parameters is passed but one of
                the parameters is missing.
                    
        """
        super().__init__(params=params)
        
    def simulate(self, prec, frac_solid_prec, mean_temp, min_temp=None,
                 max_temp=None, snow_pack_init=0, thermal_state_init=0, 
                 return_storages=False, params=None):
        pass
    
    
def _simulate_cemaneige(prec, frac_solid_prec, mean_temp, snow_pack_init,
                        thermal_state_init, params):
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    CTG = params['CTG'][0]
    Kf = params['Kf'][0]
    
    # snow pack
    G = np.zeros(num_timesteps, np.float64)
    G[0] = snow_pack_init
    
    # thermal state
    eTG = np.zeros(num_timesteps, np.float64)
    eTG[0] = thermal_state_init
    
    # outflow as sum of liquid precipitation and melt
    outflow = np.zeros(num_timesteps, np.float64)
    
    # Extrapolate Precipitation to z layer
    
    # Extrapolate Temperature to z layer
    
    # Calculate fraction of solid precipitation depending on altitude
    
    # Split input precipitation into solid and liquid precipitation
    
    # Accumulate solid precipitation to snow pack 
    
    # Calculate snow pack thermal state before melt eTG ( eTG <= 0 )
    
    # Calculate potential melt 
    
    # Calculate snow-covered area 
    
    # Calculate actual snow melt
    
    # Update snow pack
    
    # Output: Actual snow-melt + liquid precipitation
    
    
    pass
    
def _extrapolate_precipitation():
    # P * np.exp(beta_altitude*(Z_layer - Z_median(?))
    pass

def _extrapolate_temperature():
    # T + theta_altitude*(Z_median(?) - Z_layer)
    pass
        