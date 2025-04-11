# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

import numpy as np

from numba import njit, prange

@njit    
def run_icemelt(temp, snow, params):
    """Implementation of a degree-day Ice Melt Model (modified after Nepal et al. 2017).
    
    This model simulates ice melt based on a simple degree-day approach.
    
    Args:
        temp: Numpy [t, n] array containing mean temperature for each time step and layer.
        snow: Numpy [t, n] array containing snowpack storage at each time step and layer.
        params: Numpy structured array containing model parameters:
            - 'DDF': Degree-day factor (mm/°C/day).
        
    Returns:
        liquid_water: Numpy [t, n] array containing the liquid water produced from melt
                      for each timestep and layer.
    
    References:
    [1] Nepal, S., Chen, J., Penton, D. J., Neumann, L. E., Zheng, H., & Wahid, S. (2017). 
        Spatial GR4J conceptualization of the Tamor glaciated alpine catchment in Eastern Nepal: 
        evaluation of GR4JSG against streamflow and MODIS snow extent. Hydrol. Process., 31, 51–68.
        doi: 10.1002/hyp.10962.
    """

    # Number of simulation timesteps
    num_timesteps = len(temp)

    # Unpack model parameters
    ddf = params['DDF']
    
    # tbase not calibrated
    tbase = 0

    # Number of elevation layers
    num_layers = temp.shape[1]

    # outflow as sum of liquid precipitation and melt of each layer
    liquid_water = np.zeros((num_timesteps, num_layers), dtype=np.float64)

    # Calculate routine for each elevation zone independently
    for l in prange(num_layers):
        for t in range(num_timesteps):
            # Calculate actual ice melt
            melt = ddf * (temp[t, l]-tbase)
            if melt < 0:
                melt = 0
            if snow[t, l] > 1:
                liquid_water[t, l] = 0
            else:
                liquid_water[t, l] = melt
    
    return liquid_water
