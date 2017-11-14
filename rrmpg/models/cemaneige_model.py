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
def run_cemaneige(prec, mean_temp, frac_solid_prec, snow_pack_init,
                  thermal_state_init, params):
    """Implementation of the Cemaneige snow routine.
    
    This function should be called via the .simulate() function of the 
    Cemaneige class and not directly. It is kept in a separate file for less 
    confusion if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication.
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        mean_temp: Numpy [t,n] array, which contains the mean temperature for
            each elevation layer n.
        frac_solid_prec: Numpy [t,n] array, which contains the fraction of 
            solid precipitation for each elevation layer n.
        snow_pack_init: Scalar for the initial state of the snow pack.
        thermal_state_init: Scalar for the initial state of the thermal state.
        params: Numpy array of custom dtype, which contains the model parameter.
        
    Returns:
        outflow: Numpy [t] array, which contains the liquid water outflow for
            each timestep.
        G: Numpy [t] array, which contains the state of the snow pack for 
            each timestep.
        eTG: Numpy [t] array, which contains the thermal state of the snow
            pack for each timestep.
        
    """
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Number of elevation layers
    num_layers = prec.shape[1]
    
    # Unpack model parameters
    CTG = params['CTG']
    Kf = params['Kf']
    
    # snow pack of each layer
    G = np.zeros((num_timesteps, num_layers), np.float64)
    
    # thermal state of each layer
    eTG = np.zeros((num_timesteps, num_layers), np.float64)
    
    # outflow as sum of liquid precipitation and melt of each layer
    liquid_water = np.zeros((num_timesteps, num_layers), np.float64)
    
    # total outflow which is the mean of liquid water of each layer
    outflow = np.zeros(num_timesteps, np.float64)
    
    # Calculate Cemaneige routine for each elevation zone indipentendly
    for l in prange(num_layers):
        
        # Split input precipitation into solid and liquid precipitation
        snow = prec[:, l] * frac_solid_prec[:, l]
        rain = prec[:, l] - snow
        
        # calc the snow cover threshold from mean anual solid precipitation
        G_tresh = 0.9 * 365.25 * np.mean(snow)
        
        for t in range(num_timesteps):
            
            # Accumulate solid precipitation to snow pack
            if t == 0:
                G[t, l] = snow_pack_init
            else: 
                G[t, l] = G[t-1, l] + snow[t]
            
            # Calculate snow pack thermal state before melt eTG ( eTG <= 0 )
            if t == 0:
                eTG[t, l] = thermal_state_init
            else:
                eTG[t, l] = CTG * eTG[t-1, l] + (1 - CTG) * mean_temp[t, l]
            if eTG[t, l] > 0:
                eTG[t, l] = 0
            
            # Calculate potential melt 
            if eTG[t, l] == 0 and mean_temp[t, l] > 0:
                pot_melt = Kf * mean_temp[t, l]
                
                # cap the potential snow melt to the state of the snow pack
                if pot_melt > G[t, l]:
                    pot_melt = G[t, l]
            else:
                pot_melt = 0
                
            # Calculate snow-covered area 
            if G[t, l] < G_tresh:
                G_ratio = G[t, l] / G_tresh
            else:
                G_ratio = 1
                
            # Calculate actual snow melt
            melt = (0.9 * G_ratio + 0.1) * pot_melt
            
            # Update snow pack
            G[t, l] = G[t, l] - melt 
            
            # Output: Actual snow-melt + liquid precipitation
            liquid_water[t, l] = rain[t] + melt
            
    # calculate the outflow as mean of each layer
    for j in prange(num_timesteps):
        outflow[j] = np.mean(liquid_water[j, :])

    return outflow, G, eTG