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

@njit(parallel=True)
def calculate_solid_fraction(prec, altitudes, mean_temp, min_temp, max_temp):
    """Function to calculate the fraction of solid precipitation.
    
    This function is taken from the airGR R-package [1]. It calculates the
    fraction of solid precipitation as a function of the altitude and the daily
    min-/mean-/max-temperature. The elevation threshold of 1500 m, which is 
    used to apply different functions depending on the height can be found in
    [2].
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        altitudes: Numpy [n] array, with the median elevation of each 
            elevation layer.
        mean_temp: Numpy [t,n] array, which contains the daily mean temperature 
            for each elevation layer n.
        min_temp: Numpy [t,n] array, which contains the daily min temperature
            for each elevation layer n.
        max_temp: Numpy [t,n] array, which contains the daily max temperature
            for each elevation layer n.
        
    
    Returns:
        frac_solid_prec: Numpy [t,n] array, which contains the fraction of 
            solid precipitation for each elevation layer n.
    
    [1] https://odelaigue.github.io/airGR/
    [2] Audrey Valery, Vazken Andreassian, Charles Perrin. "'As simple as 
    possible but not simpler': What is useful in a temperature-based snow-
    accounting routine? Part 2 - Sensitivity analysis of the Cemaneige snow
    accounting routine in 380 Catchments." Journal of Hydrology 517 (2014) 
    1176-1187.
    """
    # elevation threshold, since calculation depends on altitude
    z_thresh = 1500
    
    # Number of elevation layer
    num_layers = len(altitudes)

    # Number of timesteps
    num_timesteps = prec.shape[0]
    
    # array for solid fraction of precipitation for each zone and timestep
    solid_fraction = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    
    for l in prange(num_layers):
        
        # apply different calculation depending of altitude of elevation zone
        if altitudes[l] < z_thresh:
        
            for t in prange(num_timesteps):
                
                # if max daily temp under 0, everything is solid
                if max_temp[t, l] <= 0:
                    solid_fraction[t, l] = 1
                
                # if min daily temp is above 0 everything is liquid    
                elif min_temp[t, l] >= 0:              
                    solid_fraction[t, l] = 0
                
                # calculate fraction between 0-1    
                else:
                    solid_fraction[t, l] = 1 - (max_temp[t, l] / 
                                               (max_temp[t, l] - 
                                                min_temp[t, l]))
                    
        else:
            
            for t in prange(num_timesteps):
                
                # if mean temp greater than +3 everything is liquid
                if mean_temp[t, l] >= 3:
                    solid_fraction[t, l] = 0
                    
                # if mean temp smaller than 0 everything is solid
                elif mean_temp[t, l] <= 0:
                    solid_fraction[t, l] = 1
                    
                # calculate fraction between 0-1
                else:
                    solid_fraction[t, l] = 1 - (mean_temp[t, l] + 1) / 4
            
    return solid_fraction
   
@njit(parallel=True)    
def extrapolate_precipitation(prec, altitudes, met_station_height):
    """Extrapolate precipitation to any given height.
    
    This function can be used to extrapolate precipitation data from the height
    of the meteorological measurement station to any given height. The routine
    is take from the Excel version, which was released by the Cemaneige's 
    authors [1].
    
    Args:
        prec: Numpy [t] array, which contains the precipitation input as 
            measured at the meteorological station.
        altitudes: Numpty [n] array of the median altitudes of each elevation
            layer.
        met_station_height: Scalar, which is the elevation above sea level of
            the meteorological station.
            
    Returns:
        layer_prec: Numpy [t,n] array, with the precipitation of each elevation
            layer n.
    
    [1] https://webgr.irstea.fr/en/modeles/modele-de-neige/
    """
    # precipitation gradient [1/m] defined in Cemaneige excel version
    beta_altitude = 0.0004
    
    # elevation threshold
    z_thresh = 4000
    
    # Number of elevation layer
    num_layers = len(altitudes)

    # Number of timesteps
    num_timesteps = prec.shape[0]
    
    # array for extrapolated precipitation of each elevation layer
    layer_prec = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    
    # different extrapolation schemes depending on layer elevation
    for l in prange(num_layers):
        
        # layer median height smaller than threshold value
        if altitudes[l] <= z_thresh:
            layer_prec[:, l] = prec * np.exp((altitudes[l] - met_station_height)
                                             * beta_altitude)
            
        # layer median greater than threshold value
        else:
            
            # elevation of meteorological station smaller than threshold
            if met_station_height <= z_thresh:
                layer_prec[:, l] = prec * np.exp((z_thresh - met_station_height)
                                             * beta_altitude)
                
            # no extrapolation if station and layer median above threshold
            else:
                layer_prec[:, l] = prec
                
    return layer_prec
        
@njit(parallel=True)
def extrapolate_temperature(min_temp, mean_temp, max_temp, altitudes,
                             met_station_height):
    """Extrapolate temperature to any given height.
    
    This function can be used to extrapolate temperature data from the height
    of the meteorological measurement station to any given height. The routine
    is take from the Excel version, which was released by the Cemaneige's 
    authors [1].
    
    Args:
        min_temp: Numpy [t] array, which contains the daily min temperature.    
        mean_temp: Numpy [t] array, which contains the daily mean temperature.
        max_temp: Numpy [t] array, which contains the daily max temperature.
        altitudes: Numpty [n] array of the median altitudes of each elevation
            layer.
        met_station_height: Scalar, which is the elevation above sea level of
            the meteorological station.
            
    Returns:
        layer_min_temp: Numpy [t,n] array, layer-wise minium daily temperature.
        layer_mean_temp: Numpy [t,n] array, layer-wise mean daily temperature.
        layer_max_temp: Numpy [t,n] array, layer-wise maximum daily temperature.
        
    [1] https://webgr.irstea.fr/en/modeles/modele-de-neige/
    """
    # temperature gradient [mm/m] defined in cema neige excel version
    theta_temp = -0.0065
    
    # Number of elevation layer
    num_layers = len(altitudes)

    # Number of timesteps
    num_timesteps = min_temp.shape[0]
    
    # initialize arrays for each variable and all layer
    layer_min_temp = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    layer_mean_temp = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    layer_max_temp = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    
    for l in prange(num_layers):
        delta_temp = (altitudes[l] - met_station_height) * theta_temp
        
        # add delta temp to each temp variable
        layer_min_temp[:, l] = min_temp + delta_temp
        layer_mean_temp[:, l] = mean_temp + delta_temp
        layer_max_temp[:, l] = max_temp + delta_temp
        
    return layer_min_temp, layer_mean_temp, layer_max_temp