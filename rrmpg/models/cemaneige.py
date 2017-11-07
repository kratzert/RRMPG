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

class Cemaneige(BaseModel):
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
        
    def simulate(self, prec, mean_temp, min_temp, max_temp, snow_pack_init=0,  
                 thermal_state_init=0, altitudes=[], met_station_height=None,
                 return_storages=False, params=None):
        
        # extrapolate data
        prec = _extrapolate_precipitation(prec, altitudes, met_station_height)

        min_temp, mean_temp, max_temp = _extrapolate_temperature(min_temp, 
                                                                 mean_temp, 
                                                                 max_temp, 
                                                                 altitudes, 
                                                                 met_station_height)
        
        frac_solid_prec = _calculate_solid_fraction(prec, altitudes, mean_temp, 
                                                    min_temp, max_temp)
        
        # If no parameters were passed, prepare array w. params from attributes
        if params is None:
            params = np.zeros(1, dtype=self._dtype)
            for param in self._param_list:
                params[param] = getattr(self, param)
        
        # Else, check the param input for correct datatype
        else:
            if params.dtype != self._dtype:
                msg = ["The model parameters must be a numpy array of the ",
                       "models own custom data type."]
                raise TypeError("".join(msg))
            # if only one parameter set is passed, expand dimensions to 1D
            if isinstance(params, np.void):
                params = np.expand_dims(params, params.ndim)
            
        
        outflow = _simulate_cemaneige(prec, frac_solid_prec, mean_temp, 
                                      snow_pack_init, thermal_state_init, 
                                      params)
    
        return np.mean(outflow, axis=1)


@njit(parallel=True)
def _simulate_cemaneige(prec, frac_solid_prec, mean_temp, snow_pack_init,
                        thermal_state_init, params):
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Number of elevation layers
    num_layers = prec.shape[1]
    
    CTG = params['CTG'][0]
    Kf = params['Kf'][0]
    
    # snow pack of each layer
    G = np.zeros((num_timesteps, num_layers), np.float64)
    
    # thermal state of each layer
    eTG = np.zeros((num_timesteps, num_layers), np.float64)
    
    # outflow as sum of liquid precipitation and melt of each layer
    liquid_water = np.zeros((num_timesteps, num_layers), np.float64)
    
    # Calculate Cemaneige routine for each elevation zone indipentendly
    for l in range(num_layers):
        
        # Split input precipitation into solid and liquid precipitation
        snow = prec[:,l] * frac_solid_prec[:,l]
        rain = prec[:,l] - snow
        
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
    
    return liquid_water

@njit(parallel=True)
def _calculate_solid_fraction(prec, altitudes, mean_temp, min_temp, max_temp):
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
def _extrapolate_precipitation(prec, altitudes, met_station_height):
    # precipitation gradient [1/m] defined in cema neige excel version
    beta_altitude = 0.0004
    
    # Number of elevation layer
    num_layers = len(altitudes)

    # Number of timesteps
    num_timesteps = prec.shape[0]
    
    # array for extrapolated precipitation of each elevation layer
    layer_prec = np.zeros((num_timesteps, num_layers), dtype=np.float64)
    
    for l in prange(num_layers):
        
        layer_prec[:, l] = prec * np.exp((altitudes[l] - met_station_height)
                                         * beta_altitude)
        
    return layer_prec
        
@njit(parallel=True)
def _extrapolate_temperature(min_temp, mean_temp, max_temp, altitudes,
                             met_station_height):
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