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

import numbers

import numpy as np

from numba import njit, prange
from scipy import optimize

from .basemodel import BaseModel
from ..utils.array_checks import validate_array_input, check_for_negatives
from ..utils.metrics import mse

class Cemaneige(BaseModel):
    """Implementation of the Cemaneige snow acounting model.
    
    This class implements the Cemaneige snow acounting model, originally 
    developed by A. Valery [1] (french) and also presented in [2] (english). 
    This model should only be used with daily data.
    
    If no model parameters are passed upon initialization, generates random
    parameter set.
    
    [1] Valéry, A. "Modélisation précipitations – débit sous influence nivale.
    Élaboration d’un module neige et évaluation sur 380 bassins versants".
    PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)
    
    [2] Audrey Valery, Vazken Andreassian, Charles Perrin. "'As simple as 
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
    _default_bounds = {'CTG': (0, 1),
                       'Kf': (0, 10)}
    
    # Custom numpy datatype needed for the numba function
    _dtype = np.dtype([('CTG', np.float64),
                       ('Kf', np.float64)])
    
    def __init__(self, params=None):
        """Initialize a Cemaneige snow acounting model object.
        
        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.
                
        Raises:
            ValueError: If a dictionary of model parameters is passed but one
                of the parameters is missing.
                    
        """
        super().__init__(params=params)
        
    def simulate(self, prec, mean_temp, min_temp, max_temp, snow_pack_init=0,  
                 thermal_state_init=0, altitudes=[], met_station_height=None,
                 return_storages=False, params=None):
        """Simulate the snow-routine of the Cemaneige model.
        
        This function checks the input data and prepares the data for the 
        actual simulation function, which is kept outside of the model class 
        (due to restrictions of Numba). Meteorological input arrays can be 
        either lists, numpy arrays or pandas Series.
        
        In the original Cemaneige model, the catchment is divided into 5 
        subareas of different elevations with the each of them having the same
        area. For each elevation layer, the snow routine is calculated
        separately. Therefore, the meteorological input is extrapolated from
        the height of the measurement station to the median height of each 
        sub-area. This feature is optional (also the number of elevation layer)
        in this implementation an can be activated if the corresponding heights 
        of each elevation layer is passed as input. In this case, also the 
        height of the measurement station must be passed.
        
        Args:
            prec: Array of daily precipitation sum [mm]
            mean_temp: Array of the mean temperature [C]
            min_temp: Array of the minimum temperature [C]
            max_temp: Array of the maximum temperature [C]
            snow_pack_init: (optional) Initial value of the snow pack storage
            thermal_state_init: (optional) Initial value of the thermal state
                of the snow pack
            altitudes: (optional) List of median altitudes of each elevation
                layer [m]
            met_station_height: (optional) Height of the meteorological
                station [m]. Must be provided, if altitudes are passed for the
                extrapolation of the meteorological data.
            return_storages: (optional) Boolean, indicating if the model 
                storages should also be returned.
            params: (optional) Numpy array of parameter sets, that will be 
                evaluated a once in parallel. Must be of the models own custom
                data type. If nothing is passed, the parameters, stored in the 
                model object, will be used.
                
        Returns:
            An array with the simulated stream flow and optional one array for 
            each of the two storages.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeError: If there is a size mismatch between meteorological
                input arrays.

        """
        # Validation check for input data
        prec = validate_array_input(prec, np.float64, 'prec')
        mean_temp = validate_array_input(mean_temp, np.float64, 'mean_temp')
        min_temp = validate_array_input(min_temp, np.float64, 'min_temp')
        max_temp = validate_array_input(max_temp, np.float64, 'max_temp')
        
        # Check if there exist negative precipitation values in the input
        if check_for_negatives(prec):
            msg = "The precipitation array contains negative values."
            raise ValueError(msg)
        
        if any(len(ar) != len(prec) for ar in [mean_temp, min_temp, max_temp]):
            msg = "All meteorological input arrays must have the same length."
            raise RuntimeError(msg)
        
        # Validate the altitude inputs
        if not isinstance(altitudes, list):
            raise TypeError("'altitudes' must be a list.")
        if len(altitudes) > 0:
            for val in altitudes:
                if not isinstance(val, numbers.Number):
                    msg = "All elements in 'altitudes must be numbers."
                    raise TypeError(msg)
            if met_station_height is None:
                msg = ["The height of the meteorological station is missing."]
                raise ValueError(msg)
            if not isinstance(met_station_height, numbers.Number):
                raise TypeError("'met_station_height' must be a number.")
        
        
        # extrapolate data if multiple elevations are provided
        if len(altitudes) > 0:
            prec = _extrapolate_precipitation(prec, 
                                              altitudes, 
                                              met_station_height)

            (min_temp, 
             mean_temp, 
             max_temp) = _extrapolate_temperature(min_temp, mean_temp,
                                                  max_temp, altitudes, 
                                                  met_station_height)
        
        raise NotImplementedError("Continue here.")
        
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