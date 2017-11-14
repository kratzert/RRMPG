# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""Interface to the Cemaneige snow routine."""

import numbers

import numpy as np

from scipy import optimize

from .basemodel import BaseModel
from .cemaneige_model import run_cemaneige
from .cemaneige_utils import (extrapolate_precipitation, 
                              extrapolate_temperature,
                              calculate_solid_fraction)
from ..utils.array_checks import validate_array_input, check_for_negatives
from ..utils.metrics import mse

class Cemaneige(BaseModel):
    """Interface to the Cemaneige snow routine.
    
    This class builds an interface to the implementation of the Cemaneige snow 
    acounting model, originally developed by A. Valery [1] (french) and also 
    presented in [2] (english). This model should only be used with daily data.
    
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
        
    def simulate(self, prec, mean_temp, min_temp, max_temp, met_station_height,
                 snow_pack_init=0, thermal_state_init=0, altitudes=[],
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
            met_station_height: Height of the meteorological station [m]. 
                Needed to calculate the fraction of solid precipitation and
                optionally for the extrapolation of the meteorological inputs.
            snow_pack_init: (optional) Initial value of the snow pack storage
            thermal_state_init: (optional) Initial value of the thermal state
                of the snow pack
            altitudes: (optional) List of median altitudes of each elevation
                layer [m]
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
            
            # convert list to numpy array
            altitudes = np.array(altitudes)
            
        # validate input of meteorological station height
        if not isinstance(met_station_height, numbers.Number):
            raise TypeError("'met_station_height' must be a Number.")
        
        # validate initial state inputs state inputs
        if not isinstance(snow_pack_init, numbers.Number):
            raise TypeError("'snow_pack_init' must be a Number.")
        if not isinstance(thermal_state_init, numbers.Number):
            raise TypeError("'thermal_state_init' must be a Number.")
        
        # make sure both are floats
        snow_pack_init = float(snow_pack_init)
        thermal_state_init = float(thermal_state_init)
        
        # extrapolate data if multiple elevations are provided
        if len(altitudes) > 0:
            prec = extrapolate_precipitation(prec, 
                                             altitudes, 
                                             met_station_height)

            (min_temp, 
             mean_temp, 
             max_temp) = extrapolate_temperature(min_temp, mean_temp,
                                                 max_temp, altitudes, 
                                                 met_station_height)
        else:
            # expand dimensions of arrays for correct indexing later
            prec = np.expand_dims(prec, axis=-1)
            mean_temp = np.expand_dims(mean_temp, axis=-1)
            min_temp = np.expand_dims(min_temp, axis=-1)
            max_temp = np.expand_dims(max_temp, axis=-1)
            
            # add height of met. station to empty altitudes list
            altitudes = np.array([met_station_height])
        
        frac_solid_prec = calculate_solid_fraction(prec, altitudes, mean_temp, 
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
        
        # Create array for each parameter set and call simulation one by one
        outflow = np.zeros((prec.shape[0], params.size), np.float64)
        if return_storages:
            G = np.zeros((prec.shape[0], len(altitudes), params.size), 
                         np.float64)
            eTG = np.zeros((prec.shape[0], len(altitudes), params.size), 
                           np.float64)
        
        # Call simulation function for each parameter set
        for i in range(params.size):    
            if return_storages:
                (outflow[:, i], 
                 G[:, :, i], 
                 eTG[:, :, i]) = run_cemaneige(prec, mean_temp, frac_solid_prec,
                                               snow_pack_init, 
                                               thermal_state_init, params[i])
                
            else:
                outflow[:, i], _, _ = run_cemaneige(prec, mean_temp,  
                                                    frac_solid_prec, 
                                                    snow_pack_init, 
                                                    thermal_state_init, 
                                                    params[i])            
            
        if return_storages:
            return outflow, G, eTG
        else:
            return outflow
        
    def fit(self, obs, prec, mean_temp, min_temp, max_temp, met_station_height,
            snow_pack_init=0, thermal_state_init=0, altitudes=[]):
        """Fit the Cemaneige model to a observed timeseries
        
        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed 
        timeseries is simulated as good as possible.        
        
        Args:
            obs: Array of the observed timeseries [mm]
            prec: Array of daily precipitation sum [mm]
            mean_temp: Array of the mean temperature [C]
            min_temp: Array of the minimum temperature [C]
            max_temp: Array of the maximum temperature [C]
            met_station_height: Height of the meteorological station [m]. 
                Needed to calculate the fraction of solid precipitation and
                optionally for the extrapolation of the meteorological inputs.
            snow_pack_init: (optional) Initial value of the snow pack storage
            thermal_state_init: (optional) Initial value of the thermal state
                of the snow pack
            altitudes: (optional) List of median altitudes of each elevation
                layer [m]
                
        Returns:
            res: A scipy OptimizeResult class object.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If there is a size mismatch between the 
                precipitation and the pot. evapotranspiration input.
        
        """
        # Validation check for input data
        obs = validate_array_input(obs, np.float64, 'obs')
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
            
            # convert list to numpy array
            altitudes = np.array(altitudes)
            
        # validate input of meteorological station height
        if not isinstance(met_station_height, numbers.Number):
            raise TypeError("'met_station_height' must be a Number.")
        
        # validate initial state inputs state inputs
        if not isinstance(snow_pack_init, numbers.Number):
            raise TypeError("'snow_pack_init' must be a Number.")
        if not isinstance(thermal_state_init, numbers.Number):
            raise TypeError("'thermal_state_init' must be a Number.")
        
        # make sure both are floats
        snow_pack_init = float(snow_pack_init)
        thermal_state_init = float(thermal_state_init)
        
        # extrapolate data if multiple elevations are provided
        if len(altitudes) > 0:
            prec = extrapolate_precipitation(prec, 
                                             altitudes, 
                                             met_station_height)

            (min_temp, 
             mean_temp, 
             max_temp) = extrapolate_temperature(min_temp, mean_temp,
                                                 max_temp, altitudes, 
                                                 met_station_height)
        else:
            # expand dimensions of arrays for correct indexing later
            prec = np.expand_dims(prec, axis=-1)
            mean_temp = np.expand_dims(mean_temp, axis=-1)
            min_temp = np.expand_dims(min_temp, axis=-1)
            max_temp = np.expand_dims(max_temp, axis=-1)
            
            # add height of met. station to empty altitudes list
            altitudes = np.array([met_station_height])
        
        frac_solid_prec = calculate_solid_fraction(prec, altitudes, mean_temp, 
                                                   min_temp, max_temp)

        # pack input arguments for scipy optimizer
        args = (obs, prec, mean_temp, frac_solid_prec, snow_pack_init, 
                thermal_state_init, self._dtype)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])
        
        # call scipy's global optimizer
        res = optimize.differential_evolution(_loss, bounds=bnds, args=args)

        return res


def _loss(X, *args):
    """Return the loss value for the current parameter set."""
    # Unpack static arrays
    obs = args[0]
    prec = args[1]
    mean_temp = args[2]
    frac_solid_prec = args[3]
    snow_pack_init = args[4]
    thermal_state_init = args[5]
    dtype = args[6]
    
    # Create custom numpy array of the model parameters
    params = np.zeros(1, dtype=dtype)
    params['CTG'] = X[0]
    params['Kf'] = X[1]
    
    # Calcuate simulated outflow
    outflow, _, _ = run_cemaneige(prec, mean_temp, frac_solid_prec, 
                                  snow_pack_init, thermal_state_init, 
                                  params[0])
    
    # calculate loss as the mean squared error
    loss_value = mse(obs, outflow)
    
    return loss_value

