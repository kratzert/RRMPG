# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

"""Implementation of the educational version of the HBV model."""

import numbers

import numpy as np

from numba import njit, prange
from scipy import optimize

from .basemodel import BaseModel
from ..utils.metrics import mse
from ..utils.array_checks import check_for_negatives, validate_array_input


class HBVEdu(BaseModel):
    """Implementation of the educational version of the HBV model.

    Original publication:
        Aghakouchak, Amir, and Emad Habib. "Application of a conceptual
        hydrologic model in teaching hydrologic processes." International
        Journal of Engineering Education 26.4 (S1) (2010).

    If no model parameters are passed upon initialization, generates random
    parameter set.

    Args:
        area: Area of the basin.
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.

    Raises:
        ValueError: If Area isn't a positive numerical value or on model
            parameter is missing in the passed dictonary.

    """

    # List of model parameters
    _param_list = ['T_t', 'DD', 'FC', 'Beta', 'C', 'PWP', 'K_0', 'K_1', 'K_2',
                   'K_p', 'L']

    # Dictionary with default parameter bounds
    _default_bounds = {'T_t': (-1, 1),
                       'DD': (3, 7),
                       'FC': (100, 200),
                       'Beta': (1, 7),
                       'C': (0.01, 0.07),
                       'PWP': (90, 180),
                       'K_0': (0.05, 0.2),
                       'K_1': (0.01, 0.1),
                       'K_2': (0.01, 0.05),
                       'K_p': (0.01, 0.05),
                       'L': (2, 5)}

    # Custom numpy datatype needed for numba input
    _dtype = np.dtype([('T_t', np.float64),
                       ('DD', np.float64),
                       ('FC', np.float64),
                       ('Beta', np.float64),
                       ('C', np.float64),
                       ('PWP', np.float64),
                       ('K_0', np.float64),
                       ('K_1', np.float64),
                       ('K_2', np.float64),
                       ('K_p', np.float64),
                       ('L', np.float64)])

    def __init__(self, area, params=None):
        """Initialize a HBVEdu model object.

        Args:
            area: Area of the basin.
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.

        Raises:
            ValueError: If Area isn't a positive numerical value or on model
                parameter is missing in the passed dictonary.

        """
        super().__init__(params=params)
        # Parse inputs
        if (isinstance(area, numbers.Number) and (area > 0)):
            self.area = area
        else:
            raise ValueError("Area must be a positiv numercial value.")

    def simulate(self, temp, prec, month, PE_m, T_m, snow_init=0, soil_init=0,
                 s1_init=0, s2_init=0, return_storage=False, params=None):
        """Simulate rainfall-runoff process for given input.

        This function bundles the model parameters and validates the
        meteorological inputs, then calls the optimized model routine. Due
        to restrictions with the use of numba, this routine is kept outside
        of this model class.
        The meteorological inputs can be either list, numpy array or pandas 
        Series.

        Args:
            temp: Array of (mean) temperature for each timestep.
            prec: Array of (summed) precipitation for each timestep.
            month: Array of integers indicating for each timestep to which
                month it belongs [1,2, ..., 12]. Used for adjusted
                potential evapotranspiration.
            PE_m: long-term mean monthly potential evapotranspiration.
            T_m: long-term mean monthly temperature.
            snow_init: (optional) Initial state of the snow reservoir.
            soil_init: (optional) Initial state of the soil reservoir.
            s1_init: (optional) Initial state of the near surface flow
                reservoir.
            s2_init: (optional) Initial state of the base flow reservoir.
            return_storage: (optional) Boolean, indicating if the model 
                storages should also be returned.
            params: (optional) Numpy array of parameter sets, that will be 
                evaluated a once in parallel. Must be of the models own custom
                data type. If nothing is passed, the parameters, stored in the 
                model object, will be used.

        Returns:
            An array with the simulated streamflow and optional one array for
            each of the four reservoirs.

        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If the monthly arrays are not of size 12 or there 
                is a size mismatch between precipitation, temperature and the
                month array.

        """
        # Validation check of the temperature, precipitation and input
        temp = validate_array_input(temp, np.float64, 'temperature')
        prec = validate_array_input(prec, np.float64, 'precipitation')
        # Check if there exist negative precipitation
        if check_for_negatives(prec):
            raise ValueError("In the precipitation array are negative values.")
        
        month = validate_array_input(month, np.int8, 'month')
        if any(len(arr) != len(temp) for arr in [prec, month]):
            msg = ["The arrays of the temperature, precipitation and month ",
                   "data must be of equal size."]
            raise RuntimeError("".join(msg))

        # Validation check for PE_m and T_m
        PE_m = validate_array_input(PE_m, np.float64, 'PE_m')
        T_m = validate_array_input(T_m, np.float64, 'T_m')
        if any(len(arr) != 12 for arr in [PE_m, T_m]):
            msg = ["The monthly potential evapotranspiration and temperature",
                   " array must be of length 12."]
            raise RuntimeError("".join(msg))

        # Check if entires of month array are between 1 and 12
        if (np.min(month) < 1) or (np.max(month) > 12):
            msg = ["The month array must be between an integer1 (Jan) and ",
                   "12 (Dec)."]
            raise ValueError("".join(msg))

        # For correct python indexing [start with 0] subtract 1 of month array
        month -= 1

        # Make sure all initial storage values are floats
        snow_init = float(snow_init)
        soil_init = float(soil_init)
        s1_init = float(s1_init)
        s2_init = float(s2_init)

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
            
        if return_storage:
            # call the actual simulation function
            qsim, snow, soil, s1, s2 = _simulate_hbv_edu(temp, prec, month, 
                                                         PE_m, T_m, snow_init,
                                                         soil_init, s1_init, 
                                                         s2_init, params)

            # TODO: conversion from qobs in m³/s for different time resolutions
            # At the moment expects daily input data
            qsim = (qsim * self.area * 1000) / (24 * 60 * 60)
            
            return qsim, snow, soil, s1, s2
        
        else:
            # call the actual simulation function
            qsim, _, _, _, _ = _simulate_hbv_edu(temp, prec, month, PE_m, T_m, 
                                                 snow_init, soil_init, s1_init, 
                                                 s2_init, params)

            # TODO: conversion from qobs in m³/s for different time resolutions
            # At the moment expects daily input data
            qsim = (qsim * self.area * 1000) / (24 * 60 * 60)
            
            return qsim

        return qsim

    def fit(self, qobs, temp, prec, month, PE_m, T_m, snow_init=0.,
            soil_init=0., s1_init=0., s2_init=0.):
        """Fit the HBVEdu model to a timeseries of discharge.

        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed 
        discharge is simulated as good as possible.

        Args:
            qobs: Array of observed streamflow discharge.
            temp: Array of (mean) temperature for each timestep.
            prec: Array of (summed) precipitation for each timestep.
            month: Array of integers indicating for each timestep to which
                month it belongs [1,2, ..., 12]. Used for adjusted
                potential evapotranspiration.
            PE_m: long-term mean monthly potential evapotranspiration.
            T_m: long-term mean monthly temperature.
            snow_init: (optional) Initial state of the snow reservoir.
            soil_init: (optional) Initial state of the soil reservoir.
            s1_init: (optional) Initial state of the near surface flow
                reservoir.
            s2_init: (optional) Initial state of the base flow reservoir.

        Returns:
            res: A scipy OptimizeResult class object.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If the monthly arrays are not of size 12 or there 
                is a size mismatch between precipitation, temperature and the
                month array.

        """
        # Validation check of the temperature, precipitation and qobs input
        temp = validate_array_input(temp, np.float64, 'temperature')
        prec = validate_array_input(prec, np.float64, 'precipitation')
        qobs = validate_array_input(qobs, np.float64, 'observed discharge')
        # Check if there exist negative precipitation
        if check_for_negatives(prec):
            raise ValueError("In the precipitation array are negative values.")
        
        month = validate_array_input(month, np.int8, 'month')
        if any(len(arr) != len(temp) for arr in [prec, month]):
            msg = ["The arrays of the temperature, precipitation and month ",
                   "data must be of equal size."]
            raise RuntimeError("".join(msg))

        # Validation check for PE_m and T_m
        PE_m = validate_array_input(PE_m, np.float64, 'PE_m')
        T_m = validate_array_input(T_m, np.float64, 'T_m')
        if any(len(arr) != 12 for arr in [PE_m, T_m]):
            msg = ["The monthly potential evapotranspiration and temperature",
                   " array must be of length 12."]
            raise RuntimeError("".join(msg))

        # Check if entires of month array are between 1 and 12
        if (np.min(month) < 1) or (np.max(month) > 12):
            msg = ["The month array must be between an integer1 (Jan) and ",
                   "12 (Dec)."]
            raise ValueError("".join(msg))

        # For correct python indexing [start with 0] subtract 1 of month array
        month -= 1

        # Make sure all initial storage values are floats
        snow_init = float(snow_init)
        soil_init = float(soil_init)
        s1_init = float(s1_init)
        s2_init = float(s2_init)

        # pack input arguments for scipy optimizer
        args = (qobs, temp, prec, month, PE_m, T_m, snow_init, soil_init,
                s1_init, s2_init, self._dtype, self.area)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])
        
        # call scipy's global optimizer
        res = optimize.differential_evolution(_loss, bounds=bnds, args=args)

        return res


def _loss(X, *args):
    """Return the loss value for the current parameter set."""
    # Unpack static arrays
    qobs = args[0]
    temp = args[1]
    prec = args[2]
    month = args[3]
    PE_m = args[4]
    T_m = args[5]
    snow_init = args[6]
    soil_init = args[7]
    s1_init = args[8]
    s2_init = args[9]
    dtype = args[10]
    area = args[11]

    # Create custom numpy array of model parameters
    params = np.zeros(1, dtype=dtype)
    params['T_t'] = X[0]
    params['DD'] = X[1]
    params['FC'] = X[2]
    params['Beta'] = X[3]
    params['C'] = X[4]
    params['PWP'] = X[5]
    params['K_0'] = X[6]
    params['K_1'] = X[7]
    params['K_2'] = X[8]
    params['K_p'] = X[9]
    params['L'] = X[10]

    # Calculate simulated streamflow
    qsim,_,_,_,_ = _simulate_hbv_edu(temp, prec, month, PE_m, T_m, snow_init,
                                     soil_init, s1_init, s2_init, params)
    
    # transform discharge to m³/s
    qsim = (qsim * area * 1000) / (24 * 60 * 60)

    # Calculate the Mean-Squared-Error as optimization criterion
    loss_value = mse(qobs, qsim)

    return loss_value


@njit(parallel=True)
def _simulate_hbv_edu(temp, prec, month, PE_m, T_m, snow_init, soil_init,
                      s1_init, s2_init, params):
    """Run the educational HBV model for given inputs and model parameters."""
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Initialize arrays for all simulations of all parameter sets.
    qsim_all = np.zeros((num_timesteps, params.size), dtype=np.float64)
    snow_all = np.zeros((num_timesteps, params.size), dtype=np.float64)
    soil_all = np.zeros((num_timesteps, params.size), dtype=np.float64)
    s1_all = np.zeros((num_timesteps, params.size), dtype=np.float64)
    s2_all = np.zeros((num_timesteps, params.size), dtype=np.float64)
    
    # Process different param sets in parallel through 'prange'-loop
    for i in prange(params.size):    
        # Unpack the model parameters
        T_t = params['T_t'][i]
        DD = params['DD'][i]
        FC = params['FC'][i]
        Beta = params['Beta'][i]
        C = params['C'][i]
        PWP = params['PWP'][i]
        K_0 = params['K_0'][i]
        K_1 = params['K_1'][i]
        K_2 = params['K_2'][i]
        K_p = params['K_p'][i]
        L = params['L'][i]
    
        # initialize empty arrays for all reservoirs and outflow
        snow = np.zeros(num_timesteps, np.float64)
        soil = np.zeros(num_timesteps, np.float64)
        s1 = np.zeros(num_timesteps, np.float64)
        s2 = np.zeros(num_timesteps, np.float64)
        qsim = np.zeros(num_timesteps, np.float64)
    
        # set initial values
        snow[0] = snow_init
        soil[0] = soil_init
        s1[0] = s1_init
        s2[0] = s2_init
    
        # Start the model simulation as loop over all timesteps
        for t in range(1, num_timesteps):
    
            # Check if temperature is below threshold
            if temp[t] < T_t:
                # accumulate snow
                snow[t] = snow[t-1] + prec[t]
                # no liquid water
                liquid_water = 0
            else:
                # melt snow
                snow[t] = max(0, snow[t-1] - DD * (temp[t] - T_t))
                # add melted snow to available liquid water
                liquid_water = prec[t] + min(snow[t-1], DD * (temp[t] - T_t))
    
            # calculate the effective precipitation
            prec_eff = liquid_water * (soil[t-1] / FC) ** Beta
    
            # Calculate the potential evapotranspiration
            pe = (1 + C * (temp[t] - T_m[month[t]])) * PE_m[month[t]]
    
            # Calculate the actual evapotranspiration
            if soil[t-1] > PWP:
                ea = pe
            else:
                ea = pe * (soil[t-1] / PWP)
    
            # calculate the actual level of the soil reservoir
            soil[t] = soil[t-1] + liquid_water - prec_eff - ea
    
            # calculate the actual level of the near surface flow reservoir
            s1[t] = (s1[t-1]
                     + prec_eff
                     - max(0, s1[t-1] - L) * K_0
                     - s1[t-1] * K_1
                     - s1[t-1] * K_p)
    
            # calculate the actual level of the base flow reservoir
            s2[t] = (s2[t-1]
                     + s1[t-1] * K_p
                     - s2[t-1] * K_2)
    
            qsim[t] = ((max(0, s1[t-1] - L)) * K_0
                       + s1[t] * K_1
                       + s2[t] * K_2)
            
        # copy results into arrays of all qsims and storages
        qsim_all[:, i] = qsim
        snow_all[:, i] = snow
        soil_all[:, i] = soil
        s1_all[:, i] = s1
        s2_all[:, i] = s2
    
    return qsim_all, snow_all, soil_all, s1_all, s2_all
