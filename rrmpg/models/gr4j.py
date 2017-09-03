# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""Implementation of the GR4J hydrological model."""

import numpy as np

from numba import njit

from .basemodel import BaseModel
from ..utils.array_checks import validate_array_input, check_for_negatives


class GR4J(BaseModel):
    """Implementation of the GR4J hydrological model.
    
    This class implements the GR4J model, as presented in [1]. This model 
    should only be used with daily data.
    
    If no model parameters are passed upon initialization, generates random
    parameter set.
    
    Args:
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.
            
    Raises:
        ValueError: If a dictionary of model parameters is passed but one of
            the parameters is missing.
            
    """
    
    # List of model parameters
    _param_list = ['x1', 'x2', 'x3', 'x4']
    
    # Dictonary with the default parameter bounds
    _default_bounds = {'x1': (100, 1200),
                       'x2': (-5, 3),
                       'x3': (20, 300),
                       'x4': (1.1, 2.9)}
    
    # Custom numpy datatype needed for the numba function
    _dtype = np.dtype([('x1', np.float64),
                       ('x2', np.float64),
                       ('x3', np.float64),
                       ('x4', np.float64)])
    
    def __init__(self, params=None):
        """Initialize a GR4J model object.
        
        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.
                
        Raises:
            ValueError: If a dictionary of model parameters is passed but one of
                the parameters is missing.
                    
        """
        super().__init__(params=params)
        
    def simulate(self, prec, pe, s_init=0, r_init=0, return_storage=False):
        """Simulate rainfall-runoff process for given input.
        
        This function bundles the model parameters and validates the 
        meteorological inputs, then calls the optimized model routine. Due to 
        restrictions with the use of Numba, this routine is kept outside 
        of this model class.
        The meteorological inputs can be either list, numpy arrays or pandas
        Series.
        
        Args: 
            prec: Array of precipitation
            pe: Array of potential evapotranspiration
            s_init: (optional) Initial value of the production storage. 
            r_init: (optional) Initial value of the routing storage.
            return_stprage: (optional) Boolean, indicating if the model 
                storages should also be returned.
                
        Returns:
            An array with the simulated streamflow and optional one array for 
            each of the two storages.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeError: If there is a size mismatch between the 
                precipitation and the pot. evapotranspiration input.
                
        """
        # Validation check of the inputs
        prec = validate_array_input(prec, np.float64, 'precipitation')
        pe = validate_array_input(pe, np.float64, 'pot. evapotranspiration')
        
        # Check if there exist negative precipitation values in the input
        if check_for_negatives(prec):
            msg = "The precipitation array contains negative values."
            raise ValueError(msg)
        
        # Check for same size of inputs
        if len(prec) != len(pe):
            msg = ["The arrays of precipitation and pot. evapotranspiration,"
                   " must be of the same size."]
            raise RuntimeError("".join(msg))
        
        # Make sure the intial storage values are floating point numbers
        s_init = float(s_init)
        r_init = float(r_init)
        
        # bundle model parameters in custom numpy array
        model_params = np.zeros(1, dtype=self._dtype)
        for param in self._param_list:
            model_params[param] = getattr(self, param)
            
        if return_storage:
            qsim, s_store, r_store = _simulate_gr4j(prec, pe, s_init, 
                                                    r_init, model_params)
            return qsim, s_store, r_store
        
        else:
            qsim, _, _ = _simulate_gr4j(prec, pe, s_init, r_init, 
                                        model_params)
            return qsim
            
@njit
def _s_curve1(t, x4):
    """Calculate the s-curve of the unit-hydrograph 1.
    
    Args:
        t: timestep
        x4: model parameter x4 of the gr4j model.
        
    """
    if t <= 0:
        return 0.
    elif t < x4:
        return (t / x4)**2.5
    else:
        return 1.

@njit 
def _s_curve2(t, x4): 
    """Calculate the s-curve of the unit-hydrograph 2.
    
    Args:
        t: timestep
        x4: model parameter x4 of the gr4j model.
        
    """
    if t <= 0:
        return 0.
    elif t <= x4:
        return 0.5 * ((t / x4) ** 2.5)
    elif t < 2*x4:
        return 1 - 0.5 * ((2 - t / x4) ** 2.5)
    else:
        return 1.

@njit           
def _simulate_gr4j(prec, pe, s_init, r_init, model_params):
    """Actual run function of the gr4j hydrological model."""
    # Unpack the model parameters
    x1 = model_params['x1'][0]
    x2 = model_params['x2'][0]
    x3 = model_params['x3'][0]
    x4 = model_params['x4'][0]
    
    # get the number of simulation timesteps
    num_timesteps = len(prec)
    
    # initialize empty arrays for production (s), routing (r) store and qsim
    s_store = np.zeros(num_timesteps, np.float64)
    r_store = np.zeros(num_timesteps, np.float64)
    qsim = np.zeros(num_timesteps, np.float64)
    
    
    # set initial values
    s_store[0] = s_init
    r_store[0] = r_init
    
    # calculate number of unit hydrograph ordinates
    num_uh1 = int(np.ceil(x4))
    num_uh2 = int(np.ceil(2*x4 + 1))
    
    # calculate the ordinates of both unit-hydrographs (eq. 16 & 17)
    uh1_ordinates = np.zeros(num_uh1, np.float64)
    uh2_ordinates = np.zeros(num_uh2, np.float64)
    
    for j in range(1, num_uh1 + 1):
        uh1_ordinates[j - 1] = _s_curve1(j, x4) - _s_curve1(j - 1, x4)
        
    for j in range(1, num_uh2 + 1):
        uh2_ordinates[j - 1] = _s_curve2(j, x4) - _s_curve2(j - 1, x4)
    
    # empty arrays to store the rain distributed through the unit hydrographs
    uh1 = np.zeros(num_uh1, np.float64)
    uh2 = np.zeros(num_uh2, np.float64)
    
    # Start the model simulation loop
    for t in range(1, num_timesteps):
        
        # Calculate netto precipitation and evaporation
        if prec[t] >= pe[t]:
            p_n = prec[t-1] - pe[t-1]
            pe_n = 0
        
            # calculate fraction of netto precipitation that fills production 
            # store (eq. 3)
            p_s = ((x1 * (1 - (s_store[t-1] / x1)**2) * np.tanh(p_n/x1)) /
                   (1 + s_store[t-1] / x1 * np.tanh(p_n / x1)))
            
            # no evaporation from production store
            e_s = 0   
        
        else:
            p_n = 0
            pe_n = pe[t-1] - prec[t-1]
            
            # calculate the fraction of the evaporation that will evaporate 
            # from the production store (eq. 4)
            e_s = ((s_store[t-1] * (2 - s_store[t-1]/x1) * np.tanh(pe_n/x1)) /
                   (1 + (1 - s_store[t-1] / x1) * np.tanh(pe_n / x1)))
            
            # no rain that is allocation to the production store
            p_s = 0
            
        # Calculate the new storage content
        s_store[t] = s_store[t-1] - e_s + p_s
        
        # calculate percolation from actual storage level
        perc = s_store[t] * (1 - (1 + (4 / 9 * s_store[t] / x1)**4)**(-0.25))
        
        # final update of the production store for this timestep
        s_store[t] = s_store[t] - perc
        
        # total quantity of water that reaches the routing
        p_r = perc + (p_n - p_s)
        
        # split this water quantity by .9/.1 for different routing (UH1 & UH2)
        p_r_uh1 = 0.9 * p_r 
        p_r_uh2 = 0.1 * p_r
        
        # update the state of the rain distributed through the unit hydrographs
        for j in range(0, num_uh1 - 1):
            uh1[j] = uh1[j + 1] + uh1_ordinates[j] * p_r_uh1
        uh1[-1] = uh1_ordinates[-1] * p_r_uh1
        
        for j in range(0, num_uh2 - 1):
            uh2[j] = uh2[j + 1] + uh2_ordinates[j] * p_r_uh2
        uh2[-1] = uh2_ordinates[-1] * p_r_uh2
        
        # calculate the groundwater exchange F (eq. 18)
        gw_exchange = x2 * (r_store[t - 1] / x3) ** 3.5
        
        # update routing store
        r_store[t] = max(0, r_store[t - 1] + uh1[0] + gw_exchange)
        
        # outflow of routing store
        q_r = r_store[t] * (1 - (1 + (r_store[t] / x3)**4)**(-0.25))
        
        # subtract outflow from routing store level
        r_store[t] = r_store[t] - q_r
        
        # calculate flow component of unit hydrograph 2
        q_d = max(0, uh2[0] + gw_exchange)
        
        # total discharge of this timestep
        qsim[t] = q_r + q_d
        
    return qsim, s_store, r_store
            
        