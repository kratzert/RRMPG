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

from scipy import optimize

from .basemodel import BaseModel
from .gr4j_model import run_gr4j
from ..utils.array_checks import validate_array_input, check_for_negatives
from ..utils.metrics import mse


class GR4J(BaseModel):
    """Implementation of the GR4J hydrological model.
    
    This class implements the GR4J model, as presented in [1]. This model 
    should only be used with daily data.
    
    If no model parameters are passed upon initialization, generates random
    parameter set.
    
    [1] Perrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement 
    of a parsimonious model for streamflow simulation." Journal of hydrology 
    279.1 (2003): 275-289.
    
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
                of the parameters is missing.
                    
        """
        super().__init__(params=params)
        
    def simulate(self, prec, etp, s_init=0., r_init=0., return_storage=False,
                 params=None):
        """Simulate rainfall-runoff process for given input.
        
        This function bundles the model parameters and validates the 
        meteorological inputs, then calls the optimized model routine. Due to 
        restrictions with the use of Numba, this routine is kept outside 
        of this model class.
        The meteorological inputs can be either list, numpy arrays or pandas
        Series.
        
        Args: 
            prec: Array of daily precipitation sum [mm]
            etp: Array of mean potential evapotranspiration [mm]
            s_init: (optional) Initial value of the production storage as 
                fraction of x1. 
            r_init: (optional) Initial value of the routing storage as fraction
                of x3.
            return_stprage: (optional) Boolean, indicating if the model 
                storages should also be returned.
            params: (optional) Numpy array of parameter sets, that will be 
                evaluated a once in parallel. Must be of the models own custom
                data type. If nothing is passed, the parameters, stored in the 
                model object, will be used.            
                
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
        etp = validate_array_input(etp, np.float64, 'pot. evapotranspiration')
        
        # Check if there exist negative precipitation values in the input
        if check_for_negatives(prec):
            msg = "The precipitation array contains negative values."
            raise ValueError(msg)
        
        # Check for same size of inputs
        if len(prec) != len(etp):
            msg = ["The arrays of precipitation and pot. evapotranspiration,"
                   " must be of the same size."]
            raise RuntimeError("".join(msg))
        
        # Make sure the intial storage values are floating point numbers
        s_init = float(s_init)
        r_init = float(r_init)
        
        # check if the intial values are in rage [0, 1]
        if (s_init < 0) or (s_init > 1):
            msg = ["The initial value of the production storage must be in ",
                   "the range [0,1]."]
            raise ValueError("".join(msg))
        
        if (r_init < 0) or (r_init > 1):
            msg = ["The initial value of the routing storage must be in the",
                   " range [0,1]."]
            raise ValueError("".join(msg))
        
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
        
        # Create output arrays
        qsim = np.zeros((prec.shape[0], params.size), np.float64)
        if return_storage:
            s_store = np.zeros((prec.shape[0], params.size), np.float64)
            r_store = np.zeros((prec.shape[0], params.size), np.float64)
            
        # call simulation function for each parameter set
        for i in range(params.size):   
            if return_storage:
                qsim[:,i], s_store[:,i], r_store[:,i] = run_gr4j(prec, etp, 
                                                                 s_init, 
                                                                 r_init, 
                                                                 params[i])
            
            else:
                qsim[:,i], _, _ = run_gr4j(prec, etp, s_init, r_init, params[i])
                return qsim
            
        if return_storage:
            return qsim, s_store, r_store
        else:
            return qsim
        
    def fit(self, qobs, prec, etp, s_init=0., r_init=0.):
        """Fit the GR4J model to a timeseries of discharge.
        
        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed 
        discharge is simulated as good as possible.
        
        Args:
            qobs: Array of observed streamflow discharge [mm/day]
            prec: Array of daily precipitation sum [mm]
            etp: Array of mean potential evapotranspiration [mm]
            s_init: (optional) Initial value of the production storage as 
                fraction of x1. 
            r_init: (optional) Initial value of the routing storage as fraction
                of x3.
        
        Returns:
            res: A scipy OptimizeResult class object.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If there is a size mismatch between the 
                precipitation and the pot. evapotranspiration input.
        
        """
        # Validation check of the inputs
        prec = validate_array_input(prec, np.float64, 'precipitation')
        etp = validate_array_input(etp, np.float64, 'pot. evapotranspiration')
        qobs = validate_array_input(qobs, np.float64, 'observed discharge')
        
        # Check if there exist negative precipitation values in the input
        if check_for_negatives(prec):
            msg = "The precipitation array contains negative values."
            raise ValueError(msg)
        
        # Check for same size of inputs
        if len(prec) != len(etp):
            msg = ["The arrays of precipitation and pot. evapotranspiration,"
                   " must be of the same size."]
            raise RuntimeError("".join(msg))
        
        # Make sure the intial storage values are floating point numbers
        s_init = float(s_init)
        r_init = float(r_init)
        
        # check if the intial values are in rage [0, 1]
        if (s_init < 0) or (s_init > 1):
            msg = ["The initial value of the production storage must be in ",
                   "the range [0,1]."]
            raise ValueError("".join(msg))
        
        if (r_init < 0) or (r_init > 1):
            msg = ["The initial value of the routing storage must be in the",
                   " range [0,1]."]
            raise ValueError("".join(msg))
        
        # pack input arguments for scipy optimizer
        args = (qobs, prec, etp, s_init, r_init, self._dtype)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])
        
        # call scipy's global optimizer
        res = optimize.differential_evolution(_loss, bounds=bnds, args=args)

        return res
    
    
def _loss(X, *args):
    """Return the loss value for the current parameter set."""
    # Unpack static arrays
    qobs = args[0]
    prec = args[1]
    etp = args[2]
    s_init = args[3]
    r_init = args[4]
    dtype = args[5]
    
    # Create custom numpy array of model parameters
    params = np.zeros(1, dtype=dtype)
    params['x1'] = X[0]
    params['x2'] = X[1]
    params['x3'] = X[2]
    params['x4'] = X[3]
    
    # Calculate simulated streamflow
    qsim, _, _ = run_gr4j(prec, etp, s_init, r_init, params[0])
    
    # Calculate the loss of the fit as the mean squared error
    loss_value = mse(qobs, qsim)
    
    return loss_value

  
