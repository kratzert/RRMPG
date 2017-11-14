# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

"""Implementation of the ABC-Model."""

import numbers

import numpy as np

from numba import njit, prange
from scipy import optimize

from .basemodel import BaseModel
from .abcmodel_model import run_abcmodel
from ..utils.metrics import mse
from ..utils.array_checks import check_for_negatives, validate_array_input


class ABCModel(BaseModel):
    """Implementation of the ABC-Model.

    This model implements the classical ABC-Model. It was developed for
    educational purpose and represents a simple linear model.

    Original Publication:
        Myron B. Fiering "Streamflow synthesis" Cambridge, Harvard University
        Press, 1967. 139 P. (1967).

    If no model parameters are passed upon initialization, generates random
    parameter set.

    Args:
        params: (optional) Dictonary containing all model parameters as a
            seperate key/value pairs.

    """

    # List of model parameters
    _param_list = ['a', 'b', 'c']

    # Dictionary with default parameter bounds
    _default_bounds = {'a': (0, 1),
                       'b': (0, 1),
                       'c': (0, 1)}

    # Custom numpy datatype needed for numba input
    _dtype = np.dtype([('a', np.float64),
                       ('b', np.float64),
                       ('c', np.float64)])

    def __init__(self, params=None):
        """Initialize an ABC-Model.

        If no parameters are passed as input arguments, random values are
        sampled that satisfy the parameter constraints of the ABC-Model.

        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.

        """
        super().__init__(params=params)
        
    def get_random_params(self, num=1):
        """Generate random sets of model parameters for the ABC-model.

        The ABC-model has specific parameter constraints, therefore we will 
        overwrite the function of the BaseModel, to generated random model
        parameters, that satisfy the ABC-Model constraints.

        Args:
            num: (optional) Integer, specifying the number of parameter sets,
                that will be generated. Default is 1.
                
        Returns:
            A dict containing one key/value pair for each model parameter.

        """
        params = np.zeros(num, dtype=self._dtype)
        
        # sample parameter 'a' between the bounds [0,1]
        params['a'][:] = np.random.uniform(low=self._default_bounds['a'][0],
                                           high=self._default_bounds['a'][1],
                                           size=num)
                # parameter 'c' must be between [0,1] and has no further constraints
        params['c'][:] = np.random.uniform(low=self._default_bounds['c'][0],
                                           high=self._default_bounds['c'][1],
                                           size=num)
        
        # Parameter b is constraint by its corresponding a parameter.
        for i in range(num):
            # sample parameter 'b' between lower bound 0 and upper bnd (1 - a)
            params['b'][i] = np.random.uniform(low=self._default_bounds['b'][0],
                                               high=(1-params['a'][i]),
                                               size=1)
        
        return params

    def simulate(self, prec, initial_state=0, return_storage=False, 
                 params=None):
        """Simulate the streamflow for the passed precipitation.

        This function makes sanity checks on the input and then calls the
        externally defined ABC-Model function.

        Args:
            prec: Precipitation data for each timestep. Can be a List, numpy
                array or pandas.Series
            initial_state: (optional) Initial value for the storage.
            return_storage: (optional) Boolean, wether or not to return the
                simulated storage for each timestep.
            params: (optional) Numpy array of parameter sets, that will be 
                evaluated a once in parallel. Must be of the models own custom
                data type. If nothing is passed, the parameters, stored in the 
                model object, will be used.

        Returns:
            An array with the simulated stream flow for each timestep and
            optional an array with the simulated storage.

        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.

        """
        # Validation check of the precipitation input
        prec = validate_array_input(prec, np.float64, 'precipitation')
        
        # Check if there exist negative precipitation
        if check_for_negatives(prec):
            raise ValueError("In the precipitation array are negative values.")

        # Validation check of the initial state
        if not isinstance(initial_state, numbers.Number) or initial_state < 0:
            msg = ["The variable 'initial_state' must be a numercial scaler ",
                   "greate than 0."]
            raise TypeError("".join(msg))

        # Cast initial state as float
        initial_state = float(initial_state)

        # Validation check of the return_storage
        if not isinstance(return_storage, bool):
            raise TypeError("The return_storage arg must be a boolean.")
        
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
            storage = np.zeros((prec.shape[0], params.size), np.float64)
            
        # call simulation function for each parameter set
        for i in range(params.size):  
        # Call ABC-model simulation function and return results
            if return_storage:
                qsim[:,i], storage[:,i] = run_abcmodel(prec, initial_state,
                                                       params[i])

            else:
                qsim[:,i], _ = run_abcmodel(prec, initial_state, params[i])
        
        if return_storage:   
            return qsim, storage
        else:
            return qsim

    def fit(self, qobs, prec, initial_state=0):
        """Fit the model to a timeseries of discharge using.

        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed 
        discharge is simulated as good as possible.

        Args:
            qobs: Array of observed streaflow discharge.
            prec: Array of precipitation data.
            initial_state: (optional) Initial value for the storage.

        Returns:
            res: A scipy OptimizeResult class object.
            
        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.

        """
        # Validation check of the inputs
        qobs = validate_array_input(qobs, np.float64, 'qobs')
        prec = validate_array_input(prec, np.float64, 'precipitation')
        
        # Check if there exist negative precipitation
        if check_for_negatives(prec):
            raise ValueError("In the precipitation array are negative values.")
        
        # Validation check of the initial state
        if not isinstance(initial_state, numbers.Number) or initial_state < 0:
            msg = ["The variable 'initial_state' must be a numercial scaler ",
                   "greate than 0."]
            raise TypeError("".join(msg))
        
        # Cast initial state as float
        initial_state = float(initial_state)

        # pack input arguments for scipy optimizer
        args = (prec, initial_state, qobs, self._dtype)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])

        # call the actual optimizer function
        res = optimize.differential_evolution(_loss, bounds=bnds, args=args)

        return res


def _loss(X, *args):
    """Return the loss value for the current parameter set."""
    # Unpack static arguments
    prec = args[0]
    initial_state = args[1]
    qobs = args[2]
    dtype = args[3]

    # Create a custom numpy array of the model parameters
    params = np.zeros(1, dtype=dtype)
    params['a'] = X[0]
    params['b'] = X[1]
    params['c'] = X[2]

    # Calculate the simulated streamflow
    qsim, _ = run_abcmodel(prec, initial_state, params[0])

    # Calculate the Mean-Squared-Error as optimization criterion
    loss_value = mse(qobs, qsim)

    return loss_value

