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

from numba import njit
from scipy.optimize import minimize

from .basemodel import BaseModel
from ..utils.metrics import nse
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

    def simulate(self, prec, initial_state=0, return_storage=False):
        """Simulate the streamflow for the passed precipitation.

        This function makes sanity checks on the input and then calls the
        externally defined ABC-Model function.

        Args:
            prec: Precipitation data for each timestep. Can be a List, numpy
                array or pandas.Series
            initial_state: (optional) Initial value for the storage.
            return_storage: (optional) Boolean, wether or not to return the
                simulated storage for each timestep.

        Returns:
            An array with the simulated stream flow for each timestep and
            optional an array with the simulated storage.

        Raises:
            ValueError: If one of the inputs is not a correct data type or
                value.

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
            raise ValueError("".join(msg))

        # Cast initial state as float
        initial_state = float(initial_state)

        # Validation check of the return_storage
        if not isinstance(return_storage, bool):
            raise ValueError("The return_storage arg must be a boolean.")

        # Create custom numpy data structure containing the model parameters
        params = np.zeros(1, dtype=self._dtype)
        for param in self._param_list:
            params[param] = getattr(self, param)

        # Call ABC-model simulation function and return results
        if return_storage:
            qsim, storage = _simulate_abc(params, prec, initial_state)
            return qsim, storage

        else:
            qsim, _ = _simulate_abc(params, prec, initial_state)
            return qsim

    def fit(self, qobs, prec, initial_state=0, x0=None, method=None):
        """Fit the model to a timeseries of discharge using.

        This functions uses scipy's minimize optimizer to find a good set of
        parameters for the model, so that the observed discharge is simulated
        as good as possible.

        Args:
            qobs: Array of observed streaflow discharge.
            prec: Array of precipitation data.
            initial_state: (optional) Initial value for the storage.
            x0: (optional) Initial guess of parameter values. If not passed,
                random parameters will be used as starting point.
            method: String specifing one of scipy's minimizer methods.

        Returns:
            res: A scipy OptimizeResult class object.

        """
        # Validation check of the inputs
        qobs = validate_array_input(qobs, np.float64, 'qobs')
        prec = validate_array_input(prec, np.float64, 'precipitation')
        
        # Check if there exist negative precipitation
        if check_for_negatives(prec):
            raise ValueError("In the precipitation array are negative values.")
        
        if not isinstance(initial_state, numbers.Number):
            raise ValueError("The initial_state must be a numerical scalar.")
        
        # Cast initial state as float
        initial_state = float(initial_state)

        # If no parameter guess were passed, generate random values
        if not x0:
            rand_params = self.get_random_params()
            x0 = [rand_params[p] for p in self._param_list]

        # pack input arguments for scipy optimizer
        args = (prec, initial_state, qobs, self._dtype)
        bnds = tuple([self._default_bounds[p] for p in self._param_list])

        # call the actual optimizer function
        res = minimize(_loss, x0, args=args, method=method, bounds=bnds)

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
    qsim = _simulate_abc(params, prec, initial_state)

    # Calculate the Nash-Sutfliff model efficiency
    loss_value = -1 * (nse(qobs, qsim))

    return loss_value


@njit
def _simulate_abc(params, prec, initial_state):
    """Run a simulation of the ABC-model for given input and params."""
    # Unpack model parameters
    a = params['a'][0]
    b = params['b'][0]
    c = params['c'][0]

    # Number of simulation timesteps
    num_timesteps = len(prec)

    # Initialize array for the simulated stream flow and the storage
    qsim = np.zeros(num_timesteps, np.float64)
    storage = np.zeros(num_timesteps, np.float64)

    # Set the initial storage value
    storage[0] = initial_state

    # Model simulation
    for t in range(1, num_timesteps):
        # Calculate the streamflow
        qsim[t] = (1 - a - b) * prec[t] + c * storage[t-1]

        # Update the storage
        storage[t] = (1 - c) * storage[t-1] + a * prec[t]

    return qsim, storage
