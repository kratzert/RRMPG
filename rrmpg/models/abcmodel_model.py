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

from numba import njit

@njit
def run_abcmodel(prec, initial_state, params):
    """Implementation of the ABC-Model.
    
    This function should be called via the .simulate() function of the ABCModel
    class and not directly. It is kept in a separate file for less confusion
    if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication [1].
    
    Args:
        prec: Numpy [t] array, which contains the precipitation input.
        initial_state: Scalar for the intial state of the storage.
        params: Numpy array of custom dtype, which contains the model parameter.
        
    Returns:
        qsim: Numpy [t] array with the simulated streamflow.
        storage: Numpy [t] array with the state of the storage of each timestep.

    [1] Myron B. Fiering "Streamflow synthesis" Cambridge, Harvard University
    Press, 1967. 139 P. (1967).
    """
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Unpack model parameters
    a = params['a']
    b = params['b']
    c = params['c']

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