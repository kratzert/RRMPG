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
    """Run a simulation of the ABC-model for given input and param sets."""
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