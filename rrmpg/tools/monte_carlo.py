# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""Implementation of Monte_Carlo-Simulation for rrmpg.models."""

import numpy as np

from ..models.basemodel import BaseModel
from ..utils.metrics import mse
from ..utils.array_checks import validate_array_input


def monte_carlo(model, num, qobs=None, **kwargs):
    """Perform Monte-Carlo-Simulation.

    This function performs a Monte-Carlo-Simulation for any given hydrological
    model of this repository.

    Args:
        model: Any instance of a hydrological model of this repository.
        num: Number of simulations.
        qobs: (optional) Array of observed streamflow.
        **kwargs: Keyword arguments, matching the inputs the model needs to
            perform a simulation (e.g. qobs, precipitation, temperature etc.).
            See help(model.simulate) for model input requirements.

    Returns:
        A dictonary containing the following two keys ['params', 'qsim']. The 
        key 'params' contains a numpy array with the model parameter of each 
        simulation. 'qsim' is a 2D numpy array with the simulated streamflow 
        for each simulation. If an array of observed streamflow is provided,
        one additional key is returned in the dictonary, being 'mse'. This key
        contains an array of the mean-squared-error for each simulation.

    Raises:
        ValueError: If any input contains invalid values.
        TypeError: If any of the inputs has a wrong datatype.

    """
    # Make sure the model contains to this repository.
    if not issubclass(model.__class__, BaseModel):
        msg = ["The model must be one of the models implemented in the ",
               "rrmpg.models module."]
        raise TypeError("".join(msg))

    # Check if n is an integer.
    if not isinstance(num, int) or num < 1:
        raise TypeError("'n' must be a positive integer greate than zero.")

    if qobs is not None: 
        # Validation check of qobs
        qobs = validate_array_input(qobs, np.float64, 'qobs')
    
    # Generate sets of random parameters
    params = model.get_random_params(num=num)

    # perform monte carlo simulation by calculating simulation of all param sets
    qsim = model.simulate(params=params, **kwargs)

    if qobs is not None:  
        # calculate nse of each simulation
        mse_values = np.zeros(num, dtype=np.float64)
        
        for n in range(num):
            mse_values[n] = mse(qobs, qsim[:, n])
            
        return {'params': params, 'qsim': qsim, 'mse': mse_values}
    
    else:
        return {'params': params, 'qsim': qsim}
