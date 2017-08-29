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
from ..utils.metrics import nse, mse
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
        one additional key is returned in the dictonary, being 'nse'. This key
        contains an array of the Nash-Sutcliff-Efficiency for each simulation.

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
    
    # Initialize arrays for the params, simulations and model efficiency
    params = np.zeros(num, dtype=model.get_dtype())
    qsim = np.zeros((len(kwargs['prec']), num), dtype=np.float64)
    nse_values = np.zeros(num, dtype=np.float64)

    # Perform monte-carlo-simulations
    for n in range(num):
        # generate random parameters
        p = model.get_random_params()

        # store parameters to params array
        for key, value in p.items():
            params[key][n] = value

        # set random params as model parameter
        model.set_params(p)

        # calculate simulation
        qsim[:, n] = model.simulate(**kwargs)
        
        if qobs is not None: 
            # calculate model efficiency
            nse_values[n] = mse(qobs, qsim[:, n])
            
    if qobs is not None:       
        return {'params': params, 'qsim': qsim, 'mse': nse_values}
    
    else:
        return {'params': params, 'qsim': qsim}
