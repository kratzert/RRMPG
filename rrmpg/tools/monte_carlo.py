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
import pandas as pd

from ..models.basemodel import BaseModel
from ..utils.metrics import nse


def monte_carlo(model, num, qobs, **kwargs):
    """Perform Monte-Carlo-Simulation.

    This function performs a Monte-Carlo-Simulation for any given hydrological
    model of this repository.

    Args:
        model: Any instance of a hydrological model of this repository.
        num: Number of simulations.
        qobs: Array of observed streamflow.
        **kwargs: Keyword arguments, matching the inputs the model needs to
            perform a simulation (e.g. qobs, precipitation, temperature etc.).
            See help(model.simulate) for model input requirements.

    Returns:
        A dictonary containing the following three keys ['params', 'nse',
        'qsim']. The key 'params' contains a numpy array with the model
        parameter of each simulation. 'nse' contains the
        Nash-Sutcliff-Efficiency of each simulation, and 'qsim' is a 2D numpy
        array with the simulated streamflow for each simulation.

    Raises:
        ValueError: For incorrect inputs.

    """
    # Make sure the model contains to this repository.
    if not issubclass(model.__class__, BaseModel):
        msg = ["The model must be one of the models implemented in the ",
               "rrmpg.models module."]
        raise ValueError("".join(msg))

    # Check if n is an integer.
    if not isinstance(num, int) or num < 1:
        raise ValueError("'n' must be a positive integer greate than zero.")

    # Validation check of qobs
    if isinstance(qobs, (list, np.ndarray, pd.Series)):
        # Try to convert as numpy array
        try:
            qobs = np.array(qobs, dtype=np.float64)
        except:
            msg = ["The data of the 'qobs' array must be must be purely ",
                   "numerical."]
            raise ValueError("".join(msg))
    else:
        msg = ["The array 'qobs' must be either a list, numpy.ndarray or ",
               "pandas.Series"]
        raise ValueError("".join(msg))

    # Initialize arrays for the params, simulations and model efficiency
    params = np.zeros(num, dtype=model.get_dtype())
    qsim = np.zeros((len(qobs), num), dtype=np.float64)
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

        # calculate model efficiency
        nse_values[n] = nse(qobs, qsim[:, n])

    return {'params': params, 'nse': nse_values, 'qsim': qsim}
