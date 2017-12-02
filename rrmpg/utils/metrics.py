# -*- coding: utf-8 -*-a 
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

"""
Implementation of evaluation metrics for e.g. hydrological model simulations.

Implemented functions:
    nse: Calculate the Nash-Sutcliffe model efficiency coefficient.
    rmse: Calculate the root mean squared error.
    mse: Calculate the mean squared error.
"""
import numpy as np

from .array_checks import validate_array_input

def nse(obs, sim):
    """Calculate the Nash-Sutcliffe model efficiency coefficient.

    Original Publication:
    Nash, J. Eamonn, and Jonh V. Sutcliffe. "River flow forecasting through
    conceptual models part I—A discussion of principles." Journal of
    hydrology 10.3 (1970): 282-290.
    
    

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The NSE value for the simulation, compared to the observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.
        RuntimeError: If all values in qobs are equal. The NSE is not defined
            for this cases.

    """
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')
    
    # denominator of the fraction term
    denominator = np.sum((obs-np.mean(obs))**2)
    
    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        msg = ["The Nash-Sutcliffe-Efficiency coefficient is not defined ",
               "for the case, that all values in the observations are equal.",
               " Maybe you should use the Mean-Squared-Error instead."]
        raise RuntimeError("".join(msg))
    
    # numerator of the fraction term
    numerator = np.sum((sim-obs)**2)

    # calculate the NSE
    nse_val = 1 - numerator/denominator
    
    return nse_val



def rmse(obs, sim):
    """Calculate the root mean squared error.

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The RMSE value for the simulation, compared to the observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.

    """
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')

    # Calculate the rmse value
    rmse_val = np.sqrt(np.mean((obs-sim)**2))

    return rmse_val


def mse(obs, sim):
    """Calculate the mean squared error.

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The MSE value for the simulation, compared to the observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.

    """
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')

    # Calculate the rmse value
    mse_val = np.mean((obs-sim)**2)

    return mse_val 