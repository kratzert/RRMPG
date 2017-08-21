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
"""
import numpy as np
import pandas as pd


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
        ValueError: If the arrays are not of equal size, have non-numeric,
            values or are not of a correct datatype.

    """
    # Check the observation input
    if isinstance(obs, (list, np.ndarray, pd.Series)):
        # Check for numerical values
        try:
            obs = np.array(obs, dtype=np.float64)
        except:
            msg = "'obs' must be an array of only numerical values."
            raise ValueError(msg)
    else:
        msg = ["'obs' must either be a list, numpy.ndarray or pandas.Series"]
        raise ValueError(msg)

    # Check the observation input
    if isinstance(sim, (list, np.ndarray, pd.Series)):
        # Check for numerical values
        try:
            sim = np.array(sim, dtype=np.float64)
        except:
            msg = "'sim' must be an array of only numerical values."
            raise ValueError(msg)
    else:
        msg = ["'sim' must either be a list, numpy.ndarray or pandas.Series"]
        raise ValueError(msg)
    
    # numerator of the fraction term
    numerator = np.sum((sim-obs)**2)
    
    # if simulation matches observation perfectly nse is defined as 1
    if numerator == 0:
        return 1
    
    # denominator of the fraction term
    denominator = np.sum((obs-np.mean(obs))**2)
    
    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0 and numerator != 0:
        return -1 * np.inf

    else:
        # calculate the NSE
        nse = 1 - (np.sum((sim-obs)**2)/np.sum((obs-np.mean(obs))**2))
        return nse
    

    return nse


def rmse(obs, sim):
    """Calculate the root mean squared error.

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The RMSE value for the simulation, compared to the observation.

    Raises:
        ValueError: If the arrays are not of equal size, have non-numeric,
            values or are not of a correct datatype.

    """
    # Check the observation input
    if isinstance(obs, (list, np.ndarray, pd.Series)):
        # Check for numerical values
        try:
            obs = np.array(obs, dtype=np.float64)
        except:
            msg = "'obs' must be an array of only numerical values."
            raise ValueError(msg)
    else:
        msg = ["'obs' must either be a list, numpy.ndarray or pandas.Series"]
        raise ValueError(msg)

    # Check the observation input
    if isinstance(sim, (list, np.ndarray, pd.Series)):
        # Check for numerical values
        try:
            sim = np.array(sim, dtype=np.float64)
        except:
            msg = "'sim' must be an array of only numerical values."
            raise ValueError(msg)
    else:
        msg = ["'sim' must either be a list, numpy.ndarray or pandas.Series"]
        raise ValueError(msg)

    # Calculate the rmse value
    rmse = np.sqrt(np.mean((obs-sim)**2))

    return rmse
