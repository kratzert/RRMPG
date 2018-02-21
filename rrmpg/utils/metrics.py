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
    calc_nse: Calculate the Nash-Sutcliffe model efficiency coefficient.
    calc_rmse: Calculate the root mean squared error.
    calc_mse: Calculate the mean squared error.
    calc_kge: Calculate the Kling-Gupta-Efficiency.
    calc_alpha_nse: Calculate the alpha decomposition of the NSE.
    calc_beta_nse: Calculate the beta decomposition of the NSE.
    calc_r: Calculate the pearson r coefficient.

"""
import numpy as np
from scipy.stats.stats import pearsonr

from .array_checks import validate_array_input

def calc_nse(obs, sim):
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
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
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



def calc_rmse(obs, sim):
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
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")

    # Calculate the rmse value
    rmse_val = np.sqrt(np.mean((obs-sim)**2))

    return rmse_val


def calc_mse(obs, sim):
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

    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    # Calculate the rmse value
    mse_val = np.mean((obs-sim)**2)

    return mse_val 


def calc_kge(obs, sim):
    """Calculate the Kling-Gupta-Efficiency.
    
    Calculate the original KGE value following [1].

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The KGE value for the simulation, compared to the observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.
        RuntimeError: If the mean or the standard deviation of the observations
            equal 0.
    
    [1] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). 
    Decomposition of the mean squared error and NSE performance criteria: 
    Implications for improving hydrological modelling. Journal of Hydrology, 
    377(1-2), 80-91.
    
    """
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
     
    mean_obs = np.mean(obs)
    if mean_obs == 0:
        msg = "KGE not definied if the mean of the observations equals 0."
        raise RuntimeError(msg)
    
    std_obs = np.std(obs)
    if std_obs == 0:
        msg = ["KGE not definied if the standard deviation of the ",
               "observations equals 0."]
        raise RuntimeError("".join(msg))
    
    r = pearsonr(obs, sim)[0]
    alpha = np.std(sim) / std_obs
    beta = np.mean(sim) / mean_obs
    
    kge_val = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    
    return kge_val     
    

def calc_alpha_nse(obs, sim):
    """Calculate the alpha decomposition of the NSE.
    
    Calculate the alpha decomposition of the NSE following [1].

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The alpha decomposition of the NSE of the simulation compared to the
        observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.
        RuntimeError: If the standard deviation of the observations
            equal 0.
    
    [1] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). 
    Decomposition of the mean squared error and NSE performance criteria: 
    Implications for improving hydrological modelling. Journal of Hydrology, 
    377(1-2), 80-91.
    
    """      
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    std_obs = np.std(obs)
    if std_obs == 0:
        msg = ["Not definied if the standard deviation of the observations ",
               "equals 0."]
        raise RuntimeError("".join(msg))
    
    return np.std(sim) / std_obs


def calc_beta_nse(obs, sim):
    """Calculate the beta decomposition of the NSE.
    
    Calculate the beta decomposition of the NSE following [1].

    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The beta decomposition of the NSE of the simulation compared to the
        observation.

    Raises:
        ValueError: If the arrays are not of equal size or have non-numeric
            values.
        TypeError: If the arrays is not a supported datatype.
        RuntimeError: If the mean or the standard deviation of the observations 
            equal 0.
    
    [1] Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F. (2009). 
    Decomposition of the mean squared error and NSE performance criteria: 
    Implications for improving hydrological modelling. Journal of Hydrology, 
    377(1-2), 80-91.
    
    """      
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    std_obs = np.std(obs)
    if std_obs == 0:
        msg = ["Not definied if the standard deviation of the observations ",
               "equals 0."]
        raise RuntimeError("".join(msg))
    
    mean_obs = np.mean(obs)
    if mean_obs == 0:
        msg = "Not definied if the mean of the observations equals 0."
        raise RuntimeError(msg)
    
    return (np.mean(sim) - mean_obs) / std_obs


def calc_r(obs, sim):
    """Calculate the pearson r coefficient.
    
    Interface to the scipy implementation of the pearson r coeffienct.
    
    Args:
        obs: Array of the observed values
        sim: Array of the simulated values

    Returns:
        The pearson r coefficient of the simulation compared to the observation.
 
    """
    # Validation check on the input arrays
    obs = validate_array_input(obs, np.float64, 'obs')
    sim = validate_array_input(sim, np.float64, 'sim')
    
    if len(obs) != len(sim):
        raise ValueError("Arrays must have the same size.")
    
    return pearsonr(obs, sim)