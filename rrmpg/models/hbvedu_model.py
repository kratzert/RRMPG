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
def run_hbvedu(temp, prec, month, PE_m, T_m, snow_init, soil_init, s1_init, 
               s2_init, params):
    """Implementation of the HBV educational model.
    
    This function should be called via the .simulate() function of the HBVEdu
    class and not directly. It is kept in a separate file for less confusion
    if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication [1].
    
    Args:
        temp: Numpy [t] array, which contains the daily mean temperature.
        prec: Numpy [t] array, which contains the daily precipitation.
        month: Numpy [t] array, containing integers that hold the number of the
            month of the current timestep (from 0 to 11 because of Pythons 
            zero indexing).
        PE_m: Numpy array of length 12, containing the long-term monthly 
            potential evapotranspiration.
        T_m: Numpy array of length 12, containing the long-term monthly 
            temperature.
        snow_init: Scalar for the initial state of the snow storage.
        soil_init: Scalar for the initial state of the soil storage.
        s1_init: Scalar for the initial state of the s1 storage.
        s2_init: Scalar for the initial state of the s2 storage.
        params: Numpy array of custom dtype, which contains the model parameter.
    
    Returns:
        qsim: Numpy [t] array with the simulated streamflow.
        snow: Numpy [t] with the state of the snow storage of each timestep.
        soil: Numpy [t] with the state of the soil storage of each timestep.
        s1: Numpy [t] with the state of the s1-storage of each timestep.
        s2: Numpy [t] with the state of the s2-storage of each timestep.
    
    [1] Aghakouchak, Amir, and Emad Habib. "Application of a conceptual 
    hydrologic model in teaching hydrologic processes." International Journal 
    of Engineering Education 26.4 (S1) (2010).
    
    """
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Unpack the model parameters
    T_t = params['T_t']
    DD = params['DD']
    FC = params['FC']
    Beta = params['Beta']
    C = params['C']
    PWP = params['PWP']
    K_0 = params['K_0']
    K_1 = params['K_1']
    K_2 = params['K_2']
    K_p = params['K_p']
    L = params['L']
    
    # initialize empty arrays for all reservoirs and outflow
    snow = np.zeros(num_timesteps, np.float64)
    soil = np.zeros(num_timesteps, np.float64)
    s1 = np.zeros(num_timesteps, np.float64)
    s2 = np.zeros(num_timesteps, np.float64)
    qsim = np.zeros(num_timesteps, np.float64)

    # set initial values
    snow[0] = snow_init
    soil[0] = soil_init
    s1[0] = s1_init
    s2[0] = s2_init

    # Start the model simulation as loop over all timesteps
    for t in range(1, num_timesteps):

        # Check if temperature is below threshold
        if temp[t] < T_t:
            # accumulate snow
            snow[t] = snow[t-1] + prec[t]
            # no liquid water
            liquid_water = 0
        else:
            # melt snow
            snow[t] = max(0, snow[t-1] - DD * (temp[t] - T_t))
            # add melted snow to available liquid water
            liquid_water = prec[t] + min(snow[t-1], DD * (temp[t] - T_t))

        # calculate the effective precipitation
        prec_eff = liquid_water * (soil[t-1] / FC) ** Beta

        # Calculate the potential evapotranspiration
        pe = (1 + C * (temp[t] - T_m[month[t]])) * PE_m[month[t]]

        # Calculate the actual evapotranspiration
        if soil[t-1] > PWP:
            ea = pe
        else:
            ea = pe * (soil[t-1] / PWP)

        # calculate the actual level of the soil reservoir
        soil[t] = soil[t-1] + liquid_water - prec_eff - ea

        # calculate the actual level of the near surface flow reservoir
        s1[t] = (s1[t-1]
                 + prec_eff
                 - max(0, s1[t-1] - L) * K_0
                 - s1[t-1] * K_1
                 - s1[t-1] * K_p)

        # calculate the actual level of the base flow reservoir
        s2[t] = (s2[t-1]
                 + s1[t-1] * K_p
                 - s2[t-1] * K_2)

        qsim[t] = ((max(0, s1[t-1] - L)) * K_0
                   + s1[t] * K_1
                   + s2[t] * K_2)
        
    return qsim, snow, soil, s1, s2
