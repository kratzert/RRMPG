import numpy as np
from numba import njit, prange

@njit
def run_cemaneigehyst(prec, mean_temp, frac_solid_prec, snow_pack_init,
                  thermal_state_init, sca_init, params):
    """Implementation of the Cemaneige snow routine with modified linear hysteresis.
    
    This function should be called via the .simulate() function of the 
    Cemaneige class and not directly. It is kept in a separate file for less 
    confusion if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publications [1] [2].
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        mean_temp: Numpy [t,n] array, which contains the mean temperature for
            each elevation layer n.
        frac_solid_prec: Numpy [t,n] array, which contains the fraction of 
            solid precipitation for each elevation layer n.
        snow_pack_init: Scalar for the initial state of the snow pack.
        thermal_state_init: Scalar for the initial state of the thermal state.
        params: Numpy array of custom dtype, which contains the model parameters.
        sca_init: Scalar for the initial state of the snow-covered area.
        params: Numpy array of custom dtype, which contains the model parameters.
        
    Returns:
        outflow: Numpy [t] array, which contains the liquid water outflow for
            each timestep.
        G: Numpy [t,n] array, which contains the state of the snow pack for 
            each timestep.
        eTG: Numpy [t,n] array, which contains the thermal state of the snow
            pack for each timestep.
        sca: Numpy [t,n] array, which contains the snow-covered area fraction for each timestep
            and elevation layer.
        rain: Numpy [t,n] array, which contains the liquid precipitation (rainfall)
            for each timestep and elevation layer.
            
    [1] Valéry, A. "Modélisation précipitations – débit sous influence nivale.
    Élaboration d’un module neige et évaluation sur 380 bassins versants".
    PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

    [2] Riboust, P., Thirel, G., Le Moine, N., Ribstein, P. "Revisiting a simple degree-day model for
    integrating satellite data: implementation of SWE-SCA hystereses". Jounral of Hydrology and Hydromenchanics,
    vol. 67, pp. 70-81, (2019)
    
    """
    # Number of simulation timesteps
    num_timesteps = len(prec)
    
    # Number of elevation layers
    num_layers = prec.shape[1]
    
    # Unpack model parameters
    CTG = params['CTG']
    Kf = params['Kf']
    Thacc = params['Thacc']
    Rsp = params['Rsp']
    
    # Snow pack of each layer ## current SWE at each timestep
    G = np.zeros((num_timesteps, num_layers), np.float64)
    
    # Thermal state of each layer
    eTG = np.zeros((num_timesteps, num_layers), np.float64)
    
    # Outflow as sum of liquid precipitation and melt of each layer
    liquid_water = np.zeros((num_timesteps, num_layers), np.float64)
    
    # Snow-covered area of each layer
    sca = np.zeros((num_timesteps, num_layers), np.float64)

    # rainfall of each layer
    rain = np.zeros((num_timesteps, num_layers), np.float64)
    
    # Total outflow which is the mean of liquid water of each layer
    outflow = np.zeros(num_timesteps, np.float64)
    
    #  Track maximum SWE before melting to determine Thmax
    swe_max = np.zeros(num_layers, np.float64)
    Thmax = np.zeros(num_layers, np.float64)
    
    # Calculate Cemaneige routine for each elevation zone independently
    for l in prange(num_layers):
        
        # Split input precipitation into solid and liquid precipitation
        snow = prec[:, l] * frac_solid_prec[:, l]
        rain[:, l] = prec[:, l] - snow

        # Calc mean annual solid precipitation for each elevation zone
        Psolannual = 365.25 * np.mean(snow)

        for t in range(num_timesteps):
            
            # Accumulate solid precipitation to snow pack
            if t == 0:
                G[t, l] = snow_pack_init
                sca[t, l] = sca_init
            else: 
                G[t, l] = G[t-1, l] + snow[t]
            
            # Calculate snow pack thermal state before melt eTG (eTG ≤ 0)
            if t == 0:
                eTG[t, l] = thermal_state_init
            else:
                eTG[t, l] = CTG * eTG[t-1, l] + (1 - CTG) * mean_temp[t, l]
            if eTG[t, l] > 0:
                eTG[t, l] = 0
            
            # Calculate potential melt 
            if eTG[t, l] == 0 and mean_temp[t, l] > 0:
                pot_melt = Kf * mean_temp[t, l]
                
                # Cap the potential snow melt to the state of the snow pack
                if pot_melt > G[t, l]:
                    pot_melt = G[t, l]
            else:
                pot_melt = 0
                        
            # Calculate snow balance (accumulation - melt)
            snow_balance = snow[t] - pot_melt # snow_balance = Delta SWEt
            
            # Calculate snow-covered area
            if snow_balance >= 0:
                # Accumulation phase
                sca[t, l] = sca[t-1, l] + snow_balance / Thacc 
                swe_max[l] = max(swe_max[l], G[t, l])  # Track max SWE before melt
            else:
                Thmelt = Psolannual * Rsp
                # Ablation phase
                if swe_max[l] > Thmelt:
                    Thmax[l] = Thmelt
                else:
                    Thmax[l] = swe_max[l]  # Max SWE before melting
                
                # Update snow-covered area
                if Thmax[l] > 0:
                    sca[t, l] = G[t, l] / Thmax[l]
                else:
                    sca[t, l] = 0

            # Ensure SCA remains in [0,1]
            sca[t, l] = min(max(sca[t, l], 0), 1)
            
            # Calculate actual snow melt
            melt = (0.9 * sca[t, l] + 0.1) * pot_melt

            # Update snow pack
            G[t, l] = G[t, l] - melt

            # Reset max SWE if snow pack is empty
            if G[t, l] == 0:
                swe_max[l] = 0

            
            # Output: liquid precipitation + actual snow melt
            liquid_water[t, l] = rain[t,l] + melt
        
    # Calculate the outflow as mean of each layer
    for j in prange(num_timesteps):
        outflow[j] = np.mean(liquid_water[j, :])

    return outflow, G, eTG, sca, rain