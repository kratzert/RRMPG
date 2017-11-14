# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

from numba import njit

from .cemaneige_model import run_cemaneige
from .gr4j_model import run_gr4j

@njit
def run_cemaneigegr4j(prec, mean_temp, etp, frac_solid_prec, snow_pack_init, 
                      thermal_state_init, s_init, r_init, params):
    """Implementation of the Cemaneige + GR4J coupled hydrological model.
    
    This function should be called via the .simulate() function of the 
    CemaneigeGR4J class and not directly. It is kept in a separate file for 
    less confusion if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication [1] and 
    [2].
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        mean_temp: Numpy [t,n] array, which contains the mean temperature for
            each elevation layer n.
        frac_solid_prec: Numpy [t,n] array, which contains the fraction of 
            solid precipitation for each elevation layer n.
        snow_pack_init: Scalar for the initial state of the snow pack.
        thermal_state_init: Scalar for the initial state of the thermal state.
        params: Numpy array of custom dtype, which contains the model parameter.
        
    Returns:
        outflow: Numpy [t] array, which contains the liquid water outflow for
            each timestep.
        G: Numpy [t] array, which contains the state of the snow pack for 
            each timestep.
        eTG: Numpy [t] array, which contains the thermal state of the snow
            pack for each timestep.
            
    [1] Valéry, A. "Modélisation précipitations – débit sous influence nivale.
    Élaboration d’un module neige et évaluation sur 380 bassins versants".
    PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)
    
    [2] Perrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement 
    of a parsimonious model for streamflow simulation." Journal of hydrology 
    279.1 (2003): 275-289.
        
    """    
    # run the cemaneige snow routine (see cemaneige_model.py)
    liquid_water, G, eTG = run_cemaneige(prec, mean_temp, frac_solid_prec, 
                                         snow_pack_init, thermal_state_init, 
                                         params)
    
    # use the output from above as input to the gr4j. (see gr4j_model.py)
    qsim, s_store, r_store = run_gr4j(liquid_water, etp, s_init, r_init, params)
    
    return qsim, G, eTG, s_store, r_store