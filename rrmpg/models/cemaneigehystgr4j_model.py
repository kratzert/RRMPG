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

from .cemaneigehyst_model import run_cemaneigehyst
from .gr4j_model import run_gr4j

@njit
def run_cemaneigehystgr4j(prec, mean_temp, etp, frac_solid_prec, snow_pack_init, 
                      thermal_state_init, sca_init, s_init, r_init, params):
    """Implementation of the Cemaneige Hysteresis + GR4J coupled hydrological model.
    
    This function should be called via the .simulate() function of the 
    CemaneigeHystGR4J class and not directly. It is kept in a separate file for 
    less confusion if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication [1], [2] and 
    [3].
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        mean_temp: Numpy [t,n] array, which contains the mean temperature for
            each elevation layer n.
        etp: Numpy [t] array, which contains the potential evapotranspiration.
        frac_solid_prec: Numpy [t,n] array, which contains the fraction of 
            solid precipitation for each elevation layer n.
        snow_pack_init: Scalar for the initial state of the snow pack.
        thermal_state_init: Scalar for the initial state of the thermal state.
        sca_init: Scalar for the initial snow-covered area fraction.
        s_init: Scalar for the initial production storage as a fraction of x1.
        r_init: Scalar for the initial routing storage as a fraction of x3.
        params: Numpy array of custom dtype, which contains the model parameter.
        
    Returns:
        qsim: Numpy [t] array, which contains the liquid water outflow for
            each timestep.
        G: Numpy [t] array, which contains the state of the snow pack for 
            each timestep.
        eTG: Numpy [t] array, which contains the thermal state of the snow
            pack for each timestep.
        s_store: Numpy [t] array, which contains the state of the production 
            storage for each timestep.
        r_store: Numpy [t] array, which contains the state of the routing 
            storage for each timestep.
        sca: Numpy [t,n] array, which contains the snow-covered area fraction 
            for each timestep and elevation layer.
        rain: Numpy [t,n] array, which contains the liquid precipitation 
            (rainfall) for each timestep and elevation layer.
            
    [1] Valéry, A. "Modélisation précipitations – débit sous influence nivale.
    Élaboration d’un module neige et évaluation sur 380 bassins versants".
    PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

    [2]  Riboust, P., Thirel, G., Le Moine, N., Ribstein, P. "Revisiting a simple degree-day model for
    integrating satellite data: implementation of SWE-SCA hystereses". Jounral of Hydrology and Hydromenchanics,
    vol. 67, pp. 70-81, (2019)
    
    [3] Perrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement 
    of a parsimonious model for streamflow simulation." Journal of hydrology 
    279.1 (2003): 275-289.
        
    """    
    # run the cemaneige snow routine (see cemaneigehyst_model.py)
    liquid_water, G, eTG, sca, rain = run_cemaneigehyst(prec, mean_temp, frac_solid_prec,
                                         snow_pack_init, thermal_state_init, sca_init, params) 
    
    # use the output from above as input to the gr4j. (see gr4j_model.py)
    qsim, s_store, r_store = run_gr4j(liquid_water, etp, s_init, r_init, params)
    
    return qsim, G, eTG, s_store, r_store, sca, rain