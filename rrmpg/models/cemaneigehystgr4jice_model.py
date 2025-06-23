# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
# Updated by Mat


from numba import njit
import numpy as np

from .cemaneigehyst_model import run_cemaneigehyst
from .gr4j_model import run_gr4j
from .icemelt_model import run_icemelt


@njit
def run_cemaneigehystgr4jice(prec, mean_temp, etp, frac_ice, frac_solid_prec, snow_pack_init, 
                      thermal_state_init, sca_init, s_init, r_init, params):
    """Implementation of the IceMelt + Cemaneige Hysteresis + GR4J coupled hydrological model.
    
    This function should be called via the .simulate() function of the 
    CemaneigeHystGR4JIce class and not directly. It is kept in a separate file for 
    less confusion if anyone wants to inspect the actual model routine.
    
    The naming of the variables is kept as in the original publication [1], [2], [3] and [4].
    
    Args:
        prec: Numpy [t,n] array, which contains the precipitation for each 
            elevation layer n.
        mean_temp: Numpy [t,n] array, which contains the mean temperature for
            each elevation layer n.
        etp: Numpy [t] array, which contains the potential evapotranspiration.
        frac_ice: Numpy [n] array, which contains the fraction of ice for 
            each elevation layer n.
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
        icemelt: Numpy [t] array, which contains the ice melt contribution for 
            each timestep.
        snowmelt: Numpy [t,n] array, which contains the snowmelt for each 
            timestep and elevation layer.
        rain: Numpy [t,n] array, which contains the liquid precipitation 
            (rainfall) for each timestep and elevation layer.
        

    [1] Nepal, S., Chen, J., Penton, D. J., Neumann, L. E., Zheng, H., & Wahid, S. (2017). 
    Spatial GR4J conceptualization of the Tamor glaciated alpine catchment in Eastern Nepal: 
    evaluation of GR4JSG against streamflow and MODIS snow extent. Hydrol. Process., 31, 51–68.
    doi: 10.1002/hyp.10962.
                          
    [2] Valéry, A. "Modélisation précipitations – débit sous influence nivale.
    Élaboration d’un module neige et évaluation sur 380 bassins versants".
    PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)
    
    [3] Perrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement 
    of a parsimonious model for streamflow simulation." Journal of hydrology 
    279.1 (2003): 275-289.

    [4] Riboust, P., Thirel, G., Le Moine, N., Ribstein, P. "Revisiting a simple degree-day model for
    integrating satellite data: implementation of SWE-SCA hystereses". Jounral of Hydrology and Hydromenchanics,
    vol. 67, pp. 70-81, (2019)
        
    """    
    # run the cemaneige snow routine (see cemaneige_model.py)
    snowmelt, G, eTG, sca, rain = run_cemaneigehyst(prec, mean_temp, frac_solid_prec,
                                         snow_pack_init, thermal_state_init, sca_init, params)
    
    # run the ice melt routine (see icemelt_model.py)                                    
    icemelt = run_icemelt(mean_temp, G, params)
    
    # calculate total icemelt
    icemelt = np.sum(icemelt * frac_ice[np.newaxis, :], axis=1)

    # calculate total liquid water
    liquid_water = snowmelt + icemelt

    # use the output from above as input to the gr4j (see gr4j_model.py)
    qsim, s_store, r_store = run_gr4j(liquid_water, etp, s_init, r_init, params)
    
    return qsim, G, eTG, s_store, r_store, sca, icemelt, snowmelt, rain