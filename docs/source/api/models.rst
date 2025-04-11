Models
======

Documentation of all hydrological models implemented in the ``rrmpg.models`` module.

ABC-Model
---------

Implementation of the model described in:

.. code-block:: none

    Myron B. Fiering "Streamflow synthesis" Cambridge, Harvard University
    Press, 1967. 139 P. (1967).

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* a: describes the fraction of precipitation that percolates through the soil to the groundwater.
* b: describes the fraction of the precipitaion that is directly lost to the atmosphere through evapotranspiration.
* c: describes the amount of groundwater that leaves the storage and drains into the stream.

Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of (summed) precipitation for each timestep. [mm/day]

Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.ABCModel
   :members:
   :inherited-members:


HBV education
-------------

Implementation of the model described in:

.. code-block:: none

    Aghakouchak, Amir, and Emad Habib. "Application of a conceptual hydrologic
    model in teaching hydrologic processes." International Journal of
    Engineering Education 26.4 (S1) (2010).

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* T_t: Threshold temperature. Decides if snow is melting or accumulating.
* DD: Degree-day factor. Indicates the decrease of the water content in the snow cover.
* FC: Field capacity. Describes the maximum soil moisture storage in the subsurface zone.
* Beta: Shape coefficient. Controls the amount of liquid water (Precipitation + melting Snow), which contributes to runoff.
* C: Improves model performance, when mean daily temperature deviates considerably from long-term mean.
* PWP: Permanent Wilting Point. Is a soil-moisture limit for evapotranspiration.
* K_0: Near surface flow storage coefficient.
* K_1: Interflow storage coefficient. K_1 should be smaller than K_0.
* K_2: Baseflow storage coefficient. K_2 should be smaller than K_1.
* K_p: Percolation storage coefficient.
* L: Threshold of the water level in the upper storage.

Model inputs for simulation:
""""""""""""""""""""""""""""

* temp: Array of (mean) temperature for each timestep.
* prec: Array of (summed) precipitation for each timestep. [mm/day]
* month: Array of integers indicating for each timestep to which month it belongs [1,2, ..., 12]. Used for adjusted potential evapotranspiration.
* PE_m: long-term mean monthly potential evapotranspiration.
* T_m: long-term mean monthly temperature.

Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.HBVEdu
   :members:
   :inherited-members:


GR4J
-------------
Implementation of the model described in:

.. code-block:: none

    Perrin, Charles, Claude Michel, and Vazken Andréassian. "Improvement of a
    parsimonious model for streamflow simulation." Journal of hydrology 279.1
    (2003): 275-289.

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* x1: maximum capacity of the production store [mm]
* x2: groundwater exchange coefficient [mm]
* x3: one day ahead maximum capacity of the routing store [mm]
* x4: time base of the unit hydrograph UH1 [days]

Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of precipitation [mm/day]
* etp: Array mean potential evapotranspiration [mm]


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.GR4J
  :members:
  :inherited-members:


Cemaneige
-------------
Implementation of the model described in:

.. code-block:: none

  Valéry, A. "Modélisation précipitations – débit sous influence nivale.
  Élaboration d’un module neige et évaluation sur 380 bassins versants".
  PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* CTG: snow-pack inertia factor
* Kf: day-degree factor


Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of daily precipitation sum [mm]
* mean_temp: Array of the mean temperature [C]
* min_temp: Array of the minimum temperature [C]
* max_temp: Array of the maximum temperature [C]
* met_station_height: Height of the meteorological station [m]. Needed to
  calculate the fraction of solid precipitation and optionally for the
  extrapolation of the meteorological inputs.
* altitudes: (optionally) List of the median elevation of each elevation layer.


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.Cemaneige
  :members:
  :inherited-members:

CemaneigeGR4J
-------------
This model couples the Cemaneige snow routine with the GR4J model into one model.

.. code-block:: none

  Valéry, A. "Modélisation précipitations – débit sous influence nivale.
  Élaboration d’un module neige et évaluation sur 380 bassins versants".
  PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* CTG: snow-pack inertia factor
* Kf: day-degree factor
* x1: maximum capacity of the production store [mm]
* x2: groundwater exchange coefficient [mm]
* x3: one day ahead maximum capacity of the routing store [mm]
* x4: time base of the unit hydrograph UH1 [days]


Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of daily precipitation sum [mm]
* mean_temp: Array of the mean temperature [C]
* min_temp: Array of the minimum temperature [C]
* max_temp: Array of the maximum temperature [C]
* etp: Array mean potential evapotranspiration [mm]
* met_station_height: Height of the meteorological station [m]. Needed to
  calculate the fraction of solid precipitation and optionally for the
  extrapolation of the meteorological inputs.
* altitudes: (optionally) List of the median elevation of each elevation layer.


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.CemaneigeGR4J
  :members:
  :inherited-members:

CemaneigeGR4JIce
----------------
This model couples the Cemaneige snow routine and Ice-Melt model with the GR4J model into one model.

.. code-block:: none

  Valéry, A. "Modélisation précipitations – débit sous influence nivale.
  Élaboration d’un module neige et évaluation sur 380 bassins versants".
  PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

  Nepal, S., Chen, J., Penton, D. J., Neumann, L. E., Zheng, H., & Wahid, S. 
  "Spatial GR4J conceptualization of the Tamor glaciated alpine catchment in Eastern Nepal: 
  evaluation of GR4JSG against streamflow and MODIS snow extent." Hydrol. Process., 31, 51–68. (2017)

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* CTG: snow-pack inertia factor
* Kf: day-degree factor
* x1: maximum capacity of the production store [mm]
* x2: groundwater exchange coefficient [mm]
* x3: one day ahead maximum capacity of the routing store [mm]
* x4: time base of the unit hydrograph UH1 [days]
* DDF: degree-day factor for ice melt


Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of daily precipitation sum [mm]
* mean_temp: Array of the mean temperature [C]
* min_temp: Array of the minimum temperature [C]
* max_temp: Array of the maximum temperature [C]
* etp: Array mean potential evapotranspiration [mm]
* frac_ice: Array of the fraction of glaciated area in each elevation band. [0 - 1]
* met_station_height: Height of the meteorological station [m]
* snow_pack_init: (optional) Initial value of the snow pack storage
* thermal_state_init: (optional) Initial value of the thermal state of the snow pack
* s_init: (optional) Initial value of the production storage as fraction of x1. 
* r_init: (optional) Initial value of the routing storage as fraction of x3.
* altitudes: (optional) List of median altitudes of each elevation layer [m]
* return_storages: (optional) Boolean, indicating if the model storages should also be returned
* params: (optional) Numpy array of parameter sets


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.CemaneigeGR4JIce
  :members:
  :inherited-members:

CemaneigeHystGR4J
-----------------
This model couples the Cemaneige snow routine (with Hysteresis) with the GR4J model into one model.

.. code-block:: none

  Valéry, A. "Modélisation précipitations – débit sous influence nivale.
  Élaboration d’un module neige et évaluation sur 380 bassins versants".
  PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

  Riboust, P., Thirel, G., Le Moine, N., Ribstein, P. "Revisiting a simple degree-day model for
  integrating satellite data: implementation of SWE-SCA hystereses". Jounral of Hydrology and Hydromenchanics,
  vol. 67, pp. 70-81. (2019)

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* CTG: snow-pack inertia factor
* Kf: day-degree factor
* Thacc: accumulation threshold [mm]
* Rsp: ratio of melt threshold and mean annual precipitation [0-1]
* x1: maximum capacity of the production store [mm]
* x2: groundwater exchange coefficient [mm]
* x3: one day ahead maximum capacity of the routing store [mm]
* x4: time base of the unit hydrograph UH1 [days]


Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of daily precipitation sum [mm]
* mean_temp: Array of the mean temperature [C]
* min_temp: Array of the minimum temperature [C]
* max_temp: Array of the maximum temperature [C]
* etp: Array mean potential evapotranspiration [mm]
* met_station_height: Height of the meteorological station [m]. 
* snow_pack_init: (optional) Initial value of the snow pack storage
* thermal_state_init: (optional) Initial value of the thermal state of the snow pack
* sca_init: (optional) Initial value of the snow covered area.
* s_init: (optional) Initial value of the production storage as fraction of x1. 
* r_init: (optional) Initial value of the routing storage as fraction of x3.
* altitudes: (optional) List of median altitudes of each elevation layer [m]
* return_storages: (optional) Boolean, indicating if the model storages should also be returned
* params: (optional) Numpy array of parameter sets


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.CemaneigeHystGR4J
  :members:
  :inherited-members:

CemaneigeHystGR4JIce
--------------------
This model couples the Cemaneige snow routine (with Hysteresis) and Ice-Melt with the GR4J model into one model.

.. code-block:: none

  Valéry, A. "Modélisation précipitations – débit sous influence nivale.
  Élaboration d’un module neige et évaluation sur 380 bassins versants".
  PhD thesis, Cemagref (Antony), AgroParisTech (Paris), 405 pp. (2010)

  Riboust, P., Thirel, G., Le Moine, N., Ribstein, P. "Revisiting a simple degree-day model for
  integrating satellite data: implementation of SWE-SCA hystereses". Jounral of Hydrology and Hydromenchanics,
  vol. 67, pp. 70-81. (2019)

  Nepal, S., Chen, J., Penton, D. J., Neumann, L. E., Zheng, H., & Wahid, S. 
  "Spatial GR4J conceptualization of the Tamor glaciated alpine catchment in Eastern Nepal: 
  evaluation of GR4JSG against streamflow and MODIS snow extent." Hydrol. Process., 31, 51–68. (2017)

Explanation of the model parameters:
""""""""""""""""""""""""""""""""""""

* CTG: snow-pack inertia factor
* Kf: day-degree factor
* Thacc: accumulation threshold [mm]
* Rsp: ratio of melt threshold and mean annual precipitation [0-1]
* x1: maximum capacity of the production store [mm]
* x2: groundwater exchange coefficient [mm]
* x3: one day ahead maximum capacity of the routing store [mm]
* x4: time base of the unit hydrograph UH1 [days]
* DDF: degree-day factor for ice melt


Model inputs for simulation:
""""""""""""""""""""""""""""

* prec: Array of daily precipitation sum [mm]
* mean_temp: Array of the mean temperature [C]
* min_temp: Array of the minimum temperature [C]
* max_temp: Array of the maximum temperature [C]
* etp: Array mean potential evapotranspiration [mm]
* frac_ice: Array of the fraction of glaciated area in each elevation band. [0 - 1]
* met_station_height: Height of the meteorological station [m]. 
* snow_pack_init: (optional) Initial value of the snow pack storage
* thermal_state_init: (optional) Initial value of the thermal state of the snow pack
* sca_init: (optional) Initial value of the snow covered area.
* s_init: (optional) Initial value of the production storage as fraction of x1. 
* r_init: (optional) Initial value of the routing storage as fraction of x3.
* altitudes: (optional) List of median altitudes of each elevation layer [m]
* return_storages: (optional) Boolean, indicating if the model storages should also be returned
* params: (optional) Numpy array of parameter sets


Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.CemaneigeHystGR4JIce
  :members:
  :inherited-members: