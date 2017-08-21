Models
======

Documentation of all hydrological models implemented in the ``rrmpg.models`` module.

ABC-Model
---------

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

Basin specification
"""""""""""""""""""
* area: Area of the basin in [m²]

Class documentation
"""""""""""""""""""

.. autoclass:: rrmpg.models.HBVEdu
   :members:
   :inherited-members:
