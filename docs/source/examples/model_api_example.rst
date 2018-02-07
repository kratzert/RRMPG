
Model API Example
=================

In this notebook, we'll explore some functionality of the models of this
package. We'll work with the coupled CemaneigeGR4j model that is
implemented in ``rrmpg.models`` module. The data we'll use, comes from
the CAMELS [1] data set. For some basins, the data is provided within
this Python library and can be easily imported using the
``CAMELSLoader`` class implemented in the ``rrmpg.data`` module.

In summary we'll look at: - How you can create a model instance. - How
we can use the CAMELSLoader. - How you can fit the model parameters to
observed discharge by: - Using one of SciPy's global optimizer -
Monte-Carlo-Simulation - How you can use a fitted model to calculate the
simulated discharge.

[1] Addor, N., A.J. Newman, N. Mizukami, and M.P. Clark, 2017: The
CAMELS data set: catchment attributes and meteorology for large-sample
studies. version 2.0. Boulder, CO: UCAR/NCAR. doi:10.5065/D6G73C3Q

.. code:: python

    # Imports and Notebook setup
    from timeit import timeit

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from rrmpg.models import CemaneigeGR4J
    from rrmpg.data import CAMELSLoader
    from rrmpg.tools.monte_carlo import monte_carlo
    from rrmpg.utils.metrics import calc_nse

Create a model
--------------

As a first step let us have a look how we can create one of the models
implemented in ``rrmpg.models``. Basically, for all models we have two
different options: 1. Initialize a model **without** specific model
parameters. 2. Initialize a model **with** specific model parameters.

The `documentation <http://rrmpg.readthedocs.io>`__ provides a list of
all model parameters. Alternatively we can look at ``help()`` for the
model (e.g. ``help(CemaneigeGR4J)``).

If no specific model parameters are provided upon intialization, random
parameters will be generated that are in between the default parameter
bounds. We can look at these bounds by calling ``.get_param_bounds()``
method on the model object and check the current parameter values by
calling ``.get_params()`` method.

For now we don't know any specific parameter values, so we'll create one
with random parameters.

.. code:: python

    model = CemaneigeGR4J()
    model.get_params()

.. parsed-literal::

    {'CTG': 0.3399735717656279,
     'Kf': 0.8724652383290821,
     'x1': 427.9652389107806,
     'x2': 0.9927197563086638,
     'x3': 288.20205223188475,
     'x4': 1.4185137324914372}

Here we can see the six model parameters of CemaneigeGR4J model and
their current values.

Using the CAMELSLoader
----------------------

To have data to start with, we can use the ``CAMELSLoader`` class to
load data of provided basins from the CAMELS dataset. To get a list of
all available basins that are provided within this library, we can use
the ``.get_basin_numbers()`` method. For now we will use the provided
basin number ``01031500``.

.. code:: python

    df = CAMELSLoader().load_basin('01031500')
    df.head()

.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }

        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>dayl(s)</th>
          <th>prcp(mm/day)</th>
          <th>srad(W/m2)</th>
          <th>swe(mm)</th>
          <th>tmax(C)</th>
          <th>tmin(C)</th>
          <th>vp(Pa)</th>
          <th>PET</th>
          <th>QObs(mm/d)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>1980-10-01</th>
          <td>41050.80</td>
          <td>0.00</td>
          <td>286.90</td>
          <td>0.0</td>
          <td>16.19</td>
          <td>4.31</td>
          <td>825.78</td>
          <td>1.5713</td>
          <td>0.5550</td>
        </tr>
        <tr>
          <th>1980-10-02</th>
          <td>40780.81</td>
          <td>2.08</td>
          <td>195.94</td>
          <td>0.0</td>
          <td>13.46</td>
          <td>5.72</td>
          <td>920.18</td>
          <td>1.2619</td>
          <td>0.4979</td>
        </tr>
        <tr>
          <th>1980-10-03</th>
          <td>40435.21</td>
          <td>5.57</td>
          <td>172.60</td>
          <td>0.0</td>
          <td>17.84</td>
          <td>8.61</td>
          <td>1128.70</td>
          <td>1.2979</td>
          <td>0.5169</td>
        </tr>
        <tr>
          <th>1980-10-04</th>
          <td>40435.21</td>
          <td>23.68</td>
          <td>170.45</td>
          <td>0.0</td>
          <td>16.28</td>
          <td>7.32</td>
          <td>1027.91</td>
          <td>1.2251</td>
          <td>1.5634</td>
        </tr>
        <tr>
          <th>1980-10-05</th>
          <td>40089.58</td>
          <td>3.00</td>
          <td>113.83</td>
          <td>0.0</td>
          <td>10.51</td>
          <td>5.01</td>
          <td>881.61</td>
          <td>0.9116</td>
          <td>2.8541</td>
        </tr>
      </tbody>
    </table>
    </div>

Next we will split the data into a calibration period, which we will use
to find a set of good model parameters, and a validation period, we will
use the see how good our model works on unseen data. As in the CAMELS
data set publication, we will use the first 15 hydrological years for
calibration. The rest of the data will be used for validation.

Because the index of the dataframe is in pandas Datetime format, we can
easily split the dataframe into two parts

.. code:: python

    # calcute the end date of the calibration period
    end_cal = pd.to_datetime(f"{df.index[0].year + 15}/09/30", yearfirst=True)

    # validation period starts one day later
    start_val = end_cal + pd.DateOffset(days=1)

    # split the data into two parts
    cal = df[:end_cal].copy()
    val = df[start_val:].copy()

Fit the model to observed discharge
-----------------------------------

As already said above, we'll look at two different methods implemented
in this library: 1. Using one of SciPy's global optimizer 2.
Monte-Carlo-Simulation

Using one of SciPy's global optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each model has a ``.fit()`` method. This function uses the global
optimizer `differential
evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__
from the scipy package to find the set of model parameters that produce
the best simulation, regarding the provided observed discharge array.
The inputs for this function can be found in the
`documentation <http://rrmpg.readthedocs.io>`__ or the ``help()``.

.. code:: python

    help(model.fit)


.. parsed-literal::

    Help on method fit in module rrmpg.models.cemaneigegr4j:

    fit(obs, prec, mean_temp, min_temp, max_temp, etp, met_station_height, snow_pack_init=0, thermal_state_init=0, s_init=0, r_init=0, altitudes=[]) method of rrmpg.models.cemaneigegr4j.CemaneigeGR4J instance
        Fit the Cemaneige + GR4J coupled model to a observed timeseries

        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed
        timeseries is simulated as good as possible.

        Args:
            obs: Array of the observed timeseries [mm]
            prec: Array of daily precipitation sum [mm]
            mean_temp: Array of the mean temperature [C]
            min_temp: Array of the minimum temperature [C]
            max_temp: Array of the maximum temperature [C]
            etp: Array of mean potential evapotranspiration [mm]
            met_station_height: Height of the meteorological station [m].
                Needed to calculate the fraction of solid precipitation and
                optionally for the extrapolation of the meteorological inputs.
            snow_pack_init: (optional) Initial value of the snow pack storage
            thermal_state_init: (optional) Initial value of the thermal state
                of the snow pack
            s_init: (optional) Initial value of the production storage as
                fraction of x1.
            r_init: (optional) Initial value of the routing storage as fraction
                of x3.
            altitudes: (optional) List of median altitudes of each elevation
                layer [m]

        Returns:
            res: A scipy OptimizeResult class object.

        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If there is a size mismatch between the
                precipitation and the pot. evapotranspiration input.


We don't know any values for the initial states of the storages, so we
will ignore them for now. For the missing mean temperature, we calculate
a proxy from the minimum and maximum daily temperature. The station
height can be retrieved from the ``CAMELSLoader`` class via the
``.get_station_height()`` method.

.. code:: python

    # calculate mean temp for calibration and validation period
    cal['tmean'] = (cal['tmin(C)'] + cal['tmax(C)']) / 2
    val['tmean'] = (val['tmin(C)'] + val['tmax(C)']) / 2

    # load the gauge station height
    height = CAMELSLoader().get_station_height('01031500')

Now we are ready to fit the model and retrieve a good set of model
parameters from the optimizer. Again, this will be done with the
calibration data. Because the model methods also except pandas Series,
we can call the function as follows.

.. code:: python

    # We don't have an initial value for the snow storage,  so we omit this input
    result = model.fit(cal['QObs(mm/d)'], cal['prcp(mm/day)'], cal['tmean'],
                       cal['tmin(C)'], cal['tmax(C)'], cal['PET'], height)

``result`` is an object defined by the scipy library and contains the
optimized model parameters, as well as some more information on the
optimization process. Let us have a look at this object:

.. code:: python

    result

.. parsed-literal::

         fun: 1.6435277126036711
         jac: array([ 0.00000000e+00,  3.68594044e-06, -1.36113343e-05, -6.66133815e-06,
           -4.21884749e-07,  7.35368810e-01])
     message: 'Optimization terminated successfully.'
        nfev: 2452
         nit: 25
     success: True
           x: array([7.60699105e-02, 4.22084687e+00, 1.45653881e+02, 1.14318020e+00,
           5.87237837e+01, 1.10000000e+00])


The relevant information here is: - ``fun`` is the final value of our
optimization criterion (the mean-squared-error in this case) -
``message`` describes the cause of the optimization termination -
``nfev`` is the number of model simulations - ``sucess`` is a flag
wether or not the optimization was successful - ``x`` are the optimized
model parameters

Next, let us set the model parameters to the optimized ones found by the
search. Therefore we need to create a dictonary containing one key for
each model parameter and as the corresponding value the optimized
parameter. As mentioned before, the list of model parameter names can be
retrieved by the ``model.get_parameter_names()`` function. We can then
create the needed dictonary by the following lines of code:

.. code:: python

    params = {}

    param_names = model.get_parameter_names()

    for i, param in enumerate(param_names):
        params[param] = result.x[i]

    # This line set the model parameters to the ones specified in the dict
    model.set_params(params)

    # To be sure, let's look at the current model parameters
    model.get_params()


.. parsed-literal::

    {'CTG': 0.07606991045128364,
     'Kf': 4.220846873695767,
     'x1': 145.6538807127758,
     'x2': 1.143180196835088,
     'x3': 58.723783711432226,
     'x4': 1.1}


Also it might not be clear at the first look, this are the same
parameters as the ones specified in ``result.x``. In ``result.x`` they
are ordered according to the ordering of the ``_param_list`` specified
in each model class, where ass the dictonary output here is
alphabetically sorted.


Monte-Carlo-Simulation
^^^^^^^^^^^^^^^^^^^^^^

Now let us have a look how we can use the Monte-Carlo-Simulation
implemented in ``rrmpg.tools.monte_carlo``.

.. code:: python

    help(monte_carlo)



.. code::

    Help on function monte_carlo in module rrmpg.tools.monte_carlo:

    monte_carlo(model, num, qobs=None, **kwargs)
        Perform Monte-Carlo-Simulation.

        This function performs a Monte-Carlo-Simulation for any given hydrological
        model of this repository.

        Args:
            model: Any instance of a hydrological model of this repository.
            num: Number of simulations.
            qobs: (optional) Array of observed streamflow.
            **kwargs: Keyword arguments, matching the inputs the model needs to
                perform a simulation (e.g. qobs, precipitation, temperature etc.).
                See help(model.simulate) for model input requirements.

        Returns:
            A dictonary containing the following two keys ['params', 'qsim']. The
            key 'params' contains a numpy array with the model parameter of each
            simulation. 'qsim' is a 2D numpy array with the simulated streamflow
            for each simulation. If an array of observed streamflow is provided,
            one additional key is returned in the dictonary, being 'mse'. This key
            contains an array of the mean-squared-error for each simulation.

        Raises:
            ValueError: If any input contains invalid values.
            TypeError: If any of the inputs has a wrong datatype.



As specified in the help text, all model inputs needed for a simulation
must be provided as keyword arguments. The keywords need to match the
names specified in the ``model.simulate()`` function. Let us create a
new model instance and see how this works for the CemaneigeGR4J model.

.. code:: python

    model2 = CemaneigeGR4J()

    # Let use run MC for 1000 runs, which is in the same range as the above optimizer
    result_mc = monte_carlo(model2, num=10000, qobs=cal['QObs(mm/d)'],
                            prec=cal['prcp(mm/day)'], mean_temp=cal['tmean'],
                            min_temp=cal['tmin(C)'], max_temp=cal['tmax(C)'],
                            etp=cal['PET'], met_station_height=height)

    # Get the index of the best fit (smallest mean squared error)
    idx = np.argmin(result_mc['mse'][~np.isnan(result_mc['mse'])])

    # Get the optimal parameters and set them as model parameters
    optim_params = result_mc['params'][idx]

    params = {}

    for i, param in enumerate(param_names):
        params[param] = optim_params[i]

    # This line set the model parameters to the ones specified in the dict
    model2.set_params(params)


Calculate simulated discharge
-----------------------------

We now have two models, optimized by different methods. Let's calculate
the simulated streamflow of each model and compare the results! Each
model has a ``.simulate()`` method, that returns the simulated discharge
for the inputs we provide to this function.

.. code:: python

    # simulated discharge of the model optimized by the .fit() function
    val['qsim_fit'] = model.simulate(val['prcp(mm/day)'], val['tmean'],
                                     val['tmin(C)'], val['tmax(C)'],
                                     val['PET'], height)

    # simulated discharge of the model optimized by monte-carlo-sim
    val['qsim_mc'] = model2.simulate(val['prcp(mm/day)'], val['tmean'],
                                     val['tmin(C)'], val['tmax(C)'],
                                     val['PET'], height)

    # Calculate and print the Nash-Sutcliff-Efficiency for both simulations
    nse_fit = calc_nse(val['QObs(mm/d)'], val['qsim_fit'])
    nse_mc = calc_nse(val['QObs(mm/d)'], val['qsim_mc'])

    print("NSE of the .fit() optimization: {:.4f}".format(nse_fit))
    print("NSE of the Monte-Carlo-Simulation: {:.4f}".format(nse_mc))


.. parsed-literal::

    NSE of the .fit() optimization: 0.8075
    NSE of the Monte-Carlo-Simulation: 0.7332


What do this number mean? Let us have a look at some window of the
simulated timeseries and compare them to the observed discharge:

.. code:: python

    # Plot last full hydrological year of the simulation
    %matplotlib notebook
    start_date = pd.to_datetime("2013/10/01", yearfirst=True)
    end_date = pd.to_datetime("2014/09/30", yearfirst=True)
    plt.plot(val.loc[start_date:end_date, 'QObs(mm/d)'], label='Qobs')
    plt.plot(val.loc[start_date:end_date, 'qsim_fit'], label='Qsim .fit()')
    plt.plot(val.loc[start_date:end_date, 'qsim_mc'], label='Qsim mc')
    plt.legend()



.. parsed-literal::

    <IPython.core.display.Javascript object>



.. raw:: html

    <img src="data:image/png;base64,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adPJ/TUIgheIDsELAEjIYx2nhME7fsn6qMcB0k4qB2+tWrV0wQUXaNeuXQUeX7dunX7yk5/o+eefDx/r2LGjKleurLJly+q8887TE088Ea47+pRmx3HUpEkTPfPMMzrttNNUsWJFDRs2TBs2bNDDDz+s0047Ta7ratasWSXOZ2bq0qWLfvOb36hcuXK6/PLLNW3aNC1btkw1atTQT37yE910001avnx5gf2GDx+u6667TmXLltXPfvYzPfbYY9/p90PwAtEheAEACXm4w+HgHZv5TdTjAGmnqCjKy8vTzn07S33Jy8tLeO5NmzapTJkyatasWZHr69Spo7POOkt5eXmaNWuWTjnlFPXv31/Z2dmaPXu22rVrF25bVPCeffbZ6tKli7KysvT3v/9dZ5xxhmrVqqVBgwZp6dKlevTRR3XFFVeUOLOZqUKFCho4cGC4T6VKlXTXXXdp9OjRyszM1E033aRatWqF+wRBoFNOOUUNGjRQZmam5s6dq6ZNmyb8ezkSwQtEh+AFACTkofaTw+D9dOG6qMcB0k5RUbRz3065vlvqy859OxOee8aMGTIzDR06tMj1rVu3lplp/fr1Gjx4sMqXL1/s93SLCt6nn346/HndunUyM73xxhvhY9OnT5eZad264v+5ZGZ6/fXXC+3Ts2fP8LEBAwboxz/+cfjzzTffrD/96U8lvPLEEbxAdAheAEBCHmg3KQzeUQvWRj0OkHbSNXhbtWolM9OWLVu0bds2VatWTeecc46efvpp9evXTzt3Hn6uooK3RYsW4c95eXkyMw0aNCh8bOXKlTIzzZs3r9gZi9tn5syZ4WPjx4/Xkf9dXK5cOfXq1Svh30NJCF4gOgQvACAh97WZGAbviHlfRz0OkHZS9ZTm3NxclSlTptjTfevUqaNzzz03/Hn//v0aM2aM6tatq4svvliVK1fWli1bJBUdvG3atClwPDsqrletWiUz05w5c4qdMZF9JkyYEIa5JJ199tkEL5AGCF4AQELubX04eIfNWRP1OEDaSeWLVt17772qUKFCsRetqlu3bpH77dixQz/84Q81ePBgSSdW8N5xxx2c0gykAYIXAJCQmq0ywuAdMvurqMcB0k4qB29WVpbOOecc3XbbbZo4caJycnI0atQoua6r6tWra/v27ZKkESNGqF27dpozZ46ys7PVqVMn/eAHP9DChQsl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" width="956">




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7f1b03d61198>



The result is not perfect, but it is not bad either! And since this
package is also about speed, let us also check how long it takes to
simulate the discharge for the entire validation period (19 years of
data).

.. code:: python

    %%timeit
    model.simulate(val['prcp(mm/day)'], val['tmean'],
                                     val['tmin(C)'], val['tmax(C)'],
                                     val['PET'], height)


.. parsed-literal::

    2.46 ms ± 42.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
