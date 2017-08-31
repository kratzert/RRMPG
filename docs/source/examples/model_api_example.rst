
Model API Example
=================

In this notebook, we'll explore some functionality of the models of this
package. We'll work with the HBV-Educational model that is implemented
in ``rrmpg.models.hbvedu``. The data we'll use, is the data that comes
with the official code release by the papers authors.

The data can be found here: http://amir.eng.uci.edu/software.php (under
"HBV Hydrological Model")

In summary we'll look at: - How you can create one of the models - How
you can fit the model parameters to observed discharge by: - Using one
of SciPy's global optimizer - Monte-Carlo-Simulation - How you can use a
fitted model to calculate the simulated discharge

.. code:: python

    # Imports and Notebook setup
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from timeit import timeit

    from rrmpg.models import HBVEdu
    from rrmpg.tools.monte_carlo import monte_carlo
    from rrmpg.utils.metrics import nse

    # Let plots appear in the notebook
    %matplotlib notebook

First we need to load the input data into memory. There are several
files we need:

-  ``inputPrecipTemp.txt`` contains all daily input data we need
   (temperature, precipitation and month, which is a number specifying
   for each day/timestep to which month it belongs from 1 to 12)
-  ``inputMonthlyTempEvap.txt`` contains the long-term mean monthly
   temperature and potential evapotranspiration
-  ``Qobs.txt`` contains the observed discharge

Also we need some of the values specified in ``IV.txt``, like the area
of the watershed and the initial storage values. These I'll specify
directly below. Note that we don't fix the model parameter
``T_t the temperature threshold``, like the authors did, but instead
optimize this parameter as well.

.. code:: python

    daily_data = pd.read_csv('/path/to/inputPrecipTemp.txt',
                             names=['date', 'month', 'temp', 'prec'], sep='\t')
    daily_data.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }

        .dataframe thead th {
            text-align: left;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>date</th>
          <th>month</th>
          <th>temp</th>
          <th>prec</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1/1/1991</td>
          <td>1</td>
          <td>-1.5</td>
          <td>0.4</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1/2/1991</td>
          <td>1</td>
          <td>-0.8</td>
          <td>10.5</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1/3/1991</td>
          <td>1</td>
          <td>-2.8</td>
          <td>0.9</td>
        </tr>
        <tr>
          <th>3</th>
          <td>1/4/1991</td>
          <td>1</td>
          <td>-3.7</td>
          <td>4.4</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1/5/1991</td>
          <td>1</td>
          <td>-6.1</td>
          <td>0.6</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    monthly = pd.read_csv('/path/to/inputMonthlyTempEvap.txt',
                          sep=' ', names=['temp', 'not_needed', 'evap'])
    monthly.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }

        .dataframe thead th {
            text-align: left;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temp</th>
          <th>not_needed</th>
          <th>evap</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.4</td>
          <td>5</td>
          <td>0.161</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.3</td>
          <td>5</td>
          <td>0.179</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2.6</td>
          <td>20</td>
          <td>0.645</td>
        </tr>
        <tr>
          <th>3</th>
          <td>6.3</td>
          <td>50</td>
          <td>1.667</td>
        </tr>
        <tr>
          <th>4</th>
          <td>10.9</td>
          <td>95</td>
          <td>3.065</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    qobs = pd.read_csv('/path/to/Qobs.txt',
                       names=['values'])
    qobs.head()




.. raw:: html

    <div>
    <style>
        .dataframe thead tr:only-child th {
            text-align: right;
        }

        .dataframe thead th {
            text-align: left;
        }

        .dataframe tbody tr th {
            vertical-align: top;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>values</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>4.5</td>
        </tr>
        <tr>
          <th>1</th>
          <td>11.0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>6.6</td>
        </tr>
        <tr>
          <th>3</th>
          <td>5.0</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4.1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # Values taken from the IV.txt
    area = 410
    soil_init = 100
    s1_init = 3
    s2_init = 10

Create a model
--------------

Now let's have a look how we can create one of the models implemented in
``rrmpg.models``. Basically for all models we have two different
options: 1. Initialize a model with all mandatory inputs but **without**
specific model parameters. 2. Initialize a model with all mandatory
inputs **with** specific model parameters.

In the `documentation <http://rrmpg.readthedocs.io>`__ you can find a
list of all model parameters or you can look at help(HBVEdu) for this
model. In the case you don't provide specific model parameters, random
parameters will be generated that are in between the default parameter
bounds. You can have a look at these bounds by calling
get\_param\_bounds() on a model.

For now we don't know any specific parameter values, so we'll create on
with random parameters. We only need to specify the watershed area for
this model.

.. code:: python

    model = HBVEdu(area=area)

Fit the model to observed discharge
-----------------------------------

As already said above, we'll look at two different methods implemented
in this package: 1. Using one of SciPy's global optimizer 2.
Monte-Carlo-Simulation

Using one of SciPy's global optimizer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each model has a ``.fit()`` method. This function uses the globel
optimizer `differential
evolution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html>`__
from the scipy package to find the set of model parameters that produce
the best simulation, regarding the provided observed discharge array.
The inputs for this function can be found in the
`documentation <http://rrmpg.readthedocs.io>`__ or the help().

.. code:: python

    help(model.fit)


.. code-block:: none

    Help on method fit in module rrmpg.models.hbvedu:

    fit(qobs, temp, prec, month, PE_m, T_m, snow_init=0.0, soil_init=0.0, s1_init=0.0, s2_init=0.0) method of rrmpg.models.hbvedu.HBVEdu instance
        Fit the model to a timeseries of discharge using.

        This functions uses scipy's global optimizer (differential evolution)
        to find a good set of parameters for the model, so that the observed
        discharge is simulated as good as possible.

        Args:
            qobs: Array of observed streaflow discharge.
            temp: Array of (mean) temperature for each timestep.
            prec: Array of (summed) precipitation for each timestep.
            month: Array of integers indicating for each timestep to which
                month it belongs [1,2, ..., 12]. Used for adjusted
                potential evapotranspiration.
            PE_m: long-term mean monthly potential evapotranspiration.
            T_m: long-term mean monthly temperature.
            snow_init: (optional) Initial state of the snow reservoir.
            soil_init: (optional) Initial state of the soil reservoir.
            s1_init: (optional) Initial state of the near surface flow
                reservoir.
            s2_init: (optional) Initial state of the base flow reservoir.

        Returns:
            res: A scipy OptimizeResult class object.

        Raises:
            ValueError: If one of the inputs contains invalid values.
            TypeError: If one of the inputs has an incorrect datatype.
            RuntimeErrror: If the monthly arrays are not of size 12 or there
                is a size mismatch between precipitation, temperature and the
                month array.



.. code:: python

    # We don't have an initial value for the snow storage,  so we omit this input
    result = model.fit(qobs.values, daily_data.temp, daily_data.prec,
                       daily_data.month, monthly.evap, monthly.temp,
                       soil_init=soil_init, s1_init=s1_init,
                       s2_init=s2_init)

``result`` is an object defined by the scipy library and contains the
optimized model parameter, as well as some more information on the
optimization prozess. Let us have a look on this object

.. code:: python

    result




.. parsed-literal::

         fun: 9.3905281887057583
     message: 'Optimization terminated successfully.'
        nfev: 10782
         nit: 49
     success: True
           x: array([ -2.91830209e-01,   3.74195970e+00,   1.98920300e+02,
             1.52260747e+00,   1.24451078e-02,   9.03672595e+01,
             5.08673767e-02,   8.15783540e-02,   1.00310869e-02,
             4.88820992e-02,   4.94212319e+00])



Some of the relevant informations are: - ``fun`` is the final value of
our optimization criterion, the mean-squared-error in this case -
``message`` describes the cause of the optimization termination -
``nfev`` is the number of model simulations - ``sucess`` is a flag
wether or not the optimization was successful - ``x`` are the optimized
model parameters

Now let's set the model parameter to the optimized parameter found by
the optimizer. Therefore we need to create a dictonary containing on key
for each model parameter and as the corresponding value the optimized
parameter. The list of model parameter names can be retrieved by the
``model.get_parameter_names()`` function. We can then create the needed
dictonary by the following lines of code.

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

    {'Beta': 1.5226074683067585,
     'C': 0.012445107819582725,
     'DD': 3.7419596954402823,
     'FC': 198.92030029792784,
     'K_0': 0.050867376726295155,
     'K_1': 0.081578354027559724,
     'K_2': 0.010031086907160772,
     'K_p': 0.048882099196888087,
     'L': 4.9421231856757295,
     'PWP': 90.367259489128287,
     'T_t': -0.29183020885043631}



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


.. code-block:: none

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
            one additional key is returned in the dictonary, being 'nse'. This key
            contains an array of the Nash-Sutcliff-Efficiency for each simulation.

        Raises:
            ValueError: If any input contains invalid values.
            TypeError: If any of the inputs has a wrong datatype.



As specified in the help text, all model inputs needed for a simulation
must be provided as keyword arguments. The keywords need to match the
names specified in the ``model.simulate()`` function. We'll create a new
model instance and see how this works for the HBVEdu model.

.. code:: python

    model2 = HBVEdu(area=area)

    # Let use run MC for 1000 runs, which is in the same range as the above optimizer
    result_mc = monte_carlo(model2, num=10000, qobs=qobs.values,
                            temp=daily_data.temp, prec=daily_data.prec,
                            month=daily_data.month, PE_m=monthly.evap,
                            T_m=monthly.temp, soil_init=soil_init,
                            s1_init=s1_init, s2_init=s2_init)

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

Now that we have two models, optimized by different methods, let's
calculate the simulated streamflow of each model and compare the
results. Each model has a ``.simulate()`` method, that returns the
simulated discharge for the inputs we provide to this function.

.. code:: python

    # simulated discharge of the model optimized by the .fit() function
    qsim_fit = model.simulate(daily_data.temp, daily_data.prec,
                              daily_data.month, monthly.evap, monthly.temp,
                              soil_init=soil_init, s1_init=s1_init,
                              s2_init=s2_init)

    # simulated discharge of the model optimized by monte-carlo-sim
    qsim_mc = model2.simulate(daily_data.temp, daily_data.prec,
                              daily_data.month, monthly.evap, monthly.temp,
                              soil_init=soil_init, s1_init=s1_init,
                              s2_init=s2_init)

    # Calculate and print the Nash-Sutcliff-Efficiency for both simulations
    nse_fit = nse(qobs.values, qsim_fit)
    nse_mc = nse(qobs.values, qsim_mc)

    print("NSE of the .fit() optimization: {:.4f}".format(nse_fit))
    print("NSE of the Monte-Carlo-Simulation: {:.4f}".format(nse_mc))


.. parsed-literal::

    NSE of the .fit() optimization: 0.6173
    NSE of the Monte-Carlo-Simulation: 0.5232


And let us finally have a look at some window of the simulated
timeseries compared to the observed discharge

.. code:: python

    # Plot last year of the simulation
    plt.plot(qobs.values[-366:], color='cornflowerblue', label='Qobs')
    plt.plot(qsim_fit[-365:], color='coral', label='Qsim .fit()')
    plt.plot(qsim_mc[-365:], color='mediumseagreen', label='Qsim mc')
    plt.legend()



.. parsed-literal::

    <IPython.core.display.Javascript object>



.. raw:: html

    <img src="data:image/png;base64,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heFXyWv0ytgfhl306DnD93sMyKi7iE3XYvBhzqZHDh54pfnik/hcnClV9Vhu1JiwNakD9x6o90YWq4amNivut1kHnDFOwzPbLChvq0VidT42XI3G7NzNeM6lGFr/5p32Db46vg6VLZ1obh//4mhMkok8b2Ikycl7By9isWUOEPAFEOEHxG0H0oJVdcr8JODyKeBmAQyJO+zLmHxhGmFiTUREROND0zR0mnvHZeazu8+GoMxOt4Jr84LqMC/0Li4MUpzLkzrNvbjYXIGI26cQcDMVcXfzcLAiE99ePIi4u9w+NtZ6LEacranG9oJz+OF6EnwvBOHtLD+3vfbWUdgSMBqYJBN53sRIkgtSVZXJoK/UsQj+A/eFDNVKts2EV5ovZqSvGt+fgYiIiCaFlg4r6g0WvU2kyuQ0sbSZunDlYSVO1V8Z70vRMUkm8rwJkSQnVufj+8sR2HkjAYcqTiK5+hzO3C1ASeV51NzMQ+eVk9DOpwA50cDxUCB+J3B4HbB/JU4c/Bxeab74Mm/nuP4MRERERESexiSZyPMmRJK85krUsHtzXkhfhrez/DDnzBZ8en4vVhWHYeO1o1iUvxteab7YWnJsXH8GIiIiIiJPY5JMNHKfisg9ETGKSJGI/NMIv25CJMmXW27jWNUZhJQdx+ZrMVhVfACL8ndj5un1eCVj6EqUri2m6sy4/gxERERERJ7GJJloZGaKiElE5onI34pIqIi0icifj+BrJ0SSPByj1Yym3laUt9WiuLkcpxuuIuneeURWqmIXwWXp6DL3jvdlEhERERF5FJNkopEpEpF9Lh//WEQaRGTVCL52UiTJRERERETEJJloJH4iIlYR8e73+QgRSRnk/j8V9QflaL8QJslERERERJMCk2Si4f2lqD+Sf+n3+a2iZpj787ff360xSSYiIiIimviYJBMN73GTZM4kExERERFNUkySiYb3uMut++OeZCIiIiKiSYJJMtHIFInIXpePfywi9cLCXUREREREUwqTZKKRmSnqfOSPRORvRCRE1BFQfzGCr2WSTEREREQ0STBJJhq5z0SkRtR5yUUi8s8j/DomyUREREREkwSTZCLPY5JMRERERDRJMEkm8jwmyUREREREkwSTZCLPY5JMRERERDRJMEkm8jwmyUREREREkwSTZCLPY5JMRERERDRJMEkm8jwmyUREREREkwST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" width="969">




.. parsed-literal::

    <matplotlib.legend.Legend at 0x7fe0768e2358>



Albeit not perfect, I guess it's not bad either for an out-of-the-box
optimization framework that works for any model. And since this package
is also about speed, let us see how long it takes to optimize the
HBV-Educational model performing a Monte-Carlo-Simulation for 10,000
runs.

.. code:: python

    %%timeit
    monte_carlo(model2, num=10000, qobs=qobs.values, temp=daily_data.temp,
                prec=daily_data.prec, month=daily_data.month, PE_m=monthly.evap,
                T_m=monthly.temp, soil_init=soil_init, s1_init=s1_init,
                s2_init=s2_init)


.. parsed-literal::

    5.01 s ± 5.07 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
