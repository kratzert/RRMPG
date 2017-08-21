# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>
"""
This module implements some utility functions for plotting.

Implemented functions:
    plot_qsim_range: Plot the range of multiple simulations and their mean.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_qsim_range(qsim, x_vals=None, qobs=None):
    """Plot the range of multiple simulations and their mean.

    This function plots the quantiles of multiple simulations as a filled area
    and the mean as a line. The (0.05, 0.95) and the (0.25, 0.75) quantile
    are plotted as different colored areas and the mean as a solid line. If
    observations are also passed, they are plotted as well as a solid line.

    Args:
        qsim: 2D array of simulations. Shape must be (num_timesteps, num_sims)
        x_vals: (optional) 1D array, that will be used as x-axes values.
            (e.g. date)
        qobs: (optional) 1D arary of oversations.

    Returns:
        A handle to the matplotlib figure.

    Raises:
        ValueError: For incorrect inputs.

    """
    # Validate inputs
    if not isinstance(qsim, np.ndarray) or (len(qsim.shape) != 2):
        raise ValueError("'qsim' must be a two dimensional numpy.ndarray.")

    if x_vals is not None:
        if not isinstance(x_vals, (list, np.ndarray, pd.Series, pd.Index)):
            msg = ["'x_vals' must be either a list, numpy.ndarray or ",
                   "pandas.Series."]
            raise ValueError("".join(msg))

    if qobs is not None:
        if isinstance(qobs, (list, np.ndarray, pd.Series)):
            try:
                qobs = np.array(qobs, dtype=np.float64)
            except:
                raise ValueError("All elements in 'qobs' must be numerical.")
        else:
            msg = ["'qobs' must be either a list, numpy.ndarray or ",
                   "pandas.Series."]
            raise ValueError("".join(msg))
        if len(qobs.shape) != 1:
            raise ValueError("'qobs' must be one dimensional.")

    # Calculate quantiles
    q05 = np.percentile(qsim, 5, axis=1)
    q25 = np.percentile(qsim, 25, axis=1)
    q75 = np.percentile(qsim, 75, axis=1)
    q95 = np.percentile(qsim, 95, axis=1)

    # If an array for the x values exist.
    if x_vals is None:
        x_vals = np.arange(qsim.shape[0])

    # Create plot
    fig, ax = plt.subplots(1)

    ax.plot(x_vals, np.mean(qsim, axis=1), color='red', label="Qsim mean",
            lw=0.5)
    if qobs is not None:
        ax.plot(x_vals, qobs, color='blue', label="Qobs", lw=0.5)

    ax.fill_between(x_vals, q05, q95, color=(1, 0, 0, 0.3),
                    label="5%/95% quantile")
    ax.fill_between(x_vals, q25, q75, color=(1, 0, 0, 0.1),
                    label="25%/75% quantile")
    plt.legend()
    plt.show()

    return fig, ax
