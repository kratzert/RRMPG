# -*- coding: utf-8 -*-
# This file is part of RRMPG.
#
# RRMPG is free software with the aim to provide a playground for experiments
# with hydrological rainfall-runoff-models while achieving competitive
# performance results.
#
# You should have received a copy of the MIT License along with RRMPG. If not,
# see <https://opensource.org/licenses/MIT>

"""Parent class for all models implemented in rrmpg.models."""

import numbers

import numpy as np

from numpy.random import uniform


class BaseModel(object):
    """Basic model class for all rainfall-runoff model.

    This class builds the core skelleton for all rainfall-runoff models
    within this project. Every other model will inherit from this class and
    therefore only have to implement the model specific methods.
    """

    # List of strings containing all model parameters
    _param_list = []

    # Dict containing the default parameter bounds
    _default_bounds = {}

    # Custom numpy datatype needed for numba input
    _dtype = np.dtype([])

    def __init__(self, params=None):
        """Initialize a new hydrological model.

        This is the base class for all hydrological models. In common for all
        models is, then upon initialization you can either pass concrete model
        parameters or if you don't, random parameters within the default bounds
        will be generated.

        Args:
            params: (optional) Dictonary containing all model parameters as a
                seperate key/value pairs.

        Raises:
            AttributeError: If a model parameter is missing in the passed 
                dictonary.

        """
        # Check if params are passed
        if params:
            # Check if a value for each parameter is passed
            missings = [p for p in self._param_list if p not in params.keys()]
            if len(missings) > 0:
                raise AttributeError("Missing the following model parameters: "
                                     "{}".format(missings))
        else:
            # Generate random model parameters
            params = self.get_random_params()

        # Parse model parameters as class attributes
        self.set_params(params)

    def get_random_params(self):
        """Generate a random set of model parameters in the default bounds.

        Samples one value for each model parameter from a uniform distribution
        betwen the default bounds.

        Returns:
            A dict containing one key/value pair for each model parameter.
            Example:
            {'param1': 0.3, 'param2': 0.2, 'param3': 0.7}

        """
        params = {}
        # sample one value for each parameter
        for param in self._param_list:
            value = uniform(low=self._default_bounds[param][0],
                            high=self._default_bounds[param][1],
                            size=1)
            params[param] = value[0]

        return params

    def get_params(self):
        """Return a dict with all model parameters and their current value."""
        params = {}
        for param in self._param_list:
            params[param] = getattr(self, param)
        return params

    def set_params(self, params):
        """Set model parameters to values passed in params.

        Args:
            params: Dictonary containing model parameters as key/value pairs.
                Can be only one model parameter or many/all. The naming of the
                parameters must be identical to the names specified in the
                _param_list. All parameter values must be numerical.

        Raises:
            ValueError: If any parameter is not a numerical value.
            AttributeError: If the parameter dictonary contains a key, that 
                doesn't match any of the parameter names.

        """
        for param, value in params.items():
            # Check if parameter is defined in the model parameter list
            if param in self._param_list:
                # Check if value is numerical
                if isinstance(value, numbers.Number):
                    setattr(self, param, value)
                else:
                    msg = ["The value of parameter '{}'".format(param),
                           "must be numerical"]
                    raise ValueError("".join(msg))
            else:
                msg = ["Unknow parameter '{}'.".format(param),
                       "Name must match one of the model parameters."
                       "Use {}".format(self.__class__.__name__),
                       ".get_parameter_names() to get a list of valid names."]
                raise AttributeError("".join(msg))

    def get_parameter_names(self):
        """Return the list of parameter names."""
        return self._param_list

    def get_default_bounds(self):
        """Return the dictionary containing the default parameter bounds."""
        return self._default_bounds

    def get_dtype(self):
        """Return the custom model datatype."""
        return self._dtype
