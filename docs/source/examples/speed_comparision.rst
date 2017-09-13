
Numba Speed-Test
================

In this notebook I'll test the speed of a simple hydrological model (the
ABC-Model [1]) implemented in pure Python, Numba and Fortran. This
should only been seen as an example of the power of numba in speeding up
array-oriented python functions, that have to be processed using loops.
This is for example the case for hydrological models that have to be
processed timestep after timestep to update model states (depending on
previous states) and calculate flows. Python is natively very slow for
this kind of functions (loops). Normally hydrological (as well as
meterological and environmental) models are implemented in Fortran or
C/C++ which are known for their speed. The downside is, that this
languages are quite harder to start with and the code often seems overly
complicated for beginner. Numba is a library that performs just-in-time
compilation on Python code and can therefore dramatically increase the
speed of Python functions (without having to change much in the code).

Anyway, this is not meant to give an introduction to numba, but just to
compare the execution speed against pure Python and Fortan. For
everybody, who is interested in further explanations on Numba see: -
`Gil Forsyth's & Lorena Barba's tutorial from the SciPy
2017 <https://www.youtube.com/watch?v=1AwG0T4gaO0>`__ - `The numba
homepage, which includes examples <https://numba.pydata.org/>`__

If you want to reproduce the results and you have installed a conda
environment using the environment.yml from the `rrmpg github
repository <https://github.com/kratzert/RRMPG/blob/master/environment.yml>`__
make sure to additionally install ``fortran-magic`` using pip:

::

    pip install -U fortran-magic

[1] Myron B. Fiering "Streamflow synthesis" Cambridge, Harvard
University Press, 1967. 139 P. (1967).

.. code:: python

    # Notebook setups
    import numpy as np

    from numba import njit, float64
    from timeit import timeit

    %load_ext fortranmagic

We'll use an array of random numbers as input for the model. Since we
only want to test the execution time, this will work for now.

.. code:: python

    # Let's an array of 10 mio values
    rain = np.random.random(size=10000000)

Next we are going to define three different functions:

1. ``abc_model_py``: An implementation of the ABC-Model using pure
   Python.
2. ``abc_model_numba``: A numba version of the ABC-model. The
   just-in-time compilation is achieved by adding a numba decorator over
   the function definition. I use the ``@njit`` to make sure an error is
   raised if numba can't compile the function.
3. ``abc_model_fortan``: A fortan version of the ABC-model.

Note how for this simple model the only difference between the pure
Python version and the Numba version is the decorator. The entire code
of the model is the same.

.. code:: python

    # pure Python implementation
    def abc_model_py(a, b, c, rain):
        outflow = np.zeros((rain.size), dtype=np.float64)
        state_in = 0
        state_out = 0
        for i in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[i]
            outflow[i] = (1 - a - b) * rain[i] + c * state_in
            state_in = state_out
        return outflow

    # numba version of the ABC-model
    @njit(['float64[:](float64,float64,float64,float64[:])'])
    def abc_model_numba(a, b, c, rain):
        outflow = np.zeros((rain.size), dtype=np.float64)
        state_in = 0
        state_out = 0
        for i in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[i]
            outflow[i] = (1 - a - b) * rain[i] + c * state_in
            state_in = state_out
        return outflow

.. code:: bash

    %%fortran

        subroutine abc_model_fortran(col_dim, a, b, c, inflow, outflow)

            integer, intent(in) :: col_dim
            real(kind = 8 ), intent(in) :: a, b, c
            real(kind = 8 ), intent(in), dimension(col_dim) :: inflow

            real(kind = 8) :: state_in, state_out
            integer :: t ! loop variable
            real(kind = 8 ) :: init_state
            real(kind = 8 ), dimension(col_dim) :: state

            real(kind = 8 ), intent(out), dimension(col_dim) :: outflow

            state_in = 0
            state_out = 0
            do t = 1,col_dim
                state_out = (1 - c) * state_in + a * inflow(t)
                outflow(t) = (1 - a - b) * inflow(t) + c * state_in
                state_in = state_out
            end do

        end subroutine

Now we'll use the ``timeit`` package to measure the execution time of
each of the functions

.. code:: python

    # Measure the execution time of the Python implementation
    py_time = %timeit -r 5 -n 10 -o abc_model_py(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    6.75 s ± 11.6 ms per loop (mean ± std. dev. of 5 runs, 10 loops each)


.. code:: python

    # Measure the execution time of the Numba implementation
    numba_time = %timeit -r 5 -n 10 -o abc_model_numba(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    30.6 ms ± 498 µs per loop (mean ± std. dev. of 5 runs, 10 loops each)


.. code:: python

    # Measure the execution time of the Fortran implementation
    fortran_time = %timeit -r 5 -n 10 -o abc_model_fortran(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    31.9 ms ± 757 µs per loop (mean ± std. dev. of 5 runs, 10 loops each)


As you can see by the raw numbers, Fortran (as expected) is the fastest,
but what is interesting, that the Numba version of the ABC-Model does
not perform much worse. Let's compare the numbers.

First we'll compare the pure Python version, against the Numba version.
Remember, everthing we did was to add a decorator to the Python
function, the rest (the magic) is done by the Numba library.

.. code:: python

    py_time.best / numba_time.best


.. parsed-literal::

    222.1521754580626



Wow, this is an over 220 x speed up by one single additional line of
code. Note that for more complicated models, we'll have to adapt the
code a bit more, but in general it will stay very close to normal Python
code.

Now let's see how the Numba version performs against Fortran, which is
still the standard in the modelling community of hydrology and
meteorology.

.. code:: python

    numba_time.best / fortran_time.best


.. parsed-literal::

    0.9627960721576471



Actually, this even surprised me. With one decorator the Python function
became faster than the Fortran file, although the difference is minimal,
but who would have guessed that we can bring Python to this speed
dimensions.

Note that this Fortran function is compiled using the GNU Fortran
compiler, which is open source and free. Using e.g. the Intel Fortran
compiler will certainly increase speed of the Fortran function, but I
think it's only fair to compare two open source and free-of-charge
versions.

**So what does this mean**

We'll see, but you'll now maybe better understand the idea of this
project. We can implement models in Python, that have the performance of
Fortran, but are easier to get started with and play around. We can run
1000s of simulations and don't have to wait for ages and we can stay the
entire time in one environment (for simulating and evaluating the
results). The hope is, that this will help fellow students/researchers
to better understand hydrological models and lose fear of what might
seem intimidating at first, follwing a quote by Richard Feynman:

**"What I can not create, I do not understand" - Richard Feynman**
