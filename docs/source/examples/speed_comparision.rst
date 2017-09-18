
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
make sure to additionally install ``cython``:

::

    conda install -c anaconda cython

[1] Myron B. Fiering "Streamflow synthesis" Cambridge, Harvard
University Press, 1967. 139 P. (1967).

.. code:: python

    # Notebook setups
    import numpy as np

    from numba import njit, float64
    from timeit import timeit

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
3. ``abc_model_fortan``: A fortan version of the ABC-model. In previous
   version this was done using the f2py module which added some overhead
   to the function call and was no fair benchmark (see [pull request #3](https://github.com/kratzert/RRMPG/pull/3)).
   Now the Fortran implementation is wrapped in a Cython function.

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

.. code:: python

    %%file abc.f90

    module abc
    use iso_c_binding, only: c_int, c_double
    implicit none
    integer, parameter :: dp = kind(0d0)
    private
    public c_abc_model_fortran

    contains


    subroutine c_abc_model_fortran(n, a, b, c, inflow, outflow) bind(c)
    integer(c_int), intent(in), value :: n
    real(c_double), intent(in), value :: a, b, c
    real(c_double), intent(in) :: inflow(n)
    real(c_double), intent(out) :: outflow(n)
    call abc_model(a, b, c, inflow, outflow)
    end subroutine


    subroutine abc_model(a, b, c, inflow, outflow)
    real(dp), intent(in) :: a, b, c, inflow(:)
    real(dp), intent(out) :: outflow(:)
    real(dp) :: state_in, state_out
    integer :: t
    state_in = 0
    do t = 1, size(inflow)
        state_out = (1 - c) * state_in + a * inflow(t)
        outflow(t) = (1 - a - b) * inflow(t) + c * state_in
        state_in = state_out
    end do
    end subroutine


    end module


.. parsed-literal::

    Overwriting abc.f90


.. code:: python

    %%file abc_py.pyx

    from numpy cimport ndarray
    from numpy import empty, size

    cdef extern:
        void c_abc_model_fortran(int n, double a, double b, double c, double *inflow, double *outflow)

    def abc_model_fortran(double a, double b, double c, ndarray[double, mode="c"] inflow):
        cdef int N = size(inflow)
        cdef ndarray[double, mode="c"] outflow = empty(N, dtype="double")
        c_abc_model_fortran(N, a, b, c, &inflow[0], &outflow[0])
        return outflow


.. parsed-literal::

    Overwriting abc_py.pyx


Compile the Fortran and Cython module

.. code:: bash

    %%bash
    set -e
    #set -x
    # Debug flags
    #FFLAGS="-Wall -Wextra -Wimplicit-interface -fPIC -fmax-errors=1 -g -fcheck=all -fbacktrace"
    #CFLAGS="-Wall -Wextra -fPIC -fmax-errors=1 -g"
    # Release flags
    FFLAGS="-fPIC -O3 -march=native -ffast-math -funroll-loops"
    CFLAGS="-fPIC -O3 -march=native -ffast-math -funroll-loops"
    gfortran -o abc.o -c abc.f90 $FFLAGS
    cython abc_py.pyx
    gcc -o abc_py.o -c abc_py.c -I$CONDA_PREFIX/include/python3.6m/ $CFLAGS
    gcc -o abc_py.so abc_py.o abc.o -L$CONDA_PREFIX/lib -lpython3.6m -lgfortran -shared

.. code:: python

    # Now we can import it like a normal Python module
    from abc_py import abc_model_fortran

Now we'll use the ``timeit`` package to measure the execution time of
each of the functions

.. code:: python

    # Measure the execution time of the Python implementation
    py_time = %timeit -o abc_model_py(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    6.94 s ± 258 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)


.. code:: python

    # Measure the execution time of the Numba implementation
    numba_time = %timeit -o abc_model_numba(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    32.6 ms ± 52.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


.. code:: python

    # Measure the execution time of the Fortran 2 implementation
    fortran_time = %timeit -o abc_model_fortran(0.2, 0.6, 0.1, rain)


.. parsed-literal::

    23.4 ms ± 934 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)


As you can see by the raw numbers, Fortran (as expected) is the fastest,
but what is interesting, that the Numba version of the ABC-Model does
not perform much worse. Let's compare the numbers.

First we'll compare the pure Python version, against the Numba version.
Remember, everthing we did was to add a decorator to the Python
function, the rest (the magic) is done by the Numba library.

.. code:: python

    py_time.best / numba_time.best




.. parsed-literal::

    205.15122150338178



Wow, this is roughly a 205 x speed up by one single additional line of
code. Note that for more complicated models, we'll have to adapt the
code a bit more, but in general it will stay very close to normal Python
code.

Now let's see how the Numba version performs against Fortran, which is
still the standard in the modelling community of hydrology and
meteorology.

.. code:: python

    numba_time.best / fortran_time.best




.. parsed-literal::

    1.451113966128858



So the Fortran implementation is still faster but not much. We only need
less than 1,5x the time of the Fortran version if we run the Python code
optimized with the Numba library.

Note that this Fortran function is compiled using the GNU Fortran
compiler, which is open source and free. Using e.g. the Intel Fortran
compiler will certainly increase speed of the Fortran function, but I
think it's only fair to compare two open source and free-of-charge
versions.

**So what does this mean**

We'll see, but you will now have maybe a better idea of this project.
The thing is, we can implement models in Python, that have roughly the
performance of Fortran, but are at the same time less complex to
implement and play around with. We can also save a lot of boilerplate
code we need with Fortran to compiler our code in the most optimal way.
We only need to follow some rules of the Numba library and for the rest,
add one decorator to the function definition. We can run 1000s of
simulations and don't have to wait for ages and we can stay the entire
time in one environment (for simulating and evaluating the results). The
hope is, that this will help fellow students/researchers to better
understand hydrological models and lose fear of what might seem
intimidating at first, follwing a quote by Richard Feynman:

**"What I can not create, I do not understand" - Richard Feynman**
