Introduction
############

This repository is a work in progress repository.

Documentation can be found here: `http://rrmpg.readthedocs.io <http://rrmpg.readthedocs.io>`_

Idea
----
One of the fundamental parts of hydrology is rainfall-runoff-modelling. The task here is to model the response of a catchment to meteorological input data and to forecast the river discharge. There are different approaches to tackle the problem, namely: conceptual models, physical-based models and data-driven models.

Although this is taught at university, often hands-on experience is missing or is done on using very simple modelling approaches. One of the main reasons I see is, that most (at least the complex ones) hydrological models are implemented in Fortran but very few students of the field of hydrology know Fortran, when they first get in touch with RR-Models. So all they can probably do is simply apply a model to their data and play manually with parameter tuning, but not explore the model and see the effect of code changes.

This might be different if there would exist well performing implementations of hydrological models in a more simplistic and readable language, such as Python.
What was hindering this step was always the speed of Python and the nature of RR-Models - they mostly have to be implemented using loops over all timesteps. And well, big surprise: Pure Python and for-loops is not the best combination in terms of performance.

This could be changed e.g. by using `Cython <http://cython.org/>`_ for the hydrological model, but this again might hinder the code understanding, since Cython adds non-pythonic complexity to the code, which might be hard for beginners to understand and therefore play/experiment with the code.

Another option could be `PyPy <http://pypy.org/>`_. The problem I see with PyPy is, that the user would be forced to install a different Python interpreter, while most I know of are quite comfortable using e.g. `Anaconda <https://www.continuum.io/anaconda-overview>`_.

`Numba <http://numba.pydata.org/>`_ is another way to speed up array-oriented and math-heavy Python code but without changing the language/interpreter and just by few code adaptions. Using numba, the code stays easily readable and therefore better understandable for novices. I won't spend much time now on explaining how numba works, but I'll definitely add further information in the future.
First performance comparisons between Fortran implementations and numba optimized Python code have shown, that the speed is roughly the same (Fortran is about ~1-2 times faster, using the GNU Fortran compiler).

**Summary**: The idea of this code repository is to provide fast (roughly the speed of Fortan) implementations of hydrological models in Python to make it easier to play and experiment with rainfall-runoff models.


What's next?
------------

During the next days/weeks I'll start to add more and more content to this repository. If you have any questions feel free to ask.


You want to contribute?
-----------------------

At the moment I'm looking for a selection of hydrological models I'll implement in Python. If you want to see any (your?) model in this project, feel free to contact me.

Contact
-------

Raise an issue here in this repository or contact me by mail f.kratzert(at)gmail.com
