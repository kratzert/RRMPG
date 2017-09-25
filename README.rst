Introduction
############

This repository is a work in progress repository.
(Due to travelling I won't be regularly online during the next weeks until approx. 23 of october).

Official documentation can be found here: `http://rrmpg.readthedocs.io <http://rrmpg.readthedocs.io>`_

Read the :ref:`Idea secion <idea>` for further information about the background and aim of this project.

Road map (pull requests welcome!)
---------------------------------
During the last days I was quite busy with my normal job and for the next weeks I'll be in holydays. But before I leave I want to write down my ideas for the further development of this project.

- Change the implementations of the simulate function to accept arrays of model parameters as optional inputs and change the implementation of the model itself (the numba optimized one) to allow inputs of multiple model parameter sets. We can then add a parallelized loop over all parameter sets using Numbas ``prange`` and add the ``parallel=True`` flag to the decorator for parallel simulations of all parameter sets. This could be useful for e.g. Monte-Carlo simulations or other optimizers, where multiple parameter sets are generated and validated at once.
- Add a routing class. This should will be one of the main focuses in the near future. I'm still undecided in how to build this class and I think this needs good planning. With this routing class it should be possible connect many lumped models (even different models for each sub-catchment since they all have a homogenious model class api) to an entire basin consisting of many sub-catchments. Since there are many ways, how the routing can be implemented, the question is if for each routing procedure one seperate class should be implemented (similar to the models with homogenious api) or one class, where the user can decide by an input argument which scheme to use. I tend to the first one, to be consistent with the model classes. There could be one base class for all routing classes as well, which implements the basic functionality, similar to the model base class.
- Add more models to the library. This point needs no further explanation I guess. Next planned is the ``CemaNeige`` snow module, which can be coupled with the ``gr4j``.
- Add example data to the code base. `here <http://tdwg.catchment.org/datasets.html>`_ are some hydrological datasets that could maybe be used. Also utility functions should be added to import the data and preprocess them correctly, so that the user has an easy start with this library.
- Start writing model descriptions in the wiki section (or even better encourage colleagues to participate!).

If you start working on any of these points I think it would be advisable to open an issue on github to let me know on what you are working. By this we can avoid to work on the same thing and better organize our energy.


.. _idea:

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


You want to contribute?
-----------------------

At the moment I'm looking for a selection of hydrological models I'll implement in Python. If you want to see any (your?) model in this project, feel free to contact me.
There is also a `How to contribute section <http://rrmpg.readthedocs.io/en/latest/contribution.html>`_ at the official documentation, were you can read more on the various ways you can contribute to this repository.

Contributors
------------
I'll add later a better looking section to the official documentation. For now I list everybody, who contributed to this repository here:

- `Ondřej Čertík <https://github.com/certik>`_ with pull request `#3 <https://github.com/kratzert/RRMPG/pull/3>`_: Optimized Fortran code and compilation procedure for fair speed comparision.
- `Daniel Klotz <https://github.com/danklotz>`_ with pull request `#4 <https://github.com/kratzert/RRMPG/pull/4>`_ and `#5 <https://github.com/kratzert/RRMPG/pull/4>`_: Both spell corrections.

Contact
-------

Raise an issue here in this repository or contact me by mail f.kratzert(at)gmail.com
