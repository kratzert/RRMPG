How to contribute
=================

Since this is a community project, I would deeply encourage any developer or hydrologist to contribute to the development of this project. Different ways of how you can contribute exist and not all require programming skills.

Spell checking
--------------

Since English is not my native-language, I'm sure there are a lot of mistakes in this documentation or the code documentation itself. Feel free to make a pull request on GitHub (or open an issue or write me an email, whatever is the most comfortable for you) and I'll gladly correct the mistakes.

Contribute to the wiki
----------------------

The :ref:`Wiki <wiki_top>` should give a more detailed description of the model, including e.g. historical background and application examples. If feasable, also visualizations of the model structure can be added. The idea is, to have a summary of the model, that provide enough information, such that anyone without previous knowledge of the model understands the model capabilities/weaknesses and knows what the model is doing. This doesn't have to be written in one push and can be extended in future commits by other contributers.

What do I need?
^^^^^^^^^^^^^^^
The entire documentation is created using Sphinx_ with reStructuredText (rst: `Wikipedia <https://en.wikipedia.org/wiki/ReStructuredText>`_ , `Quick Ref <http://docutils.sourceforge.net/docs/user/rst/quickref.html>`_), which is lightweight markup language. Basically you can write rst-files with any editor, I personally use the free Atom-Editor_. If you want to create a new entry in the wiki for one of the models, simply create a new rst-file and start writing. In the case you want to extend/edit an existing model entry, simply edit the corresponding rst-file.
To compile the documentation and create the html output from the rst-files locally you further need Python installed with various packages. I highly recommend using Anaconda_, which ships with a lot of useful packages. In general it's a good practise to have different environments of Python for different projects you are working on (Read this link_ for an introduction to Python environments) and Anaconda comes with it's own way for organising environments. In the repository I added a rtd_environment.yml_ file, which creates a new environment for you with everything you need compile the documentation locally. Simply download the rtd_environment.yml_ file, then enter the terminal and enter:

.. code-block:: bash

    conda env  create -f rtd_environment.yml


You can then activate the environment by entering:

.. code-block:: bash

    # on linux and macOS
    source activate docenv3

    # on windows
    activate docenv3


To leave an environment enter

.. code-block:: bash

    # on linux and macOS
    source deactivate

    # on windows
    deactivate


For further details on Anaconda environments see here_.

After you have made changes to the documentation and you want to see the result as a html-page, direct in the terminal to the ``rrmpg/docs`` and enter:

.. code-block:: bash

    # on linux and macOS
    make html

    # on windows
    make.bat html

If everything has compiled correctly you should find an ``index.html`` in ``rrmpg/docs/build/html``.

Other option
^^^^^^^^^^^^
Anyway, if this might seem to complicated for you, you can always send me your text by email (f.kratzert[at]gmail.com) or create an issue on GitHub and I'll do the rest.

Important note
^^^^^^^^^^^^^^
This should be commonsense but I would like to remind you to cite every work of others (may it be publications, homepages, images etc.) you use in what ever you write.

.. _Sphinx: http://www.sphinx-doc.org
.. _Atom-Editor: https://atom.io/
.. _rtd_environment.yml: https://github.com/kratzert/RRMPG/blob/master/rtd_environment.yml
.. _Anaconda: https://www.continuum.io/downloads
.. _link: http://docs.python-guide.org/en/latest/dev/virtualenvs/
.. _here: https://conda.io/docs/user-guide/tasks/manage-environments.html
