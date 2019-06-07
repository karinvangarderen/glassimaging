============
GLASSImaging
============


Automatic delineation of relevant structures for the analysis of GLASS imaging, using deep neural networks.


* Free software: Apache Software License 2.0


Using a new dataset
--------

1. Write a class extending glassimaging.dataloading.niftidataset.NiftiDataset
2. Implement the importData function, which loads examples from a file into a dataframe (see examples)

Defining a Job
--------

1. Extend the Job class
2. Implement the run function which contains the tasks to run
3. Use the self.config dictionary to define and use parameters

Running jobs using SLURM
--------


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
