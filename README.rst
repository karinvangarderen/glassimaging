============
GLASSImaging
============


Automatic delineation of low-grade glioma from MR imaging for longitudinal analysis. Using deep neural networks.


* Free software: Apache Software License 2.0

Installation
----------------------
1. Clone the repository
2. Install requirements
3. If on a cluster, make a ``platform.ini`` file in the main directory,
    which contains the name of the
    cluster. For now, either ``gpucluster`` or ``cartesius``.


Running an Experiment
----------------------
An experiment (glassimaging.execution.experiment) is a set of Jobs,
which may depend on eachother. Defining these Jobs as an Experiment makes it easier to
run them sequentially (or in parallel) on a (SLURM-based) cluster. All configuration is
defined in a JSON file.

To run a job you need to specify three arguments:

1. A name, or handle to call it by.
2. A config file
3. An output directory. A new directory will be made here, which contains all the logs
   and results

``` python -m glassimaging.execution.experiment name config.json /home/outputdir/
```


Defining a Job
-------------------

A Job (glassimaging.execution.jobs.job) is basically a script. It is defined by code, configuration and dependencies.The code
resides in the class that extends Job, which contains a 'run' function. Any results can
be written to a temporary directory which will be written to output directory
after the job is finished.

The configuration is defined by the JSON file included in the experiment. The contents of
this file may be anything, as long as it is valid JSON. Within the job you may access it
as a python dictionary.

Dependencies are results of previous jobs, for example a trained model. The dependencies
are defined in the experiment configuration. For each dependency the 'result' directory
will be copied to the temporary directory, and any file included may be accessed in the job.

To run a stand-alone job:
``` python -m glassimaging.execution.jobs.myjob name config.json /home/outputdir/
```

Using a new dataset
--------------------

1. Write a class extending glassimaging.dataloading.niftidataset.NiftiDataset
2. Implement the importData function, which loads examples from a file into a dataframe (see examples)

