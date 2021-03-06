# -*- coding: utf-8 -*-

"""Console script for glassimaging."""
import sys
import click
from glassimaging.execution.experiment import Experiment
from glassimaging.execution.utils import runJob
from glassimaging.execution.jobs.jobapply_single import JobApplySingle
from glassimaging.preprocessing.preprocess_subject import process_experiment

@click.command()
@click.option('--t1', 't1', required=True)
@click.option('--t2', 't2', required=True)
@click.option('--t1gd', 't1gd', required=True)
@click.option('--flair', 'flair', required=True)
@click.option('-m', '--modelpath', 'modelpath', required=True)
@click.option('-b', '--brainmaskpath', 'brainmaskpath', required=True)
@click.option('-c', '--configfile', 'configfile', required=True)
def applynetwork(t1, t2, t1gd, flair, modelpath, brainmaskpath, configfile):
    job = JobApplySingle(t1, t2, flair, t1gd, modelpath, brainmaskpath, configfile)
    job.run()
    return 0

if __name__ == "__main__":
    sys.exit(applynetwork())  # pragma: no cover
