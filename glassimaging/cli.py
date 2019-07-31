# -*- coding: utf-8 -*-

"""Console script for glassimaging."""
import sys
import click
from glassimaging.execution.experiment import Experiment
from glassimaging.execution.utils import runJob
from glassimaging.preprocessing.preprocess_subject import process_experiment

@click.group()
def cli():
    return 0

@cli.command()
@click.option('-c', '--configfile', 'configfile', required=True, type=str)
@click.option('-n', '--name', 'name', required=True)
@click.option('-o', '--outputdir', 'outputdir', required=True)
def experiment(configfile, name, outputdir):
    """Console script for glassimaging."""
    try:
        with open('platform.ini', 'r') as f:
            platform = f.readline()
    except:
        platform = 'unknown'
    if 'bigr-nzxt-7' in platform:
        platform = 'gpucluster'
    elif 'cartesius' in platform:
        platform = 'cartesius'
    exp = Experiment(configfile, name, outputdir, platform=platform)
    click.echo(exp.run())
    return 0

@cli.command()
@click.option('-c', '--configfile', 'configfile', required=True, type=str)
@click.option('-n', '--name', 'name', required=True)
@click.option('-o', '--outputdir', 'outputdir', required=True)
@click.option('-t', '--type', 'type', required=True)
def job(configfile, name, outputdir, type):
    runJob(type, name, configfile, outputdir)
    return 0

@cli.command()
@click.option('-h', '--xnathost', 'xnathost', required=True, type=str)
@click.option('-s', '--subject', 'subject', required=True)
@click.option('-o', '--outputdir', 'outputdir', required=True)
def preprocess(xnathost, subject, outputdir):
    process_experiment(xnathost, subject, outputdir)
    return 0



if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
