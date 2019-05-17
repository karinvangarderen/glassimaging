"""
Helper functions used during execution

@author: Karin van Garderen
"""

import os
from string import Template


""" Loads a templatefile and fills it using the provided config, and saves it to outputfile """
def createBashScript(templatefile, outputfile, config):
    with open(templatefile, 'r') as tmpfile:
        src = Template(tmpfile.read())
    result = src.substitute(config)
    with open(outputfile, 'w+') as outfile:
        outfile.write(result)
    

def runSlurmjob(outputdir, name, filename, config):
    partition = config['partition'] if 'partition' in config else 'gpu-test'
    outputfile = config['outputfile'] if 'outputfile' in config else os.path.join(outputdir, 'out.out')
    errorfile = config['errorfile'] if 'errorfile' in config else os.path.join(outputdir, 'err.log')
    timelimit = config['timelimit'] if 'timelimit' in config else '1:00:00'
    
    jobfile = os.path.join(outputdir, name+'.job')
    
    with open(jobfile, 'w+') as f:
        f.write('#!/bin/sh \n')
        f.write('#SBATCH --partition={}\n'.format(partition))
        f.write('#SBATCH -o {} \n'.format(outputfile))
        f.write('#SBATCH -e {} \n'.format(errorfile))
        f.write('#SBATCH -t {} \n'.format(timelimit))
        f.write('. /home/kvangarderen/pytorch36/bin/activate \n'.format(outputfile)) ###TODO make this generic
        f.write('python {} $1 $2 \n'.format(filename))
        f.write('deactivate \n')

