# -*- coding: utf-8 -*-

"""
Job to setup a network

@author: Karin van Garderen
"""

import sys
import os
import argparse
from glassimaging.execution.jobs.job import Job
from glassimaging.training.standardTrainer import StandardTrainer

class JobSetup(Job):
    
    def __init__(self, configfile, name,  tmpdir, homedir = None, uid = None):
        super().__init__(configfile, name,  tmpdir, homedir, uid = uid)
        
    def run(self):
        ##### Create identifiers
        modelpath = os.path.join(self.tmpdir, 'model.pt')
        
        ##### Set network specifics
        myconfig = self.config
        model_desc = myconfig["Model"]
        trainer = StandardTrainer.initFromDesc(model_desc)
        trainer.saveModel(modelpath)
        self.logger.info('model initialized and saved under ' + modelpath + '.')
                
        self.tearDown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a job.')
    parser.add_argument('name', nargs=1,
                        help='a name to call your job by.')
    parser.add_argument('configfile', nargs=1,
                        help='path to a json config file.')
    parser.add_argument('tmpdir', nargs=1,
                        help='directory for the output.')
    parser.add_argument('--h', nargs=1,
                        help='additional directory to write logs to.')
    args = parser.parse_args()
    job = JobSetup(args.configfile, args.name, args.tmpdir, args.h)
    job.run()
