# -*- coding: utf-8 -*-

"""
Job to setup a network

@author: Karin van Garderen
"""

import sys
import os
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
    name = sys.argv[1]
    configfile = sys.argv[2]
    tmpdir = sys.argv[3]
    homedir = sys.argv[4]
    job = JobSetup(configfile, name, tmpdir, homedir)
    job.run()
