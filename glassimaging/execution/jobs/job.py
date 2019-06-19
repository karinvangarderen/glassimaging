# -*- coding: utf-8 -*-
"""
Represents a runnable job. Contains logic for creating the directory and initiating the logging

@author: Karin van Garderen
"""
import logging
import sys
import os
from datetime import datetime as dt
import json
import subprocess
import torch
import numpy as np
import random

class Job():
    
    """ Create a new directory in the tmpdir, containing a log and the configfile.
        Also creates a log in the provided homedir, to enable tracking progress. """
    def __init__(self, configfile, name, tmpdir, homedir = None, uid = None):
        self.name = name
        self.uid = uid if uid else dt.now().strftime("%Y%m%d%H%M%S")
        self.datadir = tmpdir
        self.tmpdir = os.path.join(tmpdir, name)
        if not (os.path.exists(self.tmpdir)):
            os.mkdir(self.tmpdir)
        self.logger = logging.getLogger(__name__)
        ## Log to temporary directory
        fh = logging.FileHandler("{0}/{1}.log".format(self.tmpdir, self.name))
        sh = logging.StreamHandler(sys.stdout)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        sh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self.logger.setLevel(logging.DEBUG)
        ### If a homedir is set, also log to that directory. Otherwise, the homedir is the same as the tmpdir
        if homedir != None:
            self.homedir = homedir
            hh = logging.FileHandler("{0}/{1}.log".format(self.homedir, self.name))
            hh.setFormatter(formatter)
            self.logger.addHandler(hh)
            
        else:
            self.logger.info('No homedir set.')
            self.homedir = tmpdir
        self.logger.info('Logging initiated')
        gitlabel = self.getGitLabel()
        self.logger.info('Current git commit: {label}'.format(label = gitlabel))
        self.config = self.loadConfig(configfile)
        self.saveConfig(os.path.join(self.tmpdir, 'config.json'))
        if homedir != None:
            self.saveConfig(os.path.join(self.homedir, 'config.json'))
            
        if 'seed' in self.config:
            seed = self.config['seed']
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            self.logger.info('Random seed is {}'.format(seed))
        
    def loadConfig(self, path):
        with open(path, 'r') as file:
            return(json.load(file))
            
    def saveConfig(self,path):
        with open(path, 'w') as file:
            json.dump(self.config, file)
    
    """ Release log files. """  
    def tearDown(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
            
    def getGitLabel(self):
        gitlabel = subprocess.check_output(["git", "describe", "--always"]).strip()
        logging.info('Current Git Label: ' + str(gitlabel))
        return str(gitlabel)       
    
    def run(self):
        ### Depending on the job type in configfile, do something
        raise NotImplementedError
