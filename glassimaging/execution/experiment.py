# -*- coding: utf-8 -*-
"""
Responsible for spawning jobs from a configuration file and running them.
To add a new job type: write the runnable python file and add the type to the getScriptForJob function
To add a new platform: write the function to run on the platform and add a 'platform.ini' file to the root dir with the platform name

@author: Karin van Garderen
"""

from glassimaging.execution.scripts import createBashScript
import os
import logging
import subprocess
import sys
import re
import json
from datetime import datetime as dt

class Experiment():
    
    def __init__(self, configfile, name, directory, platform = 'bigr-nzxt-7'):
        ### Create a unique name to prevent duplicate folders in the tmp dir
        self.platform = platform
        if self.platform == 'gpucluster':
            os.umask(0)
        
        self.name = dt.now().strftime("%Y%m%d%H%M%S_") + name
        self.outputdir = os.path.join(directory, self.name)
        os.mkdir(self.outputdir)
        ### Platform is needed to spawn a job with the correct slurm template
        self.logger = logging.getLogger(__name__)
        self.logger.addHandler(logging.FileHandler("{0}/{1}.log".format(self.outputdir, self.name)))
        self.logger.addHandler(logging.StreamHandler(sys.stdout))
        self.logger.setLevel(logging.DEBUG)
        
        ## Load the config and fetch the job keys
        self.config = self.loadConfig(configfile)
        self.joblist = self.config["jobs"].keys()
        self.executionlist = self.config["execution"]
        
        
    def loadConfig(self, path):
        with open(path, 'r') as file:
            return json.load(file)
        
    def saveConfig(self,path, config):
        with open(path, 'w') as file:
            json.dump(config, file)

    """ Attempt to run all jobs in the experiment's config, while keeping track of dependencies """
    def run(self):
        ### Keep a list of all the jobs created with their job number
        self.jobsCreated = {jobname:False for jobname in self.joblist}
        self.jobsFinished = {jobname:False for jobname in self.joblist}
        self.jobnumbers = {}
        self.jobDirectories = {}
        ### Check whether some have already been finished
        for jobname in self.joblist:
            if 'use result' in self.config["jobs"][jobname]:
                self.jobDirectories[jobname] = self.config["jobs"][jobname]['use result']
                self.jobsFinished[jobname] = True
        ### Execute in the order of the execution list
        for joblist in self.executionlist:
            jobscripts = {}
            configfiles = {}
            copy_jobs = []
            wait_for = []
            for name in joblist:
                if 'use result' not in self.config["jobs"][name]:
                    jobscripts[name] = self.getScriptForJob(name)
                    configfiles[name] = self.prepareConfig(name)
                    for d in self.config["jobs"][name]['dependencies']:
                        if not d in self.joblist: 
                            raise Exception('Dependency {} of job {} not in the list of jobs.'.format(d, name))
                        elif d in joblist:
                            pass ## Result will be created in job
                        elif self.jobsFinished[d]:
                            if d not in copy_jobs:
                                copy_jobs.append(d)
                        elif self.jobsCreated[d]:
                            if d not in copy_jobs:
                                copy_jobs.append(d)
                            if d not in wait_for:
                                wait_for.append(d)
                        else:
                            raise KeyError('Dependency could not be satisfied: {}'.format(d))

            jobnum = self.postJob(self.platform, joblist, jobscripts, configfiles, copy_jobs, wait_for)
            for name in joblist:
                self.jobnumbers[name] = jobnum
                self.jobsCreated[name] = True

        self.tearDown()
    
    """ The configuration is composed of both a file and a specific configuration
        Both can be either present or not, if neither are present the configuration is simply empty.
        A specific configuration defined in the experiment config overwrites any included file
    """
    def prepareConfig(self,name):
        jobconfig = {}
        if 'configfile' in self.config["jobs"][name]:
            jobconfig.update(self.loadConfig(self.config["jobs"][name]['configfile']))
        if 'config' in self.config["jobs"][name]:
            jobconfig.update(self.config["jobs"][name]['config'])
        path = os.path.join(self.outputdir, name + '.json')
        self.saveConfig(path, jobconfig)
        return path
    
    """ Fetches the correct script for each job type. """        
    def getScriptForJob(self, jobname):
        jobtype = self.config["jobs"][jobname]['type']
        if jobtype == 'eval':
            return 'glassimaging.execution.jobs.jobeval'
        elif jobtype == 'train':
            return 'glassimaging.execution.jobs.jobtrain'
        elif jobtype == 'setup':
            return 'glassimaging.execution.jobs.jobsetup'
        elif jobtype == 'apply':
            return 'glassimaging.execution.jobs.jobapply'
        elif jobtype == 'train_multipath':
            return 'glassimaging.execution.jobs.jobtrain_multipath'
        
    def getExecuteString(self, platform, names, jobscripts, configfiles, job_outputdirs):
        if platform == 'cartesius':
            return self.getExecuteStringCartesius(names, jobscripts, configfiles, job_outputdirs)
        elif platform == 'gpucluster':
            return self.getExecuteStringGPUCluster(names, jobscripts, configfiles, job_outputdirs)
        
    def getExecuteStringCartesius(self, names, jobscripts, configfiles, outputdirs):
        string = ''
        for n in names:
            string = string + 'python3 -m {script} {name} {config} $TMPDIR --log {outputdir} \n'.format(\
                                                        script = jobscripts[n],\
                                                        name = n,\
                                                        config = configfiles[n],\
                                                        outputdir = outputdirs[n])
            string = string + 'cp -r $TMPDIR/{name} {outputdir}/result \n'.format(\
                                                        name = n,\
                                                        outputdir = outputdirs[n])
        return string
        
    def getExecuteStringGPUCluster(self, names, jobscripts, configfiles, outputdirs):
        string = ''
        for n in names:
            string = string + 'OMP_NUM_THREADS=1 python3 -m {script} {name} {config} $MY_TMP_DIR --log {outputdir} \n'.format(\
                                                        script = jobscripts[n],\
                                                        name = n,\
                                                        config = configfiles[n],\
                                                        outputdir = outputdirs[n])
            string = string + 'cp -r $MY_TMP_DIR/{name} {outputdir}/result \n'.format(\
                                                        name = n,\
                                                        outputdir = outputdirs[n])
        return string
        
    def getCopyString(self, platform, copy_jobs):
        if platform == 'cartesius':
            return self.getCopyStringCartesius(copy_jobs)
        elif platform == 'gpucluster':
            return self.getCopyStringGPUCluster(copy_jobs)
        
    def getCopyStringCartesius(self, copy_jobs):
        copystring = ''
        for d in copy_jobs: 
            olddir = self.jobDirectories[d]
            newdir = d
            copystring = copystring + 'cp -r {olddir}/result $TMPDIR/{newdir} \n'.format(olddir = olddir, newdir = newdir)
        return copystring
            
    def getCopyStringGPUCluster(self, copy_jobs):
        copystring = ''
        for d in copy_jobs: 
            olddir = self.jobDirectories[d]
            newdir = d
            copystring = copystring + 'cp -r {olddir}/result $MY_TMP_DIR/{newdir}\n'.format(olddir = olddir, newdir = newdir)
        return copystring
        
            
    def getSlurmConfig(self, platform):
        if platform == 'cartesius':
            return {
                'partition': 'gpu',
                'timelimit': '2-00:00:00',
                'outfile': os.path.join(self.outputdir, '%j_out.out'),
                'errfile': os.path.join(self.outputdir, '%j_err.log')
                }
        elif platform == 'gpucluster':
            return {
                'gres': 'gpu:1',
                'timelimit': '2-00:00:00',
                'outfile': os.path.join(self.outputdir, '%j_out.out'),
                'errfile': os.path.join(self.outputdir, '%j_err.log'),
                'ntasks':'6',
                'mem':'14G',
                }
            
            
    def getTemplatefile(self, platform):
        if platform == 'cartesius':
            return 'glassimaging/execution/templates/cartesius.job'
        elif platform == 'gpucluster':
            return 'glassimaging/execution/templates/cluster.job'
    """
    platform: string describing the current platform
    names: list of strings
    jobscripts: dictionary with names as keys
    configfiles: dictionary with names as keys
    copy_jobs: list of names
    wait_for: list of names
    """
    def postJob(self, platform, names, jobscripts, configfiles, copy_jobs, wait_for):
        
        ###### GPU cluster has permission issues
        if platform == 'gpucluster':
            os.umask(0)
        
        ######## Make a local directory for each subjob
        job_outputdirs = {}
        for n in names:
            job_outputdirs[n] = os.path.join(self.outputdir, n)
            os.mkdir(job_outputdirs[n])
            self.jobDirectories[n] = job_outputdirs[n]
        
        ######## Setup the command to copy old results to the node
        copystring = self.getCopyString(platform, copy_jobs)
        
        ######## Setup commands to execute the scripts and copy the results
        executestring = self.getExecuteString(platform, names, jobscripts, configfiles, job_outputdirs)
            
        ######## Combine everything to fill the template job files
        slurm_config = self.getSlurmConfig(platform)
        slurm_config['copystring'] =  copystring
        slurm_config['executestring'] =  executestring
            
        ### Fill the template file
        templatefile = self.getTemplatefile(platform)
        path_script = os.path.join(self.outputdir, names[0] + '.job')
        createBashScript(templatefile, path_script, slurm_config)
        
        #### Set the slurm dependencies for those jobs that have been created and execute accordingly
        dependencystring = ''
        for d in copy_jobs: 
                if not self.jobsFinished[d]:
                    dependencystring = dependencystring + ':' + str(self.jobnumbers[d])
        if len(dependencystring) > 0:
            dependencystring = '--dependency=afterok' + dependencystring
            command = subprocess.run(["sbatch", dependencystring, path_script], stdout=subprocess.PIPE)
        else:
            command = subprocess.run(["sbatch", path_script], stdout=subprocess.PIPE)
            
        ### Log and find the returncode which contains the job number
        self.logger.info("Submitting {name} with command: {command}".format(name = [names[0]], command=command.args))
        jobresponse = command.stdout.decode('utf-8')
        self.logger.info(jobresponse)
        jobnum = int(re.findall('[0-9]+', jobresponse)[0])
        status = command.returncode
        if (status == 0 ):
            self.logger.info("{name} is {s}".format(name =names[0], s=jobnum))
        else:
            self.logger.error("Error submitting {name}".format(name = names[0]))
        return jobnum
    
    """ Release log files. """        
    def tearDown(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)
    
if __name__ == '__main__':
    configfile = sys.argv[2]
    name = sys.argv[1]
    directory = sys.argv[3]
    try:
        with open('platform.ini', 'r') as f:
            platform = f.readline()
    except:
        platform = 'unknown'
    if 'bigr-nzxt-7' in platform:
        platform = 'gpucluster'
    elif 'cartesius' in platform:
        platform = 'cartesius'
    exp = Experiment(configfile, name, directory, platform = platform)
    exp.run()
