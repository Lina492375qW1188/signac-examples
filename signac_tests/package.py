import time
import os
import itertools
import numpy as np

class JobDict(dict):
    
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

class Job:
    
    def __init__(self, job_id, job_path, statepoint):
    
        self.job_id = job_id
        self.job_path = job_path
        
        self.statepoint_path = os.path.join(self.job_path, 'statepoint.npy')
        self.document_path = os.path.join(self.job_path, 'document.npy')
        
        if os.path.exists(self.statepoint_path):
            statepoint = np.load(self.statepoint_path, allow_pickle=True).tolist()
        
        self.sp = JobDict(statepoint)
        self.statepoint = self.sp
        
        if os.path.exists(self.document_path):
            document = np.load(self.document_path, allow_pickle=True).tolist()
        else:
            document = {}
            
        self.doc = JobDict(document)
        self.document = self.doc
        
        
    def init(self):
        
        if not os.path.isdir(self.job_path):
            os.makedirs(self.job_path)
        
        if not os.path.exists(self.statepoint_path):
            np.save(self.statepoint_path, dict(self.sp))
        else:
            print('job already exist!')
            
    def fn(self, filename):
        
        job_associated_path = os.path.join(self.job_path, filename)
        
        return job_associated_path
    
    def update(self):
        
        np.save(self.statepoint_path, dict(self.sp))
        np.save(self.document_path, dict(self.doc))
    
        
class Project:

    def __init__(self, project_name='project'):
        
        self.project_name = project_name
        self.project_path = self.project_name
        self.job_id = -1
        
        self.id2sp = {}
        self.id2job_path = {}
        
        if not os.path.isdir(self.project_path):
            os.makedirs(self.project_path)
            
    def open_job(self, statepoint):
        """
        Open a new job without overwriting existing jobs.
        """
        self.job_id += 1
        job_path = os.path.join(self.project_path, '{:06d}'.format(self.job_id))
        if not os.path.isdir(job_path):
            
            self.id2sp[self.job_id] = statepoint
            self.id2job_path[self.job_id] = job_path
            
            return Job(self.job_id, job_path, statepoint)
        
        else:
            if not self.id2sp or self.id2job_path: 
                self.get_jobs() # run only once
            statepoint_i = self.id2sp[self.job_id]
            job_path_i = self.id2job_path[self.job_id]
            
            return Job(self.job_id, job_path_i, statepoint_i)
        
    def get_jobs(self):
        """
        Get jobs in the project folder.
        """
        for job_id_i in os.listdir(self.project_name):
            
            i = int(job_id_i)
            job_path_i = os.path.join(self.project_name, job_id_i)
            statepoint_path_i = os.path.join(job_path_i, 'statepoint.npy')
            statepoint_i = np.load(statepoint_path_i, allow_pickle=True).tolist()
            
            self.id2sp[i] = statepoint_i
            self.id2job_path[i] = job_path_i
        
    def find_jobs(self, condition=None):
        """
        Find jobs which match the condition.
        """
        if condition==None:
            for job_id_i, job_path_i in self.id2job_path.items():
                statepoint_i = self.id2sp[job_id_i]
                yield Job(job_id_i, job_path_i, statepoint_i)
        
        else:
            for job_id_i, statepoint_i in self.id2sp.items():
                if all(statepoint_i[k] == v for k, v in condition.items()):
                    job_path_i = self.id2job_path[job_id_i]
                    yield Job(job_id_i, job_path_i, statepoint_i)
                    
    def find_job_ids(self, job_id_arr):
        """
        Find jobs which match the job_id.
        """
        for job_id_i in job_id_arr:
            job_path_i = self.id2job_path[job_id_i]
            statepoint_i = self.id2sp[job_id_i]
            yield Job(job_id_i, job_path_i, statepoint_i)
