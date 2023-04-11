import time
import numpy as np

from package import Project, Job
from simulation import randomize, compress, equilibriate

project = Project('test_project')
project.get_jobs()
    
for job in project.find_job_ids([1,3]):
    
    if job.doc['randomize']==False:
        randomize(job)
    if job.doc['compress']==False:
        compress(job)
    if job.doc['equilibriate']==False:
        equilibriate(job)
