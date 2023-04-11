import time
import itertools
import numpy as np

import coxeter

from package import Project
from simulation import init
from verts_lib import octa, hexbp, hexbp3 trunc_octa

# List of fixed parameter
N_particles_arr = [1000]
vertices_arr = [hexbp]
particle = coxeter.shapes.ConvexPolyhedron( vertices_arr[0] )
particle_volume_arr = [particle.volume]
pressure_arr = [12.]

# List to explore
seed_arr = [i for i in range(1,5)]
inertia_arr = []
for ri in np.array([1.]):
    diag_inertia = particle.inertia_tensor.diagonal().copy()
    diag_inertia[2] *= ri
    inertia_arr.append(diag_inertia)
    
sp_dict = list(itertools.product(N_particles_arr, 
                                 vertices_arr, 
                                 particle_volume_arr, 
                                 pressure_arr, 
                                 seed_arr, 
                                 inertia_arr))


project = Project('test_project')

for N_particles, vertices, particle_volume, pressure, seed, inertia in sp_dict:
    statepoint = dict(N_particles=N_particles, 
                      vertices=vertices,
                      particle_volume=particle_volume,
                      pressure=pressure, 
                      seed=seed,
                      inertia=inertia)

    job = project.open_job(statepoint)
    job.init()
    
for job in project.find_jobs():
    
    job.doc['init']=False
    job.doc['randomize']=False
    job.doc['compress']=False
    job.doc['equilibriate']=False
    
    particle = coxeter.shapes.ConvexPolyhedron( job.sp.vertices )
    job.doc['verts'] = particle.vertices.tolist() 
    job.doc['faces'] = particle.faces 
    job.doc['sigma'] = 2*particle.insphere_from_center.radius 
    job.doc['r_cut'] = 2*particle.circumsphere_from_center.radius + 0.15*job.doc.sigma
    job.update()
    
    init(job) # init() contains gsd.hoomd which does not support MPI.
    
    