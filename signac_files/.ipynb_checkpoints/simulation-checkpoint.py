import itertools
import numpy as np

import hoomd
import gsd.hoomd
import freud
import coxeter

import signac
import flow

device = hoomd.device.CPU()


class Project(flow.FlowProject):
    pass


@Project.operation  # Workflow step.
@Project.post.true('init')  # Post-condition in job document.
def init(job):
    
    m = 10
    spacing = 2.6 # > particle size
    L = m * spacing

    N_tot = m**3
    N_particles = job.sp.N_particles

    x = np.linspace(-L/2, L/2, m, endpoint=False) + spacing/2
    position = list(itertools.product(x, repeat=3))[:N_particles]

    snapshot = gsd.hoomd.Snapshot()
    snapshot.particles.types = ['A']
    snapshot.particles.N = N_particles
    snapshot.particles.position = position
    snapshot.particles.orientation = [1,0,0,0] * N_particles
    snapshot.particles.moment_inertia = [job.sp.inertia] * N_particles
    snapshot.configuration.box = [L, L, L, 0, 0, 0]
    verts = np.array(job.doc.verts).tolist()
    snapshot.particles.type_shapes = [{'type': 'ConvexPolyhedron',
                                       'rounding_radius': 0.,
                                       'vertices': verts}]

    with gsd.hoomd.open(name=job.fn('lattice.gsd'), mode='xb') as f:
        f.append(snapshot)
        
    job.document['init'] = True
    

@Project.operation  # Workflow step.
@Project.pre.true('init')  # Pre-condition in job document.
@Project.post.true('randomize')  # Post-condition in job document.
def randomize(job):
    
    # integration timestep
    dt = 0.0005
    # coupling
    kT = 1.0
    tau = 100*dt
    # randomizing time
    t_rand = 1e+4
    # particle attributes
    verts = np.array(job.doc.verts)
    faces = np.array(job.doc.faces)
    sigma = float(job.doc.sigma)
    r_cut = float(job.doc.r_cut)
    # energy constant
    epsilon = 1.0
    
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    sim.create_state_from_gsd(filename=job.fn('lattice.gsd'))
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), 
                                          kT=kT)
    
    # neighboring list
    nl = hoomd.md.nlist.Cell(buffer=2.5*sigma)
    # potential
    alj = hoomd.md.pair.aniso.ALJ(nl)
    alj.r_cut[('A', 'A')] = r_cut
    alj.params[('A', 'A')] = dict(epsilon=epsilon, 
                                  sigma_i=sigma, 
                                  sigma_j=sigma, 
                                  alpha=0)
    alj.shape['A'] = dict(vertices=verts, 
                          faces=faces, 
                          rounding_radii=0.) # rounding_radii=0.15*(sigma/2)

    nvt = hoomd.md.methods.NVT(filter=hoomd.filter.All(), 
                               kT=kT, 
                               tau=tau)
    
    integrator = hoomd.md.Integrator(dt=dt, 
                                     methods=[nvt],
                                     forces=[alj],
                                     integrate_rotational_dof=True)
    
    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=['timestep', 'walltime'])
    logger.add(alj, quantities=['type_shapes'])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    logger.add(thermodynamic_properties)
    
    gsd_writer = hoomd.write.GSD(filename=job.fn('randomize.gsd'),
                                 trigger=hoomd.trigger.Periodic(100),
                                 log=logger,
                                 mode='xb')
    
    sim.operations.writers.append(gsd_writer)
    sim.operations.integrator = integrator
    sim.operations.computes.append(thermodynamic_properties)
    
    sim.run(t_rand)
    
    job.document['randomize'] = True
    
    
@Project.operation
@Project.pre.true('randomize')
@Project.post.true('compressed_step')
def compress(job):
    
    randomized = gsd.hoomd.open(job.fn('randomize.gsd'), 'rb')
    p_init = randomized[-1].log['md/compute/ThermodynamicQuantities/pressure'][0]
    
    # integration timestep
    dt = 0.0005
    # coupling
    kT = 1.0
    tau = 100*dt
    # reduced pressure
    p_init = p_init
    p_second = 3*kT/job.sp.particle_volume
    tauS = 1000*dt
    # compressing time
    t_ramp = int(1e4)
    # particle attributes
    verts = np.array(job.doc.verts)
    faces = np.array(job.doc.faces)
    sigma = float(job.doc.sigma)
    r_cut = float(job.doc.r_cut)
    # energy constant
    epsilon = 1.0
    
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    sim.create_state_from_gsd(filename=job.fn('randomize.gsd'), frame=-1)
    
    nl = hoomd.md.nlist.Cell(buffer=2.5*sigma)
    alj = hoomd.md.pair.aniso.ALJ(nl)
    alj.r_cut[('A', 'A')] = r_cut
    alj.params[('A', 'A')] = dict(epsilon=epsilon, 
                                  sigma_i=sigma, 
                                  sigma_j=sigma, 
                                  alpha=0)
    alj.shape['A'] = dict(vertices=verts, 
                          faces=faces, 
                          rounding_radii=0.) # rounding_radii=0.15*(sigma/2)
    
    S = hoomd.variant.Ramp(A=p_init, 
                           B=p_second, 
                           t_start=sim.timestep, 
                           t_ramp=t_ramp)
    npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), 
                               kT=kT, 
                               tau=tau, 
                               S=S, 
                               tauS=tauS,
                               couple='xyz')
    
    integrator = hoomd.md.Integrator(dt=dt, 
                                     methods=[npt],
                                     forces=[alj],
                                     integrate_rotational_dof=True)
    
    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=['timestep', 'walltime'])
    logger.add(alj, quantities=['type_shapes'])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    logger.add(thermodynamic_properties)

    gsd_writer = hoomd.write.GSD(filename=job.fn('compress.gsd'),
                                 trigger=hoomd.trigger.Periodic(100),
                                 log=logger,
                                 mode='xb')
    
    sim.operations.writers.append(gsd_writer)
    sim.operations.integrator = integrator
    sim.operations.computes.append(thermodynamic_properties)
    
    sim.run(t_ramp)
    
    job.document['compressed_step'] = sim.timestep
    
    
N_EQUIL_STEPS = 1e5  # Number of timesteps to run during equilibration.
HOOMD_RUN_WALLTIME_LIMIT = 30  # Time in seconds at which to stop the operation.
@Project.operation
@Project.pre.after(compress)
@Project.post.true('equilibriate')
def equilibriate(job):
    
    end_step = job.document['compressed_step'] + N_EQUIL_STEPS
    
    # integration timestep
    dt = 0.0005
    # coupling
    kT = 1.0
    tau = 100*dt
    # reduced pressure
    pressure = job.sp.pressure*kT/job.sp.particle_volume
    tauS = 1000*dt
    # equilibriating time
    t_eq = int(1e4)
    # particle attributes
    verts = np.array(job.doc.verts)
    faces = np.array(job.doc.faces)
    sigma = float(job.doc.sigma)
    r_cut = float(job.doc.r_cut)
    # energy constant
    epsilon = 1.0
    
    sim = hoomd.Simulation(device=device, seed=job.sp.seed)
    
    if job.isfile('restart.gsd'):
        # Read the final system configuration from a previous execution.
        sim.create_state_from_gsd(filename=job.fn('restart.gsd'))
    else:
        # Or read `compressed.gsd` for the first execution of equilibrate.
        sim.create_state_from_gsd(filename=job.fn('compress.gsd'), frame=-1)
    
    nl = hoomd.md.nlist.Cell(buffer=2.5*sigma)
    alj = hoomd.md.pair.aniso.ALJ(nl)
    alj.r_cut[('A', 'A')] = r_cut
    alj.params[('A', 'A')] = dict(epsilon=epsilon, 
                                  sigma_i=sigma, 
                                  sigma_j=sigma, 
                                  alpha=0)
    alj.shape['A'] = dict(vertices=verts, 
                          faces=faces, 
                          rounding_radii=0.) # rounding_radii=0.15*(sigma/2)
    
    npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), 
                               kT=kT, 
                               tau=tau, 
                               S=pressure, 
                               tauS=tauS,
                               couple='xyz')
    
    integrator = hoomd.md.Integrator(dt=dt, 
                                     methods=[npt],
                                     forces=[alj],
                                     integrate_rotational_dof=True)
    
    logger = hoomd.logging.Logger()
    logger.add(sim, quantities=['timestep', 'walltime'])
    logger.add(alj, quantities=['type_shapes'])
    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    logger.add(thermodynamic_properties)

    gsd_writer = hoomd.write.GSD(filename=job.fn('equilibriate.gsd'),
                                 trigger=hoomd.trigger.Periodic(100),
                                 log=logger,
                                 mode='ab')
    
    sim.operations.writers.append(gsd_writer)
    sim.operations.integrator = integrator
    sim.operations.computes.append(thermodynamic_properties)
    
    try:
        # Loop until the simulation reaches the target timestep.
        while sim.timestep < end_step:
            # Run the simulation in chunks of 10,000 time steps.
            sim.run(min(10_000, end_step - sim.timestep))
            if (sim.device.communicator.walltime + sim.walltime >= HOOMD_RUN_WALLTIME_LIMIT):
                break
    
    finally:
        hoomd.write.GSD.write(state=sim.state,
                              mode='wb',
                              filename=job.fn('restart.gsd'))

        job.document['timestep'] = sim.timestep
    
    job.document['equilibriate'] = (job.document.get('timestep', 0) - job.document['compressed_step'] >= N_EQUIL_STEPS)
    