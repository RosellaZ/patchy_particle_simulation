import jax.numpy as jnp
import numpy as onp
from jax import random
from jax import jit
from jax import lax
from jax import vmap
from jax.config import config
config.update("jax_enable_x64", True)
import time
import pickle
from jax_md import space, energy, simulate, rigid_body
import os
import sys
from pathlib import Path

##### parameters ######
N = int(int(sys.argv[1]) * 8 + 12)
print(f"number of particle = {N}")
num_density = 0.4
dim = 2
num_patch = 2
center_particle_rad = 1.0 # radius of the central particle

r_cutoff = 1.0

max_energy_patch = 20.0
patch_energies = jnp.array([0.0, max_energy_patch, 0.0])
max_energy_center = 10000.0

get_box_size = lambda phi, N, rad: jnp.sqrt( N * jnp.pi * rad**2 / phi)
box_size = get_box_size(num_density, N, center_particle_rad)

dt = 1e-3
kT = 1.0

key = random.PRNGKey(0)
ensemble_size = 50
loop_batch = 4
key, split = random.split(key)
sim_keys = random.split(split, ensemble_size)

n_steps = 40

##### forward simulation #####
@jit
def thetas_to_shape(thetas, radius = 1.0):
  # map array of thetas to a RigidPointUnion object
  patch_positions = jnp.zeros((len(thetas), 2), dtype = jnp.float64)
  patch_positions = patch_positions.at[:,0].set(radius * jnp.cos(thetas))
  patch_positions = patch_positions.at[:,1].set(radius * jnp.sin(thetas))
  positions = jnp.concatenate((jnp.array([[0.0, 0.0]]), patch_positions), axis = 0)
  species = jnp.arange(len(thetas) + 1)
  mass = jnp.zeros(len(thetas) + 1)
  mass = mass.at[0].set(1.0)
  shape = rigid_body.point_union_shape(positions, mass).set(point_species = species)
  return shape

def gen_init_pos(N, box_size, key):
  newkey, split = random.split(key)
  pos = random.uniform(newkey, (N, dim), maxval = box_size, dtype=jnp.float64)
  ort = random.uniform(split, (N, ), minval = -jnp.pi, maxval = jnp.pi, dtype=jnp.float64)
  body = rigid_body.RigidBody(pos, ort)
  return body
v_gen_init_pos = jit(vmap(gen_init_pos, (None, None, 0)), static_argnums = 0)

@jit
def sys_loss(center, orient, gamma = 0.1, delta = 0.1, clip_val = 1):
  displacement, shift = space.periodic(box_size)

  # first component: map distance
  vdisp = space.map_product(displacement)
  ds = space.distance(vdisp(center, center))

  # second component: map the difference between orientation (should be pi/2)
  diff_vec = lambda vec_1, vec_2 : vec_1 - vec_2
  vdis_vec = space.map_product(diff_vec)
  dtheta = vdis_vec(orient, orient)
  dtheta = jnp.mod(dtheta+jnp.pi, 2*jnp.pi) - jnp.pi

  # third component: map the difference between the angle between the egde and the orientation (should be pi/4)
  dss=vdisp(center, center)
  delta_xy = jnp.transpose(jnp.reshape(dss, (-1,2)))
  diff_angle = jnp.reshape(jnp.arctan2(delta_xy[1], delta_xy[0]), (len(center), len(center))) - orient[:, None]
  diff_angle = jnp.mod(diff_angle+jnp.pi, 2*jnp.pi) - jnp.pi

  loss_list = (ds - 2)**2 + gamma * (abs(dtheta) - 0.5 * jnp.pi)**2 + delta * (abs(diff_angle) - jnp.pi*0.25)**2
  loss_list = loss_list[~onp.eye(loss_list.shape[0],dtype=bool)].reshape(loss_list.shape[0],-1)
  loss = jnp.sum(jnp.sort(jnp.clip(loss_list, None, clip_val))[:,:2])
  return loss

def run_simulation_patchy_particle(theta, energy_patch, key, num_steps, init_pos, max_energy_center = 10000.0, kT = 1.0, r_cutoff = 1.0):
  displacement_fn, shift_fn = space.periodic(box_size)

  assert len(energy_patch) == int((1+num_patch)*num_patch/2), "length of the interaction matrix does not match the number of patches!"
  ind_i, ind_j = jnp.triu_indices(num_patch)
  eng_m = jnp.zeros((num_patch, num_patch))
  eng_m = eng_m.at[ind_i, ind_j].set(energy_patch)
  eng_patch = eng_m.at[ind_j, ind_i].set(energy_patch)

  morse_eps = jnp.pad(eng_patch, pad_width = (1, 0)) # define energy range
  pair_energy_morse = energy.morse_pair(displacement_fn, species = 1+num_patch, sigma = 0.0, epsilon = morse_eps, alpha = 9.0, r_cutoff = r_cutoff)

  # centers interact via soft sphere potential
  soft_sphere_eps = jnp.zeros((1+num_patch, 1+num_patch))
  soft_sphere_eps = soft_sphere_eps.at[0, 0].set(max_energy_center) # define energy range
  pair_energy_soft_sphere = energy.soft_sphere_pair(displacement_fn, species = 1+num_patch, sigma = center_particle_rad*2, epsilon = soft_sphere_eps)

  pair_energy_fn = lambda R, **kwargs: pair_energy_morse(R, **kwargs) + pair_energy_soft_sphere(R, **kwargs)

  thetas = jnp.array([-0.5*theta, 0.5*theta], dtype=jnp.float64)
  shape = thetas_to_shape(thetas, center_particle_rad)

  energy_fn = rigid_body.point_energy(pair_energy_fn, shape)

  init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
  
  key, split = random.split(key)
  state = init_fn(split, init_pos, mass = shape.mass())

  apply_fn = jit(apply_fn)
  do_step = lambda state, t: (apply_fn(state), [sys_loss(state.position.center, state.position.orientation), energy_fn(state.position)])
  do_step = jit(do_step)
  state, [losss, energies] = lax.scan(do_step, state, jnp.arange(num_steps))
  return state, losss, energies

run_simulation_patchy_particle = jit(run_simulation_patchy_particle, static_argnums=3)
v_sim = jit(vmap(run_simulation_patchy_particle, (None, None, 0, None, 0, None, None, None)), static_argnums = 3)


##### run simulation and store results #####
# DIR_NAME = '/content/drive/MyDrive/Simulation_Results/' + 'square' + '_kT{}_nparticle{}_nsteps{}_batchsize{}'.format(kT, N, n_steps, ensemble_size)
DIR_NAME = '../Simulation_Results/' + 'square' + '_kT{}_nparticle{}_nsteps{}_batchsize{}'.format(kT, N, n_steps, ensemble_size)
p = Path(DIR_NAME)
if not p.exists():
  os.mkdir(DIR_NAME)

sweep_thetas = jit(vmap(v_sim, (0, None, None, None, None, None, None, None)), static_argnums = 3)
theta_list = jnp.linspace(75, 150, 6) *jnp.pi/180
init_poss = v_gen_init_pos(N, box_size, random.split(split, ensemble_size))

start = time.time()
state_afters, lossss, energies = sweep_thetas(theta_list, patch_energies, sim_keys, n_steps, init_poss, max_energy_center, kT, r_cutoff)
end = time.time()
duration = end - start
print(f"Sweeping {len(theta_list)} different thetas, each with ensemble size = {ensemble_size}, takes {duration} seconds in total")

loss_file = DIR_NAME + '/loss'
energy_file = DIR_NAME + '/energy'
center_file = DIR_NAME + '/center_pos'
orient_file = DIR_NAME + '/orient'
pickle.dump(lossss, open(loss_file, 'wb'))
pickle.dump(energies, open(energy_file, 'wb'))
pickle.dump(state_afters.position.center, open(center_file, 'wb'))
pickle.dump(state_afters.position.orientation, open(orient_file, 'wb'))

