import jax.numpy as jnp
import numpy as onp
import jax
import functools
from jax import random
from jax import jit
from jax import lax
from jax import vmap
from jax.example_libraries import optimizers
from jax.config import config
config.update("jax_enable_x64", True)
import time
from jax_md import space, energy, simulate, rigid_body
import os
from os.path import exists
from pathlib import Path

"""Utilities for computing gradients."""
 
from typing import Callable
 
def _first_arg_partial(f, *args, **kwargs):
  def f_(x):
    return f(x, *args, **kwargs)
  return f_
 
def _split_and_pack_like(j, x):
  leaves, structure = jax.tree_flatten(x)
  sizes = [leaf.size for leaf in leaves]
  split = jnp.split(j, onp.cumsum(sizes), axis=-1)
  reshaped = [s.reshape(s.shape[:-1] + y.shape) for s, y in zip(split, leaves)]
  return jax.tree_unflatten(structure, reshaped)
 
def _tangents_like(x):
  eye = onp.eye(sum([leaf.size for leaf in jax.tree_leaves(x)]))
  return _split_and_pack_like(eye, x)
 
def value_and_jacfwd(f: Callable) -> Callable:
  """Returns a function that computes the Jacobian for the first argument, along with the value of the function."""
  def val_and_jac(*args, **kwargs):
    partial_f = _first_arg_partial(f, *args[1:], **kwargs)
    tangents = _tangents_like(args[0])
    jvp = functools.partial(jax.jvp, partial_f, (args[0],))
    y, jac = jax.vmap(jvp, out_axes=-1)((tangents,))
    y = jax.tree_map(lambda x: x[..., 0], y)
    jac = jax.tree_map(lambda j: _split_and_pack_like(j, args[0]), jac)
    return y, jac
  return val_and_jac

##### parameters #####
N = 20
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


n_steps = 40000 #40000
n_steps_opt = 1000 #1000
opt_steps = 100


##### simulation #####
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
  # generate an initial configuration of the particles (both x,y coordinate and angular psi)

  newkey, split = random.split(key)
  pos = random.uniform(newkey, (N, dim), maxval = box_size, dtype=jnp.float64)
  # ort = random.uniform(split, (N, ), minval = 0, maxval = 2*jnp.pi, dtype=jnp.float64)
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

def avg_loss(R_batched, O_batched):
  losses = vmap(sys_loss, (0,0))(R_batched, O_batched)
  return jnp.mean(losses)

def run_simulation_patchy_particle(theta, energy_patch, key, num_steps, init_pos, max_energy_center = 10000.0, kT = 1.0, r_cutoff = 1.0):
  # note here the init_pos should include both the physical coordination (x,y) of each particle, and the angular position of the particle, psi

  displacement_fn, shift_fn = space.periodic(box_size)

  # patches interact via morse potential
  # eng_patch = jnp.array([[0.0, max_energy_patch], [max_energy_patch, 0.0]], dtype=jnp.float64)
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

  thetas = jnp.array([0.0, theta], dtype=jnp.float64)
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

##### optimization #####

def get_mean_loss(theta, sim_keys, num_steps_opt, init_poss):
  states, losss, energies = v_sim(theta, patch_energies, sim_keys, num_steps_opt, init_poss, max_energy_center, kT, r_cutoff)
  mean_loss = avg_loss(states.position.center, states.position.orientation)
  return mean_loss

get_mean_loss = jit(value_and_jacfwd(jit(get_mean_loss, static_argnums=2)), static_argnums=2)

OPT_DIR_NAME = '../Simulation_Results/' + 'square_Opt_kT{}_nparticle{}_nsteps{}_noptsteps{}_nopt{}_batchsize{}'.format(kT, N, n_steps, n_steps_opt, opt_steps, ensemble_size)
# OPT_DIR_NAME = '/content/drive/MyDrive/Simulation_Results/' + 'square_Opt_kT{}_nparticle{}_nsteps{}_noptsteps{}_nopt{}_batchsize{}'.format(kT, N, n_steps, n_steps_opt, opt_steps, ensemble_size)
p = Path(OPT_DIR_NAME)
if not p.exists():
  os.mkdir(OPT_DIR_NAME)

def optimization(input_params, opt_steps, key, learning_rate = 0.1, resume = False):
  # define learning rate function
  ind = int(opt_steps/3)
  learning_rate_schedule = jnp.ones(opt_steps) * learning_rate
  learning_rate_schedule = learning_rate_schedule.at[ind:2*ind].set(learning_rate * 0.5)
  learning_rate_schedule = learning_rate_schedule.at[2*ind:].set(learning_rate * 0.1)
  learning_rate_fn = lambda i: learning_rate_schedule[i]
  opt_init, opt_update, get_params = optimizers.adam(step_size = learning_rate_fn)

  loss_file = OPT_DIR_NAME + '/loss' + str(learning_rate) + '.txt'
  grad_file = OPT_DIR_NAME + '/grad' + str(learning_rate) + '.txt'
  param_file = OPT_DIR_NAME + '/param' + str(learning_rate) + '.txt'
  
  # Check if these files exist, if yes delete them
  def clear_files(filenames):
    for fn in filenames:
      if exists(fn):
        os.remove(fn)
  if not resume:
    clear_files([loss_file, grad_file, param_file])

  opt_state = opt_init(input_params)

  def step(i, opt_state, key):
    params = get_params(opt_state)
    loss = 0
    # grad = []
    grad = 0
    for j in range(loop_batch):
      key, pos_key, split = random.split(key, 3)
      key_batches = random.split(split, ensemble_size)
      init_positions = v_gen_init_pos(N, box_size, random.split(pos_key, ensemble_size))
      state_after, _, _ = v_sim(params, patch_energies, key_batches, n_steps, init_positions, max_energy_center, kT, r_cutoff)
      final_positions = state_after.position
      key, split = random.split(key)
      l, g = get_mean_loss(params, random.split(split, ensemble_size), n_steps_opt, final_positions)
      loss += l
      grad += g
      # grad += [g]
    # grad = jnp.mean(jnp.array(grad), axis = 0)
    loss = loss/loop_batch
    grad = grad/loop_batch
    opt_state = opt_update(i, grad, opt_state)

    with open(loss_file, 'a') as out1:
      out1.write("{}".format(loss)+'\n')
    
    with open(grad_file, 'a') as out2:
      out2.write("{}".format(grad)+'\n')
    
    with open(param_file, 'a') as out3:
      # temp_param = onp.array(params).tolist()
      # separator = ', '
      out3.write("{}".format(params)+'\n')
      # out3.write(separator.join(['{}'.format(temp) for temp in temp_param]) + '\n')

    return opt_state, [loss, grad]
  
  min_loss_params = input_params
  min_loss = 1e6

  print(f"I'll start optimization iteration now! The learning rate is {learning_rate}.")

  loss_array = jnp.ones(opt_steps)

  for i in range(opt_steps):
    key, split = random.split(key)
    start = time.time()
    new_opt_state, [loss, grad] = step(i, opt_state, split)
    end = time.time()
    loss_array = loss_array.at[i].set(loss)
    print(f"@Opt step {i}")
    if loss < min_loss:
      min_loss = loss
      min_loss_params = get_params(opt_state)
      print("YEAH!! The loss is decreasing!")
    print(f"This step takes {end - start} seconds") 
    print(f"the loss is {loss}, the gradients are {grad}, and the params is {get_params(opt_state)}\n")
    opt_state = new_opt_state
  
  return loss_array, min_loss_params

key = random.PRNGKey(129)
key, split = random.split(key)
params = random.uniform(split, maxval = jnp.pi)
print(params)

# start = time.time()
# key, split = random.split(key)
# loss_array, min_loss_params = optimization(params, opt_steps, split, learning_rate = 0.1, resume = False)
# end = time.time()
# duration = end - start
# print(f"Learning rate = 0.1, Optimization {opt_steps} steps, simulation steps {n_steps}, sim_opt_steps {n_steps_opt}, with ensemble size = {ensemble_size}, takes {duration} seconds in total")

previous_sim_param = onp.loadtxt(OPT_DIR_NAME + '/param' + str(0.01) + '.txt')
print(len(previous_sim_param))
print(previous_sim_param[-1])
min_loss_params = previous_sim_param[-1]

# start = time.time()
key, split = random.split(key)
# loss_array, min_loss_params = optimization(min_loss_params, opt_steps - len(previous_sim_param), split, learning_rate = 0.05, resume = True)
# end = time.time()
# duration = end - start
# print(f"Learning rate = 0.05, Optimization {opt_steps} steps, simulation steps {n_steps}, sim_opt_steps {n_steps_opt}, with ensemble size = {ensemble_size}, takes {duration} seconds in total")

start = time.time()
key, split = random.split(key)
loss_array, min_loss_params = optimization(min_loss_params, opt_steps - len(previous_sim_param), split, learning_rate = 0.01, resume = True)
end = time.time()
duration = end - start
print(f"Learning rate = 0.01, Optimization {opt_steps} steps, simulation steps {n_steps}, sim_opt_steps {n_steps_opt}, with ensemble size = {ensemble_size}, takes {duration} seconds in total")

