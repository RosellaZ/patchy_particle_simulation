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
import pickle
config.update("jax_debug_nans", True)
from jax_md import space, energy, minimize, quantity, simulate, rigid_body
import os
import sys
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

N = 20
num_density = 0.3
dim = 2
num_patch = 3
center_particle_rad = 1.0 # radius of the central particle

def thetas_to_shape(thetas, radius = 1.0, center_mass = 1.0, label = 1):
      # map array of thetas to a RigidPointUnion object
  patch_positions = jnp.zeros((len(thetas), 2), dtype = jnp.float64)
  patch_positions = patch_positions.at[:,0].set(radius * jnp.cos(thetas))
  patch_positions = patch_positions.at[:,1].set(radius * jnp.sin(thetas))
  positions = jnp.concatenate((jnp.array([[0.0, 0.0]]), patch_positions), axis = 0)
  if label == 1:
    species = jnp.arange(len(thetas) + 1)
  elif label == 2:
    species = jnp.arange(len(thetas) + 1, 2*len(thetas) + 2)
  else:
    return "Label of the particle is invalid!"
  mass = jnp.zeros(len(thetas) + 1)
  mass = mass.at[0].set(center_mass)
  shape = rigid_body.point_union_shape(positions, mass).set(point_species = species)
  return shape

def thetas_to_shape_seed(theta, radius = 1.0, center_mass = 1.0):
  positions = jnp.array([[0.0, radius], [0.0, -radius], [radius * jnp.sin(theta), radius *(1- jnp.cos(theta))], [radius * jnp.sin(theta), -radius *(1- jnp.cos(theta))]])
  species = jnp.array([4, 4, 2, 1])
  mass = jnp.array([center_mass, center_mass, 0.0, 0.0])
  shape = rigid_body.RigidPointUnion(points=positions,
                          masses=mass,
                          point_count=jnp.array([len(positions)]),
                          point_offset=jnp.array([0])).set(point_species = species)
  return shape

def gen_init_pos(N, box_size, key):
  # generate an initial configuration of the particles (both x,y coordinate and angular psi)

  newkey, split = random.split(key)
  pos = random.uniform(newkey, (N, dim), maxval = box_size, dtype=jnp.float64)
  pos = pos.at[0].set(jnp.array([0.5*box_size, 0.5*box_size], dtype=jnp.float64))

  ort = random.uniform(split, (N, ), minval = -jnp.pi, maxval = jnp.pi, dtype=jnp.float64)
  ort = ort.at[0].set(0.0)

  body = rigid_body.RigidBody(pos, ort)
  return body
v_gen_init_pos = jit(vmap(gen_init_pos, (None, None, 0)), static_argnums = 0)

# parameters
thetas = jnp.array([-jnp.pi*0.5, 0.5*jnp.pi, 0], dtype=jnp.float64)

shape = rigid_body.concatenate_shapes(thetas_to_shape_seed(jnp.pi*0.5, center_particle_rad), thetas_to_shape(thetas, center_particle_rad, label = 1), thetas_to_shape(thetas, center_particle_rad, label = 2))

alpha_morse = 7.0
alpha_soft = 2.0
r_cutoff = 1.0
threshold = 0.202

weak_e = 4.0
strong_e = 20.0
max_energy_center = 10000.0

get_box_size = lambda phi, N, rad: jnp.sqrt( N * jnp.pi * rad**2 / phi)
box_size = get_box_size(num_density, N, center_particle_rad)

dt = 1e-3
kT = 1.0

ensemble_size = 50
loop_batch = 4

n_steps = 40000 #40000
n_steps_opt = 1000 #1000
opt_steps = 100

lr_list = jnp.array([0.5, 0.1, 0.05, 0.01, 0.005])
ind = int(sys.argv[1])

from typing import Tuple, Optional, Callable

from jax_md import util
from jax import tree_map

Array = util.Array

RigidBody = rigid_body.RigidBody
RigidPointUnion = rigid_body.RigidPointUnion
transform = rigid_body.transform
 
### SLIGHTLY MODIFIED RIGID BODY CODE.
 
def union_to_points(body: RigidBody,
                    shape: RigidPointUnion,
                    shape_species: Optional[onp.ndarray]=None,
                    **kwargs,
                    ) -> Tuple[Array, Optional[Array]]:
  """Transforms points in a RigidPointUnion to world space."""
  if shape_species is None:
    position = vmap(transform, (0, None))(body, shape)
    point_species = shape.point_species
    body_id = jnp.arange(len(body.center))
    body_id = jnp.broadcast_to(body_id[:, None], position.shape[:-1])
    body_id = jnp.reshape(body_id, (-1,))
    if point_species is not None:
      point_species = shape.point_species[None, :]
      point_species = jnp.broadcast_to(point_species, position.shape[:-1])
      point_species = jnp.reshape(point_species, (-1,))
    position = jnp.reshape(position, (-1, position.shape[-1]))
    return position, point_species, body_id
  elif isinstance(shape_species, onp.ndarray):
    shape_species_types = onp.unique(shape_species)
    shape = tree_map(lambda x: onp.array(x), shape)
 
    point_position = []
    point_species = []
    body_ids = []
 
    body_id = jnp.arange(len(body.center))

    for s in shape_species_types:
      cur_shape = shape[s]
      pos = vmap(transform, (0, None))(body[shape_species == s], cur_shape)
 
      this_body_id = body_id[shape_species == s]
      this_body_id = jnp.broadcast_to(this_body_id[:, None], pos.shape[:-1])
      body_ids += [jnp.reshape(this_body_id, (-1,))]
 
      ps = cur_shape.point_species
      if ps is not None:
        ps = cur_shape.point_species[None, :]
        ps = jnp.broadcast_to(ps, pos.shape[:-1])
        point_species += [jnp.reshape(ps, (-1,))]
 
      pos = jnp.reshape(pos, (-1, pos.shape[-1]))
      point_position += [pos]
    point_position = jnp.concatenate(point_position)
    point_species = jnp.concatenate(point_species) if point_species else None
    body_ids = jnp.concatenate(body_ids)
    return point_position, point_species, body_ids
  else:
    raise NotImplementedError('Shape species must either be None or of type '
                              'onp.ndarray since it must be specified ahead '
                              f'of compilation. Found {type(shape_species)}.')
 
# Energy Functions
 
 
def point_energy(energy_fn: Callable[..., Array],
                 shape: RigidPointUnion,
                 shape_species: Optional[onp.ndarray]=None
                 ) -> Callable[..., Array]:
  """Produces a RigidBody energy given a pointwise energy and a point union.
  This function takes takes a pointwise energy function that computes the
  energy of a set of particle positions along with a RigidPointUnion
  (optionally with shape species information) and produces a new energy
  function that computes the energy of a collection of rigid bodies.
  Args:
    energy_fn: An energy function that takes point positions and produces a
      scalar energy function.
    shape: A RigidPointUnion shape that contains one or more shapes defined as
      a union of point masses.
    shape_species: An optional array specifying the composition of the system
      in terms of shapes.
  Returns:
    An energy function that takes a `RigidBody` and produces a scalar energy
    energy.
  """
  def wrapped_energy_fn(body, **kwargs):
    pos, point_species, body_id = union_to_points(body, shape, shape_species)
    if point_species is None:
      return energy_fn(pos, **kwargs)
    return energy_fn(pos, shape = shape, species=point_species, body_id=body_id, **kwargs)
  return wrapped_energy_fn


################ Simulation #################
def run_simulation(key, num_steps, init_pos, init_species, params, threshold = 0.202, weak_e = 4.0, kT = 1.0, alpha_soft = 2.0, r_cutoff = 1.0):
  alpha_morse, strong_e = params
  displacement_fn, shift_fn = space.periodic(box_size)

  energy_patch0 = jnp.array([0.0, weak_e, 0.0, 0.0, 0.0, weak_e])
  energy_patch1 = jnp.array([0.0, strong_e, 0.0, 0.0, 0.0, strong_e])

  @jit
  def gen_morse_eps(energy_patch):
    ind_i, ind_j = jnp.triu_indices(num_patch)
    eng = jnp.zeros((num_patch, num_patch))
    eng = eng.at[ind_i, ind_j].set(energy_patch)
    eng_patch = eng.at[ind_j, ind_i].set(energy_patch)
    morse_eps_sub = jnp.pad(eng_patch, pad_width = (1, 0)) # define energy range
    return morse_eps_sub
  morse_eps = jnp.vstack((jnp.hstack((gen_morse_eps(energy_patch0), gen_morse_eps(energy_patch1))), jnp.hstack((gen_morse_eps(energy_patch1), gen_morse_eps(energy_patch1)))))
  pair_energy_morse = energy.morse_pair(displacement_fn, species = (1+num_patch)*2, sigma = 0.0, epsilon = morse_eps, alpha = alpha_morse, r_cutoff = r_cutoff)

  # centers interact via soft sphere potential
  soft_sphere_eps = jnp.zeros((1+num_patch, 1+num_patch))
  soft_sphere_eps = soft_sphere_eps.at[0, 0].set(max_energy_center) # define energy range
  soft_sphere_eps = jnp.vstack((jnp.hstack((soft_sphere_eps, soft_sphere_eps)), jnp.hstack((soft_sphere_eps, soft_sphere_eps))))
  pair_energy_soft_sphere = energy.soft_sphere_pair(displacement_fn, species = (1+num_patch)*2, sigma = center_particle_rad*2, alpha = alpha_soft, epsilon = soft_sphere_eps)

  pair_energy_fn = lambda R, **kwargs: pair_energy_morse(R, **kwargs) + pair_energy_soft_sphere(R, **kwargs)
  
  @jit
  def E(R, shape, body_type, species=None, body_id=None, **kwargs):
    ind = (body_type[body_id].reshape((-1,num_patch+1))*4 + jnp.arange(1+num_patch)).reshape((1,-1))[0]
    species = (shape.point_species)[ind]
    return pair_energy_fn(R, species=species, **kwargs)

  species0 = onp.where(onp.arange(N) < 1,0,1)
  energy_fn = point_energy(E, shape, shape_species = species0)

  @jit
  def update_species2(body, pre_species):
    seed = rigid_body.RigidBody(body.center[0], body.orientation[0])
    seed_pos = transform(seed, thetas_to_shape_seed(jnp.pi*0.5, center_particle_rad))
    body = rigid_body.RigidBody(body.center[1:,:], body.orientation[1:])
    vdisp = space.map_product(displacement_fn)

    middle = jnp.mean(seed_pos[:2], axis = 0)
    all_pos = vmap(transform, (0, None))(body, thetas_to_shape(thetas, center_particle_rad))
    all_pos0 = jnp.moveaxis(all_pos, 0, 1)
    patch1_pos = jnp.concatenate((seed_pos[-1][None,:], all_pos0[1,:,:]))
    patch2_pos = jnp.concatenate((seed_pos[-2][None,:], all_pos0[2,:,:]))
    patch3_pos = jnp.concatenate((middle[None,:], all_pos0[3,:,:]))
    center_pos = jnp.concatenate((seed_pos[:2], all_pos0[0,:,:]))

    ds21 = space.distance(vdisp(patch2_pos, patch1_pos))
    ds12 = space.distance(vdisp(patch1_pos, patch2_pos))
    ds3 = space.distance(vdisp(patch3_pos, patch3_pos))

    ds00 = abs(space.distance(vdisp(center_pos, center_pos))-2*jnp.sqrt(2)*center_particle_rad)
    ds0 = ds00[1:,1:]
    ds0 = ds0.at[:,0].set(jnp.where(ds00[1:,0]<ds00[1:,1], ds00[1:,0],ds00[1:,1]))

    temp = jnp.where(ds21 < ds12, ds21, ds12)

    def addition_species(mat):
      temp001 = jnp.where((pre_species == 1)[:, None] + jnp.zeros((N,N)), mat, 100000)
      temp002 = jnp.where((jnp.remainder(pre_species,2) == 0)[None,:] + jnp.zeros((N,N)), temp001, 100000)
      temp003 = 10000 - jnp.where(temp002 < threshold, 0, 10000)
      return jnp.clip(jnp.sum(temp003, -1), None, 1)

    final_species = pre_species + jnp.where(addition_species(temp) < addition_species(ds0), addition_species(temp), addition_species(ds0))
    return final_species

  init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, dt, kT)
  key, split = random.split(key)
  state = init_fn(split, init_pos, mass=shape.mass(jnp.where(jnp.arange(N) < 1, 0, 1)), body_type = init_species)
  apply_fn = jit(apply_fn)

  @jit
  def do_step(state_zipped, t):
    state = state_zipped[0]
    pre_species = state_zipped[1]
    species = update_species2(state.position, pre_species)
    # return [apply_fn(state, body_type = species), species], [state.position, species, energy_fn(state.position, body_type = species)]
    return [apply_fn(state, body_type = species), species], energy_fn(state.position, body_type = species)

  [state, species], energiess = lax.scan(do_step, [state, init_species], jnp.arange(num_steps))
  return state, species, energiess
run_simulation = jit(run_simulation, static_argnums = (1,))
v_sim = jit(vmap(run_simulation, (0, None, 0, 0, None, None, None, None, None, None)), static_argnums=(1,))

def get_mean_loss(params, sim_keys, num_steps_opt, init_poss, init_speciess):
  _, _, energiess = v_sim(sim_keys, num_steps_opt, init_poss, init_speciess, params, threshold, weak_e, kT, alpha_soft, r_cutoff)
  mean_loss = jnp.mean(energiess[:,-10:])/params[1]
  return mean_loss

get_mean_loss = jit(value_and_jacfwd(jit(get_mean_loss, static_argnums=2)), static_argnums=2)

OPT_DIR_NAME = '../Simulation_Results/' + 'opt_f_kT{}_N{}_nsteps{}_noptsteps{}_nopt{}_batchsize{}_lr{}'.format(kT, N, n_steps, n_steps_opt, opt_steps, ensemble_size * loop_batch, lr_list[ind])
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
  param_file = OPT_DIR_NAME + '/params' + str(learning_rate) + '.txt'
  
  def clip_gradient(g, clip=10000.0):
    return jnp.array(jnp.where(jnp.abs(g) > clip, jnp.sign(g)*clip, g)) 
  
  # Check if these files exist, if yes delete them
  def clear_files(filenames):
    for fn in filenames:
      if exists(fn):
        os.remove(fn)
  if not resume:
    clear_files([loss_file, grad_file, param_file])

  opt_state = opt_init(input_params)
  init_sps = jnp.repeat(onp.where(onp.arange(N) < 2,0,1)[jnp.newaxis,...], ensemble_size, axis=0)

  def step(i, opt_state, key):
    params = get_params(opt_state)
    loss = 0
    grad = []
    # grad = 0
    for j in range(loop_batch):
      key, pos_key, split = random.split(key, 3)
      key_batches = random.split(split, ensemble_size)
      init_positions = v_gen_init_pos(N, box_size, random.split(pos_key, ensemble_size))
      state_after, species_after, _ = v_sim(key_batches, n_steps, init_positions, init_sps, params, threshold, weak_e, kT, alpha_soft, r_cutoff)
      final_positions = state_after.position
      key, split = random.split(key)
      l, g = get_mean_loss(params, random.split(split, ensemble_size), n_steps_opt, final_positions, species_after)
      loss += l
      # grad += clip_gradient(g)
      grad += [clip_gradient(g)]
    grad = jnp.mean(jnp.array(grad), axis = 0)
    loss = loss/loop_batch
    # grad = grad/loop_batch
    opt_state = opt_update(i, grad, opt_state)

    with open(loss_file, 'a') as out1:
      out1.write("{}".format(loss)+'\n')
    
    with open(grad_file, 'a') as out2:
      temp_grad = onp.array(grad).tolist()
      separator = ', '
      out2.write(separator.join(['{}'.format(temp) for temp in temp_grad]) + '\n')
      # out2.write("{}".format(grad)+'\n')
    
    with open(param_file, 'a') as out3:
      temp_param = onp.array(params).tolist()
      separator = ', '
      # out3.write("{}".format(params)+'\n')
      out3.write(separator.join(['{}'.format(temp) for temp in temp_param]) + '\n')

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

key = random.PRNGKey(1021459)
key, split = random.split(key)
params = jnp.array([alpha_morse, strong_e])

start = time.time()
key, split = random.split(key)
loss_array, min_loss_params = optimization(params, opt_steps, split, learning_rate = lr_list[ind], resume = False)
end = time.time()
duration = end - start
print(f"Learning rate = lr_list[ind], Optimization {opt_steps} steps, simulation steps {n_steps}, sim_opt_steps {n_steps_opt}, with ensemble size = {ensemble_size}, takes {duration} seconds in total")
