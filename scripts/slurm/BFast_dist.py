import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
import jax.numpy as jnp
import numpy as np
import BFast
from BFast.core.jax_utils import show_hlo_info

jax.distributed.initialize() # on a slurm-based cluster this should find all nodes related to the job

res = 1536
boxsize = 1000.
shape = (res,)*3
sharding = BFast.get_sharding()

if jax.process_index()==0: print(sharding)

def sharded_random_field():
    field = jax.random.normal(jax.random.PRNGKey(0), shape=shape)
    return field

sharded_random_field.jit = jax.jit(sharded_random_field, out_shardings=sharding)
if jax.process_index()==0: show_hlo_info(sharded_random_field.jit)

sharded_field = sharded_random_field.jit()
if jax.process_index()==0: print(sharded_field.addressable_shards[0].data.shape)

b_bins = jnp.arange(1,res//3,15)
B_info = BFast.get_triangles(b_bins, False)
if jax.process_index()==0: print(B_info['bin_edges'].shape[0]-1, B_info['triangle_centers'].shape[0])

if jax.process_index()==0: show_hlo_info(BFast.bispectrum.jit, sharded_field, boxsize, **B_info, fast=True, compute_norm=False, sharding=sharded_field.sharding)

result = BFast.bispectrum.jit(sharded_field, boxsize, **B_info, fast=True, compute_norm=False, sharding=sharded_field.sharding)
if jax.process_index()==0: print(result)