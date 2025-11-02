import jax
import jax.numpy as jnp
import numpy as np
import BFast
from BFast.core.utils import shard_3D_array

jax.distributed.initialize()

res = 1024
shape = (res,)*3
# np.random.seed(42)
# field = np.random.normal(size=shape)
field = np.zeros(shape)

sharded_field = shard_3D_array(field)
print(sharded_field.addressable_shards[0].data.shape)

b_bins = jnp.arange(1,59)
B_info = BFast.get_triangles(b_bins, False)
print(B_info['bin_edges'].shape[0]-1, B_info['triangle_centers'].shape)

result = BFast.Bk.jit(sharded_field, **B_info, fast=False, compute_norm=True, sharding=sharded_field.sharding)
print(result)