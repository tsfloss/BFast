import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec, NamedSharding, Mesh
from jax.experimental import mesh_utils
from .sharded_fft import jitted_irfftn, jitted_rfftn

def get_kmesh(res, sharding=None):
    kx = jnp.fft.fftfreq(res, 1/res)
    kz = jnp.fft.rfftfreq(res, 1/res)
    kvec = jnp.array(jnp.meshgrid(kx, kx, kz, indexing='ij'))
    kmag = jnp.sqrt((kvec**2).sum(0))
    if sharding is not None: 
        kmag = jax.device_put(kmag, sharding)
    return kmag

def shard_3D_array(array):
    device_mesh = Mesh(jax.devices(), axis_names=('gpus',))
    sharding = NamedSharding(device_mesh, PartitionSpec(None, 'gpus'))
    return jax.device_put(array, sharding)

def get_ffts(dim, sharding=None):
    if sharding is not None and dim==3:
        device_mesh = sharding.mesh
        rfftn = lambda x: jitted_rfftn(device_mesh)(x)
        irfftn_batch = lambda x: jnp.array([jitted_irfftn(device_mesh)(i) for i in x])
        irfftn = lambda x: jitted_irfftn(device_mesh)(x)
    else:
        fft_axes = tuple(range(-dim,0))
        rfftn = lambda x: jnp.fft.rfftn(x, axes=fft_axes)
        irfftn_batch = lambda x: jnp.array([jnp.fft.irfftn(i, axes=fft_axes) for i in x])
        irfftn = lambda x: jnp.fft.irfftn(x, axes=fft_axes)
    return rfftn, irfftn, irfftn_batch