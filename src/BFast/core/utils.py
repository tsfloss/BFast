import jax.numpy as jnp
import jax
from jax.sharding import PartitionSpec, NamedSharding, Mesh
from jax.experimental import mesh_utils
from .sharded_fft import jitted_irfftn, jitted_rfftn

def get_kmesh(dim, res):
    kx = jnp.fft.fftfreq(res, 1/res)
    kz = jnp.fft.rfftfreq(res, 1/res)
    k_tuple = get_fourier_tuple(dim, kx, kz)
    kmesh = jnp.array(jnp.meshgrid(*k_tuple, indexing='ij'))
    return kmesh

def get_kmag(dim, res):
    kmesh = get_kmesh(dim, res)
    kmag = jnp.sqrt((kmesh**2).sum(0))
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
        # irfftn_batch = lambda x: jnp.array([jnp.fft.irfftn(i, axes=fft_axes) for i in x])
        irfftn_batch = lambda x: jnp.fft.irfftn(x, axes=fft_axes)
        irfftn = lambda x: jnp.fft.irfftn(x, axes=fft_axes)
    return rfftn, irfftn, irfftn_batch

def get_fourier_tuple(dim, x,z):
    fourier_tuple = list((x,)*(dim-1))
    fourier_tuple.append(z)
    fourier_tuple = tuple(fourier_tuple)
    return fourier_tuple

def get_mas_kernel(mas_order, dim, res):
    kmesh = get_kmesh(dim, res)
    return (jnp.sinc(kmesh/res)**(-mas_order)).prod(0)