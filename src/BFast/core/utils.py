import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, NamedSharding, Mesh

from .sharded_fft import irfftn_unjitted, rfftn_unjitted


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
        jitted_rfftn = jax.jit(
            rfftn_unjitted,
            # donate_argnums=0,  # doesn't help
            # in_shardings=sharding,
            out_shardings=sharding
        )
        jitted_irfftn = jax.jit(
            irfftn_unjitted,
            # donate_argnums=0,  # doesn't help
            # in_shardings=sharding,
            out_shardings=sharding
        )
        rfftn = jitted_rfftn
        irfftn = jitted_irfftn
    else:
        fft_axes = tuple(range(-dim,0))
        rfftn = lambda x: jnp.fft.rfftn(x, axes=fft_axes)
        irfftn = lambda x: jnp.fft.irfftn(x, axes=fft_axes)
    return rfftn, irfftn

def get_fourier_tuple(dim, x,z):
    fourier_tuple = list((x,)*(dim-1))
    fourier_tuple.append(z)
    fourier_tuple = tuple(fourier_tuple)
    return fourier_tuple

def get_mas_kernel(mas_order, dim, res):
    kmesh = get_kmesh(dim, res)
    return (jnp.sinc(kmesh/res)**(-mas_order)).prod(0)

def bin_field(field_k, kmag, bin_low, bin_high, irfftn):
    return irfftn((kmag >= bin_low) * (kmag < bin_high)*field_k)
