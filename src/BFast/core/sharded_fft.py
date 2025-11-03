## Author: Lukas Winkler (University of Vienna), 2025

from typing import Callable

import jax
from jax import jit
from jax.experimental import mesh_utils
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh


def fft_partitioner(fft_func: Callable[[jax.Array], jax.Array], partition_spec: P, sharding_rule=None):
    # @jax.custom_batching.sequential_vmap
    @custom_partitioning
    def func(x):
        return fft_func(x)

    def supported_sharding(sharding, shape):
        if sharding is None:
            print("infer_sharding_from_operands got empty object")
            num_gpus = jax.device_count()
            devices = mesh_utils.create_device_mesh((num_gpus,))
            mesh = Mesh(devices, axis_names=('gpus',))

            return NamedSharding(mesh, partition_spec)
        return NamedSharding(sharding.mesh, partition_spec)

    def partition(mesh, arg_shapes, result_shape):
        # result_shardings = jax.tree.map(lambda x: x.sharding, result_shape)
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return mesh, fft_func, supported_sharding(arg_shardings[0], arg_shapes[0]), (
            supported_sharding(arg_shardings[0], arg_shapes[0]),)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        arg_shardings = jax.tree.map(lambda x: x.sharding, arg_shapes)
        return supported_sharding(arg_shardings[0], arg_shapes[0])

    func.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule=sharding_rule
    )
    return func


def fftn_XY(x):
    return jax.numpy.fft.fftn(x, axes=[-3, -2])


def fftn_XY_norm_forward(x):
    # return jax.numpy.fft.fftn(x, axes=[-3, -2], norm='forward')
    return jax.numpy.fft.fftn(x, axes=[-3, -2]) / (x.shape[-3] * x.shape[-2])


def rfft_Z(x):
    return jax.numpy.fft.rfft(x, axis=-1)


def rfft_Z_norm_forward(x):
    # return jax.numpy.fft.rfft(x, axis=-1, norm='forward')
    return jax.numpy.fft.rfft(x, axis=-1) / (x.shape[-1])


def ifftn_XY(x):
    return jax.numpy.fft.ifftn(x, axes=[-3, -2])


def ifftn_XY_norm_forward(x):
    # return jax.numpy.fft.ifftn(x, axes=[-3, -2],norm="forward")
    return jax.numpy.fft.ifftn(x, axes=[-3, -2]) * (x.shape[-3] * x.shape[-2])


def irfft_Z(x):
    return jax.numpy.fft.irfft(x, axis=-1)


def irfft_Z_norm_forward(x):
    # return jax.numpy.fft.irfft(x, axis=-1,norm='forward')
    return jax.numpy.fft.irfft(x, axis=-1) * (2 * (x.shape[-1] - 1))


fftn_XY = fft_partitioner(fftn_XY, P(None, None, "gpus"), sharding_rule="x y z -> x y z")
fftn_XY_norm_forward = fft_partitioner(fftn_XY_norm_forward, P(None, None, "gpus"), sharding_rule="x y z -> x y z")
rfft_Z = fft_partitioner(rfft_Z, P(None, "gpus"), sharding_rule="x y z -> x y z_new")
rfft_Z_norm_forward = fft_partitioner(rfft_Z_norm_forward, P(None, "gpus"), sharding_rule="x y z -> x y z_new")
ifftn_XY = fft_partitioner(ifftn_XY, P(None, None, "gpus"), sharding_rule="x y z -> x y z")
ifftn_XY_norm_forward = fft_partitioner(ifftn_XY_norm_forward, P(None, None, "gpus"), sharding_rule="x y z -> x y z")
irfft_Z = fft_partitioner(irfft_Z, P(None, "gpus"), sharding_rule="x y z -> x y z_new")
irfft_Z_norm_forward = fft_partitioner(irfft_Z_norm_forward, P(None, "gpus"), sharding_rule="x y z -> x y z_new")

### VJP for rfftn

rfft_Z_with_vjp = jax.custom_vjp(rfft_Z)


def _rfft_Z_fwd(x):
    n = x.shape[-1]
    assert n % 2 == 0
    # print("_rfft_Z_fwd", x.shape, x.dtype)
    # return jax.numpy.fft.rfft(x, axis=-1), n
    return rfft_Z(x), n


def _rfft_Z_bwd(_, g):
    g = g.at[..., 1:-1].multiply(0.5)
    # print("_rfft_Z_bwd", g.shape, g.dtype)
    # output=jax.numpy.fft.irfft(g, axis=-1, norm="forward")
    output = irfft_Z_norm_forward(g)
    return (output,)


rfft_Z_with_vjp.defvjp(_rfft_Z_fwd, _rfft_Z_bwd)

fftn_XY_with_vjp = jax.custom_vjp(fftn_XY)


def _fftn_XY_fwd(x):
    # print("_fftn_XY_fwd", x.shape, x.dtype)
    return fftn_XY(x), None  # Nothing needs to be saved
    # return jax.numpy.fft.fftn(x, axes=[-3, -2]), None


def _fftn_XY_bwd(_, g):
    # print("_fftn_XY_bwd", g.shape, g.dtype)
    return (ifftn_XY_norm_forward(g.conj()),)
    # return (jax.numpy.fft.ifftn(g.conj(), axes=[-3, -2], norm="forward"),)
    # return (jax.numpy.fft.ifftn(g.conj(), axes=[-3, -2])*(g.shape[-3]*g.shape[-2]),)


fftn_XY_with_vjp.defvjp(_fftn_XY_fwd, _fftn_XY_bwd)

### VJP for irfftn

irfft_Z_with_vjp = jax.custom_vjp(irfft_Z)


def _irfft_Z_fwd(x):
    # print("_irfft_Z_fwd", x.shape, x.dtype)
    # return jax.numpy.fft.rfft(x, axis=-1), n
    return irfft_Z(x), None


def _irfft_Z_bwd(_, g):
    # print("_irfft_Z_bwd", g.shape, g.dtype)
    # output=jax.numpy.fft.irfft(g, axis=-1, norm="forward")
    output = rfft_Z_norm_forward(g)
    return (output,)


irfft_Z_with_vjp.defvjp(_irfft_Z_fwd, _irfft_Z_bwd)

ifftn_XY_with_vjp = jax.custom_vjp(ifftn_XY)


def _ifftn_XY_fwd(x):
    # print("_ifftn_XY_fwd", x.shape, x.dtype)
    return ifftn_XY(x), None  # Nothing needs to be saved
    # return jax.numpy.fft.fftn(x, axes=[-3, -2]), None


def _ifftn_XY_bwd(_, g):
    g = g.at[..., 1:-1].multiply(2)
    # print("_ifftn_XY_bwd", g.shape, g.dtype)
    return (fftn_XY_norm_forward(g).conj(),)
    # return (jax.numpy.fft.ifftn(g.conj(), axes=[-3, -2], norm="forward"),)
    # return (jax.numpy.fft.ifftn(g.conj(), axes=[-3, -2])*(g.shape[-3]*g.shape[-2]),)


ifftn_XY_with_vjp.defvjp(_ifftn_XY_fwd, _ifftn_XY_bwd)



def _rfftn(x):
    x = rfft_Z_with_vjp(x)
    x = fftn_XY_with_vjp(x)
    return x


def _irfftn(x):
    x = ifftn_XY_with_vjp(x)
    x = irfft_Z_with_vjp(x)
    return x


def jitted_rfftn(device_mesh: Mesh):
    sharding = NamedSharding(device_mesh, P(None, "gpus"))
    return jit(
        _rfftn,
        # donate_argnums=0,  # doesn't help
        in_shardings=sharding,
        out_shardings=sharding
    )


def jitted_irfftn(device_mesh: Mesh):
    sharding = NamedSharding(device_mesh, P(None, "gpus"))
    return jit(
        _irfftn,
        # donate_argnums=0,  # doesn't help
        # in_shardings=sharding,
        out_shardings=sharding
    )