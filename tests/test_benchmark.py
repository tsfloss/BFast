import os
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest
from jax import NamedSharding, P
from pytest_jax_bench import JaxBench

from BFast.core.powerspectrum import powerspectrum
from BFast.core.bispectrum import get_triangles, bispectrum
from shared_test_setup import shared_test_setup
from BFast.core.utils import shard_3D_array

use_gpu=os.environ.get('USE_GPU',"false")=="true"
if not use_gpu:
    shared_test_setup()

@pytest.mark.parametrize("sharded", [False, True])
def test_powerspectrum_sharded_bench(request, sharded):
    dim = 3
    res = 64
    boxsize = 1000.
    mas_order = 2
    multipole_axis = 0

    field = jax.random.normal(jax.random.PRNGKey(2), (res,) * dim)
    # input is 2MB (with double precision)
    if sharded:
        field = shard_3D_array(field)
    bin_edges = jnp.arange(1, res // 2 + 1)

    def bench_func(field, boxsize, sharding):
        return powerspectrum(field, boxsize, bin_edges, mas_order=mas_order, multipole_axis=multipole_axis, sharding=sharding)

    jitted_bench = jax.jit(bench_func, static_argnames=["boxsize", "sharding"])

    comp = jitted_bench.lower(field, boxsize, field.sharding).compile()
    hlo = comp.as_text()
    assert "all-gather" not in hlo

    jb = JaxBench(request, jit_rounds=5, jit_warmup=1, eager_rounds=0, eager_warmup=0)
    jb.measure(fn=bench_func, fn_jit=jitted_bench, field=field, boxsize=boxsize, sharding=field.sharding)

@pytest.mark.parametrize("sharded", [False, True])
def test_bispectrum_sharded_bench(request, sharded):
    dim = 3
    res = 64
    boxsize = 1000.
    mas_order = 2

    field = jax.random.normal(jax.random.PRNGKey(2), (res,) * dim)
    # input is 2MB (with double precision)
    if sharded:
        field = shard_3D_array(field)
    bin_edges = jnp.arange(1, res // 3 + 1)

    B_info = get_triangles(bin_edges, open_triangles=True)

    def bench_func(field, boxsize, B_info, sharding):
        B_norm = bispectrum(field, boxsize, **B_info, mas_order=mas_order, compute_norm=True, sharding=sharding)
        B_field = bispectrum(field, boxsize, **B_info, mas_order=mas_order, compute_norm=False, sharding=sharding)
        B_field['Bk'] /= B_norm['Bk']
        return B_field

    jitted_bench = jax.jit(bench_func, static_argnames=["boxsize", "sharding"])

    comp = jitted_bench.lower(field, boxsize, B_info, field.sharding).compile()
    hlo = comp.as_text()
    assert "all-gather" not in hlo

    jb = JaxBench(request, jit_rounds=5, jit_warmup=1, eager_rounds=0, eager_warmup=0)
    jb.measure(fn=bench_func, fn_jit=jitted_bench, field=field, boxsize=boxsize, B_info=B_info, sharding=field.sharding)

def test_benchmark_just_triangles(request):
    jb = JaxBench(request, jit_rounds=0, jit_warmup=0, eager_rounds=10, eager_warmup=1)
    res = 64

    bin_edges = jnp.arange(1, res // 3 + 1)

    jb.measure(fn=get_triangles, bin_edges=bin_edges)
