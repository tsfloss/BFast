import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import BFast
from BFast.core.utils import shard_3D_array

dim = 3
res = 64
boxsize = 1000.
mas_order = 2
multipole_axis = 0
field = jax.random.normal(jax.random.PRNGKey(2),(res,)*dim, dtype=jnp.float32).astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) # jnp.float32 to make sure that the random field is the same in single and double precision modes
field = shard_3D_array(field)
sharding = field.sharding
bin_edges = jnp.arange(1,res//3+1)
B_info = BFast.get_triangles(bin_edges, open_triangles=True)

ref_triangles_onlyclosed = jnp.load(f"tests/reference_data/triangles_res{res}_onlyclosed.npz")
ref_triangles_openclosed = jnp.load(f"tests/reference_data/triangles_res{res}_openclosed.npz")
ref_Pk = jnp.load(f"tests/reference_data/Pk_dim{dim}_res{res}_mas{mas_order}_multipole{multipole_axis}.npz")
ref_Bk_norm = jnp.load(f"tests/reference_data/Bk_norm_PB_dim{dim}_res{res}_openclosed.npz")
ref_Bk_meas = jnp.load(f"tests/reference_data/Bk_measured_PB_dim{dim}_res{res}_mas{mas_order}_openclosed.npz")

def test_Pk():
    bin_edges = jnp.arange(1,res//2+1)
    result = BFast.Pk(field, boxsize, bin_edges, mas_order=mas_order, multipole_axis=multipole_axis, sharding=sharding)
    assert jnp.allclose(result['k'], ref_Pk['k'])
    assert jnp.allclose(result['norm'], ref_Pk['norm'])
    assert jnp.allclose(result['Pk0'], ref_Pk['Pk0'])
    assert jnp.allclose(result['Pk2'], ref_Pk['Pk2'])
    assert jnp.allclose(result['Pk4'], ref_Pk['Pk4'])

def test_Bk_triangles_onlyclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=False)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_onlyclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_onlyclosed['triangle_indices'])

def test_Bk_triangles_openclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=True)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_openclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_openclosed['triangle_indices'])

def test_Bk_norm_unjitted_fast():
    result = BFast.Bk(field, boxsize, **B_info, mas_order=mas_order, fast=True, only_B=False, compute_norm=True, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_norm['Bk'])

def test_Bk_norm_unjitted_slow():
    result = BFast.Bk(field, boxsize, **B_info, mas_order=mas_order, fast=False, only_B=False, compute_norm=True, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_norm['Bk'])

def test_Bk_norm_jitted_fast():
    result = BFast.Bk.jit(field, boxsize, **B_info, mas_order=mas_order, fast=True, only_B=False, compute_norm=True, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_norm['Bk'])

def test_Bk_norm_jitted_slow():
    result = BFast.Bk.jit(field, boxsize, **B_info, mas_order=mas_order, fast=False, only_B=False, compute_norm=True, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_norm['Bk'])

def test_Bk_meas_unjitted_fast():
    result = BFast.Bk(field, boxsize, **B_info, mas_order=mas_order, fast=True, only_B=False, compute_norm=False, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_meas['Bk'])

def test_Bk_meas_unjitted_slow():
    result = BFast.Bk(field, boxsize, **B_info, mas_order=mas_order, fast=False, only_B=False, compute_norm=False, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_meas['Bk'])

def test_Bk_meas_jitted_fast():
    result = BFast.Bk.jit(field, boxsize, **B_info, mas_order=mas_order, fast=True, only_B=False, compute_norm=False, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_meas['Bk'])

def test_Bk_meas_jitted_slow():
    result = BFast.Bk.jit(field, boxsize, **B_info, mas_order=mas_order, fast=False, only_B=False, compute_norm=False, sharding=sharding)
    assert jnp.allclose(result['Pk'], ref_Bk_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_Bk_meas['Bk'])