import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import BFast

dim = 3
res = 64
boxsize = 1000.
mas_order = 2
field = jax.random.normal(jax.random.PRNGKey(2),(res,)*dim, dtype=jnp.float32).astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) # jnp.float32 to make sure that the random field is the same in single and double precision modes
bin_edges = jnp.arange(1,res//3+1)
B_info = BFast.get_triangles(bin_edges, open_triangles=True)

ref_triangles_onlyclosed = jnp.load(f"tests/reference_data/triangles_res{res}_onlyclosed.npz")
ref_triangles_openclosed = jnp.load(f"tests/reference_data/triangles_res{res}_openclosed.npz")
ref_norm = jnp.load(f"tests/reference_data/norm_PB_dim{dim}_res{res}_openclosed.npz")
ref_meas = jnp.load(f"tests/reference_data/measured_PB_dim{dim}_res{res}_mas{mas_order}_openclosed.npz")

def test_triangles_onlyclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=False)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_onlyclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_onlyclosed['triangle_indices'])

def test_triangles_openclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=True)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_openclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_openclosed['triangle_indices'])

def test_norm_unjitted_fast():
    result = BFast.Bk(field, boxsize, mas_order, **B_info, fast=True, only_B=False, compute_norm=True)
    assert jnp.allclose(result['Pk'], ref_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_norm['Bk'])

def test_norm_unjitted_slow():
    result = BFast.Bk(field, boxsize, mas_order, **B_info, fast=False, only_B=False, compute_norm=True)
    assert jnp.allclose(result['Pk'], ref_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_norm['Bk'])

def test_norm_jitted_fast():
    result = BFast.Bk.jit(field, boxsize, mas_order, **B_info, fast=True, only_B=False, compute_norm=True)
    assert jnp.allclose(result['Pk'], ref_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_norm['Bk'])

def test_norm_jitted_slow():
    result = BFast.Bk.jit(field, boxsize, mas_order, **B_info, fast=False, only_B=False, compute_norm=True)
    assert jnp.allclose(result['Pk'], ref_norm['Pk'])
    assert jnp.allclose(result['Bk'], ref_norm['Bk'])

def test_meas_unjitted_fast():
    result = BFast.Bk(field, boxsize, mas_order, **B_info, fast=True, only_B=False, compute_norm=False)
    assert jnp.allclose(result['Pk'], ref_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_meas['Bk'])

def test_meas_unjitted_slow():
    result = BFast.Bk(field, boxsize, mas_order, **B_info, fast=False, only_B=False, compute_norm=False)
    assert jnp.allclose(result['Pk'], ref_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_meas['Bk'])

def test_meas_jitted_fast():
    result = BFast.Bk.jit(field, boxsize, mas_order, **B_info, fast=True, only_B=False, compute_norm=False)
    assert jnp.allclose(result['Pk'], ref_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_meas['Bk'])

def test_meas_jitted_slow():
    result = BFast.Bk.jit(field, boxsize, mas_order, **B_info, fast=False, only_B=False, compute_norm=False)
    assert jnp.allclose(result['Pk'], ref_meas['Pk'])
    assert jnp.allclose(result['Bk'], ref_meas['Bk'])