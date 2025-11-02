import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import BFast

res = 64
field = jax.random.normal(jax.random.PRNGKey(2),(res,)*3)
bin_edges = jnp.arange(1,res//3+1)
B_info = BFast.get_triangles(bin_edges, open_triangles=False)

ref_triangles_onlyclosed = jnp.load(f"tests/triangles_{res}_onlyclosed.npz")
ref_triangles_openclosed = jnp.load(f"tests/triangles_{res}_openclosed.npz")
ref_norm = jnp.load(f"tests/norm_{res}.npy")
# ref_Bk = 

def test_triangles_onlyclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=False)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_onlyclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_onlyclosed['triangle_indices'])

def test_triangles_openclosed():
    B_info = BFast.get_triangles(bin_edges, open_triangles=True)
    assert jnp.allclose(B_info['triangle_centers'], ref_triangles_openclosed['triangle_centers'])
    assert jnp.allclose(B_info['triangle_indices'], ref_triangles_openclosed['triangle_indices'])

def test_norm_unjitted_fast():
    Bk = BFast.Bk(field, **B_info, fast=True, compute_norm=True)
    assert jnp.allclose(Bk, ref_norm)

def test_norm_unjitted_slow():
    Bk = BFast.Bk(field, **B_info, fast=False, compute_norm=True)
    assert jnp.allclose(Bk, ref_norm)

def test_norm_jitted_fast():
    Bk = BFast.Bk.jit(field, **B_info, fast=True, compute_norm=True)
    assert jnp.allclose(Bk, ref_norm)

def test_norm_jitted_slow():
    Bk = BFast.Bk.jit(field, **B_info, fast=False, compute_norm=True)
    assert jnp.allclose(Bk, ref_norm)