## Author: Thomas Flöss (University of Vienna), 2025
import jax
import jax.numpy as jnp
from .utils import get_kmesh, get_ffts, shard_3D_array

def get_triangles(bin_edges, open_triangles=True):
    ot = 1*open_triangles
    nbins = bin_edges.shape[0] - 1

    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    widths = 0.5 * (bin_edges[1:] - bin_edges[:-1])

    x = jnp.arange(nbins)
    i, j, l = jnp.meshgrid(x,x,x, indexing='ij')

    mask = (j <= i) * (l <= j) # keep only k_3 < k_2 < k_1
    a, b, c = mids[i], mids[j], mids[l]
    wa, wb, wc = widths[i], widths[j], widths[l]

    valid = (a - ot * wa) < ((b + ot * wb) + (c + ot * wc)) # keep only (nearly-)closed triangles
    mask = mask & valid
    triangle_centers = jnp.stack([a[mask], b[mask], c[mask]], axis=-1)
    triangle_indices = jnp.stack([i[mask], j[mask], l[mask]], axis=-1)
    return {'bin_edges' : bin_edges, 'triangle_centers' : triangle_centers, 'triangle_indices' : triangle_indices}

def Bk(field, bin_edges, triangle_centers, triangle_indices, fast=True, sharding=None, compute_norm=False):
    dim = len(field.shape)
    res = field.shape[0]
    rfftn, irfftn, irfftn_batch = get_ffts(dim,sharding) 
        
    bin_edges = bin_edges[:,None,None,None]
    bin_low = bin_edges[:-1]
    bin_high = bin_edges[1:]

    kmag = get_kmesh(res, sharding)

    if compute_norm==True:
        field = jnp.ones((res,res,res//2+1),  dtype=jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64)
        if sharding is not None:
            field = jax.device_put(field, sharding)

    else:
        field = rfftn(field)
    
    if fast:
        fields = (kmag >= bin_low) * (kmag < bin_high) * field
        fields = irfftn_batch(fields)
        _, Bk = jax.lax.scan(
            jax.checkpoint(lambda c, t: (0., fields[t].prod(0).sum())),
            init=0.,
            xs=triangle_indices)

    else:
        def field_cache(i_curr, i_prev, curr_bin_index, prev_bin_index, binned_fields):
            curr_bin = curr_bin_index[i_curr]
            prev_bin = prev_bin_index[i_prev]
            return jax.lax.cond(curr_bin == prev_bin,
                                lambda _: binned_fields[i_prev],
                                lambda _: irfftn((kmag >= bin_low[curr_bin]) * (kmag < bin_high[curr_bin])*field),
                                operand=None)
        
        def _B(binned_fields, i):
            curr_bin_index = triangle_indices[i]
            prev_bin_index = triangle_indices[i-1]

            #check whether side 0 is the same as previous side 0
            field1 = field_cache(0, 0, curr_bin_index, prev_bin_index, binned_fields)
            #check whether side 1 is the same as previous side 1
            field2 = field_cache(1, 1, curr_bin_index, prev_bin_index, binned_fields) 
            #check whether side 2 is the same as previous side 1
            field3 = field_cache(2, 1, curr_bin_index, prev_bin_index, binned_fields) 
            
            return (field1, field2, field3), (field1 * field2 * field3).sum()
        _, Bk = jax.lax.scan(jax.checkpoint(_B),
                             init=(jnp.zeros((res,res,res)),)*3,
                             xs=jnp.arange(triangle_indices.shape[0]))
        
    Bk *= res**6.
    return Bk

Bk.jit = jax.jit(Bk, static_argnames=('fast','sharding','compute_norm'))
    