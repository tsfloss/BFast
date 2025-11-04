## Author: Thomas Flöss (University of Vienna), 2025
import jax
import jax.numpy as jnp
from .utils import get_kmag, get_ffts, get_fourier_tuple, get_mas_kernel, bin_field

def get_triangles(bin_edges, equilateral=False, open_triangles=True):
    ot = 1*open_triangles
    nbins = bin_edges.shape[0] - 1

    mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    widths = 0.5 * (bin_edges[1:] - bin_edges[:-1])

    x = jnp.arange(nbins)
    i, j, l = jnp.meshgrid(x,x,x, indexing='ij')

    mask = (j <= i) * (l <= j) # keep only k_3 < k_2 < k_1
    a, b, c = mids[i], mids[j], mids[l]
    wa, wb, wc = widths[i], widths[j], widths[l]

    if equilateral:
         valid = (a==b)&(b==c) # keep only equilateral triangles
    else:
         valid = (a - ot * wa) < ((b + ot * wb) + (c + ot * wc)) # keep only (nearly-)closed triangles
    mask = mask & valid
    triangle_centers = jnp.stack([a[mask], b[mask], c[mask]], axis=-1)
    triangle_indices = jnp.stack([i[mask], j[mask], l[mask]], axis=-1)
    return {'bin_edges' : bin_edges, 'triangle_centers' : triangle_centers, 'triangle_indices' : triangle_indices}

def bin_field_or_take_previous(i_curr, i_prev, curr_bin_index, prev_bin_index, bin_low, bin_high, field, previous_fields, kmag, irfftn):
            curr_bin = curr_bin_index[i_curr]
            prev_bin = prev_bin_index[i_prev]
            return jax.lax.cond(curr_bin == prev_bin,
                                lambda _: previous_fields[i_prev],
                                lambda _: bin_field(field, kmag, bin_low[curr_bin], bin_high[curr_bin], irfftn),
                                operand=None)

def Bk(field : jax.Array, boxsize : float, mas_order : int, bin_edges : jax.Array, triangle_centers : jax.Array, triangle_indices : jax.Array,\
        fast : bool = True, only_B : bool = True, compute_norm : bool = False, sharding : jax.sharding.NamedSharding | None = None) -> dict: 
    dim = len(field.shape)
    res = field.shape[0]
    kF = 2*jnp.pi/boxsize
    rfftn, irfftn = get_ffts(dim, sharding)
    fourier_shape = get_fourier_tuple(dim, res, res//2+1)

    bin_edges = bin_edges[:,None,None,None]
    bin_low = bin_edges[:-1]
    bin_high = bin_edges[1:]
    nbins = bin_low.shape[0]

    kmag = get_kmag(dim, res)

    if compute_norm==True:
        field = jnp.ones(fourier_shape,  dtype=jnp.complex128 if jax.config.jax_enable_x64 else jnp.complex64)
        if sharding is not None:
            field = jax.device_put(field, sharding)

    else:
        field = rfftn(field)
        if mas_order > 0:
            field *= get_mas_kernel(mas_order, dim, res)
    
    if fast:
        _, fields = jax.lax.scan(lambda c, i: (0., bin_field(field, kmag, bin_low[i], bin_high[i], irfftn)),
                                 init=0.,
                                 xs=jnp.arange(nbins))

        if not only_B:
             _, Pk = jax.lax.scan(jax.checkpoint(lambda c, i: (0., (fields[i]**2.).sum())),
                                  init=0.,
                                  xs=jnp.arange(nbins))
                                  
        _, Bk = jax.lax.scan(jax.checkpoint(lambda c, t: (0., fields[t].prod(0).sum())),
                             init=0.,
                             xs=triangle_indices)
        
    else:
        if not only_B:
            def _P(carry, i):
                return (0., (bin_field(field, kmag, bin_low[i], bin_high[i], irfftn)**2.).sum())
            _, Pk = jax.lax.scan(jax.checkpoint(_P),
                                  init=0.,
                                  xs=jnp.arange(nbins))    
        def _B(binned_fields, i):
            curr_bin_index = triangle_indices[i]
            prev_bin_index = triangle_indices[i-1]

            #check whether side 0 is the same as previous side 0
            field1 = bin_field_or_take_previous(0, 0, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            #check whether side 1 is the same as previous side 1
            field2 = bin_field_or_take_previous(1, 1, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            #check whether side 2 is the same as previous side 1
            field3 = bin_field_or_take_previous(2, 1, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            
            return (field1, field2), (field1 * field2 * field3).sum()
        _, Bk = jax.lax.scan(jax.checkpoint(_B),
                             init=(jnp.zeros((res,res,res)),)*2,
                             xs=jnp.arange(triangle_indices.shape[0]))

    results = {'triangle_centers' : triangle_centers * kF}
    if compute_norm:    
        results['Bk'] = Bk * res**3. * res**3.
        if not only_B:
             results['Pk'] = Pk * res**3.
    else:
        results['Bk'] = Bk * (boxsize**2./res)**3.
        if not only_B:
            results['Pk'] = Pk * (boxsize/res)**3.

    return results

Bk.jit = jax.jit(Bk, static_argnames=('mas_order','fast','sharding','only_B','compute_norm'))

