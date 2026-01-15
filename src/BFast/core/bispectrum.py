## Author: Thomas Flöss (University of Vienna), 2025
import jax
import jax.numpy as jnp
from .utils import get_kmag, get_ffts, get_fourier_tuple, get_mas_kernel, bin_field, shard_3D_array

def get_triangles(bin_edges, equilateral=False, open_triangles=True):
    '''
    Generate triangle configurations for bispectrum estimation based on given k-bin edges.

    Parameters
    ----------
    bin_edges : jax.Array
        The edges of the k-bins (in units of the fundamental mode of the field, kF = 2*pi/boxsize) to use for the bispectrum estimation.
    equilateral : bool, optional
        Whether to only consider equilateral triangles. Default is False.
    open_triangles : bool, optional
        Whether to include open triangles in the bispectrum estimation. Default is True.
    '''

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

def Bk(field : jax.Array, boxsize : float, bin_edges : jax.Array, mas_order : int = 0,\
        open_triangles : bool = True, equilateral : bool = False, fast : bool = True, only_B : bool = False, jit : bool = True, sharded : bool = False) -> dict:
    '''
    Compute the bispectrum B(k1,k2,k3) of a given density field.
    This high-level function wraps around the low-level bispectrum computation function
    and handles the triangle configuration generation, JIT compilation and sharding.
    It also computes the normalization of the bispectrum and power spectrum estimates.
    If you evaluate the bispectrum multiple times with the same triangle configuration,
    it is more efficient to precompute the triangle information using `get_triangles`
    and then pass the resulting dictionary to the low-level 'bispectrum' function directly, computing the normalization once, and reusing it for subsequent calls.

    Example usage, similar to what this high-level `Bk` function does internally:
    ```
    bin_edges = jnp.arange(1, res//3 + 1)
    B_field = Bk(field, boxsize, bin_edges, mas_order=2, open_triangles=True, fast=True, only_B=False, jit=True, sharded=False
    ```)

    Parameters
    ----------
    field : jax.Array
        The input density field. Can be 1D, 2D or 3D.
    boxsize : float
        The physical size of the box in the same units as the density field.
    bin_edges : jax.Array
        The edges of the k-bins (in units of the fundamental mode of the field, kF = 2*pi/boxsize) to use for the bispectrum estimation.
        For example jnp.arange(1, res//3 + 1, 3) for a field of size 'res' in each dimension, computes the bispectrum in bins of width 3*kF, starting from kF up to res/3*kF.
    mas_order : int, optional
        The order of the mass assignment scheme to correct for, e.g. (2, 3, 4 for CIC, TSC, PCS, respectively). Default is 0 (no mass assignment correction).
    open_triangles : bool, optional
        Whether to include open triangles in the bispectrum estimation. Default is True.
    equilateral : bool, optional
        Whether to only consider equilateral triangles. Default is False.
    fast : bool, optional
        Whether to use the fast bispectrum computation method. Default is True.
        The fast algorithm pre-bins the density field for each k-bin only once, and then re-uses these binned fields to compute the bispectrum for all triangle configurations.
        This is much more efficient when many triangle configurations are to be evaluated, but uses more memory (depending on the number of k-bins).
        The slow algorithm re-bins the density field for each triangle configuration separately, which is less memory-intensive, but slower when many triangles are to be evaluated.
    only_B : bool, optional
        Whether to only compute the bispectrum B(k1,k2,k3) and not the power spectra P(k_i) (useful for shot-noise contribution). Default is False.
    jit : bool, optional
        Whether to JIT compile the bispectrum computation using jax.jit. Default is True
    sharded : bool, optional
        Whether to shard the input field across multiple devices (only implemented/useful for 3D fields). Default is False.
        For this high-level function, sharding always uses all available devices that jax.devices(), so make sure to set up the jax environment according to your preferences before calling this function.
        Note that for sharding to truly operate in a fully distributed manner, jit must be set to True!
    
    Return
    ------
    dict
        A dictionary containing the triangle centers 'triangle_centers', bispectrum 'Bk' and, if only_B is False, the power spectra 'Pk'.
    '''
    
    dim = len(field.shape)
    res = field.shape[0]
    assert bin_edges[-1] <= res/3, "The maximum bin edge must be less than or equal to res/3 in order to avoid aliasing effects."
    assert mas_order >= 0, "MAS order must be non-negative."

    if sharded:
        assert dim == 3, "Sharded Bk computation is only implemented for 3D fields."
        assert jit, "For sharded Bk computation, jit must be set to True to enable a properly distributed computation."
        field = shard_3D_array(field)
    else:
        sharding = None

    B_info = get_triangles(bin_edges, open_triangles=True)

    if jit:
        _bispectrum = bispectrum.jit
    else:
        _bispectrum = bispectrum

    B_norm = _bispectrum(field, boxsize, **B_info, mas_order=mas_order, fast=fast, only_B=only_B, compute_norm=True, sharding=field.sharding)
    B_field = _bispectrum(field, boxsize, **B_info, mas_order=mas_order, fast=fast, only_B=only_B, compute_norm=False, sharding=field.sharding)

    B_field['Bk'] /= B_norm['Bk']
    if not only_B: 
        B_field['Pk'] /= B_norm['Pk']

    return B_field

def bispectrum(field : jax.Array, boxsize : float, bin_edges : jax.Array, triangle_centers : jax.Array, triangle_indices : jax.Array,\
        mas_order : int = 0, fast : bool = True, only_B : bool = True, compute_norm : bool = False, sharding : jax.sharding.NamedSharding | None = None) -> dict:
    '''
    Compute the bispectrum B(k1,k2,k3) of a given density field.
    This is the low-level bispectrum computation function.
    It is more efficient when computing the bispectrum for a fixed set of triangle configurations multiple times, where the normalization can be computed once and reused.
    This function can be used directly if the triangle configurations have already been generated using `get_triangles`.
    For the high-level function that handles triangle configuration generation, JIT compilation and sharding, see `Bk`.

    Example usage, similar to what the high-level `Bk` function does internally:
    ```
    bin_edges = jnp.arange(1, res//3 + 1)
    B_info = get_triangles(bin_edges, open_triangles=True)

    if sharded:
        field = shard_3D_array(field)
        sharding = field.sharding
    else:
        sharding = None

    B_norm = bispectrum.jit(field, boxsize, **B_info, mas_order=2, compute_norm=True, sharding=sharding)
    B_field = bispectrum.jit(field, boxsize, **B_info, mas_order=2, compute_norm=False, sharding=sharding)
    B_field['Bk'] /= B_norm['Bk']
    ```

    Parameters
    ----------
    field : jax.Array
        The input density field. Can be 1D, 2D or 3D
    boxsize : float
        The physical size of the box in the same units as the density field.
    bin_edges : jax.Array
        The edges of the k-bins (in units of the fundamental mode of the field, kF = 2*pi/boxsize) to use for the bispectrum estimation.
        For example jnp.arange(1, res//3 + 1, 3) for a field of size 'res' in each dimension, computes the bispectrum in bins of width 3*kF, starting from kF up to res/3*kF.
    triangle_centers : jax.Array
        The centers of the triangle configurations to use for the bispectrum estimation. Computed using `get_triangles`.
    triangle_indices : jax.Array
        The indices of the triangle configurations to use for the bispectrum estimation. Computed using `get_triangles`.
    mas_order : int, optional
        The order of the mass assignment scheme to correct for, e.g. (2, 3, 4 for CIC, TSC, PCS, respectively). Default is 0 (no mass assignment correction).
    fast : bool, optional
        Whether to use the fast bispectrum computation method. Default is True.
        The fast algorithm pre-bins the density field for each k-bin only once, and then re-uses these binned fields to compute the bispectrum for all triangle configurations.
        This is much more efficient when many triangle configurations are to be evaluated, but uses more memory (depending on the number of k-bins).
        The slow algorithm re-bins the density field for each triangle configuration separately, which is less memory-intensive, but slower when many triangles are to be evaluated.
    only_B : bool, optional
        Whether to only compute the bispectrum B(k1,k2,k3) and not the power spectra P(k_i) (useful for shot-noise contribution). Default is True.
    compute_norm : bool, optional
        Whether to compute the normalization of the bispectrum and power spectrum estimates. Default is False.
    sharding : jax.sharding.NamedSharding | None, optional
        The sharding to use for the input field. Default is None (no sharding).
        Contrary to the high-level 'Bk' function, here you have to set up the sharding yourself if desired (e.g. by sharding the input field using `shard_3D_array`).
        If you want to provide your own sharding, note that the sharded FFTs require the sharding to be along the second ('y') axis.
        Note that for sharding to truly operate in a fully distributed manner, the input field must be sharded and the function must be JIT-compiled using jax.jit!

    Return
    ------
    dict
        A dictionary containing the triangle centers 'triangle_centers', bispectrum 'Bk' and, if only_B is False, the power spectra 'Pk'.
    '''

    dim = len(field.shape)
    res = field.shape[0]
    kF = 2*jnp.pi/boxsize
    V_cell = (boxsize/res)**dim
    rfftn, irfftn = get_ffts(dim, sharding)
    fourier_shape = get_fourier_tuple(dim, res, res//2+1)

    bin_edges = bin_edges[:,None,None,None]
    bin_low = bin_edges[:-1]
    bin_high = bin_edges[1:]
    nbins = bin_low.shape[0]
    ntriangles = triangle_indices.shape[0]

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

            _, Pk = jax.lax.scan(jax.checkpoint(_P), init=0., xs=jnp.arange(nbins))
            
        def _B(binned_fields, i):
            curr_bin_index = triangle_indices[i]
            prev_bin_index = triangle_indices[i-1]
            # Below we check whether we can reuse previously binned fields or need to re-bin
            # The logic can be seen from the triangles from the 'get_triangles' function:
            # For k1 > k2 > k3, k1 changes least frequently, then k2, so these can often be reused from 'themselves', while, k3 changes almost all the time, but ocassionally matches k2.
            # Check whether side 0 (k1) is the same as previous side 0 (k1)
            field1 = bin_field_or_take_previous(0, 0, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            # check whether side 1 (k2) is the same as previous side 1 (k2)
            field2 = bin_field_or_take_previous(1, 1, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            # check whether side 2 (k3) is the same as previous side 1 (k2)
            field3 = bin_field_or_take_previous(2, 1, curr_bin_index, prev_bin_index, bin_low, bin_high, field, binned_fields, kmag, irfftn) 
            
            return (field1, field2), (field1 * field2 * field3).sum()

        _, Bk = jax.lax.scan(jax.checkpoint(_B),
                             init=(jnp.zeros((res,res,res)),)*2,
                             xs=jnp.arange(ntriangles))

    results = {'triangle_centers' : triangle_centers * kF}
    if compute_norm:    
        results['Bk'] = Bk * res**dim * res**dim
        if not only_B:
             results['Pk'] = Pk * res**dim
    else:
        results['Bk'] = Bk * boxsize**dim * V_cell
        if not only_B:
            results['Pk'] = Pk * V_cell

    return results

bispectrum.jit = jax.jit(bispectrum, static_argnames=('mas_order','fast','sharding','only_B','compute_norm'))

