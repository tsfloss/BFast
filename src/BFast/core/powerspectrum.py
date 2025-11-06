## Author: Thomas Flöss (University of Vienna), 2025
import jax
import jax.numpy as jnp
from .utils import get_kmag, get_kmesh, get_ffts, get_mas_kernel, shard_3D_array

def Pk(field : jax.Array, boxsize : float, bin_edges : jax.Array, mas_order : int = 0, multipole_axis : int | None = None, jit : bool = True, sharded : bool = False) -> dict:
    '''
    Computes the power spectrum of density field.
    This high-level function wraps around the low-level `powerspectrum` function, adding options for JIT compilation and sharded computation.

    Parameters
    ----------
    field : jax.Array
        The input density field (can be 1D, 2D, or 3D).
    boxsize : float
        The physical size of the box in the same units as the density field.
    bin_edges : jax.Array
        The edges of the k-bins (in units of the fundamental mode of the field, kF = 2*pi/boxsize) to use for the power spectrum estimation.
    mas_order : int, optional
        The order of the mass assignment scheme to correct for, e.g. (2, 3, 4 for CIC, TSC, PCS, respectively). Default is 0 (no mass assignment correction).
    multipole_axis : int or None, optional
        If specified, computes the multipole moments of the power spectrum along the given axis (0, 1, or 2 for 3D fields). Default is None (only monopole).
    jit : bool, optional
        If True, JIT-compiles the power spectrum computation for improved performance. Default is True.
    sharded : bool, optional
        If True, shards the input field across available devices for distributed computation (only implemented for 3D fields). Default is False.

    Returns
    -------
    dict
        A dictionary containing the computed power spectrum components and related information:
        - 'k': The effective k-values for each bin.
        - 'norm': The number of modes in each k-bin.
        - 'Pk0': The monopole moment of the power spectrum.
        - 'Pk2': The quadrupole moment of the power spectrum (if multipole_axis is specified).
        - 'Pk4': The hexadecapole moment of the power spectrum (if multipole_axis is specified).
    '''

    dim = len(field.shape)
    res = field.shape[0]

    assert bin_edges[-1] <= res//2, "The maximum bin edge must be less than or equal to res/2 in order to avoid aliasing effects."
    assert mas_order >= 0, "MAS order must be non-negative."
    assert multipole_axis is None or (0 <= multipole_axis < dim), "Multipole axis must be None or an integer between 0 and dim-1."

    if sharded:
        assert dim == 3, "Sharded Pk computation is only implemented for 3D fields."
        field = shard_3D_array(field)
        sharding = field.sharding
    else:
        sharding = None

    if jit:
        _P = powerspectrum.jit
    else:
        _P = powerspectrum

    return _P(field, boxsize, bin_edges, mas_order=mas_order, multipole_axis=multipole_axis, sharding=sharding)

def powerspectrum(field : jax.Array, boxsize : float, bin_edges : jax.Array, mas_order : int = 0, multipole_axis : int | None = None,\
       sharding : jax.sharding.NamedSharding | None = None) -> dict:
    '''
    Low-level function to compute the power spectrum of a density field.
    
    Parameters
    ----------
    field : jax.Array
        The input density field (can be 1D, 2D, or 3D).
    boxsize : float
        The physical size of the box in the same units as the density field.
    bin_edges : jax.Array
        The edges of the k-bins (in units of the fundamental mode of the field, kF = 2*pi/boxsize) to use for the power spectrum estimation.
    mas_order : int, optional
        The order of the mass assignment scheme to correct for, e.g. (2, 3, 4 for CIC, TSC, PCS, respectively). Default is 0 (no mass assignment correction).
    multipole_axis : int or None, optional
        If specified, computes the multipole moments of the power spectrum along the given axis (0, 1, or 2 for 3D fields). Default is None (only monopole).
    sharding : jax.sharding.NamedSharding or None, optional
        If specified, uses the given sharding for distributed computation. Default is None (no sharding).
        
    Returns
    -------
    dict
        A dictionary containing the computed power spectrum components and related information:
        - 'k': The effective k-values for each bin.
        - 'norm': The number of modes in each k-bin.
        - 'Pk0': The monopole moment of the power spectrum.
        - 'Pk2': The quadrupole moment of the power spectrum (if multipole_axis is specified).
        - 'Pk4': The hexadecapole moment of the power spectrum (if multipole_axis is specified).
    '''
    
    dim = len(field.shape)
    res = field.shape[0]
    V_cell = (boxsize/res)**dim
    kF = 2*jnp.pi/boxsize
    rfftn, irfftn = get_ffts(dim,sharding=sharding)

    bin_edges = bin_edges.astype(jnp.float32).at[-1].add(-1e-5)

    kmag = get_kmag(dim, res)
    counts, _ = jnp.histogram(kmag, bin_edges, weights=jnp.ones_like(kmag).at[...,1:-1].multiply(2.))

    field = rfftn(field) / res**(dim/2)
    if mas_order > 0:
        field *= get_mas_kernel(mas_order, dim, res)

    k_eff, _ = jnp.histogram(kmag, bin_edges, weights=kmag.at[...,1:-1].multiply(2.))

    ampl2 = (field*field.conj()).real.at[...,1:-1].multiply(2.)
    Pk0_unnorm, _ = jnp.histogram(kmag, bin_edges, weights=ampl2)
    
    result = {}
    result['norm'] = counts
    result['k'] = k_eff * kF / counts
    result['Pk0'] = Pk0_unnorm / counts * V_cell

    if multipole_axis is not None:
        k_los = get_kmesh(dim, res)[multipole_axis]
        mu2 = (k_los/kmag)**2.
        Pk2_unnorm, _ = jnp.histogram(kmag, bin_edges, weights=ampl2 * (3.*mu2-1.)/2.)
        Pk4_unnorm, _ = jnp.histogram(kmag, bin_edges, weights=ampl2 * (35.*mu2**2. - 30.*mu2 + 3.)/8.)
        result['Pk2'] = 5. * Pk2_unnorm / counts * V_cell
        result['Pk4'] = 9. * Pk4_unnorm / counts * V_cell
    
    return result

powerspectrum.jit = jax.jit(powerspectrum, static_argnames=('sharding','mas_order','multipole_axis'))
