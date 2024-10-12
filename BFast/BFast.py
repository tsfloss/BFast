import os
os.environ['JAX_ENABLE_X64']= 'True'
import numpy as np
import jax.numpy as jnp
import jax
from jax_tqdm import scan_tqdm

def Pk(delta,BoxSize,multipole_axis=None,MAS=None,left_inclusive=True,precision='float32'):
    """
    Computes binned powerspectrum of field

    Parameters:
    -----------
    delta: array
        Real field to compute powerspectrum of
    BoxSize:
        Size of box in comoving units (Mpc/h) such that power spectrum has units (Mpc/h)^3
    multipole_axis: int, optional (default=None):
        Axis along which to compute power spectrum multipoles. Can be None, 0, 1 or 2. If None, only the monopole is computed.
    MAS: str, optional (default=None)
        Mass Assignment Scheme window function to compensate for (options are NGP,CIC,TSC,PCS)
    left_inclusive: bool, optional (default=True)
        If True, uses left-inclusive bins. If False, uses right-inclusive bins instead.

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (Nbins,3) containing the power spectrum and related information.
        The columns contain: {mean k of the bin, P(k), number of modes in bin}
    """
    assert multipole_axis in (None,0,1,2), "choose a valid value for multipole_axis (None, 0, 1 or 2)"
    grid = delta.shape[0]
    kF = 2*jnp.pi/BoxSize
    
    kx = jnp.fft.fftfreq(grid,1./grid,dtype=jnp.float64)
    kmesh = jnp.meshgrid(kx,kx,kx,indexing='ij',sparse=True)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)

    kmax = jnp.array(kgrid.max(),jnp.int64)

    if precision=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif precision=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128

    delta = jnp.asarray(delta,dtype=rtype)
    delta = jnp.fft.fftn(delta)
                      
    if left_inclusive:
        bin_lower = jnp.arange(1,kmax,dtype=jnp.float64)
        
        def _P(fft2,i):
            binned_field = fft2*(kgrid >= bin_lower[i])*(kgrid < bin_lower[i]+1.)
            return fft2, 0.5*jnp.sum(binned_field,dtype=rtype)
    else:
        bin_lower = jnp.arange(0,kmax,dtype=jnp.float64)
        
        def _P(fft2,i):
            binned_field = fft2*(kgrid > bin_lower[i])*(kgrid <= bin_lower[i]+1.)
            return fft2, 0.5*jnp.sum(binned_field,dtype=rtype)

    Nbins = bin_lower.shape[0]

    if MAS:
        p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            delta *= jnp.sinc(kmesh[i]/grid)**(-p_MAS)
        delta = delta.astype(ctype)

    delta = jnp.abs(delta)**2.
    delta, Pk_0 = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))

    if multipole_axis != None: 
        mu2 = (kmesh[multipole_axis] / kgrid)**2.
        delta_l = (delta * (3.*mu2-1.)/2.)
        delta_l, Pk_2 = jax.lax.scan(f=_P,init=delta_l,xs=jnp.arange(Nbins))

        delta_l = (delta * (35.*mu2*mu2 - 30.*mu2 + 3.)/8.)
        delta_l, Pk_4 = jax.lax.scan(f=_P,init=delta_l,xs=jnp.arange(Nbins))
        del delta_l
    
    delta = jnp.ones_like(delta)
    delta, counts = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))

    delta, k_means = jax.lax.scan(f=_P,init=kgrid,xs=jnp.arange(Nbins))

    k_means = kF * k_means / counts
    Pk_0 = Pk_0 / counts * BoxSize**3 / grid**3 / grid**3

    if multipole_axis != None:
        Pk_2 = 5. * Pk_2 / counts * BoxSize**3 / grid**6
        Pk_4 = 9. * Pk_4 / counts * BoxSize**3 / grid**6
        return np.array([k_means,Pk_0, Pk_2, Pk_4, counts],dtype=rtype).T
    else:
        return np.array([k_means,Pk_0, counts],dtype=rtype).T

def xPk(delta1,delta2,BoxSize,MAS=[None,None],left_inclusive=True,precision='float32'):
    """
    Computes binned cross-powerspectrum of two fields

    Parameters:
    -----------
    delta1: array
        Real field to compute powerspectrum of
    delta2: array
        Real field to compute powerspectrum of
    BoxSize:
        Size of box in comoving units (Mpc/h) such that power spectrum has units (Mpc/h)^3
    MAS: list of [str,str], optional (default=[None,None])
        Mass Assignment Scheme window function to compensate for (options are NGP,CIC,TSC,PCS)
    left_inclusive: bool, optional (default=True)
        If True, uses left-inclusive bins. If False, uses right-inclusive bins instead.

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (Nbins,6) containing the power spectrum and related information.
        The columns contain: {mean k of the bin, powerspectrum of delta1, powerspectrum of delta2, cross-powerspectrum, cross-correlation, number of modes in bin}
    """
    assert delta1.shape == delta2.shape, "shapes of fields mismatch"
    assert len(MAS) == 2, "supply valid MAS values for the two fields, e.g. [None,None] or ['CIC','CIC']"
    grid = delta1.shape[0]
    kF = 2*jnp.pi/BoxSize
    
    kx = jnp.fft.fftfreq(grid,1./grid,dtype=jnp.float64)
    kmesh = jnp.meshgrid(kx,kx,kx,indexing='ij',sparse=True)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)   

    kmax = jnp.array(kgrid.max(),jnp.int64)

    if precision=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif precision=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128

    delta1 = jnp.asarray(delta1,dtype=rtype)
    delta1 = jnp.fft.fftn(delta1)

    delta2 = jnp.asarray(delta2,dtype=rtype)
    delta2 = jnp.fft.fftn(delta2)
                      
    if left_inclusive:
        bin_lower = jnp.arange(1,kmax,dtype=jnp.float64)
        
        def _P(fft2,i):
            binned_field = fft2*(kgrid >= bin_lower[i])*(kgrid < bin_lower[i]+1.)
            return fft2, 0.5*jnp.sum(binned_field)
    else:
        bin_lower = jnp.arange(0,kmax,dtype=jnp.float64)
        
        def _P(fft2,i):
            binned_field = fft2*(kgrid > bin_lower[i])*(kgrid <= bin_lower[i]+1.)
            return fft2, 0.5*jnp.sum(binned_field)

    Nbins = bin_lower.shape[0]

    if MAS[0] or MAS[1]:
        
        for i in range(3):
            mas_fac = jnp.sinc(kmesh[i]/grid)

        if MAS[0]:
            p_MAS_1 = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS[0]]
            delta1 *= mas_fac**(-p_MAS_1)
            delta1 = delta1.astype(ctype)
        if MAS[1]:
            p_MAS_2 = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS[1]]
            delta2 *= mas_fac**(-p_MAS_2)
            delta2 = delta2.astype(ctype)

        del mas_fac

    del kmesh
    
    delta = jnp.abs(delta1)**2.
    delta, Pk1 = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))
    delta = jnp.abs(delta2)**2.
    delta, Pk2 = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))
    delta = 0.5*(delta1 * delta2.conj() + delta2 * delta1.conj()).real
    delta, Pkx = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))

    _, counts = jax.lax.scan(f=_P,init=jnp.ones_like(delta),xs=jnp.arange(Nbins))

    _, k_means = jax.lax.scan(f=_P,init=kgrid,xs=jnp.arange(Nbins))
    
    k_means = kF * k_means / counts
    Pk1 = Pk1 / counts * BoxSize**3 / grid**6
    Pk2 = Pk2 / counts * BoxSize**3 / grid**6
    Pkx = Pkx / counts * BoxSize**3 / grid**6
    Ck = Pkx/jnp.sqrt(Pk1*Pk2)
    return np.array([k_means,Pk1, Pk2, Pkx, Ck, counts],dtype=rtype).T

def Bk(delta,BoxSize,fc,dk,Nbins,triangle_type='All',open_triangles=True,MAS=None,fast=True,precision='float32',file_path="./",verbose=False):
    """
    Computes binned bispectrum of field for given binning and triangles

    Parameters:
    -----------
    delta: array
        Real field to compute bispectrum of
    BoxSize:
        Size of box in comoving units (Mpc/h) such that power spectrum has units (Mpc/h)^3 and bispectrum has units (Mpc/h)^6
    fc: float
        Center of first bin in units of the fundamental mode.
    dk: float
        Width of the bin in units of the fundamental mode.
    Nbins: int
        Total number of momentum bins such that bins are given by [(fc + i)±dk/2 for i in range(Nbins)].
    triangle_type: str, optional (default='All')
        Type of triangles to include in the bispectrum calculation. 
        Options: 'All' (include all shapes of triangles),
                 'Squeezed' (include only triangles k_1 < k_2 = k_3), 
                 'Equilateral' (include only triangles k_1 = k_2 = k_3).
    open_triangles: bool, optional (default=True)
        If True, includes triangles of which the bin centers do not form a closed triangles, but still form closed triangles somewhere within the bins (see Biagetti '21)
    MAS: str, optional (default=None)
        Mass Assignment Scheme window function to compensate for (options are NGP,CIC,TSC,PCS)
    fast: bool, optional (default=True)
        If True, uses the fast algorithm that precomputes all bins. If False, use the slower algorithm for larger grid-size and/or Nbins.
    precision: str, optional (default='float32')
        Precision of the computation, affects the speed of the algorithm
    file_path: str, optional (default='./')
        Where to find/save counts file, if file_path=None no file will be used
    verbose: bool, optional (default=False)
        If True, print progress statements (only for slow algorithm)

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (len(counts['bin_centers']),8) containing the bispectrum and related information.
        The columns contain: {bin centers in units of kF, P(k1), P(k2), P(k3), B(k1,k2,k3), number of triangles in bin}
        
    Notes:
    --------
    The first time the computation for a certain binning is being done, 
    this function will first compute the necessary mode counts for power spectrum and bispectrum normalization. 
    This is saved in a file in the local directory for later use, when measuring from other density fields but with the same binning.
    """
    fc = float(fc)
    dk = float(dk)
    grid = delta.shape[0]
    kF = 2*jnp.pi/BoxSize
    kmax = (fc + dk * (Nbins-1) + dk/2)*kF

    compute_counts=True
    
    if file_path:
        file_name = file_path + f"BFast_BkCounts_Grid{int(grid)}_BoxSize{float(BoxSize):.2f}_BinSize{dk:.2f}kF_FirstCenter{fc:.2f}kF_NumBins{int(Nbins)}_TriangleType{triangle_type}_OpenTriangles{open_triangles}_Precision{precision}.npy"
    
        if os.path.exists(file_name):
            if verbose: print(f"Loading Counts from {file_name}")
            counts = np.load(file_name,allow_pickle=True).item()
            compute_counts=False

    
    if compute_counts:
        if verbose: print("No counts file found, computing this first!")
        counts = {}
        if triangle_type=='All':
            counts['bin_centers'] = np.array([(i,j,l)\
                                              for i in fc+np.arange(0, (Nbins))*dk \
                                              for j in np.arange(fc, i+1, dk)\
                                              for l in np.arange(fc, j+1, dk) if i<=j+l+open_triangles*dk]) 

        elif triangle_type=='Squeezed':
            counts['bin_centers'] = np.array([(i,i,j)\
                                              for ji,j in enumerate(fc + np.arange(0,Nbins)*dk)\
                                              for i in fc+np.arange(ji+1,Nbins)*dk])
        elif triangle_type=='Equilateral':
            counts['bin_centers'] = np.array([(i,i,i)\
                                              for i in fc+np.arange(0,Nbins)*dk])
            
    if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type}), with kmax = {kmax:.4f}")
    bin_indices = jnp.array(((counts['bin_centers'] - fc) // dk),dtype=jnp.int32)

    if fast:
        _Bk_fn = _Bk_jax_fast
    else:
        _Bk_fn = _Bk_jax
    
    if compute_counts:
        Pk, Bk = _Bk_fn(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,precision,verbose)
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6

        if file_path:
            np.save(file_name,counts)
            print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False

    Pk, Bk = _Bk_fn(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,precision,verbose)
        
    result = np.ones((len(counts['bin_centers']),8),dtype=Bk.dtype)
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices] * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result

def _Bk_jax_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,dtype,verbose):
    if dtype=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif dtype=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128

    kx = jnp.fft.fftfreq(grid,1./grid,dtype=rtype)
    kz = jnp.fft.rfftfreq(grid,1./grid,dtype=rtype)
    kmesh = jnp.asarray(jnp.meshgrid(kx,kx,kz,indexing='ij'),dtype=rtype)
    del kx, kz
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)
    
    bools = jnp.asarray([(kgrid >= fc + i*dk - dk/2)*(kgrid < fc + i*dk + dk/2) for i in jnp.arange(Nbins)],dtype=ctype)
    del kgrid

    if compute_counts:
        binned_delta = bools
    else:
        delta = jnp.fft.rfftn(delta.astype(rtype))

        if MAS:
            p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
            delta *= (jnp.sinc(kmesh/grid)**(-p_MAS)).prod(0)

        delta = delta.astype(ctype)
        binned_delta = jnp.einsum("jkl,mjkl->mjkl",delta,bools)
        
    del kmesh, bools
    
    binned_delta = jnp.fft.irfftn(binned_delta,axes=[1,2,3])
    
    Pk = jnp.sum(binned_delta**2.,axis=(1,2,3),dtype=rtype)
    _, Bk = jax.lax.scan(f= (lambda carry, bin_index: (carry,jnp.sum(jnp.prod(binned_delta[bin_index],axis=0),dtype=rtype))),init=0.,xs=bin_indices)
    
    return np.asarray(Pk), np.asarray(Bk)

def _Bk_jax(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,dtype,verbose):
    if dtype=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif dtype=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128

    kx = jnp.fft.fftfreq(grid,1./grid,dtype=rtype)
    kz = jnp.fft.rfftfreq(grid,1./grid,dtype=rtype)
    kmesh = jnp.meshgrid(kx,kx,kz,indexing='ij',sparse=True)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)
    
    bin_center = jnp.array([fc+i*dk for i in jnp.arange(Nbins)],dtype=rtype)
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2
    
    if compute_counts:
        delta = jnp.ones((grid,grid,grid//2 + 1),dtype=ctype)
    else:
        delta = jnp.fft.rfftn(delta.astype(rtype))

        if MAS:
            p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
            delta *= (jnp.sinc(kmesh[0]/grid)**(-p_MAS) * jnp.sinc(kmesh[1]/grid)**(-p_MAS) * jnp.sinc(kmesh[2]/grid)**(-p_MAS))
        
    delta = delta.astype(ctype)
        
    del kmesh

    def _bin_field(field,low,high):
        return jnp.fft.irfftn(field*(kgrid >= low)*(kgrid < high))

    def _P(carry,i):
        return carry, jnp.sum(_bin_field(delta,bin_lower[i],bin_upper[i])**2.,dtype=rtype)
    
    def _B(carry, i):
        prev_bin_index = bin_indices[i-1]
        bin_index = bin_indices[i]

        field1 = jax.lax.cond(bin_index[0] == prev_bin_index[0], lambda _: carry[0], 
                         lambda _: _bin_field(delta,bin_lower[bin_index[0]],bin_upper[bin_index[0]]),
                         operand=None)

        field2 = jax.lax.cond(bin_index[1] == prev_bin_index[1], lambda _: carry[1], 
                        lambda _: _bin_field(delta,bin_lower[bin_index[1]],bin_upper[bin_index[1]]),
                         operand=None)
        
        field3 = jax.lax.cond(bin_index[2] == prev_bin_index[2], lambda _: carry[2], 
                         lambda _: _bin_field(delta,bin_lower[bin_index[2]],bin_upper[bin_index[2]]),
                         operand=None)
        
        return (field1,field2,field3), jnp.sum(field1 * field2 * field3,dtype=rtype)

    if verbose: _P = scan_tqdm(Nbins)(_P)
    _, Pk = jax.lax.scan(f=_P,init=0.,xs=jnp.arange(Nbins))
    
    carry = (jnp.zeros((grid,grid,grid),dtype=rtype),)*3
    if verbose: _B = scan_tqdm(bin_indices.shape[0],print_rate=bin_indices.shape[0]//10)(_B)
    _, Bk = jax.lax.scan(f=_B,init=carry,xs=jnp.arange(bin_indices.shape[0]))
    
    return np.asarray(Pk), np.asarray(Bk)
