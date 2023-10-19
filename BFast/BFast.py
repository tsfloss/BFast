import os
os.environ['JAX_ENABLE_X64']= 'True'
import numpy as np
import jax.numpy as jnp
import jax
from jax_tqdm import scan_tqdm

def Pk(delta,BoxSize,MAS=None,left_inclusive=True,precision='float32'):
    """
    Computes binned bispectrum of field for given binning and triangles

    Parameters:
    -----------
    delta: array
        Real field to compute bispectrum of
    BoxSize:
        Size of box in comoving units (Mpc/h) such that power spectrum has units (Mpc/h)^3 and bispectrum has units (Mpc/h)^6
    MAS: str, optional (default=None)
        Mass Assignment Scheme window function to compensate for (options are NGP,CIC,TSC,PCS)
    left_inclusive: bool, optional (default=True)
        If True, uses left-inclusive bins. If False, uses right-inclusive bins instead.

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (Nbins,3) containing the power spectrum and related information.
        The columns contain: {mean k of the bin, P(k1), number of modes in bin}
    """
    
    grid = delta.shape[0]
    kF = 2*jnp.pi/BoxSize
    
    kx = jnp.fft.fftfreq(grid,1./grid)
    kmesh = jnp.array(jnp.meshgrid(kx,kx,kx,indexing='ij'),dtype=jnp.float64)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)   

    kmax = jnp.array(kgrid.max(),jnp.int64)+1

    if precision=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif precision=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128

    delta = jnp.array(delta,dtype=rtype)
    delta = jnp.fft.fftn(delta)
                      
    if left_inclusive:
        bin_lower = jnp.arange(1,kmax,dtype=jnp.float64)
        
        def _P(fft,i):
            binned_field = fft*(kgrid >= bin_lower[i])*(kgrid < bin_lower[i]+1.)
            return fft, jnp.sum(jnp.real(binned_field)**2. + jnp.imag(binned_field)**2.)/2
    else:
        bin_lower = jnp.arange(0,kmax,dtype=jnp.float64)
        
        def _P(fft,i):
            binned_field = fft*(kgrid > bin_lower[i])*(kgrid <= bin_lower[i]+1.)
            return fft, jnp.sum(jnp.real(binned_field)**2. + jnp.imag(binned_field)**2.)/2

    Nbins = bin_lower.shape[0]

    if MAS:
        p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in jnp.arange(3):
            delta *= jnp.sinc(kmesh[i]/grid)**(-p_MAS)


    del kmesh
    
    _, Pk = jax.lax.scan(f=_P,init=delta,xs=jnp.arange(Nbins))
    
    ones_fft = jnp.ones_like(delta)
    _, counts = jax.lax.scan(f=_P,init=ones_fft,xs=jnp.arange(Nbins))

    ks_fft = jnp.array(ones_fft * jnp.sqrt(kgrid),dtype=ctype)
    _, k_means = jax.lax.scan(f=_P,init=ks_fft,xs=jnp.arange(Nbins))
    
    k_means = kF * k_means / counts
    Pk = Pk / counts * BoxSize**3 / grid**6
    
    return jnp.array([k_means,Pk, counts]).T

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
                 'Squeezed' (include only triangles k_1 > k_2 = k_3), 
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
        Where to find/save counts file
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
    grid = delta.shape[0]
    file_name = file_path + f"BFast_BkCounts_Grid{int(grid)}_BoxSize{float(BoxSize):.2f}_BinSize{float(dk):.2f}kF_FirstCenter{float(fc):.2f}kF_NumBins{int(Nbins)}_TriangleType{triangle_type}_OpenTriangles{open_triangles}_Precision{precision}.npy"
    
    if os.path.exists(file_name):
        if verbose: print(f"Loading Counts from {file_name}")
        counts = np.load(file_name,allow_pickle=True).item()
        compute_counts=False
    else:
        compute_counts=True
    
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
            
    if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
    bin_indices = jnp.array(((counts['bin_centers'] - fc) // dk),dtype=jnp.int64)

    if fast:
        _Bk_fn = _Bk_jax_fast
    else:
        _Bk_fn = _Bk_jax
    
    if compute_counts:
        Pk, Bk = _Bk_fn(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,precision,verbose)
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6
        np.save(file_name,counts)
        print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False
    
    Pk, Bk = _Bk_fn(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,precision,verbose)
        
    result = np.ones((len(counts['bin_centers']),8))
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices]  * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result

def _Bk_jax_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,dtype,verbose):
    kx = jnp.fft.fftfreq(grid,1./grid)
    kz = jnp.fft.rfftfreq(grid,1./grid)
    kmesh = jnp.array(jnp.meshgrid(kx,kx,kz,indexing='ij'),dtype=jnp.float64)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)

    if dtype=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif dtype=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128
    
    bin_center = jnp.array([fc+i*dk for i in jnp.arange(Nbins)],dtype=jnp.float64)
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2
    bools = jnp.array([(kgrid >= bin_lower[i])*(kgrid < bin_upper[i]) for i in jnp.arange(Nbins)],dtype=ctype)

    if compute_counts:
        binned_delta = bools
    else:
        delta = jnp.fft.rfftn(jnp.array(delta,dtype=rtype))

        if MAS:
            p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
            delta *= (jnp.sinc(kmesh/grid)**(-p_MAS)).prod(0)
        
        binned_delta = jnp.einsum("jkl,mjkl->mjkl",jnp.array(delta,dtype=ctype),bools)
        
    del kmesh
    
    binned_delta = jnp.fft.irfftn(binned_delta,axes=[1,2,3])
    
    _, Pk = jax.lax.scan(f= (lambda carry, bin: (carry,jnp.sum(binned_delta[bin]**2.))),init=0.,xs=jnp.arange(Nbins))
    _, Bk = jax.lax.scan(f= (lambda carry, bin_index: (carry,jnp.sum(jnp.prod(binned_delta[bin_index],axis=0)))),init=0.,xs=bin_indices)
    
    return Pk, Bk

def _Bk_jax(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,dtype,verbose):
    kx = jnp.fft.fftfreq(grid,1./grid)
    kz = jnp.fft.rfftfreq(grid,1./grid)
    kmesh = jnp.meshgrid(kx,kx,kz,indexing='ij')
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)

    if dtype=='float32':
        rtype = jnp.float32
        ctype = jnp.complex64
    elif dtype=='float64':
        rtype = jnp.float64
        ctype = jnp.complex128
    
    bin_center = jnp.array([fc+i*dk for i in jnp.arange(Nbins)],dtype=jnp.float64)
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2
    
    if compute_counts:
        delta = jnp.ones((grid,grid,grid//2 + 1),dtype=ctype)
    else:
        delta = jnp.fft.rfftn(jnp.array(delta,dtype=rtype))
        
        if MAS:
            p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
            for i in jnp.arange(3):
                delta *= jnp.sinc(kmesh[i]/grid)**(-p_MAS)

    del kmesh

    delta = jnp.array(delta,dtype=ctype)

    def _P(carry,i):
        return carry, jnp.sum(jnp.fft.irfftn(delta*(kgrid >= bin_lower[i])*(kgrid < bin_upper[i]))**2.)

    if verbose: _P = scan_tqdm(Nbins)(_P)
    _, Pk = jax.lax.scan(f=_P,init=0.,xs=jnp.arange(Nbins))
    
    def _B(carry,i):
        bin_index = bin_indices[i]
        field1 = jnp.fft.irfftn(delta*(kgrid >= bin_lower[bin_index[0]])*(kgrid < bin_upper[bin_index[0]]))
        field2 = jnp.fft.irfftn(delta*(kgrid >= bin_lower[bin_index[1]])*(kgrid < bin_upper[bin_index[1]]))
        field3 = jnp.fft.irfftn(delta*(kgrid >= bin_lower[bin_index[2]])*(kgrid < bin_upper[bin_index[2]]))
        return carry, jnp.sum(field1*field2*field3)

    if verbose: _B = scan_tqdm(bin_indices.shape[0])(_B)
    _, Bk = jax.lax.scan(f=_B,init=0.,xs=jnp.arange(bin_indices.shape[0]))

    return Pk, Bk
