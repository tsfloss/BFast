import os
os.environ['JAX_ENABLE_X64']= 'True'
import numpy as np
import jax.numpy as jnp
import jax
from jax_tqdm import scan_tqdm

def _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose):
    kF = 2*jnp.pi/BoxSize
    
    kx = 2*jnp.pi * jnp.fft.fftfreq(grid,BoxSize/grid)
    kz = 2*jnp.pi * jnp.fft.rfftfreq(grid,BoxSize/grid)
    kmesh = jnp.array(jnp.meshgrid(kx,kx,kz,indexing='ij'),dtype=jnp.float64)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)

    bin_center = jnp.array([fc+i*dk for i in jnp.arange(Nbins)],dtype=jnp.float64)
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2
    bools = jnp.array([(kgrid >= kF*bin_lower[i])*(kgrid < kF*bin_upper[i]) for i in jnp.arange(Nbins)],dtype=jnp.complex64)

    if compute_counts:
        masked_maps_fft = bools
    else:
        map_fft = jnp.array(jnp.fft.rfftn(delta),dtype=jnp.complex64)

        if MAS:
            p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
            fac = jnp.pi * kmesh/kF/grid
            mas_fac = (fac/jnp.sin(fac))**p_MAS
            mas_fac = jnp.array(jnp.where(jnp.isnan(mas_fac),1.,mas_fac),dtype=jnp.complex64)
            mas_fac = jnp.prod(mas_fac,0)
            map_fft *= mas_fac
            del mas_fac
        
        masked_maps_fft = jnp.einsum("jkl,mjkl->mjkl",map_fft,bools)
        
    del kmesh
    
    masked_maps = jnp.fft.irfftn(masked_maps_fft,axes=[1,2,3])
    
    _, Pk = jax.lax.scan(f= (lambda carry, bin: (carry,jnp.sum(masked_maps[bin]**2.))),init=0.,xs=jnp.arange(Nbins))
    _, Bk = jax.lax.scan(f= (lambda carry, bin_index: (carry,jnp.sum(jnp.prod(masked_maps[bin_index],axis=0)))),init=0.,xs=bin_indices)
    
    return Pk, Bk

def Bk_fast(delta,BoxSize,fc,dk,Nbins,triangle_type='All',MAS=None,verbose=False):
    """
    Computes binned bispectrum of field for given binning and triangles

    Parameters:
    -----------
    fc: float
        Center of first bin in units of the fundamental mode.
    dk: float
        Width of the bin in units of the fundamental mode.
    Nbins: int
        Total number of momentum bins such that bins are given by kf*[(fc + i)±dk/2 for i in range(Nbins)].
    triangle_type: str, optional (default='All')
        Type of triangles to include in the bispectrum calculation. 
        Options: 'All' (include all shapes of triangles),
                 'Squeezed' (only triangles k_1 > k_2 = k_3), 
                 'Equilateral' (include only triangles k_1 = k_2 = k_3).
    MAS: str, optional (default=None)
        Mass Assignment Scheme to compensate for (options are NGP,CIC,TSC,PCS)
    verbose: bool, optional (default=False)
        If True, print progress statements.

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (len(counts['bin_centers']),8) containing the bispectrum and related information.
        The columns contain: bin centers, P(k1), P(k2), P(k3), B(k1,k2,k3), counts_B.
        
    Notes:
    --------
    The first time the computation for a certain binning is being done, 
    this function will first compute the necessary mode counts for power spectrum and bispectrum normalization. 
    This is saved in a file in the local directory for later use, when measuring from other density fields but with the same binning.
    """
    grid = delta.shape[0]
    file_name = f"BFast_BkCounts_LBox{float(BoxSize):.2f}_Grid{int(grid)}_Binning{float(dk):.2f}kF_fc{float(fc):.2f}_NBins{int(Nbins)}_TriangleType{triangle_type}.npy"

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
                                              for l in np.arange(fc, j+1, dk) if i<=j+l+dk]) 
                                                #the +dk allows for open bins (Biagetti '21)

        elif triangle_type=='Squeezed':
            counts['bin_centers'] = np.array([(i,i,j)\
                                              for ji,j in enumerate(fc + np.arange(0,Nbins)*dk)\
                                              for i in fc+np.arange(ji+1,Nbins)*dk])
        elif triangle_type=='Equilateral':
            counts['bin_centers'] = np.array([(i,i,i)\
                                              for i in fc+np.arange(0,Nbins)*dk])
            
    if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
    bin_indices = jnp.array(((counts['bin_centers'] - fc) // dk),dtype=jnp.int64)
    
    if compute_counts:
        Pk, Bk = _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6
        np.save(file_name,counts)
        print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False
    
    Pk, Bk = _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
        
    result = np.ones((len(counts['bin_centers']),8))
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices]  * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result

def _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose):
    kF = 2*jnp.pi/BoxSize
    
    kx = 2*jnp.pi * jnp.fft.fftfreq(grid,BoxSize/grid)
    kz = 2*jnp.pi * jnp.fft.rfftfreq(grid,BoxSize/grid)
    kmesh = jnp.array(jnp.meshgrid(kx,kx,kz,indexing='ij'),dtype=jnp.float64)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)

    bin_center = jnp.array([fc+i*dk for i in jnp.arange(Nbins)],dtype=jnp.float64)
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2
    
    map_fft = jnp.array(jnp.fft.rfftn(delta),dtype=jnp.complex64)
    
    if compute_counts:
        map_fft = jnp.array(jnp.ones_like(map_fft),dtype=jnp.complex64)
        
    if MAS:
        p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in jnp.arange(3):
            fac = jnp.pi * kmesh[i]/kF/grid
            mas_fac = (fac/jnp.sin(fac))**p_MAS
            mas_fac = jnp.array(jnp.where(jnp.isnan(mas_fac),1.,mas_fac),dtype=jnp.complex64)
            map_fft *= mas_fac
        del mas_fac

    del kmesh

    @scan_tqdm(Nbins)
    def _P(carry,i):
        return carry, jnp.sum(jnp.array(jnp.fft.irfftn(map_fft*(kgrid >= kF*bin_lower[i])*(kgrid < kF*bin_upper[i])),dtype=jnp.float64)**2.)

    _, Pk = jax.lax.scan(f=_P,init=0.,xs=jnp.arange(Nbins))
    
    @scan_tqdm(bin_indices.shape[0])
    def _B(carry,i):
        bin_index = bin_indices[i]
        field1 = jnp.array(jnp.fft.irfftn(map_fft*(kgrid >= kF*bin_lower[bin_index[0]])*(kgrid < kF*bin_upper[bin_index[0]])),dtype=jnp.float32)
        field2 = jnp.array(jnp.fft.irfftn(map_fft*(kgrid >= kF*bin_lower[bin_index[1]])*(kgrid < kF*bin_upper[bin_index[1]])),dtype=jnp.float32)
        field3 = jnp.array(jnp.fft.irfftn(map_fft*(kgrid >= kF*bin_lower[bin_index[2]])*(kgrid < kF*bin_upper[bin_index[2]])),dtype=jnp.float32)
        return carry, jnp.sum(field1*field2*field3)   
        
    _, Bk = jax.lax.scan(f=_B,init=0.,xs=jnp.arange(bin_indices.shape[0]))

    return Pk, Bk

def Bk(delta,BoxSize,fc,dk,Nbins,triangle_type='All',MAS=None,verbose=False):
    """
    Computes binned bispectrum of field for given binning and triangles

    Parameters:
    -----------
    fc: float
        Center of first bin in units of the fundamental mode.
    dk: float
        Width of the bin in units of the fundamental mode.
    Nbins: int
        Total number of momentum bins such that bins are given by kf*[(fc + i)±dk/2 for i in range(Nbins)].
    triangle_type: str, optional (default='All')
        Type of triangles to include in the bispectrum calculation. 
        Options: 'All' (include all shapes of triangles),
                 'Squeezed' (only triangles k_1 > k_2 = k_3), 
                 'Equilateral' (include only triangles k_1 = k_2 = k_3).
    MAS: str, optional (default=None)
        Mass Assignment Scheme to compensate for (options are NGP,CIC,TSC,PCS)
    verbose: bool, optional (default=False)
        If True, print progress statements.

    Returns:
    --------
    result: numpy.ndarray
        An array of shape (len(counts['bin_centers']),8) containing the bispectrum and related information.
        The columns contain: bin centers, P(k1), P(k2), P(k3), B(k1,k2,k3), counts_B.
        
    Notes:
    --------
    The first time the computation for a certain binning is being done, 
    this function will first compute the necessary mode counts for power spectrum and bispectrum normalization. 
    This is saved in a file in the local directory for later use, when measuring from other density fields but with the same binning.
    """
    grid = delta.shape[0]
    file_name = f"BFast_BkCounts_LBox{float(BoxSize):.2f}_Grid{int(grid)}_Binning{float(dk):.2f}kF_fc{float(fc):.2f}_NBins{int(Nbins)}_TriangleType{triangle_type}.npy"
        
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
                                              for l in np.arange(fc, j+1, dk) if i<=j+l+dk]) 
                                                #the +dk allows for open bins (Biagetti '21)

        elif triangle_type=='Squeezed':
            counts['bin_centers'] = np.array([(i,i,j)\
                                              for ji,j in enumerate(fc + np.arange(0,Nbins)*dk)\
                                              for i in fc+np.arange(ji+1,Nbins)*dk])
        elif triangle_type=='Equilateral':
            counts['bin_centers'] = np.array([(i,i,i)\
                                              for i in fc+np.arange(0,Nbins)*dk])
            
    if verbose: print(f"Considering {len(counts['bin_centers'])} Triangle Configurations ({triangle_type})")
    bin_indices = jnp.array(((counts['bin_centers'] - fc) // dk),dtype=jnp.int64)
    
    if compute_counts:
        Pk, Bk = _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6
        np.save(file_name,counts)
        print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False
    
    Pk, Bk = _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
        
    result = np.ones((len(counts['bin_centers']),8))
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices]  * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result

def Pk(delta,BoxSize,MAS=None):
    grid = delta.shape[0]
    kF = 2*jnp.pi/BoxSize
    
    kx = jnp.fft.fftfreq(grid,1./grid)
    kmesh = jnp.array(jnp.meshgrid(kx,kx,kx,indexing='ij'),dtype=jnp.float64)
    kgrid = jnp.sqrt(kmesh[0]**2. + kmesh[1]**2. + kmesh[2]**2.)   

    kmax = jnp.array(kgrid.max(),jnp.int64)+1
    bin_lower = jnp.arange(1,kmax,dtype=jnp.float64)
    bin_upper = bin_lower+1.
    Nbins = bin_lower.shape[0]
    
    def _P(fft,i):
        binned_field = fft*(kgrid >= bin_lower[i])*(kgrid < bin_upper[i])
        return fft, jnp.sum(jnp.real(binned_field)**2. + jnp.imag(binned_field)**2.)/2

    map_fft = jnp.array(jnp.fft.fftn(delta),dtype=jnp.complex128)

    if MAS:
        p_MAS = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in jnp.arange(3):
            fac = jnp.pi * kmesh[i]/grid
            mas_fac = (fac/jnp.sin(fac))**p_MAS
            mas_fac = jnp.array(jnp.where(jnp.isnan(mas_fac),1.,mas_fac),dtype=jnp.complex128)
            map_fft *= mas_fac
        del mas_fac

    del kmesh
    
    _, Pk = jax.lax.scan(f=_P,init=map_fft,xs=jnp.arange(Nbins))
    
    ones_fft = jnp.ones((grid,grid,grid),dtype=jnp.complex128)
    _, counts = jax.lax.scan(f=_P,init=ones_fft,xs=jnp.arange(Nbins))

    ks_fft = ones_fft * jnp.sqrt(kgrid)
    _, k_means = jax.lax.scan(f=_P,init=ks_fft,xs=jnp.arange(Nbins))
    
    k_means = kF * k_means / counts
    Pk = Pk / counts * BoxSize**3 / grid**6
    
    return jnp.array([k_means,Pk, counts]).T