import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts):
    kF = 2*np.pi/BoxSize
    
    kx = 2*np.pi * np.fft.fftfreq(grid,BoxSize/grid)
    kz = 2*np.pi * np.fft.rfftfreq(grid,BoxSize/grid)
    kmesh = tf.meshgrid(kx,kx,kz,indexing='ij')
    kgrid = tf.cast(tf.sqrt(tf.square(kmesh[0])+tf.square(kmesh[1])+tf.square(kmesh[2])),dtype=tf.float32)
    
    map_fft = tf.expand_dims(tf.signal.rfft3d(delta),axis=0)
    
    if MAS:
        p = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            fac = np.pi * kmesh[i]/kF/grid
            mas_fac = (fac/tf.math.sin(fac))**p
            mas_fac = tf.where(tf.math.is_nan(mas_fac),1.,mas_fac,tf.float32)
            mas_fac = tf.cast(mas_fac,dtype=tf.complex64)
            map_fft *= mas_fac
    
    if compute_counts:
        map_fft = tf.ones_like(map_fft)
        
    bin_center = tf.constant([fc+i*dk for i in range(Nbins)])
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2

    masked_maps_fft = [tf.where(tf.math.logical_and(kgrid >= kF*bin_lower[i],kgrid < kF*bin_upper[i]),map_fft,0.+0.j) for i in range(Nbins)]
    masked_maps_fft = tf.concat(masked_maps_fft,axis=0)
    masked_maps = tf.signal.irfft3d(masked_maps_fft)
    
    Pk = tf.stack([tf.reduce_sum(masked_maps[i]**2) for i in range(Nbins)])
    Bk = tf.stack([tf.reduce_sum(masked_maps[bin_indices[i,0]] * masked_maps[bin_indices[i,1]] * masked_maps[bin_indices[i,2]]) for i in range(bin_indices.shape[0])])
    return Pk, Bk

def Bk_fast(delta,BoxSize,fc,dk,Nbins,triangle_type,MAS,verbose=False):
    grid = delta.shape[0]
    file_name = f"BFast_BkCounts_LBox{int(BoxSize)}_Grid{int(grid)}_Binning{int(dk)}kF_fc{int(fc)}_NBins{int(Nbins)}_TriangleType{triangle_type}.npy"
        
    if os.path.exists(file_name):
        if verbose: print(f"Loading Counts from {file_name}")
        counts = np.load(file_name,allow_pickle=True).item()
        compute_counts=False
    else:
        print(f"No precomputed counts found. Computing this instead of bispectrum")
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
    bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
    
    if compute_counts:
        Pk, Bk = _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts)
        Pk = Pk.numpy()
        Bk = Bk.numpy()
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6
        np.save(file_name,counts)
        print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False
    
    Pk, Bk = _Bk_TF_fast(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts)
    Pk = Pk.numpy()
    Bk = Bk.numpy()
        
    result = np.ones((len(counts['bin_centers']),8))
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices]  * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result


def _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose):
    kF = 2*np.pi/BoxSize
    
    kx = 2*np.pi * np.fft.fftfreq(grid,BoxSize/grid)
    kz = 2*np.pi * np.fft.rfftfreq(grid,BoxSize/grid)
    kmesh = tf.meshgrid(kx,kx,kz,indexing='ij')
    kgrid = tf.cast(tf.sqrt(tf.square(kmesh[0])+tf.square(kmesh[1])+tf.square(kmesh[2])),dtype=tf.float32)
    
    map_fft = tf.expand_dims(tf.signal.rfft3d(delta),axis=0)
    
    if compute_counts:
        map_fft = tf.ones_like(map_fft)
    elif MAS:
        map_fft = tf.expand_dims(tf.signal.rfft3d(delta),axis=0)
        p = {'NGP':1., "CIC":2., "TSC":3., "PCS":4.}[MAS]
        for i in range(3):
            fac = np.pi * kmesh[i]/kF/grid
            mas_fac = (fac/tf.math.sin(fac))**p
            mas_fac = tf.where(tf.math.is_nan(mas_fac),1.,mas_fac,tf.float32)
            mas_fac = tf.cast(mas_fac,dtype=tf.complex64)
            map_fft *= mas_fac      
            
    bin_center = tf.constant([fc+i*dk for i in range(Nbins)])
    bin_lower = bin_center - dk/2
    bin_upper = bin_center + dk/2

    Pk = []
    for i in tqdm(range(Nbins),disable= not verbose):
        Pk.append(tf.reduce_sum(tf.signal.irfft3d(tf.where(tf.math.logical_and(kgrid >= kF*bin_lower[i],kgrid < kF*bin_upper[i]),map_fft,0.+0.j))**2))
    Pk = tf.stack(Pk)
    Bk = []
    for i in tqdm(range(bin_indices.shape[0]),disable= not verbose):
        Bk.append(tf.reduce_sum(\
            tf.signal.irfft3d(tf.where(tf.math.logical_and(kgrid >= kF*bin_lower[bin_indices[i,0]],kgrid < kF*bin_upper[bin_indices[i,0]]),map_fft,0.+0.j))\
                                *  tf.signal.irfft3d(tf.where(tf.math.logical_and(kgrid >= kF*bin_lower[bin_indices[i,1]],kgrid < kF*bin_upper[bin_indices[i,1]]),map_fft,0.+0.j))\
                                *  tf.signal.irfft3d(tf.where(tf.math.logical_and(kgrid >= kF*bin_lower[bin_indices[i,2]],kgrid < kF*bin_upper[bin_indices[i,2]]),map_fft,0.+0.j))))
    Bk = tf.stack(Bk)
    return Pk, Bk

def Bk(delta,BoxSize,fc,dk,Nbins,triangle_type,MAS,verbose=False):
    
    grid = delta.shape[0]
    file_name = f"BFast_BkCounts_LBox{int(BoxSize)}_Grid{int(grid)}_Binning{int(dk)}kF_fc{int(fc)}_NBins{int(Nbins)}_TriangleType{triangle_type}.npy"
        
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
    bin_indices = ((counts['bin_centers'] - fc) // dk).astype(np.int64)
    
    if compute_counts:
        Pk, Bk = _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
        Pk = Pk.numpy()
        Bk = Bk.numpy()
        counts['counts_P'] = Pk * grid**3
        counts['counts_B'] = Bk * grid**6
        np.save(file_name,counts)
        print(f"Saved Triangle Counts to {file_name}")
    
    compute_counts=False
    
    Pk, Bk = _Bk_TF(delta,BoxSize,grid,fc,dk,Nbins,MAS,bin_indices,compute_counts,verbose)
    Pk = Pk.numpy()
    Bk = Bk.numpy()
        
    result = np.ones((len(counts['bin_centers']),8))
    result[:,:3] = counts['bin_centers']
    result[:,3:6] = (Pk/counts['counts_P'])[bin_indices]  * BoxSize**3 / grid**3
    result[:,6] = Bk * BoxSize**6 / counts['counts_B'] / grid**3
    result[:,7] = counts['counts_B']
    
    return result