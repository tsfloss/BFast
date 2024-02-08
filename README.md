# BFast
A fast GPU based bispectrum estimator implemented with jax.

It contains a fast, more memory heavy algorithm, that can compute the bispectrum of 2276 triangle configurations in a 256^3 box in less than a second on a V100/A100, using float32 precision (~1.5x for float64).

There is also a slower, memory efficient algorithm for higher resolution grids or more bins. It computes the same 2276 triangle configurations in a 512^3 box in under 20 seconds on an A100, using float32 precision (~1.5x for float64).

Requirements:
- numpy
- jax
- jax-tqdm
- matplotlib (for example notebook)

Installation: clone the repository and cd into the directory then install using 'pip install .' (optionally add the -e flag to install in developer mode)

# Demonstration


```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import BFast
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.device_count()
```




    1




```python
BoxSize = 1000.
kF = 2*np.pi/BoxSize
grid = 256
```


```python
df = np.load(f"df_m_256_PCS_z=0.npy")
df.dtype
```




    dtype('float32')




```python
help(BFast.Bk)
```

    Help on function Bk in module BFast.BFast:
    
    Bk(delta, BoxSize, fc, dk, Nbins, triangle_type='All', open_triangles=True, MAS=None, fast=True, precision='float32', file_path='./', verbose=False)
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
    



```python
%time Bks_32 = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=True,precision='float32',verbose=True)
%time Bks_64 = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=True,precision='float64',verbose=True)
```

    No counts file found, computing this first!
    Considering 2276 Triangle Configurations (All), with kmax = 0.5184
    Saved Triangle Counts to ./BFast_BkCounts_Grid256_BoxSize1000.00_BinSize3.00kF_FirstCenter3.00kF_NumBins27_TriangleTypeAll_OpenTrianglesTrue_Precisionfloat32.npy
    CPU times: user 2.87 s, sys: 753 ms, total: 3.62 s
    Wall time: 5.52 s
    No counts file found, computing this first!
    Considering 2276 Triangle Configurations (All), with kmax = 0.5184
    Saved Triangle Counts to ./BFast_BkCounts_Grid256_BoxSize1000.00_BinSize3.00kF_FirstCenter3.00kF_NumBins27_TriangleTypeAll_OpenTrianglesTrue_Precisionfloat64.npy
    CPU times: user 2.62 s, sys: 788 ms, total: 3.41 s
    Wall time: 5.25 s



```python
plt.semilogy(Bks_32[:,-2])
```




    [<matplotlib.lines.Line2D at 0x7f3f442a4eb0>]




    
![png](example_notebook_files/example_notebook_7_1.png)
    


## The first time jax compiles certain parts and compute triangle counts, a next call is blazing fast:


```python
%time Bks_32 = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=True,precision='float32',verbose=False)
%time Bks_64 = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=True,precision='float64',verbose=False)
```

    CPU times: user 408 ms, sys: 167 ms, total: 576 ms
    Wall time: 521 ms
    CPU times: user 565 ms, sys: 303 ms, total: 868 ms
    Wall time: 806 ms


## Float32 precision is very accurate but faster:


```python
plt.semilogy(np.abs((Bks_32[:,-2]-Bks_64[:,-2])/Bks_64[:,-2]))
```




    [<matplotlib.lines.Line2D at 0x7f3f3c3f6fb0>]




    
![png](example_notebook_files/example_notebook_11_1.png)
    


## There is also a slower but more memory friendly algorithm for larger boxes or more bins


```python
%time Bks_32_slow = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=False,precision='float32',verbose=False)
%time Bks_64_slow = BFast.Bk(df,BoxSize,3.,3.,27,'All',MAS='PCS',fast=False,precision='float64',verbose=False)
```

    CPU times: user 3.7 s, sys: 67.8 ms, total: 3.76 s
    Wall time: 4.9 s
    CPU times: user 5.61 s, sys: 72 ms, total: 5.68 s
    Wall time: 6.92 s


## Again, Float32 precision is very accurate but faster:


```python
plt.semilogy(np.abs((Bks_32_slow[:,-2]-Bks_64_slow[:,-2])/Bks_64_slow[:,-2]))
```




    [<matplotlib.lines.Line2D at 0x7f3f1466a320>]




    
![png](example_notebook_files/example_notebook_15_1.png)
    


## There is also a power spectrum method with a binning of kF:


```python
help(BFast.Pk)
```

    Help on function Pk in module BFast.BFast:
    
    Pk(delta, BoxSize, MAS=None, left_inclusive=True, precision='float32')
        Computes binned powerspectrum of field
        
        Parameters:
        -----------
        delta: array
            Real field to compute powerspectrum of
        BoxSize:
            Size of box in comoving units (Mpc/h) such that power spectrum has units (Mpc/h)^3
        MAS: str, optional (default=None)
            Mass Assignment Scheme window function to compensate for (options are NGP,CIC,TSC,PCS)
        left_inclusive: bool, optional (default=True)
            If True, uses left-inclusive bins. If False, uses right-inclusive bins instead.
        
        Returns:
        --------
        result: numpy.ndarray
            An array of shape (Nbins,3) containing the power spectrum and related information.
            The columns contain: {mean k of the bin, P(k), number of modes in bin}
    



```python
Pks_32_left = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=True,precision='float32')
Pks_32_right = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=False,precision='float32')
Pks_64_left = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=True,precision='float64')
Pks_64_right = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=False,precision='float64')

plt.loglog(Pks_32_left[:,0],Pks_32_left[:,1])
plt.loglog(Pks_32_right[:,0],Pks_32_right[:,1])
```




    [<matplotlib.lines.Line2D at 0x7f3f1449e4a0>]




    
![png](example_notebook_files/example_notebook_18_1.png)
    



```python
%time Pks_32_left = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=True,precision='float32')
%time Pks_32_right = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=False,precision='float32')
%time Pks_64_left = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=True,precision='float64')
%time Pks_64_right = BFast.Pk(df,1000.,MAS='PCS',left_inclusive=False,precision='float64')
```

    CPU times: user 329 ms, sys: 8.39 ms, total: 337 ms
    Wall time: 258 ms
    CPU times: user 313 ms, sys: 10.4 ms, total: 323 ms
    Wall time: 248 ms
    CPU times: user 303 ms, sys: 18.4 ms, total: 321 ms
    Wall time: 281 ms
    CPU times: user 294 ms, sys: 25.8 ms, total: 320 ms
    Wall time: 280 ms


## Float32 precision is very accurate, but the speed up is minimal in this case (at this grid size!)


```python
plt.loglog(Pks_64_left[:,0],np.abs((Pks_32_left[:,-2]-Pks_64_left[:,-2])/Pks_64_left[:,-2]))
plt.show()
plt.loglog(Pks_64_right[:,0],np.abs((Pks_32_right[:,-2]-Pks_64_right[:,-2])/Pks_64_right[:,-2]))
plt.show()
```


    
![png](example_notebook_files/example_notebook_21_0.png)
    



    
![png](example_notebook_files/example_notebook_21_1.png)
    


## Finally, there is also a cross powerspectrum:


```python
%time Px = BFast.xPk(df,df**2.,1000.,MAS=['PCS','PCS'])
```

    CPU times: user 634 ms, sys: 16.5 ms, total: 651 ms
    Wall time: 1.05 s



```python
plt.loglog(Px[:,0],Px[:,1],label='Pk 1')
plt.loglog(Px[:,0],Px[:,2],label='Pk 2')
plt.loglog(Px[:,0],Px[:,3],label='Pk x')
plt.legend()
plt.show()
```


    
![png](example_notebook_files/example_notebook_24_0.png)
    



```python
plt.semilogx(Px[:,0],Px[:,4],label="correlation coefficient")
plt.ylim(0,1)
plt.legend()
plt.show()
```


    
![png](example_notebook_files/example_notebook_25_0.png)
    

