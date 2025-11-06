# BFast

BFast is a FFT-based bispectrum estimator written entirely in jax, allowing for easy (multi-)GPU acceleration.

## Example usage


```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"

import jax
jax.config.update('jax_enable_x64', False)
import jax.numpy as jnp
import BFast
from BFast.core.jax_utils import show_hlo_info
import matplotlib.pyplot as plt
%matplotlib inline
```

We consider a bispectrum estimation in three dimensions


```python
dim = 3
res = 256
boxsize = 1000.
```

We compute the bispectrum up to the estimator limit of $k_{\rm{max}} = \frac{2}{3}k_{\rm{Nyquist}}$.

BFast expects bin edges in units of the fundamental mode $k_{F} = \frac{2*\pi}{L}$, where $L$ is the boxsize.

To compute the bispectrum up to said limit, with a linear binning of width $3k_{\rm{F}}$, we can take the following binning:


```python
bins = jnp.arange(1, res//3, 3)
nbins = bins.shape[0] - 1
nbins
```




    27



The closed (and nearly-closed ('open')) triangle configurations of bins can be obtained as follows, and yields 2276 triangle configurations for this binning


```python
%time B_info = BFast.get_triangles(bins, open_triangles=True)
B_info['triangle_centers'].shape
```

    CPU times: user 627 ms, sys: 58 ms, total: 685 ms
    Wall time: 1.03 s





    (2276, 3)



Let's set up a field to compute the bispectrum of:


```python
field = jax.random.normal(jax.random.PRNGKey(2),(res,)*dim, dtype=jnp.float32)
```

### Single bispectrum estimation

To estimate the bispectrum, we provide two different implementations of the algorithm.

The fast (```fast=True```) algorithm precomputes the binned fields, and then loops over triangle configurations.
Storing the binned fields obviously comes with a memory cost.

The slower algorithm instead (smartly) recomputes the binned fields per triangle configuration, saving a significant amount of memory at high resolution and/or many bins.

The high-level function ```BFast.Bk``` computes the triangle configuration, bispectrum normalization, and estimates the bispectrum, given some binning.

When the argument ```Only_B=False``` is passed, the code will also estimate the power spectrum of the bins, which can be useful for shot-noise subtraction/modeling.

The codes outputs a dictionary with the triangle centers of the configurations (now in units of h/Mpc), the corresponding bispectrum ```'Bk'```, and when ```Only_B=False``` the power spectrum ```'Pk'```

The timings shown below are for a single NVIDIA A100 with 64GB of memory.



```python
%time results_fast = BFast.Bk(field, boxsize, bins, fast=True, jit=True, only_B=False)
%timeit BFast.Bk(field, boxsize, bins, mas_order=2, fast=True, jit=True, only_B=False)
results_fast.keys()
```

    CPU times: user 1.04 s, sys: 337 ms, total: 1.37 s
    Wall time: 1.51 s
    692 ms ± 144 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)





    dict_keys(['Bk', 'Pk', 'triangle_centers'])




```python
%time results_slow = BFast.Bk(field, boxsize, bins, fast=False, jit=True)
%time BFast.Bk(field, boxsize, bins, fast=False, jit=True)
```

    CPU times: user 7.49 s, sys: 34.2 ms, total: 7.53 s
    Wall time: 7.65 s
    CPU times: user 6.9 s, sys: 8.34 ms, total: 6.91 s
    Wall time: 6.94 s





    {'Bk': Array([-24938.418  ,  78022.22   , -36774.977  , ...,   -176.56902,
               368.32904,  -1803.0212 ], dtype=float32),
     'Pk': Array([51.238613, 63.30222 , 61.94866 , 60.457092, 60.146046, 59.6874  ,
            60.288227, 59.36148 , 59.279675, 59.472816, 60.645836, 59.347973,
            59.783005, 59.61014 , 59.85768 , 59.635014, 59.459198, 59.50441 ,
            59.758785, 59.753773, 59.716793, 59.546345, 59.47245 , 59.43342 ,
            59.730103, 59.50768 , 59.559887], dtype=float32),
     'triangle_centers': Array([[0.01570796, 0.01570796, 0.01570796],
            [0.03455752, 0.01570796, 0.01570796],
            [0.03455752, 0.03455752, 0.01570796],
            ...,
            [0.5057965 , 0.5057965 , 0.46809736],
            [0.5057965 , 0.5057965 , 0.4869469 ],
            [0.5057965 , 0.5057965 , 0.5057965 ]],      dtype=float32, weak_type=True)}



### Repeated bispectrum estimations

One may also want to compute the bispectrum for multiple fields. In this case using the high-level function ```BFast.Bk``` is wasteful, as the triangle configurations and corresponding bispectrum normalization only have to be computed once.

In this case, one can use the lower-level function ```BFast.bispectrum``` to once precompute the normalization (with ```compute_norm=True```), given the triangle configurations obtained above using ```BFast.get_triangles```. A jit version is already made available with ```BFast.bispectrum.jit```.

As expected it is roughly twice as fast as ```BFast.Bk```, because it does not compute the normalization:


```python
norm = BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=True, only_B=False)
%time results_unnormalized = BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=False, only_B=False)
%timeit BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=False)
```

    CPU times: user 382 ms, sys: 162 ms, total: 544 ms
    Wall time: 581 ms
    330 ms ± 122 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


The normalized bispectra and power spectra can then be obtained by normalizing with the precomputed norm


```python
results_normalized = {}
results_normalized['Pk'] = results_unnormalized['Pk']/norm['Pk']
results_normalized['Bk'] = results_unnormalized['Bk']/norm['Bk']
```

And we can check whether all results so far agree


```python
for key in results_normalized.keys():
    assert jnp.allclose(results_fast[key], results_slow[key])
    assert jnp.allclose(results_fast[key], results_normalized[key], rtol=5e-4)
    print(key,'are the same')
```

    Pk are the same
    Bk are the same


Finally, let's check the memory usage. We see that the slow algorithm uses less than a quarter of the memory, at the cost of being almost 10 times as slow.


```python
show_hlo_info(BFast.bispectrum.jit, field, boxsize, **B_info, fast=True, only_B=False, width=0)
show_hlo_info(BFast.bispectrum.jit, field, boxsize, **B_info, fast=False, only_B=False, width=0)
```

    --------  Memory usage of bispectrum  ---------
    const : 68 B
    code  : 39.5 kB
    temp  : 1.9 GB
    arg   : 64.1 MB
    output: 35.7 kB
    alias : 0 B
    peak  : 2.0 GB



    

    


    --------  Memory usage of bispectrum  ---------
    const : 64 B
    code  : 44.5 kB
    temp  : 417.3 MB
    arg   : 64.1 MB
    output: 35.7 kB
    alias : 0 B
    peak  : 481.3 MB



    

    


### Sharded (Multi-GPU)

In three dimension, for large fields, one might want to distribute the calculation over multiple devices/GPUs, to reduce the memory footprint.

To this end, we use the convenient sharding functionality of jax.
For the high-level function ```BFast.Bk```, the user only has to pass ```jit=True, sharded=True```, and the code will automatically distribute the calculation over all devices that jax has detected (i.e. those reported by ```jax.devices()```). To use change the amount of devices used, simply set the available devices through the environment variable ```CUDA_VISIBLE_DEVICES```, before launching the code.

In our case, we see that we have four devices available


```python
jax.devices()
```




    [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2), CudaDevice(id=3)]



We can now simply run the estimator as before, with ```jit=True, sharded=True```


```python
%time results_fast_sharded = BFast.Bk(field, boxsize, bins, fast=True, jit=True, sharded=True, only_B=False)
%timeit BFast.Bk(field, boxsize, bins, fast=True, jit=True, sharded=True, only_B=False)
```

    CPU times: user 2.16 s, sys: 2.6 s, total: 4.77 s
    Wall time: 3.59 s
    309 ms ± 778 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)


For the case of repeated bispectrum calculations, the user has manually provide the sharded field and sharding details to```BFast.bispectrum.jit```.

To simplify this as much as possible, we provide the function ```BFast.shard_3D_array```, which can then be passed together with the ```sharding``` argument to ```BFast.bispectrum.jit```. 

***Note that for a properly distributed computation, one has to use the jitted version!***


```python
field = BFast.shard_3D_array(field)
sharding = field.sharding
```


```python
%time BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=False, only_B=False, sharding=sharding)
%timeit BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=False, only_B=False, sharding=sharding)
```

    CPU times: user 732 ms, sys: 226 ms, total: 958 ms
    Wall time: 614 ms
    146 ms ± 122 μs per loop (mean ± std. dev. of 7 runs, 10 loops each)



```python
show_hlo_info(BFast.bispectrum.jit, field, boxsize, **B_info, mas_order=2, sharding=sharding, fast=True, width=0)
```

    --------  Memory usage of bispectrum  ---------
    const : 104 B
    code  : 70.3 kB
    temp  : 505.7 MB
    arg   : 16.1 MB
    output: 35.6 kB
    alias : 0 B
    peak  : 505.3 MB



    

    



```python
for key in results_normalized.keys():
    assert jnp.allclose(results_fast[key], jax.device_get(results_fast_sharded[key]),rtol=5e-4)
    print(key,'are the same')
```

    Pk are the same
    Bk are the same


We see that for this case, we get a factor >2 speedup and of course a quarter of the memory footprint per device!

Finally, let's try something more impressive


```python
dim = 3
res = 840
boxsize = 1000.
```


```python
bins = jnp.arange(1, res//3, 3)
nbins = bins.shape[0] - 1
nbins
```




    92




```python
%time B_info = BFast.get_triangles(bins, open_triangles=True)
B_info['triangle_centers'].shape
```

    CPU times: user 669 ms, sys: 25.1 ms, total: 695 ms
    Wall time: 1.04 s





    (72289, 3)




```python
field = jax.random.normal(jax.random.PRNGKey(2),(res,)*dim, dtype=jnp.float32)
field = BFast.shard_3D_array(field)
```


```python
%time BFast.bispectrum.jit(field, boxsize, **B_info, fast=True, compute_norm=False, only_B=False, sharding=field.sharding);
```

    CPU times: user 2min 35s, sys: 3min 7s, total: 5min 43s
    Wall time: 1min 26s



```python
show_hlo_info(BFast.bispectrum.jit, field, boxsize, **B_info, fast=True, only_B=False, sharding=field.sharding, width=0)
```

    --------  Memory usage of bispectrum  ---------
    const : 68 B
    code  : 61.0 kB
    temp  : 53.3 GB
    arg   : 566.9 MB
    output: 1.1 MB
    alias : 0 B
    peak  : 53.3 GB



    

    


We see that the fast algorithm can be used to compute around $72000$ triangles at a resolution of $840^3$ in 1.5 minutes!
