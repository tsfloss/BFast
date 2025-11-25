# BFast

BFast is an FFT-based bispectrum estimator written entirely in jax, allowing for easy (multi-)GPU acceleration and gradients.

## Installation

To install BFast simply run
```
pip install git+https://github.com/tsfloss/BFast.git
```
or clone the repository and ```pip install .``` from there (optionally with ```-e```).

To benefit from GPU acceleration, make sure [jax is installed appropriately](https://docs.jax.dev/en/latest/installation.html#installation), e.g.
```
pip install -U "jax[cuda13]" # or "jax[cuda12]"
```

## Quickstart

Estimating a bispectrum can be as simple as


```python
import BFast
import jax.numpy as np

field = YOUR_DENSITY_FIELD_HERE  # Your D-dimensional field (1D, 2D, or 3D array)
boxsize = 1000. # The physical side length of your field in arbitrary units (e.g. Mpc/h)

# Bin edges in units of the fundamental mode kF = 2π/boxsize
# This estimates the bispectrum in linear bins [kF, 3kF, 6kF, 9kF, ..., (resolution//3)kF]
bin_edges = jnp.arange(1, resolution//3, 3) 

Bk_results = BFast.Bk(density_field, boxsize=boxsize, bin_edges=bin_edges) # returns a dictionary with bispectrum results

# Triangle configurations as a (N_triangles, 3) array of k-values in units of [boxsize]^-1 (e.g. (h/Mpc))
Bk_results['triangles'] 

# Estimated (N_triangle) bispectrum values in units of [boxsize]^(2*D) (e.g. (Mpc/h)^(2*D))
Bk_results['Bk'] 

# Estimated (N_bins) power spectrum values in units of [boxsize]^D (e.g. (Mpc/h)^D)
BK_results['Pk'] 
```

See the notebooks and scripts folders for more detailed examples, e.g. on how to efficiently compute the bispectrum from many fields, and how to use distributed resources.


