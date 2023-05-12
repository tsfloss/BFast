# BFast
A fast GPU based bispectrum estimator implemented using TensorFlow.
It contains a fast, memory heavy algorithm Bk_fast, that can compute the bispectrum of 2276 triangle configurations in a 256^3 box in a bit more than a second on an Nvidia A100 40GB GPU, taking close to 8GB of memory. This algorithm is too memory heavy for higher resolution grids.

There is also a slower, memory efficient algorithm for higher resolution grids. It computes the same 2276 triangle configurations in a 512^3 box in around 45 seconds using only 6 GB of memory, whereas one CPU core would take around 10 minutes (using the fast algorithm on CPU as implemented in https://github.com/tsfloss/DensityFieldTools).

Requirements:
- numpy
- tensorflow
- tqdm
- matplotlib (for example notebook)
