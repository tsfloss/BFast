import argparse
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import BFast
jnp.set_printoptions(linewidth=200, threshold=10, suppress=True)

def main(dim, res, boxsize, mas_order, multipole_axis, slow=False, save=False):
    print(slow)
    print("Dimensions:", dim)
    print("Resolution:", res)
    print("Boxsize:", boxsize)
    print("MAS kernel order:", mas_order)
    print("Pk multipole axis:", multipole_axis)
    field = jax.random.normal(jax.random.PRNGKey(2),(res,)*dim, dtype=jnp.float32).astype(jnp.float64 if jax.config.jax_enable_x64 else jnp.float32) # jnp.float32 to make sure that the random field is the same in single and double precision modes
    bin_edges = jnp.arange(1,res//2+1)
    result_Pk = BFast.Pk(field, boxsize, bin_edges, mas_order=mas_order, multipole_axis=multipole_axis, sharding=None)
    print("k:", result_Pk['k'])
    print("Pk norm:", result_Pk['norm'])
    print("Pk0:", result_Pk['Pk0'])
    print("Pk2:", result_Pk['Pk2'])
    print("Pk4:", result_Pk['Pk4'], end='\n\n')
    if save: jnp.savez(f'reference_data/Pk_dim{dim}_res{res}_mas{mas_order}_multipole{multipole_axis}', **result_Pk)

    bin_edges = jnp.arange(1,res//3+1)
    print("Number of bins:", bin_edges.shape[0] -1)
    print("Bin edges:", bin_edges)
    B_info_onlyclosed = BFast.get_triangles(bin_edges, equilateral=False, open_triangles=False)
    B_info_openclosed = BFast.get_triangles(bin_edges, equilateral=False, open_triangles=True)

    if save: jnp.savez(f'reference_data/triangles_res{res}_onlyclosed', **B_info_onlyclosed)
    if save: jnp.savez(f'reference_data/triangles_res{res}_openclosed', **B_info_openclosed)

    print("Number of triangles (only closed):", B_info_onlyclosed['triangle_centers'].shape[0])
    print("Number of triangles (including open):", B_info_openclosed['triangle_centers'].shape[0], end='\n\n')

    results_norm = BFast.Bk.jit(field, boxsize, **B_info_openclosed, mas_order=mas_order, fast=not slow, only_B=False, compute_norm=True)
    print("Pk norm:", results_norm['Pk'])
    print("Bk norm:", results_norm['Bk'], end='\n\n')

    if save: jnp.savez(f'reference_data/Bk_norm_PB_dim{dim}_res{res}_openclosed', **results_norm)

    results_unnorm = BFast.Bk.jit(field, boxsize, **B_info_openclosed, mas_order=mas_order, fast=not slow, only_B=False, compute_norm=False)
    print("Pk unnormalized:", results_unnorm['Pk'])
    print("Bk unnormalized:", results_unnorm['Bk'], end='\n\n')

    if save: jnp.savez(f'reference_data/Bk_measured_PB_dim{dim}_res{res}_mas{mas_order}_openclosed', **results_unnorm)

    print("Pk normalized:", results_unnorm['Pk']/results_norm['Pk'])
    print("Bk normalized:", results_unnorm['Bk']/results_norm['Bk'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute BFast power spectrum and bispectrum.")
    parser.add_argument("--dim", type=int, default=3, help="Dimension of the field")
    parser.add_argument("--res", type=int, default=64, help="Resolution of the field")
    parser.add_argument("--boxsize", type=float, default=1000.0, help="Physical box size")
    parser.add_argument("--mas_order", type=int, default=2, help="MAS assignment kernel order (NGP=1, CIC=2, TSC=3, PCS=4)")
    parser.add_argument("--multipole_axis", type=int, default=0, help="Axis along which to compute the power spectrum multipoles")
    parser.add_argument("--slow", action="store_true", help="Whether to use the slow algorithm")
    parser.add_argument("--save", action="store_true", help="Whether to save the results to .npz files")
    

    args = parser.parse_args()
    main(args.dim, args.res, args.boxsize, args.mas_order, args.multipole_axis, args.slow, args.save)