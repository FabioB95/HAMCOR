"""centroid.py — calcola il centroide pesato (R_c, z_c) da uno stato salvato."""
import numpy as np

def compute_centroid(grid, w_best):
    R_c = float(np.sum(w_best * grid.R) / np.sum(w_best))
    z_c = float(np.sum(w_best * grid.z) / np.sum(w_best))
    return R_c, z_c

# subito dopo "best = all_results[best_idx]; w_best = best.final_state.emissivity"
# in ciascuno degli script (run_mrk335.py, run_iras13224.py, run_mcg6.py, run_1h0707.py):
R_c, z_c = compute_centroid(GRID, w_best)
print(f"  Centroide: (R_c, z_c) = ({R_c:.2f}, {z_c:.2f}) r_g")