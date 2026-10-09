#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import numpy as np
from kerr_lags import corona_to_disk_time

# corona tipo griglia test: R=5,z=2 → r,theta
R, z, phi_c = 5.0, 2.0, 0.0
r_c = np.sqrt(R**2 + z**2)
theta_c = np.arctan2(R, z)
r_d, phi_d = 6.0, 0.5

print(f"corona r={r_c:.3f} theta={theta_c:.3f} phi={phi_c}")
print(f"disk   r={r_d} phi={phi_d}")
t = corona_to_disk_time(r_c, theta_c, phi_c, r_d, phi_d, a=0.0)
print(f"t_corona_disk = {t}")