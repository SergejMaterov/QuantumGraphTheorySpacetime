# appendix_D4_compute_Lambda.py
import numpy as np
import pandas as pd

hbar = 1.054571817e-34   # J*s
eV = 1.602176634e-19     # J
ell_p = 1.616255e-35     # m

def E_from_freq(f_hz, angular=False):
    # if angular=False, assume f_hz is ordinary frequency in Hz: use omega=2*pi*f
    # if angular=True, f_hz is already angular frequency
    omega = 2*np.pi*f_hz if not angular else f_hz
    return hbar * omega  # in joules

def Lambda_uv(E_J, ell0_m):
    return E_J * (ell0_m/ell_p)**2  # joules

# parameter grid (example)
freqs = np.array([1e6, 20e6, 100e6, 1e9])  # Hz
ell0_factors = np.logspace(1, 30, num=10)  # factor * ell_p

rows = []
for f in freqs:
    E = E_from_freq(f, angular=False)
    for factor in ell0_factors:
        ell0 = factor * ell_p
        Lam_J = Lambda_uv(E, ell0)
        Lam_eV = Lam_J / eV
        rows.append({"freq_Hz": f, "ell0_factor": factor, "ell0_m": ell0,
                     "E_J_J": E, "Lambda_eV": Lam_eV, "Lambda_TeV": Lam_eV/1e12})
df = pd.DataFrame(rows)
df.to_csv("D4_Lambda_grid.csv", index=False)
print("Wrote D4_Lambda_grid.csv")
