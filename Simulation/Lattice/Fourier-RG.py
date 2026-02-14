#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def laplacian_dispersion(L):
    k_vals = 2*np.pi*np.arange(L)/L
    lambdas = []
    ks = []

    for nx in range(L):
        for ny in range(L):
            kx = k_vals[nx]
            ky = k_vals[ny]

            lam = 4 - 2*np.cos(kx) - 2*np.cos(ky)

            # радиальный |k|
            k_mag = np.sqrt(
                (2*np.sin(kx/2))**2 +
                (2*np.sin(ky/2))**2
            )

            ks.append(k_mag)
            lambdas.append(lam)

    return np.array(ks), np.array(lambdas)


def fourier_rg(L, b=2):
    ks, lambdas = laplacian_dispersion(L)

    # cutoff
    k_max = ks.max()
    cutoff = k_max / b

    mask = ks <= cutoff

    ks_low = ks[mask]
    lambdas_low = lambdas[mask]

    # rescale momentum
    ks_rescaled = ks_low * b

    # rescale operator
    lambdas_rescaled = lambdas_low * b**2

    return ks_rescaled, lambdas_rescaled


def spectral_dimension(lambdas):
    eigs = np.sort(lambdas)
    eigs = eigs[eigs > 1e-12]

    if len(eigs) < 10:
        return np.nan

    lambda_min = eigs.min()
    lambda_max = eigs.max()

    tmin = 10.0 / lambda_max
    tmax = 0.1 / lambda_min

    if tmin >= tmax:
        return np.nan

    tlist = np.exp(np.linspace(np.log(tmin), np.log(tmax), 100))
    P = np.array([np.mean(np.exp(-eigs*t)) for t in tlist])

    coef = np.polyfit(np.log(tlist), np.log(P), 1)
    slope = coef[0]

    return -2*slope


def main():

    L = 64

    print("Original lattice L =", L)

    ks, lambdas = laplacian_dispersion(L)
    ds_orig = spectral_dimension(lambdas)

    print("d_s original =", ds_orig)

    ks_rg, lambdas_rg = fourier_rg(L, b=2)
    ds_rg = spectral_dimension(lambdas_rg)

    print("d_s after Fourier-RG =", ds_rg)

    # сравнение дисперсии
    plt.figure()
    plt.scatter(ks, lambdas, s=5, label="original")
    plt.scatter(ks_rg, lambdas_rg, s=5, label="RG")
    plt.xlabel("|k|")
    plt.ylabel("lambda")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
