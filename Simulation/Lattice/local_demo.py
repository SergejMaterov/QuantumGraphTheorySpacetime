#!/usr/bin/env python3
"""
local_demo.py

Local demo: build Lx x Ly square lattice, compute weighted Laplacian spectrum,
spectral density, return probability P(t), fit spectral dimension d_s,
perform block coarse-graining and compare spectra.

Usage:
  python local_demo.py --Lx 8 --Ly 8 --block 2 --noise 0.05 --outdir results
"""

import os
import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import csv

plt.style.use("ggplot")
np.random.seed(12345)

def make_square_lattice(Lx, Ly, periodic=False):
    G = nx.grid_2d_graph(Lx, Ly, periodic=periodic)
    # relabel to ints 0..N-1 and return positions mapping
    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    pos = {mapping[n]: (float(n[0]), float(n[1])) for n in mapping}
    return G, pos

def adjacency_matrix_weighted(G, noise_sigma=0.0):
    # Build weighted adjacency as scipy csr matrix
    n = G.number_of_nodes()
    A = sp.lil_matrix((n,n), dtype=float)
    for u,v in G.edges():
        w = 1.0
        if noise_sigma > 0.0:
            w = w * (1.0 + np.random.normal(loc=0.0, scale=noise_sigma))
            if w < 0:
                w = 0.0
        A[u,v] = w
        A[v,u] = w
    return A.tocsr()

def laplacian_from_weighted_adjacency(W):
    degs = np.array(W.sum(axis=1)).flatten()
    D = sp.diags(degs)
    L = D - W
    return L.tocsr()

def compute_spectrum(L, k=None):
    n = L.shape[0]
    if k is None or k >= n:
        # use dense eigh
        Ld = L.toarray()
        vals, vecs = eigh(Ld)
        return np.real(vals), vecs
    else:
        vals, vecs = spla.eigsh(L, k=k, which='SM')
        idx = np.argsort(vals)
        return np.real(vals[idx]), vecs[:, idx]

def spectral_density(eigs, bins=60):
    hist, edges = np.histogram(eigs, bins=bins, density=True)
    centers = 0.5*(edges[:-1]+edges[1:])
    return centers, hist

def return_probability(eigs, tlist):
    # P(t) = (1/N) sum_i exp(-lambda_i t)
    N = len(eigs)
    P = np.array([np.mean(np.exp(-eigs * t)) for t in tlist])
    return P

def fit_spectral_dimension(tlist, P):
    mask = (P > 0) & (tlist > 0)
    if mask.sum() < 3:
        return np.nan, 0.0, None
    X = np.log(tlist[mask]).reshape(-1,1)
    y = np.log(P[mask])
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    d_s = -2.0 * slope
    r2 = reg.score(X, y)
    return float(d_s), float(r2), reg

def block_coarse_grain(G, pos, Lx, Ly, block):
    assert Lx % block == 0 and Ly % block == 0
    bx = Lx // block
    by = Ly // block
    # mapping node -> (x,y)
    inv = {}
    for node, (x,y) in pos.items():
        inv[(int(x), int(y))] = node
    B = nx.Graph()
    for ix in range(bx):
        for iy in range(by):
            bnode = ix*by + iy
            B.add_node(bnode)
    block_edges = {}
    for u, v in G.edges():
        xu, yu = pos[u]; xv, yv = pos[v]
        bu = (int(xu)//block, int(yu)//block)
        bv = (int(xv)//block, int(yv)//block)
        if bu != bv:
            bi = bu[0]*by + bu[1]
            bj = bv[0]*by + bv[1]
            if bi > bj:
                bi, bj = bj, bi
            block_edges[(bi,bj)] = block_edges.get((bi,bj), 0) + 1
    for (bi,bj), w in block_edges.items():
        B.add_edge(bi, bj, weight=float(w))
    # block positions for plotting
    bpos = {}
    for ix in range(bx):
        for iy in range(by):
            bnode = ix*by + iy
            bpos[bnode] = (ix + 0.5, iy + 0.5)
    return B, bpos

def save_spectrum_csv(eigs, path):
    with open(path, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["index", "lambda"])
        for i, val in enumerate(eigs):
            w.writerow([i, float(val)])

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main(args):
    ensure_dir(args.outdir)
    G, pos = make_square_lattice(args.Lx, args.Ly, periodic=False)
    print(f"Built lattice {args.Lx}x{args.Ly} with N={G.number_of_nodes()} nodes")

    W = adjacency_matrix_weighted(G, noise_sigma=args.noise)
    Lmat = laplacian_from_weighted_adjacency(W)
    eigs, vecs = compute_spectrum(Lmat)
    print("Computed full spectrum (len={})".format(len(eigs)))

    # save spectrum
    save_spectrum_csv(eigs, os.path.join(args.outdir, "spectrum.csv"))

    # Plot lattice
    plt.figure(figsize=(5,5))
    nx.draw(G, pos=pos, node_size=60, with_labels=False)
    plt.title(f"{args.Lx}x{args.Ly} lattice")
    plt.savefig(os.path.join(args.outdir, "lattice.png"), dpi=200)
    plt.close()

    # Spectrum plot
    plt.figure(figsize=(6,3.5))
    plt.plot(eigs, '.-')
    plt.xlabel("index")
    plt.ylabel("lambda")
    plt.title("Laplacian spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "spectrum.png"), dpi=200)
    plt.close()

    # Spectral density
    centers, hist = spectral_density(eigs, bins=args.bins)
    plt.figure(figsize=(5,3.5))
    plt.plot(centers, hist, '-')
    plt.xlabel("lambda")
    plt.ylabel("rho(lambda)")
    plt.title("Spectral density")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "spectral_density.png"), dpi=200)
    plt.close()

    # P(t) and d_s
    tlist = np.logspace(-3, 1, 60)
    P = return_probability(eigs, tlist)
    d_s, r2, reg = fit_spectral_dimension(tlist, P)
    plt.figure(figsize=(6,4))
    plt.loglog(tlist, P, 'o-', label='P(t)')
    if reg is not None:
        yfit = np.exp(reg.intercept_ + reg.coef_[0]*np.log(tlist))
        plt.loglog(tlist, yfit, '--', label=f'fit d_s={d_s:.3f}, R2={r2:.3f}')
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.title("Return probability and spectral-dimension fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "P_t.png"), dpi=200)
    plt.close()

    # coarse-grain
    B, bpos = block_coarse_grain(G, pos, args.Lx, args.Ly, args.block)
    WB = adjacency_matrix_weighted(B, noise_sigma=0.0)  # block adjacency uses integer weights
    LB = laplacian_from_weighted_adjacency(WB)
    eigsB, _ = compute_spectrum(LB)

    # save block spectrum
    save_spectrum_csv(eigsB, os.path.join(args.outdir, "spectrum_block.csv"))

    # compare spectra plot
    plt.figure(figsize=(6,3.5))
    k = min(len(eigs), len(eigsB))
    plt.plot(eigs[:k], '.-', label='orig (first k)')
    plt.plot(np.arange(k), eigsB[:k], 'o-', label='block (first k)')
    plt.legend()
    plt.title("Original vs block spectrum (first k)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "spectrum_compare.png"), dpi=200)
    plt.close()

    # residual simple measure
    k = min(len(eigs), len(eigsB))
    resid = np.linalg.norm(eigs[:k] - eigsB[:k]) / (np.linalg.norm(eigs[:k]) + 1e-12)
    with open(os.path.join(args.outdir, "residual.txt"), "w") as f:
        f.write(f"relative_residual_first_{k} = {resid:.6e}\n")
        f.write(f"d_s = {d_s}, R2 = {r2}\n")
    print("Saved results to", args.outdir)
    print(f"relative residual (first {k}): {resid:.3e}")
    print(f"fitted d_s = {d_s:.3f}, R2 = {r2:.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Local lattice diagnostics demo")
    p.add_argument("--Lx", type=int, default=8, help="Lattice X size")
    p.add_argument("--Ly", type=int, default=8, help="Lattice Y size")
    p.add_argument("--block", type=int, default=2, help="block size for coarse-grain (must divide Lx,Ly)")
    p.add_argument("--noise", type=float, default=0.0, help="multiplicative noise sigma for edge weights")
    p.add_argument("--bins", type=int, default=40, help="bins for spectral density")
    p.add_argument("--outdir", type=str, default="results", help="output directory")
    args = p.parse_args()
    main(args)
