# diagnostics_batch.py
# Diagnostics for representative motifs:
# square 2D, cubic 3D, chimera-like, random-regular z=6, Watts-Strogatz, Barabasi-Albert.
# Outputs saved to ./diagnostics_output/<motif>/ : spectrum.png, P_of_t.png, rho_lambda.csv, P_of_t.csv, eigvals.txt
#
# Requirements: Python 3.8+, numpy, scipy, networkx, matplotlib, pandas, scikit-learn
# Install dependencies:
# pip install numpy scipy networkx matplotlib pandas scikit-learn

import os, numpy as np, networkx as nx, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

out_root = "diagnostics_output"
os.makedirs(out_root, exist_ok=True)

TS = np.logspace(-3, 1, 50)
t_ref = 1.0  # weight parameter for eps_spec

def compute_diagnostics(G, motif_name):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    zbar = 2*m/n
    A = nx.to_numpy_array(G, dtype=float)
    degs = A.sum(axis=1)
    D = np.diag(degs)
    L = D - A
    eigvals = np.linalg.eigvalsh(L)
    # spectral density
    bins = 120
    vals, edges = np.histogram(eigvals, bins=bins, density=True)
    centers = 0.5*(edges[:-1]+edges[1:])
    # P(t)
    P = np.array([np.mean(np.exp(-eigvals * t)) for t in TS])
    # fit lnP vs lnT
    X = np.log(TS).reshape(-1,1)
    y = np.log(P)
    reg = LinearRegression().fit(X, y)
    slope = reg.coef_[0]
    ds = -2.0 * slope
    r2 = float(reg.score(X,y))
    # epsilon_spec via weight w = exp(-lambda * t_ref)
    w = np.exp(-eigvals * t_ref)
    total_w = float(w.sum())
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, 0.95*total_w))
    if idx >= len(eigvals):
        Lambda = float(eigvals[-1])
        eps_spec = 0.0
    else:
        Lambda = float(eigvals[idx])
        low = float(cum[idx])
        high = float(total_w - low)
        eps_spec = float(high/low) if low>0 else float('inf')
    # Save outputs
    outdir = os.path.join(out_root, motif_name.replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"lambda": centers, "rho": vals}).to_csv(os.path.join(outdir, "rho_lambda.csv"), index=False)
    pd.DataFrame({"t": TS, "P": P}).to_csv(os.path.join(outdir, "P_of_t.csv"), index=False)
    np.savetxt(os.path.join(outdir, "eigvals.txt"), eigvals)
    plt.figure(figsize=(5,3))
    plt.plot(centers, vals, lw=1)
    plt.xlabel("lambda")
    plt.ylabel("rho(lambda)")
    plt.title(motif_name)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "spectrum.png"), dpi=150)
    plt.close()
    plt.figure(figsize=(5,3))
    plt.loglog(TS, P, 'o-')
    plt.xlabel("t")
    plt.ylabel("P(t)")
    plt.title(f"P(t) - {motif_name}\nd_s={ds:.3f}, R2={r2:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "P_of_t.png"), dpi=150)
    plt.close()
    summary = {
        "motif": motif_name,
        "n": int(n),
        "m": int(m),
        "zbar": float(zbar),
        "d_s": float(ds),
        "R2": float(r2),
        "Lambda": float(Lambda),
        "eps_spec": float(eps_spec)
    }
    return summary

# Create motifs and run diagnostics
results = []

# 1) square 2D 20x20
G_sq = nx.grid_2d_graph(20,20)
G_sq = nx.convert_node_labels_to_integers(G_sq)
results.append(compute_diagnostics(G_sq, "square_2D_20x20"))

# 2) cubic 3D 8x8x8
G_cub = nx.grid_graph([8,8,8])
G_cub = nx.convert_node_labels_to_integers(G_cub)
results.append(compute_diagnostics(G_cub, "cubic_3D_8x8x8"))

# 3) chimera-like 8x8 cells (K4,4 per cell)
def make_chimera(L):
    G = nx.Graph()
    node_id = 0
    cell_nodes = {}
    for i in range(L):
        for j in range(L):
            nodes = list(range(node_id, node_id+8))
            G.add_nodes_from(nodes)
            left = nodes[:4]; right = nodes[4:8]
            for u in left:
                for v in right:
                    G.add_edge(u,v)
            cell_nodes[(i,j)] = nodes
            node_id += 8
    for i in range(L):
        for j in range(L):
            nodes = cell_nodes[(i,j)]
            right = nodes[4:8]
            if i+1 < L:
                neigh_left = cell_nodes[(i+1,j)][:4]
                for u in right:
                    for v in neigh_left:
                        G.add_edge(u,v)
            if j+1 < L:
                neigh_right = cell_nodes[(i,j+1)][4:8]
                left = nodes[:4]
                for u in left:
                    for v in neigh_right:
                        G.add_edge(u,v)
    return G

G_chim = make_chimera(8)
results.append(compute_diagnostics(G_chim, "chimera_like_8x8"))

# 4) random-regular n=500 z=6
G_rr = nx.random_regular_graph(d=6, n=500, seed=123)
results.append(compute_diagnostics(G_rr, "random_regular_n500_z6"))

# 5) Watts-Strogatz n=500 k=6 p=0.1
G_ws = nx.watts_strogatz_graph(n=500, k=6, p=0.1, seed=123)
results.append(compute_diagnostics(G_ws, "watts_strogatz_n500_k6_p0.1"))

# 6) Barabasi-Albert n=500 m=3
G_ba = nx.barabasi_albert_graph(n=500, m=3, seed=123)
results.append(compute_diagnostics(G_ba, "barabasi_albert_n500_m3"))

df = pd.DataFrame(results)
df.to_csv(os.path.join(out_root, "summary_table.csv"), index=False)
print("Saved summary_table.csv to", out_root)
print(df.to_string(index=False))
