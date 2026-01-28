# run_chimera_diagnostics.py
# Generates chimera-like graphs for several seeds and runs diagnostics.
import os, numpy as np, networkx as nx, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

out_root = "chimera_runs"
os.makedirs(out_root, exist_ok=True)

# Parameters (edit if need)
L = 8
intra_w = 1.0
inter_w = 0.4
weight_perturb = 0.05
periodic = True
seeds = [1,2,3]   # создаст chimera_L8_seed1..seed3
TS = np.logspace(-3, 1, 50)
t_ref = 1.0

def make_chimera_like(L=8, intra_w=1.0, inter_w=0.4, periodic=True, weight_perturb=0.05, seed=None):
    np.random.seed(seed)
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
                    G.add_edge(u, v, weight=intra_w)
            cell_nodes[(i,j)] = nodes
            node_id += 8
    for i in range(L):
        for j in range(L):
            nodes = cell_nodes[(i,j)]
            left = nodes[:4]; right = nodes[4:8]
            i2 = (i+1) % L if periodic else i+1
            if i2 < L:
                neigh_left = cell_nodes[(i2,j)][:4]
                for u in right:
                    for v in neigh_left:
                        G.add_edge(u, v, weight=inter_w)
            j2 = (j+1) % L if periodic else j+1
            if j2 < L:
                neigh_right = cell_nodes[(i,j2)][4:8]
                for u in left:
                    for v in neigh_right:
                        G.add_edge(u, v, weight=inter_w)
    # perturb weights
    if weight_perturb and weight_perturb>0:
        for (u,v,d) in list(G.edges(data=True)):
            w = d.get('weight',1.0)
            factor = 1.0 + np.random.normal(0, weight_perturb)
            factor = max(0.01, factor)
            G[u][v]['weight'] = float(w*factor)
    return G, cell_nodes

def compute_diagnostics(G, cell_nodes, motif_name, outdir):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    zbar = 2*m/n
    nodes = list(G.nodes()); idx = {node:i for i,node in enumerate(nodes)}
    W = np.zeros((n,n), dtype=float)
    for u,v,d in G.edges(data=True):
        i = idx[u]; j = idx[v]
        w = d.get('weight',1.0)
        W[i,j] = w; W[j,i] = w
    degs = W.sum(axis=1); D = np.diag(degs); Lmat = D - W
    eigvals = np.linalg.eigvalsh(Lmat)
    # spectral density
    vals, edges = np.histogram(eigvals, bins=120, density=True)
    centers = 0.5*(edges[:-1]+edges[1:])
    # P(t)
    P = np.array([np.mean(np.exp(-eigvals * t)) for t in TS])
    # fit lnP vs ln t
    X = np.log(TS).reshape(-1,1); y = np.log(P)
    reg = LinearRegression().fit(X,y); slope = reg.coef_[0]; ds = -2.0*slope; r2 = float(reg.score(X,y))
    # eps_spec
    w = np.exp(-eigvals * t_ref); total_w = float(w.sum()); cum = np.cumsum(w)
    idx_cut = int(np.searchsorted(cum, 0.95*total_w))
    if idx_cut >= len(eigvals):
        Lambda = float(eigvals[-1]); eps_spec = 0.0
    else:
        Lambda = float(eigvals[idx_cut])
        low = float(cum[idx_cut]); high = float(total_w - low)
        eps_spec = float(high/low) if low>0 else float('inf')
    # anisotropy (embed cells in 2D)
    coords = np.zeros((n,2), dtype=float)
    for (i,j), nodes_cell in cell_nodes.items():
        cx, cy = float(i), float(j)
        offs = np.array([-0.3, -0.1, 0.1, 0.3])
        for k in range(4):
            node = nodes_cell[k]; coords[idx[node]] = [cx-0.2, cy+offs[k]]
        for k in range(4):
            node = nodes_cell[4+k]; coords[idx[node]] = [cx+0.2, cy+offs[k]]
    S = np.zeros((2,2), dtype=float); total_w_edges = 0.0
    for u,v,d in G.edges(data=True):
        i = idx[u]; j = idx[v]; vec = coords[j] - coords[i]; w = d.get('weight',1.0)
        S += w * np.outer(vec, vec); total_w_edges += w
    if total_w_edges>0: S = S/total_w_edges
    eigs_S = np.linalg.eigvalsh(S)
    anisotropy = float(abs(eigs_S[1]-eigs_S[0]) / ((eigs_S[1]+eigs_S[0])/2.0 + 1e-12))
    # save
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame({"lambda": centers, "rho": vals}).to_csv(os.path.join(outdir,"rho_lambda.csv"), index=False)
    pd.DataFrame({"t": TS, "P": P}).to_csv(os.path.join(outdir,"P_of_t.csv"), index=False)
    np.savetxt(os.path.join(outdir,"eigvals.txt"), eigvals)
    with open(os.path.join(outdir,"summary_run.txt"), "w") as f:
        f.write(f"motif={motif_name}\nn={n}\nm={m}\nzbar={zbar}\nd_s={ds}\nR2={r2}\nLambda={Lambda}\neps_spec={eps_spec}\nanisotropy={anisotropy}\n")
    # plots
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5,3)); plt.plot(centers, vals); plt.xlabel("lambda"); plt.ylabel("rho(lambda)")
    plt.title(motif_name); plt.tight_layout(); plt.savefig(os.path.join(outdir,"spectrum.png"), dpi=150); plt.close()
    plt.figure(figsize=(5,3)); plt.loglog(TS, P,'o-'); plt.xlabel("t"); plt.ylabel("P(t)")
    plt.title(f"P(t) - {motif_name}\\nd_s={ds:.3f}, R2={r2:.3f}"); plt.tight_layout(); plt.savefig(os.path.join(outdir,"P_of_t.png"), dpi=150); plt.close()
    return {"motif": motif_name, "n": n, "m": m, "zbar": zbar, "d_s": ds, "R2": r2, "Lambda": Lambda, "eps_spec": eps_spec, "anisotropy": anisotropy}

# run seeds
summaries = []
for s in seeds:
    motif_name = f"chimera_L{L}_seed{s}"
    outdir = os.path.join(out_root, motif_name)
    G, cell_nodes = make_chimera_like(L=L, intra_w=intra_w, inter_w=inter_w, periodic=periodic, weight_perturb=weight_perturb, seed=s)
    summary = compute_diagnostics(G, cell_nodes, motif_name, outdir)
    summaries.append(summary)
df = pd.DataFrame(summaries)
df.to_csv(os.path.join(out_root,"summary_chimera.csv"), index=False)
print("Done. Outputs in", out_root)
