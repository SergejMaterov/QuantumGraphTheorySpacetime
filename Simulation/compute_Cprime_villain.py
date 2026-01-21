# compute_Cprime_villain.py (2026 Sergej Materov)
import networkx as nx
import numpy as np, pandas as pd, math, time, os
from pathlib import Path
import matplotlib.pyplot as plt

def build_chimera_like(cell_rows, cell_cols, cell_size=4):
    G = nx.Graph()
    for ci in range(cell_rows):
        for cj in range(cell_cols):
            for part in [0,1]:
                for idx in range(cell_size):
                    G.add_node((ci,cj,part,idx))
            for i in range(cell_size):
                for j in range(cell_size):
                    G.add_edge((ci,cj,0,i),(ci,cj,1,j))
    for ci in range(cell_rows):
        for cj in range(cell_cols):
            if cj+1 < cell_cols:
                for i in range(cell_size):
                    G.add_edge((ci,cj,0,i),(ci,cj+1,0,i))
            if ci+1 < cell_rows:
                for j in range(cell_size):
                    G.add_edge((ci,cj,1,j),(ci+1,cj,1,j))
    return nx.convert_node_labels_to_integers(G)

def build_grid(rows, cols):
    G = nx.grid_2d_graph(rows, cols)
    return nx.convert_node_labels_to_integers(G)

def build_rand_reg(z, N=200, seed=0):
    return nx.random_regular_graph(z, N, seed=seed)

def find_4cycles_edges(G):
    cycles = set()
    nodes = list(G.nodes())
    for u in nodes:
        for v in G.neighbors(u):
            if v <= u: continue
            for w in G.neighbors(v):
                if w==u: continue
                for x in G.neighbors(w):
                    if x==v or x==u: continue
                    if G.has_edge(x,u):
                        edges = []
                        cyc = (u,v,w,x)
                        for i in range(4):
                            a = cyc[i]; b = cyc[(i+1)%4]
                            edges.append(tuple(sorted((a,b))))
                        cycles.add(frozenset(edges))
    return [list(es) for es in cycles]

def build_link_L(G):
    edges = list(G.edges())
    m = len(edges)
    edge_to_idx = {}
    for i,(a,b) in enumerate(edges):
        edge_to_idx[(a,b)] = i
        edge_to_idx[(b,a)] = i
    A = np.zeros((m,m), dtype=float)
    for i,(a,b) in enumerate(edges):
        for j,(c,d) in enumerate(edges):
            if i==j: continue
            if a==c or a==d or b==c or b==d:
                A[i,j] = 1.0
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    return edges, edge_to_idx, L

def run_for_graph(G, name, freqs_hz, T=2.7, max_cycles=1000):
    h_ev_s = 4.135667696e-15
    kB_eV_per_K = 8.617333262145e-5
    beta = 1.0/(kB_eV_per_K * T)
    cycles = find_4cycles_edges(G)
    if len(cycles)==0:
        return []
    sel_cycles = cycles if len(cycles)<=max_cycles else cycles[:max_cycles]
    edges, edge_to_idx, L_link = build_link_L(G)
    m = len(edges)
    P = len(sel_cycles)
    V = np.zeros((m,P))
    for ci, edge_list in enumerate(sel_cycles):
        for e in edge_list:
            idx = edge_to_idx.get(e, None)
            if idx is None:
                idx = edge_to_idx.get((e[1], e[0]), None)
            if idx is not None:
                V[idx,ci] = 1.0
    out = []
    for f in freqs_hz:
        EJ = h_ev_s * f
        a = beta * EJ
        b = 0.1 * a
        M = a * np.eye(m) + b * L_link
        X = np.linalg.solve(M, V)
        S_local = np.sum(V * X, axis=0)
        Savg = float(np.mean(S_local))
        Cprime_local = 0.5 * Savg
        C_geom_per_node = len(cycles)/G.number_of_nodes()
        kappa_node = (beta * EJ)**2 * Cprime_local * C_geom_per_node if Cprime_local>0 else 0.0
        alpha = 1.0/(8.0*math.pi*kappa_node) if kappa_node>0 else float('inf')
        out.append({
            'graph': name,
            'N_nodes': G.number_of_nodes(),
            'N_edges': G.number_of_edges(),
            'C_geom_per_node': C_geom_per_node,
            'cycles_used': P,
            'freq_Hz': f,
            'EJ_eV': EJ,
            'betaEJ': beta*EJ,
            'Cprime_local': Cprime_local,
            'kappa_node': kappa_node,
            'alpha': alpha
        })
    return out

def main():
    freqs = [10e9, 50e9, 130e9]  # adjust if needed
    motifs = {
        'chimera_like_5x5': build_chimera_like(5,5,4),
        'square_grid_20x20': build_grid(20,20),
        'rand_reg_3': build_rand_reg(3,200,seed=1),
        'rand_reg_4': build_rand_reg(4,200,seed=2),
        'rand_reg_6': build_rand_reg(6,200,seed=3),
        'rand_reg_8': build_rand_reg(8,200,seed=4)
    }
    results = []
    outdir = Path('./results')
    outdir.mkdir(exist_ok=True)
    for name,G in motifs.items():
        print("Running for", name)
        r = run_for_graph(G, name, freqs, T=2.7, max_cycles=1000)
        results += r
        # quick save
        pd.DataFrame(results).to_csv(outdir/'Cprime_partial_results.csv', index=False)
    df = pd.DataFrame(results)
    df.to_csv(outdir/'Cprime_exact_results.csv', index=False)
    print("Saved results to", outdir/'Cprime_exact_results.csv')

if __name__=='__main__':
    main()
