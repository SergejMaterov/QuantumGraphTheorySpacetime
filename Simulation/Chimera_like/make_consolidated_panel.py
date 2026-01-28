#!/usr/bin/env python3
"""
make_consolidated_panel.py

Scan a base directory for subfolders matching pattern '*seed*' (e.g. chimera_L8_seed1, chimera_L8_seed2, ...),
load per-seed diagnostics (rho_lambda.csv, P_of_t.csv, eigvals.txt, summary_run.txt if present),
compute per-seed fits (ln P vs ln t -> d_s), assemble mean±std, and produce a consolidated PNG panel
and a summary CSV.

Usage:
    1) Edit BASE_DIR to point at the folder that contains subfolders chimera_L8_seed1, ...
    2) Install dependencies (one-time):
         pip install numpy pandas matplotlib scikit-learn
    3) Run:
         python make_consolidated_panel.py
Output:
    - BASE_DIR/consolidated_panel_all_seeds.png
    - BASE_DIR/summary_consolidated.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------- CONFIGURE HERE --------
BASE_DIR = "chimera_runs"   # directory that contains chimera_L8_seed1, chimera_L8_seed2, ...
OUT_PNG = os.path.join(BASE_DIR, "consolidated_panel_all_seeds.png")
OUT_CSV = os.path.join(BASE_DIR, "summary_consolidated.csv")
TS_COL_NAME = "t"
P_COL_NAME = "P"
# --------------------------------

def find_seed_dirs(base_dir):
    # find folders matching '*seed*' pattern
    candidates = sorted(glob.glob(os.path.join(base_dir, "*seed*")))
    dirs = [d for d in candidates if os.path.isdir(d)]
    return dirs

def load_seed_data(seed_dir):
    # expects rho_lambda.csv and P_of_t.csv at least
    rho_path = os.path.join(seed_dir, "rho_lambda.csv")
    p_path = os.path.join(seed_dir, "P_of_t.csv")
    eig_path = os.path.join(seed_dir, "eigvals.txt")
    summary_path = os.path.join(seed_dir, "summary_run.txt")  # optional
    if not os.path.exists(rho_path) or not os.path.exists(p_path):
        return None
    rho = pd.read_csv(rho_path)
    p = pd.read_csv(p_path)
    eigs = None
    if os.path.exists(eig_path):
        try:
            eigs = np.loadtxt(eig_path)
        except Exception:
            eigs = None
    summary = {}
    if os.path.exists(summary_path):
        # summary_run.txt has lines like key=value
        try:
            with open(summary_path, "r") as f:
                for line in f:
                    if "=" in line:
                        k,v = line.strip().split("=",1)
                        try:
                            summary[k.strip()] = float(v.strip())
                        except:
                            summary[k.strip()] = v.strip()
        except:
            summary = {}
    return {"dir": seed_dir, "rho": rho, "p": p, "eigs": eigs, "summary": summary}

def fit_ds_from_P(p_df):
    # expects columns 't' and 'P' (floats). Fit ln P = a + b ln t -> d_s = -2 b
    try:
        t = p_df[TS_COL_NAME].values
        P = p_df[P_COL_NAME].values
        mask = (P > 0) & (t > 0)
        if mask.sum() < 3:
            return np.nan, np.nan
        X = np.log(t[mask]).reshape(-1,1)
        y = np.log(P[mask])
        reg = LinearRegression().fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        d_s = -2.0 * slope
        r2 = float(reg.score(X, y))
        return float(d_s), float(r2)
    except Exception:
        return np.nan, np.nan

def compute_eps_spec_from_eigs(eigs, t_ref=1.0, threshold=0.95):
    # w(lambda) = exp(-lambda * t_ref), choose Lambda where cumulative weight >= threshold
    if eigs is None or len(eigs)==0:
        return np.nan, np.nan
    w = np.exp(-np.array(eigs) * t_ref)
    total = w.sum()
    cum = np.cumsum(w)
    idx = int(np.searchsorted(cum, threshold * total))
    if idx >= len(eigs):
        Lambda = float(eigs[-1])
        eps_spec = 0.0
    else:
        Lambda = float(eigs[idx])
        low = float(cum[idx])
        high = float(total - low)
        eps_spec = float(high/low) if low>0 else float('inf')
    return Lambda, eps_spec

def gather_all(base_dir):
    dirs = find_seed_dirs(base_dir)
    data = []
    for d in dirs:
        info = load_seed_data(d)
        if info is None:
            print(f"Skipping {d}: missing rho or P files.")
            continue
        data.append(info)
    return data

def build_summary_table(data_list):
    rows = []
    for info in data_list:
        seed_name = os.path.basename(info["dir"])
        # compute ds and r2 from p if not already present in summary
        d_s, r2 = fit_ds_from_P(info["p"])
        # use summary values if present (they may be more precise)
        s = info.get("summary", {})
        # attempt to get zbar, anisotropy, eps_spec from per-seed summary_run.txt
        zbar = s.get("zbar", np.nan)
        anisotropy = s.get("anisotropy", np.nan)
        # eps_spec: try summary or compute from eigs
        if "eps_spec" in s:
            eps_spec = s.get("eps_spec", np.nan)
            Lambda = s.get("Lambda", np.nan)
        else:
            Lambda, eps_spec = compute_eps_spec_from_eigs(info["eigs"])
        # If summary included d_s or R2, prefer those
        if "d_s" in s:
            try:
                d_s = float(s.get("d_s"))
            except:
                pass
        if "R2" in s:
            try:
                r2 = float(s.get("R2"))
            except:
                pass
        rows.append({
            "seed": seed_name,
            "dir": info["dir"],
            "n_points_rho": len(info["rho"]),
            "d_s": d_s, "R2": r2,
            "Lambda": Lambda, "eps_spec": eps_spec,
            "zbar": zbar, "anisotropy": anisotropy
        })
    df = pd.DataFrame(rows)
    return df

def plot_panel(data_list, summary_df, out_png):
    # overlay spectra (left), overlay P(t) and fits (right), table bottom
    plt.figure(figsize=(11,6))
    # left: spectra overlay
    ax1 = plt.subplot2grid((3,4),(0,0),rowspan=2,colspan=2)
    for info in data_list:
        rho = info["rho"]
        ax1.plot(rho.iloc[:,0], rho.iloc[:,1], lw=1, alpha=0.7, label=os.path.basename(info["dir"]))
    ax1.set_xlabel("λ (Laplacian eigenvalue)")
    ax1.set_ylabel("ρ(λ)")
    ax1.set_title("Spectral density — all seeds (overlay)")
    ax1.legend(fontsize='small', ncol=2, loc='best')

    # right: P(t) overlay + fit per seed
    ax2 = plt.subplot2grid((3,4),(0,2),rowspan=2,colspan=2)
    for info in data_list:
        p = info["p"]
        ts = p[TS_COL_NAME].values
        Pvals = p[P_COL_NAME].values
        ax2.loglog(ts, Pvals, 'o-', markersize=3, alpha=0.6, label=os.path.basename(info["dir"]))
        # fit line
        d_s, r2 = fit_ds_from_P(p)
        try:
            mask = (Pvals>0) & (ts>0)
            X = np.log(ts[mask]).reshape(-1,1)
            y = np.log(Pvals[mask])
            reg = LinearRegression().fit(X,y)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            t_fit = np.array([ts[mask].min()*0.9, ts[mask].max()*1.1])
            ln_fit = intercept + slope * np.log(t_fit)
            P_fit = np.exp(ln_fit)
            ax2.loglog(t_fit, P_fit, '--', linewidth=1, alpha=0.7)
        except Exception:
            pass
    ax2.set_xlabel("t")
    ax2.set_ylabel("P(t)")
    ax2.set_title("Return probability P(t) — overlay")

    # bottom: text table with mean ± std
    ax_table = plt.subplot2grid((3,4),(2,0),colspan=4)
    ax_table.axis('off')
    if not summary_df.empty:
        numeric = summary_df[["d_s","R2","eps_spec","anisotropy","zbar"]].astype(float)
        mean_vals = numeric.mean()
        std_vals = numeric.std()
        lines = [
            "Summary (mean ± std) across seeds:",
            f"d_s: {mean_vals['d_s']:.4f} ± {std_vals['d_s']:.4f}",
            f"R2: {mean_vals['R2']:.4f} ± {std_vals['R2']:.4f}",
            f"eps_spec: {mean_vals['eps_spec']:.4f} ± {std_vals['eps_spec']:.4f}",
            f"anisotropy: {mean_vals['anisotropy']:.4f} ± {std_vals['anisotropy']:.4f}",
            f"zbar: {mean_vals['zbar']:.4f} ± {std_vals['zbar']:.4f}",
            "",
            f"Detected seed folders: {len(summary_df)}"
        ]
    else:
        lines = ["No seed summaries available."]
    ax_table.text(0.01, 0.5, "\n".join(lines), fontsize=10, va='center', family='monospace')

    plt.tight_layout(rect=[0,0.03,1,0.97])
    plt.suptitle("Chimera diagnostics — all seeds", y=0.995, fontsize=12)
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote", out_png)

def main():
    if not os.path.exists(BASE_DIR):
        raise SystemExit(f"Base directory '{BASE_DIR}' not found. Edit BASE_DIR at top of script.")
    data_list = gather_all(BASE_DIR)
    if len(data_list) == 0:
        raise SystemExit("No seed folders with rho/P files found under BASE_DIR.")
    summary_df = build_summary_table(data_list)
    # try to compute eps_spec from eigs for any missing values if necessary
    # ensure eps_spec column present
    if "eps_spec" not in summary_df.columns:
        summary_df["eps_spec"] = np.nan
    # save consolidated CSV
    summary_df.to_csv(OUT_CSV, index=False)
    print("Saved consolidated summary:", OUT_CSV)
    # plot consolidated panel
    plot_panel(data_list, summary_df, OUT_PNG)

if __name__ == "__main__":
    main()
