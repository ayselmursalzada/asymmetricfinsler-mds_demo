"""
SOURCE ATTRIBUTION
══════════════════
[ISUMAP]  Joharinad & Jost, "IsUMap: Manifold Learning Leveraging
          Vietoris-Rips Filtrations", AAAI 2025.
            - Row-wise normalisation: D[i,j] = D_raw[i,j] / max_k(D_raw[i,k])
            - Asymmetric distance matrix construction philosophy

[OURS]    Adaptations for migration data:
            - Log-ratio base distance: D_raw[i,j] = log(max+1) - log(F[i,j]+1)
            - UN bilateral migration format (edges.csv) reader
            - zero-both pair identification utilities

load_asymmetric_graph.py
========================
Loads the bilateral migration dataset (edges.csv) and converts it into
an asymmetric distance matrix suitable for Finsler MDS.

Data format
-----------
Columns: source, target, migration_YEAR_total/male/female  (multiple years)
Each row is a directed edge:  source → target  with migration counts.

Distance conversion
-------------------
Migration flow F[i,j] = # people migrating from country i to country j.
Higher flow ↔ more "closeness" between countries.

We convert to distances via the log-ratio transform:

    D[i,j] = log(max_flow + 1) − log(F[i,j] + 1)

so  D[i,j] = 0  when  F[i,j] = max_flow  (closest pair)
and D[i,j] = log(max_flow+1) when F[i,j] = 0.

Missing directed edges (no migration recorded) are assigned the maximum
distance rather than infinity, so every pair of countries has a finite
distance (no Dijkstra needed).

The matrix is intentionally asymmetric: D[i,j] ≠ D[j,i] because
migration from country A to B ≠ migration from B to A.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_migration_graph(
    filepath: str,
    year: int = 2015,
    gender: str = "total",
    log_transform: bool = True,
    normalize: str = "isumap",
    verbose: bool = True,
) -> dict:
    """
    Load the edges.csv migration dataset and build an asymmetric distance matrix.

    Parameters
    ----------
    filepath     : path to edges.csv
    year         : migration year to use — one of {1990, 1995, 2000, 2005, 2010, 2015}
    gender       : "total" | "male" | "female"
    log_transform: if True, log-ratio base distance (default); else linear
    normalize    : "isumap" (default) — row-wise D[i,j]/max_k(D[i,k])
                   "none"  — use D_raw directly
    verbose      : print summary statistics

    Returns
    -------
    dict with:
      "D_asym"  : (n, n) asymmetric distance matrix
      "D_sym"   : (n, n) symmetrised distance matrix  ½(D + Dᵀ)
      "F"       : (n, n) raw flow matrix (migration counts)
      "nodes"   : list of node IDs (country integer codes)
      "n_nodes" : int
      "year"    : int
    """
    # ── Load CSV ──────────────────────────────────────────────────────────────
    col_names = [
        "source", "target",
        "m1990_total", "m1990_male", "m1990_female",
        "m1995_total", "m1995_male", "m1995_female",
        "m2000_total", "m2000_male", "m2000_female",
        "m2005_total", "m2005_male", "m2005_female",
        "m2010_total", "m2010_male", "m2010_female",
        "m2015_total", "m2015_male", "m2015_female",
    ]
    df = pd.read_csv(filepath, comment="#", header=None, names=col_names)

    flow_col = f"m{year}_{gender}"
    if flow_col not in df.columns:
        raise ValueError(f"Column '{flow_col}' not found. "
                         f"Available years: 1990,1995,2000,2005,2010,2015. "
                         f"Available genders: total, male, female.")

    # ── Build node index ──────────────────────────────────────────────────────
    all_nodes = sorted(set(df["source"]) | set(df["target"]))
    n = len(all_nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}

    # ── Build flow matrix F[i,j] = migration from i→j ─────────────────────────
    F = np.zeros((n, n), dtype=np.float64)
    for _, row in df.iterrows():
        i = node_to_idx[row["source"]]
        j = node_to_idx[row["target"]]
        F[i, j] = max(float(row[flow_col]), 0.0)  # clamp negatives if any

    # Diagonal is self-migration = 0
    np.fill_diagonal(F, 0.0)

    # ── Convert flows to distances ─────────────────────────────────────────────
    max_flow = F.max()

    # ── Base distance ─────────────────────────────────────────────────────────
    if log_transform:
        # D_raw[i,j] = log(max_flow+1) - log(F[i,j]+1)  [OURS]
        # Log-ratio: large flow → small distance. +1 avoids log(0).
        # Zero flow gets max distance, max flow gets distance 0.
        log_max = np.log(max_flow + 1.0)
        D_raw = log_max - np.log(F + 1.0)
    else:
        # D_raw[i,j] = 1 - F[i,j] / max_flow
        D_raw = 1.0 - F / max_flow
    np.fill_diagonal(D_raw, 0.0)

    # ── Normalization (Parvaneh's suggestion) ──────────────────────────────────
    if normalize == "isumap":
        # Row-wise: D[i,j] = D_raw[i,j] / max_k(D_raw[i,k])
        # Each source i is scaled by its own farthest neighbour distance.
        # Mirrors ISUMAP local metric D_ij/d_ik (Joharinad & Jost).  [ISUMAP]
        D_asym = isumap_normalize(D_raw)
    else:
        D_asym = D_raw.copy()

    np.fill_diagonal(D_asym, 0.0)

    D_sym = 0.5 * (D_asym + D_asym.T)

    # ── Summary ───────────────────────────────────────────────────────────────
    if verbose:
        asym_gap = np.abs(D_asym - D_asym.T)
        nonzero_flows = (F > 0).sum()
        print(f"Migration graph loaded  ({flow_col})")
        print(f"  Nodes          : {n}")
        print(f"  Directed edges : {len(df)}")
        print(f"  Non-zero flows : {nonzero_flows}  ({100*nonzero_flows/n**2:.1f}% of all pairs)")
        print(f"  Max flow       : {int(max_flow):,}")
        print(f"  Normalize      : {normalize}")
        print(f"  D_asym range   : [{D_asym.min():.3f}, {D_asym.max():.3f}]")
        print(f"  Mean |D−Dᵀ|    : {asym_gap.mean():.4f}  (asymmetry measure)")
        print(f"  Max  |D−Dᵀ|    : {asym_gap.max():.4f}")

    return {
        "D_asym":  D_asym,
        "D_sym":   D_sym,
        "F":       F,
        "nodes":   all_nodes,
        "n_nodes": n,
        "year":    year,
    }


def load_all_years(filepath: str, gender: str = "total",
                   verbose: bool = False) -> dict:
    """
    Load the migration graph for all available years.

    Returns a dict year → graph_dict.
    """
    years = [1990, 1995, 2000, 2005, 2010, 2015]
    graphs = {}
    for y in years:
        graphs[y] = load_migration_graph(filepath, year=y, gender=gender,
                                          verbose=verbose)
        if verbose:
            print(f"  Year {y} loaded.")
    return graphs


def asymmetry_stats(D_asym: np.ndarray) -> dict:
    """
    Compute scalar asymmetry statistics for a distance matrix.

    Returns
    -------
    dict with 'mean_asym', 'max_asym', 'relative_asym' (Hermitian gap / symmetric norm)
    """
    S = 0.5 * (D_asym + D_asym.T)   # symmetric part
    A = 0.5 * (D_asym - D_asym.T)   # skew-symmetric part
    n = D_asym.shape[0]
    mask = ~np.eye(n, dtype=bool)

    return {
        "mean_asym":     np.abs(A[mask]).mean(),
        "max_asym":      np.abs(A[mask]).max(),
        "relative_asym": np.linalg.norm(A) / (np.linalg.norm(S) + 1e-12),
    }


def isumap_normalize(D_raw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise normalization: D[i,j] = D_raw[i,j] / max_k(D_raw[i,k])

    For each source i, divide all distances by the maximum distance
    from i to any other point (the farthest neighbour of i).
    This mirrors the ISUMAP local metric: D[i,j] = d_ij / d_ik  [ISUMAP]
    where x_k is the farthest neighbour of x_i.

    Properties:
      - D[i,j] in [0, 1] for all i, j
      - D[i,i] = 0
      - Asymmetric: D[i,j] != D[j,i] in general
      - No global max_flow dominance
    """
    D = D_raw.copy()
    np.fill_diagonal(D, 0.0)
    row_max = D.max(axis=1, keepdims=True)           # (n, 1)
    row_max = np.maximum(row_max, eps)               # avoid div by zero
    return D / row_max

