"""
demo_finsler.py — Finsler MDS on bilateral migration data.
Visualization: only Y embedding, no drift arrows.
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cdist

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from load_asymmetric_graph import load_migration_graph, asymmetry_stats
from finsler_mds import finsler_mds, _cmds_init, _compute_B_from_Y, finsler_asymmetric_distances

OUT_DIR = HERE / "finsler_results"
OUT_DIR.mkdir(exist_ok=True)
EDGES_CSV = Path("/mnt/user-data/uploads/edges.csv")

def stress(D_target, D_rec):
    mask = ~np.eye(len(D_target), dtype=bool)
    return np.sqrt(((D_target[mask]-D_rec[mask])**2).sum()) / (np.sqrt((D_target[mask]**2).sum())+1e-12)

graph  = load_migration_graph(EDGES_CSV, year=2015, verbose=True)
D_asym = graph["D_asym"]
D_sym  = 0.5*(D_asym+D_asym.T)
nodes  = graph["nodes"]

print(f"\nRelative asymmetry: {asymmetry_stats(D_asym)['relative_asym']*100:.2f}%")

Y_cmds       = _cmds_init(D_sym, d=2)
stress_cmds  = stress(D_asym, cdist(Y_cmds, Y_cmds))
print(f"cMDS stress  : {stress_cmds:.4f}")

result        = finsler_mds(D_asym, d=2, n_epochs=3000, lr=5e-3,
                             max_epochs_no_improvement=300, min_epochs=200, verbose=True)
Y_finsler     = result["Y"]
B_final       = _compute_B_from_Y(Y_finsler, D_asym)
D_finsler_rec = finsler_asymmetric_distances(Y_finsler, B_final)
stress_fin    = stress(D_asym, D_finsler_rec)
print(f"Finsler stress: {stress_fin:.4f}  (improvement: {(stress_cmds-stress_fin)/stress_cmds*100:.1f}%)")

colors = (np.array(nodes)-min(nodes))/(max(nodes)-min(nodes))
cmap   = plt.cm.viridis

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Finsler MDS — Bilateral Migration 2015", fontsize=14, fontweight="bold")

axes[0].scatter(Y_cmds[:,0], Y_cmds[:,1], c=colors, cmap=cmap, s=25, alpha=0.8, linewidths=0.3, edgecolors='k')
axes[0].set_title(f"Classical MDS (symmetric)\nstress = {stress_cmds:.3f}"); axes[0].axis("equal")

sc = axes[1].scatter(Y_finsler[:,0], Y_finsler[:,1], c=colors, cmap=cmap, s=25, alpha=0.8, linewidths=0.3, edgecolors='k')
axes[1].set_title(f"Finsler MDS (asymmetric)\nstress = {stress_fin:.3f}"); axes[1].axis("equal")
plt.colorbar(sc, ax=axes[1], label="country node ID", shrink=0.8)

axes[2].plot(result["loss_history"], color="steelblue", lw=1.5)
axes[2].set_title("Loss curve"); axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "finsler_migration_2015.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
