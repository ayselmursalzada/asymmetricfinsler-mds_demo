"""Fast Vicari ALS on K matrix."""
import sys, numpy as np, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
OUT = Path("finsler_results_vicari"); OUT.mkdir(exist_ok=True)
EDGES = Path("/mnt/user-data/uploads/edges.csv")
from load_asymmetric_graph import load_migration_graph

g = load_migration_graph(EDGES, year=2015, normalize='none', verbose=False)
F = g['F']; n = g['n_nodes']
D_raw = g['D_asym']
K = 0.5*(D_raw - D_raw.T)
norm2 = np.sum(K**2)+1e-12

def obj(K, labels, C):
    sv2=0.
    for c in range(C):
        for cp in range(c+1,C):
            Ic=np.where(labels==c)[0]; Icp=np.where(labels==cp)[0]
            if not len(Ic) or not len(Icp): continue
            s=np.linalg.svd(K[np.ix_(Ic,Icp)],compute_uv=False)
            if len(s): sv2+=s[0]**2
    return sv2

def loss(K,labels,C): return 1.-2.*obj(K,labels,C)/norm2

def batch_step(K,labels,C):
    N=K.shape[0]; new=labels.copy()
    for i in range(N):
        bo=-np.inf; bc=labels[i]
        for ct in range(C):
            tr=labels.copy(); tr[i]=ct
            if (tr==labels[i]).sum()==0: continue
            o=obj(K,tr,C)
            if o>bo: bo=o; bc=ct
        new[i]=bc
    return new

def als_vicari(K,C,n_starts=10,max_iter=30,tol=1e-6):
    best_loss=np.inf; best_labs=None; t0=time.time()
    for s in range(n_starts):
        labs=KMeans(n_clusters=C,random_state=s*7+C*13,n_init=10).fit_predict(K)
        prev=np.inf
        for it in range(max_iter):
            labs=batch_step(K,labs,C)
            cur=loss(K,labs,C)
            if abs(prev-cur)<tol: break
            prev=cur
        if cur<best_loss: best_loss=cur; best_labs=labs.copy()
        print(f"  s={s+1}/{n_starts}  loss={cur:.4f}  best={best_loss:.4f}  {time.time()-t0:.0f}s")
    return best_labs,best_loss

print(f"\nn={n}  ||K||_F={np.sqrt(norm2):.3f}")

# Scree C=2..5
print("\n=== Scree (C=2..5, 5 starts) ===")
scree=[]
for C in range(2,6):
    print(f"\nC={C}")
    labs,lo=als_vicari(K,C,n_starts=5,max_iter=20)
    gof=(1-lo)*100
    net=[]
    for c in range(C):
        Ic=np.where(labs==c)[0]; Jc=np.where(labs!=c)[0]
        net.append(F[np.ix_(Ic,Jc)].sum()-F[np.ix_(Jc,Ic)].sum())
    sizes=[(labs==c).sum() for c in range(C)]
    print(f"  GoF={gof:.1f}%  sizes={sizes}  net={[f'{v/1e6:+.1f}M' for v in net]}")
    scree.append((C,lo,gof,labs.copy(),net))

fig,ax=plt.subplots(figsize=(6,4))
ax.plot([x[0] for x in scree],[x[1] for x in scree],'o-',color="steelblue",lw=2,ms=8)
for x in scree:
    ax.annotate(f"{x[2]:.1f}%",(x[0],x[1]),textcoords="offset points",xytext=(6,3),fontsize=9)
ax.set_xlabel("C"); ax.set_ylabel("Relative loss")
ax.set_title("Scree — Vicari ALS on K (n=232)")
ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(OUT/"vicari_scree.png",dpi=150,bbox_inches="tight"); plt.close(fig)
print("Saved scree")

# Best C with more starts
diffs=np.diff([x[1] for x in scree])
best_C=scree[int(np.argmin(diffs))+1][0]
print(f"\n=== C={best_C} with 30 starts ===")
labs_best,loss_best=als_vicari(K,best_C,n_starts=30,max_iter=40)
gof_best=(1-loss_best)*100
print(f"Final: C={best_C}  GoF={gof_best:.1f}%")

net_best=[]
for c in range(best_C):
    Ic=np.where(labs_best==c)[0]; Jc=np.where(labs_best!=c)[0]
    nf=F[np.ix_(Ic,Jc)].sum()-F[np.ix_(Jc,Ic)].sum()
    net_best.append(nf)
    print(f"  C{c+1}: {len(Ic):3d} countries  net={nf/1e6:+.2f}M  {'ORIGIN' if nf>0 else 'DEST'}")

# Flow heatmap
mat=np.zeros((best_C,best_C))
for ci in range(best_C):
    for cj in range(best_C):
        if ci==cj: continue
        Ii=np.where(labs_best==ci)[0]; Ij=np.where(labs_best==cj)[0]
        if len(Ii) and len(Ij): mat[ci,cj]=F[np.ix_(Ii,Ij)].mean()

fig2,ax2=plt.subplots(figsize=(5,4))
im=ax2.imshow(mat,cmap="YlOrRd",aspect="auto")
lstr=[f"C{c+1}\n({'orig' if net_best[c]>0 else 'dest'})" for c in range(best_C)]
ax2.set_xticks(range(best_C)); ax2.set_yticks(range(best_C))
ax2.set_xticklabels(lstr,fontsize=9); ax2.set_yticklabels(lstr,fontsize=9)
for i in range(best_C):
    for j in range(best_C):
        ax2.text(j,i,f"{mat[i,j]/1e3:.0f}k",ha="center",va="center",fontsize=9)
plt.colorbar(im,ax=ax2,label="mean migrants/pair")
ax2.set_title(f"C={best_C}: between-cluster flows"); fig2.tight_layout()
fig2.savefig(OUT/"vicari_flow_heatmap.png",dpi=150,bbox_inches="tight"); plt.close(fig2)
print("Saved heatmap\nAll done.")
