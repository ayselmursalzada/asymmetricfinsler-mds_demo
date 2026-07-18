import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_finsler_distances(Y: np.ndarray, B: np.ndarray, eps: float) -> np.ndarray:
    """F(i→j) = ‖yⱼ - yᵢ‖ + ⟨bᵢ, êᵢⱼ⟩  (n×n matrix)"""
    diff   = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    r      = np.sqrt((diff ** 2).sum(-1))
    r_safe = np.maximum(r, eps)
    e      = diff / r_safe[:, :, np.newaxis]
    proj   = (B[:, np.newaxis, :] * e).sum(-1)
    F      = np.maximum(r + proj, 0.0)
    np.fill_diagonal(F, 0.0)
    return F


def _compute_net_flow(D_asym: np.ndarray):
    """
    net_flow[i] = total inflow - total outflow for country i.
    a_i          = net_flow normalised to [0, 1].
    """
    inflow   = D_asym.sum(axis=0)
    outflow  = D_asym.sum(axis=1)
    net_flow = inflow - outflow
    lo, hi   = net_flow.min(), net_flow.max()
    a_i      = (net_flow - lo) / (hi - lo + 1e-12)
    return net_flow, a_i


def _init_centroids(
    Y: np.ndarray,
    F_dist: np.ndarray,
    net_flow: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Seed centroids at the two extremes of the migration spectrum,
    then fill remaining slots with Finsler K-Means++.

    centroid 0 → strongest net attractor  (max net_flow)
    centroid 1 → strongest net emitter    (min net_flow)
    centroid 2+ → Finsler K-Means++ spread
    """
    n      = Y.shape[0]
    chosen = [int(np.argmax(net_flow)), int(np.argmin(net_flow))]
    chosen = list(dict.fromkeys(chosen))          # deduplicate if n=1
    centroids = [Y[i].copy() for i in chosen]

    while len(chosen) < k:
        dist_to_chosen = F_dist[:, chosen]
        D_sq           = dist_to_chosen.min(axis=1) ** 2
        for idx in chosen:
            D_sq[idx] = 0.0
        total = D_sq.sum()
        if total == 0.0:
            probs         = np.ones(n) / n
            for idx in chosen:
                probs[idx] = 0.0
            probs /= probs.sum()
        else:
            probs = D_sq / total
        nxt = int(rng.choice(n, p=probs))
        chosen.append(nxt)
        centroids.append(Y[nxt].copy())

    return np.array(centroids)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def asymmetric_kmeans(
    Y: np.ndarray,
    B: np.ndarray,
    D_asym: np.ndarray,
    k: int = 2,
    max_iter: int = 100,
    eps: float = 1e-8,
    n_init: int = 10,
    random_state: int = 42,
) -> dict:
    """
    Asymmetric K-Means on Finsler MDS output.

    Assignment rule:
      Each point is assigned to the centroid with the smallest
      *net-flow-weighted* Finsler distance:

          cost(i → centroid j) = F(i→j) / (a_i + eps)

      Points with high a_i (strong attractors) are pulled more
      tightly toward their centroid; points with low a_i
      (strong emitters) are assigned more loosely.

    Centroid update:
      Weighted mean of member points, weights = a_i.

    Parameters
    ----------
    Y            : (n, d) Finsler MDS embedding
    B            : (n, d) drift vectors  (‖bᵢ‖ < 1)
    D_asym       : (n, n) original asymmetric flow matrix
    k            : number of clusters
    max_iter     : max iterations per restart
    eps          : numerical floor
    n_init       : independent restarts (best inertia kept)
    random_state : seed

    Returns
    -------
    dict:
      labels         : (n,) cluster assignments
      centroids      : (k, d) final centroids
      a_i            : (n,) attractor scores ∈ [0,1]
      net_flow       : (n,) raw net inflow per country
      F_dist         : (n, n) Finsler distance matrix
      inertia        : weighted within-cluster Finsler distance (lower=better)
      best_run       : winning restart index
      all_inertias   : inertia per restart
    """
    Y      = np.array(Y,      dtype=np.float64)
    B      = np.array(B,      dtype=np.float64)
    D_asym = np.array(D_asym, dtype=np.float64)
    n, d   = Y.shape
    rng    = np.random.default_rng(random_state)

    # ── Step 1: Finsler distances in embedding space ──────────────────────
    F_dist = _compute_finsler_distances(Y, B, eps)

    # ── Step 2: Migration-grounded asymmetry scores ───────────────────────
    net_flow, a_i = _compute_net_flow(D_asym)

    print(f"net_flow : min={net_flow.min():.1f}  "
          f"mean={net_flow.mean():.1f}  max={net_flow.max():.1f}")
    print(f"a_i      : min={a_i.min():.3f}  "
          f"mean={a_i.mean():.3f}  max={a_i.max():.3f}")
    print(f"Attractors (net>0): {(net_flow>0).sum()}   "
          f"Emitters  (net<0): {(net_flow<0).sum()}")

    # ── Step 3: Multi-restart optimisation ───────────────────────────────
    best_labels   = None
    best_centroids = None
    best_inertia  = np.inf
    best_run_idx  = 0
    all_inertias  = []

    for run in range(n_init):

        centroids = _init_centroids(Y, F_dist, net_flow, k, rng)
        labels    = np.zeros(n, dtype=int)
        converged = False

        for epoch in range(max_iter):
            old_labels = labels.copy()

            # ── Assignment: nearest centroid in weighted Finsler distance ──
            # F_ic[i,j] = Finsler distance from point i to centroid j
            diff_c = centroids[np.newaxis, :, :] - Y[:, np.newaxis, :]  # (n,k,d)
            r_c    = np.linalg.norm(diff_c, axis=2)                      # (n,k)
            e_c    = diff_c / (r_c[:, :, np.newaxis] + eps)
            F_ic   = np.maximum(
                r_c + (B[:, np.newaxis, :] * e_c).sum(axis=2), 0.0
            )                                                             # (n,k)

            # Weight by a_i: strong attractors get pulled harder
            cost   = F_ic / (a_i[:, np.newaxis] + eps)                  # (n,k)
            labels = np.argmin(cost, axis=1)                             # (n,)

            # ── Centroid update: a_i-weighted mean ────────────────────────
            new_centroids = centroids.copy()
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    w = a_i[mask]
                    if w.sum() == 0:
                        w = np.ones_like(w)
                    new_centroids[j] = np.average(Y[mask], axis=0, weights=w)
            centroids = new_centroids

            if np.array_equal(labels, old_labels):
                converged = True
                break

        # Inertia: sum of weighted Finsler distances to assigned centroid
        diff_c  = centroids[np.newaxis, :, :] - Y[:, np.newaxis, :]
        r_c     = np.linalg.norm(diff_c, axis=2)
        e_c     = diff_c / (r_c[:, :, np.newaxis] + eps)
        F_ic    = np.maximum(r_c + (B[:, np.newaxis, :] * e_c).sum(axis=2), 0.0)
        inertia = float((a_i * F_ic[np.arange(n), labels]).sum())
        all_inertias.append(inertia)

        sizes  = np.bincount(labels, minlength=k).tolist()
        status = "converged" if converged else f"max_iter={max_iter}"
        print(f"  Run {run+1:2d}/{n_init}  [{status:>16}]  "
              f"inertia={inertia:.2f}  sizes={sizes}")

        if inertia < best_inertia:
            best_inertia   = inertia
            best_labels    = labels.copy()
            best_centroids = centroids.copy()
            best_run_idx   = run

    print(f"\n✓ Best run: {best_run_idx+1}  (inertia={best_inertia:.2f})")
    print(f"  Final cluster sizes: {np.bincount(best_labels, minlength=k).tolist()}")

    return {
        "labels":        best_labels,
        "centroids":     best_centroids,
        "a_i":           a_i,
        "net_flow":      net_flow,
        "F_dist":        F_dist,
        "inertia":       best_inertia,
        "best_run":      best_run_idx,
        "all_inertias":  all_inertias,
    }