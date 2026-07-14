"""
finsler_mds_freeB.py
====================
Finsler MDS with jointly optimised Y and B (free parameters).

═══════════════════════════════════════════════════════════════════════
SOURCE ATTRIBUTION
═══════════════════════════════════════════════════════════════════════

This implementation builds on three sources:

[DAGES]   Dagès et al., "Finsler Multi-Dimensional Scaling", CVPR 2025.
          arXiv:2503.18010. Original Finsler MDS formulation.

[ISUMAP]  Joharinad & Jost, "IsUMap: Manifold Learning Leveraging
          Vietoris-Rips Filtrations", AAAI 2025. Distance matrix
          construction pipeline for asymmetric data.

[OURS]    Adaptations for sparse asymmetric graphs:
            - pair_mask: exclude zero-both (incomparable) pairs from loss
            - free-B: remove frozen-B approximation, jointly optimise Y and B
            - direction accuracy diagnostic to evaluate asymmetry encoding
            - D_sym = ½(D+Dᵀ) initialisation for asymmetric input

Key difference from frozen-B (v5):
  v5:     B = f(Y, D) recomputed each epoch, ∂B/∂Y assumed zero → Y learns
          symmetric MDS, asymmetry is a post-hoc correction in B
  free-B: B is a free (n, d) parameter, jointly optimised with Y via Adam →
          Y receives asymmetric gradients, embedding encodes directional info

This distinction was discovered empirically: frozen-B direction accuracy
fell below random chance (37.8%), revealing the embedding was discarding
asymmetric information rather than preserving it.
═══════════════════════════════════════════════════════════════════════

Loss (masked — observed pairs only):
  L(Y, B) = Σ_{(i,j) observed} (F̃(i→j) − D[i,j])²
  F̃(i→j) = ‖yⱼ − yᵢ‖ + ⟨bᵢ, êᵢⱼ⟩          [DAGES Eq. 1]

Gradients:
  ∂L/∂bᵢ = 2·Σⱼ res[i,j]·êᵢⱼ               [OURS — not in frozen-B]
  ∂L/∂yᵢ = source + target contributions     [DAGES — standard MDS grad]

Randers validity: ‖bᵢ‖ ≤ 1 − δ              [DAGES constraint]
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# Source: standard algebraic construction, used by DAGES as initialisation.
#         We apply it to D_sym = ½(D+Dᵀ) because the input D_asym is
#         asymmetric. [OURS: choice of D_sym for asymmetric input]
# ─────────────────────────────────────────────────────────────────────────────
def _cmds_init(D_sym: np.ndarray, d: int) -> np.ndarray:
    """Classical MDS initialisation from symmetric distance matrix."""
    n = D_sym.shape[0]
    H = -0.5 * D_sym ** 2
    H = H - H.mean(axis=1, keepdims=True) - H.mean(axis=0, keepdims=True) + H.mean()
    vals, vecs = np.linalg.eigh(H)
    idx = np.argsort(vals)[::-1][:d]
    return (vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser
# Source: standard gradient-based optimiser.
# Extended here to handle multiple parameter arrays [Y, B] simultaneously.
# [OURS: multi-parameter extension to jointly update Y and B]
# ─────────────────────────────────────────────────────────────────────────────
class _Adam:
    def __init__(self, shapes, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.t = 0
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]

    def step(self, params, grads):
        self.t += 1
        out = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g ** 2
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            out.append(p - self.lr * mh / (np.sqrt(vh) + self.eps))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Loss and gradient computation
# ─────────────────────────────────────────────────────────────────────────────
def finsler_loss_grad_freeB(Y, B, D_asym, loss_mask, eps=1e-8):
    """
    Compute L(Y, B) and gradients dL/dY, dL/dB.

    Returns: loss (float), grad_Y (n,d), grad_B (n,d)
    """
    # ── Forward pass ─────────────────────────────────────────────────────────
    # Source: DAGES Eq. 1 — Randers distance formula
    # F̃(i→j) = ‖yⱼ−yᵢ‖ + ⟨bᵢ, êᵢⱼ⟩
    # This is the core Finsler (Randers) metric. Unchanged from DAGES.
    diff   = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]   # (n,n,d): diff[i,j]=yⱼ−yᵢ
    r      = np.sqrt((diff ** 2).sum(-1))                  # (n,n): Euclidean distance
    r_safe = np.maximum(r, eps)
    e      = diff / r_safe[:, :, np.newaxis]               # (n,n,d): unit vectors êᵢⱼ
    proj   = (B[:, np.newaxis, :] * e).sum(-1)             # (n,n): ⟨bᵢ, êᵢⱼ⟩ inner product
    F_mat  = r + proj                                       # (n,n): Finsler distances [DAGES]

    # ── Loss ─────────────────────────────────────────────────────────────────
    # Source: DAGES — squared stress loss
    # OURS: * loss_mask — multiply by 0/1 mask to exclude zero-both pairs
    # (incomparable pairs: F[i,j]=0 and F[j,i]=0 carry no asymmetric signal)
    # Without masking, zero pairs dominate the gradient and suppress the
    # asymmetric structure. This is our key adaptation for sparse graphs.
    res  = (F_mat - D_asym) * loss_mask    # residuals, zero for excluded pairs
    loss = float((res ** 2).sum())

    # ── Gradient w.r.t. Y ────────────────────────────────────────────────────
    # Source: DAGES — derived from ∂F̃(i→j)/∂yᵢ and ∂F̃(i→j)/∂yⱼ [DAGES]
    # Each node i appears as both source (in pair i→j) and target (in pair j→i).
    # Because res[i,j] ≠ res[j,i] in general (asymmetric residuals), yᵢ receives
    # an asymmetric gradient — this is what makes the embedding encode direction.
    # In frozen-B, B is recomputed from Y so ∂B/∂Y is implicitly ignored,
    # making the net gradient symmetric. Here B is free, so asymmetry propagates.
    perp   = B[:, np.newaxis, :] - proj[:, :, np.newaxis] * e   # component of bᵢ ⊥ êᵢⱼ
    grad_Y = 2.0 * (res[:, :, np.newaxis] *
                    (-e - perp / r_safe[:, :, np.newaxis])).sum(axis=1)   # source contrib
    grad_Y += 2.0 * (res.T[:, :, np.newaxis] *
                     (e.transpose(1, 0, 2) +
                      perp.transpose(1, 0, 2) / r_safe.T[:, :, np.newaxis])).sum(axis=1)  # target contrib

    # ── Gradient w.r.t. B ────────────────────────────────────────────────────
    # Source: OURS — this line does not exist in frozen-B.
    # ∂F̃(i→j)/∂bᵢ = êᵢⱼ  (unit direction from i to j)
    # ∂L/∂bᵢ = 2·Σⱼ res[i,j]·êᵢⱼ
    #
    # Intuition: if D[i,j] < D[j,i] (more flow i→j than j→i), residuals
    # res[i,j] are systematically positive → bᵢ is pushed toward êᵢⱼ →
    # the i→j direction becomes cheaper in the Finsler metric.
    # Over many epochs, bᵢ converges to the direction of net emigration from i.
    grad_B = 2.0 * (res[:, :, np.newaxis] * e).sum(axis=1)   # (n,d) [OURS]

    return loss, grad_Y, grad_B


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────
def finsler_mds_freeB(
    D_asym: np.ndarray,
    d: int = 2,
    n_epochs: int = 3000,
    lr: float = 5e-3,
    clip_delta: float = 0.01,
    pair_mask: np.ndarray = None,
    max_epochs_no_improvement: int = 300,
    min_epochs: int = 200,
    B_init: np.ndarray = None,
    verbose: bool = True,
) -> dict:
    """
    Finsler MDS with jointly optimised Y and B (free parameters).

    Parameters
    ----------
    D_asym     : (n, n) asymmetric distance matrix
                 Constructed via log-ratio + ISUMAP row-normalisation [ISUMAP]
    d          : embedding dimension
    lr         : Adam learning rate
    clip_delta : Randers validity margin — ||bᵢ|| <= 1 - clip_delta  [DAGES]
    pair_mask  : (n, n) bool — True pairs enter loss (None = all off-diag)
                 Set to (F>0)|(F.T>0) to exclude zero-both pairs  [OURS]
    B_init     : (n, d) initial drift matrix (None = zeros)
                 Zero init because B is learned from data, not from Y  [OURS]
    verbose    : print progress every 200 epochs

    Returns
    -------
    dict: Y, B, Y_init, B_init, loss_history, b_norms
    """
    D_asym = np.array(D_asym, dtype=np.float64)
    n = D_asym.shape[0]

    # Build float loss mask  [OURS]
    mask = pair_mask.astype(np.float64) if pair_mask is not None else 1.0 - np.eye(n)

    # Initialise Y from cMDS on symmetric part  [OURS: use D_sym not D_asym]
    # D_sym = ½(D+Dᵀ) gives a valid symmetric distance matrix for initialisation.
    Y      = _cmds_init(0.5 * (D_asym + D_asym.T), d)
    Y_init = Y.copy()

    # Initialise B to zero  [OURS]
    # Unlike frozen-B where B=f(Y,D) at init, here B starts at zero and is
    # learned purely from the asymmetric residuals in the data.
    B            = np.zeros((n, d)) if B_init is None else B_init.copy()
    B_init_saved = B.copy()

    limit = 1.0 - clip_delta                          # Randers upper bound  [DAGES]
    opt   = _Adam([Y.shape, B.shape], lr=lr)          # joint optimiser  [OURS]

    best_loss, best_Y, best_B = np.inf, Y.copy(), B.copy()
    no_imp, history = 0, []

    if verbose:
        n_pairs = int(mask.sum())
        print(f"\n-- Finsler MDS free-B  (n={n}, d={d}) --")
        print(f"   pair_mask: {n_pairs}/{n*(n-1)} pairs ({n_pairs/n/(n-1)*100:.1f}%)")
        print(f"   lr={lr}  clip_delta={clip_delta}  max_epochs={n_epochs}")

    for epoch in range(n_epochs):

        # Compute loss and gradients for both Y and B  [OURS: B included]
        loss, gY, gB = finsler_loss_grad_freeB(Y, B, D_asym, mask)

        # Simultaneous Adam update for Y and B  [OURS: B updated here]
        [Y, B] = opt.step([Y, B], [gY, gB])

        # Randers validity clipping: ‖bᵢ‖ ≤ 1 − clip_delta  [DAGES]
        # Required for positive-definiteness of the Randers metric.
        # Applied per-node (per-node B is our extension; DAGES uses global ω).
        norms = np.linalg.norm(B, axis=1, keepdims=True)
        B = B * np.where(norms > limit, limit / np.maximum(norms, 1e-12), 1.0)

        history.append(float(loss))

        # Best-model tracking (standard early stopping)
        if loss < best_loss:
            best_loss = loss
            best_Y, best_B = Y.copy(), B.copy()
            no_imp = 0
        else:
            no_imp += 1

        if verbose and epoch % 200 == 0:
            bn = np.linalg.norm(B, axis=1)
            print(f"    epoch {epoch:4d}  loss={loss:.4f}  "
                  f"||B||_F={np.linalg.norm(B):.3f}  "
                  f"mean||bi||={bn.mean():.3f}  max||bi||={bn.max():.3f}")

        if no_imp >= max_epochs_no_improvement and epoch >= min_epochs:
            if verbose:
                print(f"  Early stop epoch {epoch}  best={best_loss:.4f}")
            break

    b_norms = np.linalg.norm(best_B, axis=1)
    if verbose:
        print(f"  Final ||B||_F={np.linalg.norm(best_B):.3f}  "
              f"mean||bi||={b_norms.mean():.3f}  max||bi||={b_norms.max():.3f}  "
              f"invalid={(b_norms>=1).sum()}")

    return {"Y": best_Y, "B": best_B, "Y_init": Y_init,
            "B_init": B_init_saved, "loss_history": history, "b_norms": b_norms}


# ─────────────────────────────────────────────────────────────────────────────
# Distance reconstruction
# Source: DAGES — same formula as forward pass, wrapped as utility function.
# ─────────────────────────────────────────────────────────────────────────────
def finsler_distances_freeB(Y: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Reconstruct asymmetric Finsler distances from embedding.

    D_F[i,j] = ||yⱼ − yᵢ|| + <bᵢ, êᵢⱼ>     [DAGES Eq. 1]

    Returns (n, n) matrix, diagonal = 0.
    """
    diff   = Y[np.newaxis, :, :] - Y[:, np.newaxis, :]
    r_safe = np.maximum(np.sqrt((diff ** 2).sum(-1)), eps)
    e      = diff / r_safe[:, :, np.newaxis]
    F      = r_safe + (B[:, np.newaxis, :] * e).sum(-1)
    np.fill_diagonal(F, 0.0)
    return F

