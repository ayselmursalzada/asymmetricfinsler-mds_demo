

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ----------------------------
# Dataset
# ----------------------------
class IndexDataset(Dataset):
    """D matrisinden batch halinde alt-matris döndürür."""
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def __len__(self):
        return self.distance_matrix.shape[0]

    def __getitem__(self, idx):
        return None, idx

    def collate_fn(self, batch):
        _, indices = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
        submatrix = self.distance_matrix[indices][:, indices]
        return indices, submatrix


# ----------------------------
# Kayıp fonksiyonları
# ----------------------------
class SammonLoss(nn.modules.loss._Loss):
    """Sammon stress (normalize edilmiş MSE biçimi)."""
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        notnull = target != 0
        x = input[notnull]/torch.sqrt(target[notnull])
        y = torch.sqrt(target[notnull])
        return nn.functional.mse_loss(x, y, reduction=self.reduction)


def sammon2(input, target, eps=1e-10):
    r = (input - target)**2
    r = r / (target + eps)
    r = torch.sum(r)
    r = r / torch.sum(target)
    return r


def ratio_loss(input, target, eps=1e-10):
    r = input / (target + eps)
    r = (r - 1)**2
    return torch.sum(r)


# ----------------------------
# Yardımcı
# ----------------------------
def convert_to_torch_float32(x):
    if isinstance(x, np.ndarray):
        if x.dtype != np.float32:
            x = x.astype('float32')
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    raise TypeError("Input must be numpy array or torch tensor.")


@torch.no_grad()
def compute_drift_vectors_torch(X, P_asym, orientation="dissimilarity",
                                normalizer="n", eps=1e-12, device=None, dtype=torch.float32):
    """
    Drift vektörleri (skew-symmetric kısma göre): N = 0.5*(P - P^T)
    d_i = (1/n) * sum_j sign * N[i,j] * (x_j - x_i)/||x_j - x_i||
    orientation: "similarity" => +1, "dissimilarity" => -1
    """
    if isinstance(X, np.ndarray): X = torch.from_numpy(X)
    if isinstance(P_asym, np.ndarray): P_asym = torch.from_numpy(P_asym)
    if device is None: device = X.device if isinstance(X, torch.Tensor) else "cpu"

    X = X.to(device=device, dtype=dtype).contiguous()
    P_asym = P_asym.to(device=device, dtype=dtype).contiguous()

    n, d = X.shape
    N = 0.5 * (P_asym - P_asym.T)
    sign = +1.0 if orientation == "similarity" else -1.0
    denom = float(n) if normalizer == "n" else float(max(n - 1, 1))

    Dvec = torch.zeros_like(X)
    for i in range(n):
        acc = torch.zeros(d, device=device, dtype=dtype)
        xi = X[i]
        for j in range(n):
            if i == j:
                continue
            aij = X[j] - xi
            nrm = torch.linalg.norm(aij)
            if nrm < eps:
                continue
            bij = aij / nrm
            acc += sign * N[i, j] * bij
        Dvec[i] = acc / denom

    return Dvec, N


# ----------------------------
# MDS (SGD)
# ----------------------------
def sgd_mds(D, initialData,
            n_epochs: int = 1000,
            lr: float = 1e-2,
            batch_size=None,
            max_epochs_no_improvement: int = 100,
            loss='MSE',
            saveloss: bool = False,     # burada dosyaya kaydetme yok; sadece API korundu
            verbose: bool = True,
            min_epochs: int = 100,
            P_asym=None,
            orientation: str = "dissimilarity",
            return_drift: bool = False):
    """
    Simetrik hedef mesafe matrisi D'ye karşılık SGD ile MDS.
    return_drift=True ve P_asym verildiğinde drift vektörleri de döner.
    """
    # tip/dtype hazırlık
    dataDtype = initialData.dtype if isinstance(initialData, np.ndarray) else initialData.detach().cpu().numpy().dtype
    init = convert_to_torch_float32(initialData)
    D = convert_to_torch_float32(D)

    # NaN temizliği
    nan_mask = torch.isnan(init)
    if nan_mask.any():
        init[nan_mask] = torch.rand(nan_mask.sum())

    # batch boyutu
    N = D.shape[0]
    if batch_size is None:
        batch_size = max(1, round(N/10))

    # model ve optimizer
    X = torch.nn.Parameter(init)
    optimizer = optim.Adam([X], lr=lr)

    # dataloader
    ds = IndexDataset(D)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=ds.collate_fn)

    # loss seçimi
    if loss == 'MSE':
        loss_fn = nn.MSELoss()
    elif loss == 'Sammon':
        loss_fn = SammonLoss()
    elif loss == 'Sammon2':
        loss_fn = sammon2
    elif loss == 'ratio':
        loss_fn = ratio_loss
    elif isinstance(loss, nn.modules.loss._Loss):
        loss_fn = loss
    else:
        raise NotImplementedError

    best_total = float('inf')
    no_imp = 0
    it = tqdm(range(n_epochs)) if verbose else range(n_epochs)

    for epoch in it:
        total = 0.0
        for indices, Dsub in dl:
            optimizer.zero_grad()
            Y = X[indices]
            YD = torch.cdist(Y, Y)
            l = loss_fn(YD, Dsub)
            l.backward()
            optimizer.step()
            total += float(l.item())

        if total < best_total:
            best_total, no_imp = total, 0
        else:
            no_imp += 1

        if (no_imp == max_epochs_no_improvement) and (epoch >= min_epochs):
            if verbose:
                print(f"Convergence. Early stopping at epoch {epoch}!")
            break

    X_np = X.detach().cpu().numpy().astype(dataDtype)

    if return_drift and (P_asym is not None):
        Dvec, N_skew = compute_drift_vectors_torch(X_np, P_asym, orientation=orientation)
        return X_np, Dvec.cpu().numpy(), N_skew.cpu().numpy()

    return X_np
