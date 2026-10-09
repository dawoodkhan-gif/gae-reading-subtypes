"""train_gae_v1.py

Extracts 8-ROI resting-state connectivity from the ds004765 BIDS images,
trains ten graph autoencoders (seeds 42-51), averages their node embeddings
and assigns k-means clusters (k = 3).

This is the exact pipeline used for the revised manuscript. See README.md,
"Known limitations": the MNI ROI coordinates are applied to native-space
(unnormalised) BOLD images.

Usage:
    python src/train_gae_v1.py --bids /path/to/ds004765 --out results
"""
import argparse

import os, glob, warnings, sys, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from nilearn.maskers import NiftiSpheresMasker
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

ap = argparse.ArgumentParser()
ap.add_argument('--bids', required=True, help='root of the ds004765 BIDS dataset')
ap.add_argument('--out', default='results', help='output directory')
args = ap.parse_args()
BASE, OUT = args.bids, args.out
os.makedirs(OUT, exist_ok=True)

print(f'Base exists: {os.path.exists(BASE)}')
print(f'Subjects: {len(glob.glob(f"{BASE}/sub-*"))}')

# ---------------------------------------------------------------------
# ROI definitions
# ---------------------------------------------------------------------
TR, RADIUS = 2.0, 6
ROIS = {
    'VWFA_L':     (-44, -58, -12),
    'VWFA_R':     ( 44, -58, -12),
    'IFG_op_L':   (-48,   8,  20),
    'IFG_tr_L':   (-48,  28,  16),
    'IFG_op_R':   ( 48,   8,  20),
    'IFG_tr_R':   ( 48,  28,  16),
    'AngularG_L': (-44, -64,  32),
    'Fusiform_L': (-38, -50, -24),
}
ROI_NAMES = list(ROIS.keys())
N_ROIS = 8

def fisher_z(r):
    r = np.clip(r, -0.999, 0.999)
    return 0.5 * np.log((1 + r) / (1 - r))

def compute_fc(ts):
    r = np.corrcoef(ts.T)
    np.fill_diagonal(r, 0.0)
    z = fisher_z(r)
    np.fill_diagonal(z, 0.0)
    return z

def minmax(A):
    lo, hi = A.min(), A.max()
    return (A - lo) / (hi - lo) if hi - lo > 1e-12 else A

# ---------------------------------------------------------------------
# Extract FC matrices
# ---------------------------------------------------------------------
print('\n=== Extracting FC matrices ===')
subs, fcs = [], []
for sd in sorted(glob.glob(f'{BASE}/sub-*')):
    sid = os.path.basename(sd)
    cands = [b for b in glob.glob(f'{sd}/func/*task-rest_bold.nii.gz') if '_mni' not in b]
    if not cands:
        print(f'  [skip] {sid}: no BOLD')
        continue
    try:
        masker = NiftiSpheresMasker(
            seeds=[ROIS[k] for k in ROI_NAMES], radius=RADIUS,
            standardize=False, detrend=True,
            high_pass=0.01, low_pass=0.1, t_r=TR,
            memory=os.path.join(OUT, 'nilearn_cache'), verbose=0,
        )
        ts = masker.fit_transform(cands[0])
        fcs.append(minmax(compute_fc(ts)))
        subs.append(sid)
        print(f'  [ok]   {sid}')
    except Exception as e:
        print(f'  [err]  {sid}: {e}')

if len(fcs) < 6:
    sys.exit('[fatal] fewer than 6 subjects succeeded')

fcs = np.stack(fcs)
print(f'\nFC tensor: {fcs.shape}')
np.savez(f'{OUT}/fc_matrices.npz', fcs=fcs, subjects=np.array(subs))

# ---------------------------------------------------------------------
# Graph Autoencoder
# ---------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

HIDDEN, LATENT = 32, 16
SEEDS = list(range(42, 52))

class GCNLayer(nn.Module):
    def __init__(self, fin, fout):
        super().__init__()
        self.lin = nn.Linear(fin, fout, bias=False)
    def forward(self, x, A_norm):
        return F.relu(torch.bmm(A_norm, self.lin(x)))

class GraphAutoencoder(nn.Module):
    def __init__(self, n, hidden=HIDDEN, latent=LATENT):
        super().__init__()
        self.gc1 = GCNLayer(n, hidden)
        self.gc2 = GCNLayer(hidden, latent)
    @staticmethod
    def norm_adj(A):
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
        Ah = A + I
        D  = Ah.sum(-1)
        Dn = torch.diag_embed(1.0 / torch.sqrt(D + 1e-8))
        return Dn @ Ah @ Dn
    def encode(self, A):
        An = self.norm_adj(A)
        return self.gc2(self.gc1(A, An), An)
    def forward(self, A):
        z = self.encode(A)
        return torch.sigmoid(z @ z.transpose(-1, -2)), z

def train_one(A_tr, A_va, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    m = GraphAutoencoder(N_ROIS).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3,
                           betas=(0.9, 0.999), weight_decay=1e-5)
    crit = nn.MSELoss()
    A_tr_t = torch.tensor(A_tr, dtype=torch.float32, device=DEVICE)
    A_va_t = torch.tensor(A_va, dtype=torch.float32, device=DEVICE)
    ckpt = f'{OUT}/gae_seed{seed}.pt'
    best, wait = np.inf, 0
    for epoch in range(300):
        m.train(); opt.zero_grad()
        A_hat, _ = m(A_tr_t)
        loss = crit(A_hat, A_tr_t)
        loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            A_hat_va, _ = m(A_va_t)
            vloss = crit(A_hat_va, A_va_t).item()
        if vloss < best - 1e-5:
            best, wait = vloss, 0
            torch.save(m.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= 15: break
    return best

print('\n=== Training 10 GAEs ===')
mses = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fcs))
    n_tr = int(0.8 * len(idx))
    tr, va = idx[:n_tr], idx[n_tr:]
    mses.append(train_one(fcs[tr], fcs[va], seed))
    print(f'  seed {seed}: val MSE = {mses[-1]:.5f}')

print(f'\nMean val MSE: {np.mean(mses):.5f} ± {np.std(mses):.5f}')

# ---------------------------------------------------------------------
# Embeddings and clustering
# ---------------------------------------------------------------------
embeds = []
for seed in SEEDS:
    m = GraphAutoencoder(N_ROIS).to(DEVICE)
    m.load_state_dict(torch.load(f'{OUT}/gae_seed{seed}.pt'))
    m.eval()
    with torch.no_grad():
        _, Z = m(torch.tensor(fcs, dtype=torch.float32, device=DEVICE))
    embeds.append(Z.cpu().numpy())

Z_flat = np.mean(embeds, 0).reshape(len(fcs), -1)
np.savez(f'{OUT}/embeddings.npz', Z=Z_flat, subjects=np.array(subs))

labels = KMeans(3, n_init=100, random_state=42).fit_predict(Z_flat)
np.savez(f'{OUT}/clusters.npz', labels=labels, subjects=np.array(subs))

print(f'\nEmbeddings shape: {Z_flat.shape}')
print(f'Cluster sizes: {np.bincount(labels).tolist()}')
print(f'\nAll training outputs saved to {OUT}')