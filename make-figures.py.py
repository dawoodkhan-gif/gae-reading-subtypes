"""make_figures_v1.py

Produces Figures 2-10 and results_summary.json from the outputs of
train_gae_v1.py and the OSF behavioural file (osf.io/yhf2e).
Computations are identical to those used for the revised manuscript;
changes from the notebook version are limited to command-line paths,
clearer axis labels, and FDR q-values added to results_summary.json.

Usage:
    python src/make_figures_v1.py --out results \
        --behav "/path/to/Behavioural Measures/BehaviouralIDs_ForOSF_Final_Updated.xlsx"
"""
import argparse

import os, glob, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.sparse.csgraph import shortest_path
from scipy.stats import f_oneway, pearsonr
from scipy.stats import false_discovery_control

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='results')
ap.add_argument('--behav', required=True, help='BehaviouralIDs_ForOSF_Final_Updated.xlsx')
args = ap.parse_args()
OUT = args.out
FIG_DIR = f'{OUT}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

ROI_NAMES = ['VWFA_L','VWFA_R','IFG_op_L','IFG_tr_L',
             'IFG_op_R','IFG_tr_R','AngularG_L','Fusiform_L']
N_ROIS = 8

# ---------------------------------------------------------------------
# Verify inputs
# ---------------------------------------------------------------------
print('=== Verifying inputs ===')
for p in [f'{OUT}/fc_matrices.npz',
          f'{OUT}/embeddings.npz',
          f'{OUT}/clusters.npz',
          args.behav]:
    print(f'  {os.path.exists(p)}  {p}')

fc_data  = np.load(f'{OUT}/fc_matrices.npz', allow_pickle=True)
fcs      = fc_data['fcs']
subs     = list(fc_data['subjects'])

cl_data  = np.load(f'{OUT}/clusters.npz', allow_pickle=True)
labels   = cl_data['labels']

emb_data = np.load(f'{OUT}/embeddings.npz', allow_pickle=True)
Z_flat   = emb_data['Z']

print(f'\nFC tensor: {fcs.shape}')
print(f'Cluster sizes: {np.bincount(labels).tolist()}')

# ---------------------------------------------------------------------
# Real OSF behavioural file
# ---------------------------------------------------------------------
beh = pd.read_excel(args.behav)
beh = beh[beh['IncludedinFinalRSFCanalysis'] == 1].copy()
beh['SubID_str'] = beh['SubID'].astype(str).str.replace(r'\.0$', '', regex=True)
beh['subject']   = beh['SubID_str'].apply(
    lambda x: f'sub-{x}' if x and x.lower() not in ('nan','none','') else None)
beh = beh.dropna(subset=['subject']).set_index('subject')

MEASURES = {
    'towre_word':     'TOWRE_Words_RawScore_n108',
    'towre_nonword':  'TOWRE_Nonwords_RawScore_n66',
    'pseudomorpheme': 'Pseudomorpheme_AccCost',
    'spelling':       'Spelling_prop_n40',
    'nwrep':          'CToPP_NWRep_PropExcNR',
    'vocabulary':     'Vocabulary_prop_n40',
    'spoon_first':    'Spoonerisms_firstletter_prop',
    'spoon_last':     'Spoonerisms_lastletter_prop',
}
behav_cols = list(MEASURES.keys())
beh_clean = beh[list(MEASURES.values())].rename(
    columns={v: k for k, v in MEASURES.items()})

# =====================================================================
# Fig 2 — Reconstruction error (from saved checkpoints)
# =====================================================================
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GCNLayer(nn.Module):
    def __init__(self, fin, fout):
        super().__init__(); self.lin = nn.Linear(fin, fout, bias=False)
    def forward(self, x, A_norm):
        return F.relu(torch.bmm(A_norm, self.lin(x)))

class GraphAutoencoder(nn.Module):
    def __init__(self, n, hidden=32, latent=16):
        super().__init__()
        self.gc1 = GCNLayer(n, hidden)
        self.gc2 = GCNLayer(hidden, latent)
    @staticmethod
    def norm_adj(A):
        B, N, _ = A.shape
        I = torch.eye(N, device=A.device).unsqueeze(0).expand(B, N, N)
        Ah = A + I; D = Ah.sum(-1)
        Dn = torch.diag_embed(1.0 / torch.sqrt(D + 1e-8))
        return Dn @ Ah @ Dn
    def encode(self, A):
        An = self.norm_adj(A)
        return self.gc2(self.gc1(A, An), An)
    def forward(self, A):
        z = self.encode(A)
        return torch.sigmoid(z @ z.transpose(-1, -2)), z

SEEDS = list(range(42, 52))
mses = []
for seed in SEEDS:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(fcs))
    n_tr = int(0.8 * len(idx))
    tr, va = idx[:n_tr], idx[n_tr:]
    m = GraphAutoencoder(N_ROIS).to(DEVICE)
    ckpt = f'{OUT}/gae_seed{seed}.pt'
    if not os.path.exists(ckpt):
        mses.append(np.nan); continue
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
    with torch.no_grad():
        A_hat, _ = m(torch.tensor(fcs[va], dtype=torch.float32, device=DEVICE))
        v = nn.MSELoss()(A_hat, torch.tensor(fcs[va], dtype=torch.float32, device=DEVICE)).item()
    mses.append(v)
print(f'\nMSE per seed: {np.round(mses, 4).tolist()}')

fig, ax = plt.subplots(figsize=(8, 5))
valid_mses = [m for m in mses if not np.isnan(m)]
valid_seeds = [s for s, m in zip(SEEDS, mses) if not np.isnan(m)]
ax.bar(range(len(valid_mses)), valid_mses, color='#4C72B0', edgecolor='black')
ax.axhline(np.mean(valid_mses), color='r', ls='--',
           label=f'Mean = {np.mean(valid_mses):.4f}')
ax.set_xticks(range(len(valid_mses)))
ax.set_xticklabels([str(s) for s in valid_seeds], rotation=45)
ax.set_xlabel('Random seed'); ax.set_ylabel('Validation MSE')
ax.set_title('Reconstruction error across GAE seeds')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig2_reconstruction_error.png', dpi=150)
plt.close()
print('saved fig2_reconstruction_error.png')

# =====================================================================
# Fig 3 — k selection
# =====================================================================
ks = list(range(2, 8))
sil, db, inertia = [], [], []
for k in ks:
    km = KMeans(k, n_init=50, random_state=0).fit(Z_flat)
    sil.append(silhouette_score(Z_flat, km.labels_))
    db.append(davies_bouldin_score(Z_flat, km.labels_))
    inertia.append(km.inertia_)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].plot(ks, inertia, 'o-'); ax[0].axvline(3, ls='--', c='r')
ax[0].set_title('Elbow'); ax[0].set_xlabel('k'); ax[0].set_ylabel('Inertia')  # k-means within-cluster sum of squares
ax[1].plot(ks, sil, 'o-'); ax[1].axvline(3, ls='--', c='r')
ax[1].set_title('Silhouette'); ax[1].set_xlabel('k'); ax[1].set_ylabel('Score')
ax[2].plot(ks, db, 'o-'); ax[2].axvline(3, ls='--', c='r')
ax[2].set_title('Davies-Bouldin'); ax[2].set_xlabel('k'); ax[2].set_ylabel('Index')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig3_k_selection.png', dpi=150)
plt.close()
print('saved fig3_k_selection.png')

# =====================================================================
# Fig 4 — t-SNE
# =====================================================================
ts = TSNE(2, perplexity=min(15, len(Z_flat)-1), random_state=0).fit_transform(Z_flat)
fig = plt.figure(figsize=(7, 6))
for lab in sorted(set(labels)):
    m = labels == lab
    plt.scatter(ts[m,0], ts[m,1], label=f'Cluster {lab} (n={m.sum()})',
                alpha=0.75, s=60)
plt.title('Latent GAE embeddings (t-SNE)')
plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2')
plt.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig4_tsne.png', dpi=150)
plt.close()
print('saved fig4_tsne.png')

# =====================================================================
# Fig 5 — connectivity edges per cluster
# =====================================================================
edge_pairs = [(0,2),(0,3),(0,6),(7,6),(0,7),(2,3)]
edge_labels = ['VWFA_L–IFG_op_L','VWFA_L–IFG_tr_L','VWFA_L–AngularG_L',
               'Fusiform_L–AngularG_L','VWFA_L–Fusiform_L','IFG_op_L–IFG_tr_L']
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(edge_pairs))
w = 0.8 / len(set(labels))
for i, lab in enumerate(sorted(set(labels))):
    W_avg = fcs[labels == lab].mean(0)
    vals = [W_avg[a, b] for a, b in edge_pairs]
    ax.bar(x + i*w - 0.4 + w/2, vals, w, label=f'Cluster {lab}')
ax.set_xticks(x)
ax.set_xticklabels(edge_labels, rotation=20, ha='right')
ax.set_ylabel('Mean connectivity (Fisher z, rescaled 0-1)')
ax.set_title('Key connectivity edges per cluster (from FC matrices)')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig5_connectivity_edges.png', dpi=150)
plt.close()
print('saved fig5_connectivity_edges.png')

# =====================================================================
# Fig 6 — RSFC heatmaps per cluster
# =====================================================================
fig, axs = plt.subplots(1, len(set(labels)), figsize=(6*len(set(labels)), 5))
if len(set(labels)) == 1: axs = [axs]
for i, lab in enumerate(sorted(set(labels))):
    W = fcs[labels == lab].mean(0)
    sns.heatmap(W, ax=axs[i], cmap='RdBu_r', center=0,
                xticklabels=ROI_NAMES, yticklabels=ROI_NAMES,
                vmin=-1, vmax=1, cbar=(i == len(set(labels))-1))
    axs[i].set_title(f'Cluster {lab} (n={(labels==lab).sum()})')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig6_rsfc_heatmaps.png', dpi=150)
plt.close()
print('saved fig6_rsfc_heatmaps.png')

# =====================================================================
# Fig 7 — graph metrics
# =====================================================================
def gmetrics(W):
    W = np.abs(W); np.fill_diagonal(W, 0); W = (W + W.T)/2
    D = 1.0/(W + 1e-9); np.fill_diagonal(D, 0)
    dist = shortest_path(D, method='D', directed=False)
    inv = 1.0/(dist + 1e-9); np.fill_diagonal(inv, 0)
    ge = inv.sum()/(N_ROIS*(N_ROIS-1))
    iu = np.triu_indices(N_ROIS, 1)
    pl = dist[iu].mean()
    return ge, pl

rows = []
for i, W in enumerate(fcs):
    ge, pl = gmetrics(W)
    rows.append({'subject': subs[i], 'subtype': int(labels[i]),
                 'global_eff': ge, 'path_len': pl})
df_metrics = pd.DataFrame(rows)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i, m in enumerate(['global_eff','path_len']):
    sns.barplot(data=df_metrics, x='subtype', y=m, ax=axs[i],
                errorbar='sd', palette='Set2')
    axs[i].set_title({'global_eff': 'Global efficiency', 'path_len': 'Characteristic path length'}[m])
    axs[i].set_xlabel('Cluster')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig7_graph_metrics.png', dpi=150)
plt.close()
print('saved fig7_graph_metrics.png')

# =====================================================================
# Merge neural + behavioural
# =====================================================================
df_neural = pd.DataFrame({'subject': subs, 'subtype': labels}).set_index('subject')
df_merged = df_neural.join(beh_clean, how='inner')
print(f'\nSubjects with neural + behavioural: {len(df_merged)}')

# =====================================================================
# Fig 8 — behaviour by cluster
# =====================================================================
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for ax_, col in zip(axs.flat, behav_cols):
    sns.barplot(data=df_merged, x='subtype', y=col, ax=ax_,
                errorbar='se', palette='Set2')
    groups = [df_merged.loc[df_merged.subtype == s, col].dropna()
              for s in sorted(df_merged.subtype.unique())]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        F, p = f_oneway(*groups)
        ax_.set_xlabel(f'F={F:.2f}, p={p:.3f}')
    ax_.set_title(col)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig8_behaviour_by_cluster.png', dpi=150)
plt.close()
print('saved fig8_behaviour_by_cluster.png')

# =====================================================================
# Fig 9 — violins
# =====================================================================
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
for ax_, col in zip(axs.flat, behav_cols):
    sns.violinplot(data=df_merged, x='subtype', y=col, ax=ax_,
                   palette='Set2', cut=0)
    ax_.set_title(col)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig9_behaviour_violins.png', dpi=150)
plt.close()
print('saved fig9_behaviour_violins.png')

# =====================================================================
# Fig 10 — brain-behaviour correlations
# =====================================================================
VWFA_IFG     = np.array([float((W[0,2]+W[0,3])/2) for W in fcs])
VWFA_Angular = np.array([float(W[0,6]) for W in fcs])
df_feat = pd.DataFrame({'subject': subs,
                        'VWFA_IFG': VWFA_IFG,
                        'VWFA_Angular': VWFA_Angular}).set_index('subject')
df_full = df_feat.join(beh_clean, how='inner')

pairs = [('VWFA_IFG','towre_nonword'),
         ('VWFA_IFG','spoon_first'),
         ('VWFA_Angular','spoon_last'),
         ('VWFA_Angular','vocabulary')]
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
r_table = []
for ax_, (nc, bc) in zip(axs.flat, pairs):
    x = df_full[nc].values; y = df_full[bc].values
    ax_.scatter(x, y, alpha=0.7, s=55)
    r, p = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax_.plot(xs, np.polyval(z, xs), 'k--', alpha=0.6)
    ax_.set_xlabel(nc); ax_.set_ylabel(bc)
    ax_.set_title(f'r={r:.2f}, p={p:.3f}')
    r_table.append((nc, bc, r, p))
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/fig10_brain_behaviour.png', dpi=150)
plt.close()
print('saved fig10_brain_behaviour.png')

# =====================================================================
# Results summary
# =====================================================================
summary = {
    'fc_shape': list(fcs.shape),
    'cluster_sizes': np.bincount(labels).tolist(),
    'mse_per_seed': [float(m) for m in mses],
    'mse_mean': float(np.nanmean(mses)),
    'mse_std': float(np.nanstd(mses)),
    'k_selection': {str(k): {'silhouette': float(s), 'db': float(d)}
                    for k, s, d in zip(ks, sil, db)},
    'graph_metrics_by_cluster': df_metrics.groupby('subtype').mean(numeric_only=True).round(3).to_dict(),
    'pearson_correlations': [
        {'neural': nc, 'behavioural': bc, 'r': float(r), 'p': float(p)}
        for (nc, bc, r, p) in r_table],
    'anova_by_measure': {},
}
for col in behav_cols:
    groups = [df_merged.loc[df_merged.subtype == s, col].dropna()
              for s in sorted(df_merged.subtype.unique())]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) >= 2:
        F, p = f_oneway(*groups)
        summary['anova_by_measure'][col] = {'F': float(F), 'p': float(p)}
# Benjamini-Hochberg FDR across the eight ANOVAs and across the four correlations
cols_ = list(summary['anova_by_measure'])
q_ = false_discovery_control([summary['anova_by_measure'][c]['p'] for c in cols_])
for c, q in zip(cols_, q_):
    summary['anova_by_measure'][c]['q_fdr'] = float(q)
q_ = false_discovery_control([d['p'] for d in summary['pearson_correlations']])
for d, q in zip(summary['pearson_correlations'], q_):
    d['q_fdr'] = float(q)
summary['n_behaviour'] = int(len(df_merged))

with open(f'{OUT}/results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nsaved {OUT}/results_summary.json')

print('done')
