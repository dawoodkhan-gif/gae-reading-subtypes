"""fig1_rois_v1.py - plots the eight ROI coordinates (as used in train_gae_v1.py)
on an MNI152 glass brain. Shows the intended locations only; in the analysis the
coordinates were applied to native-space images (see README, Known limitations)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nilearn import plotting

ROIS = {'VWFA_L': (-44, -58, -12), 'VWFA_R': (44, -58, -12),
        'IFG_op_L': (-48, 8, 20), 'IFG_tr_L': (-48, 28, 16),
        'IFG_op_R': (48, 8, 20), 'IFG_tr_R': (48, 28, 16),
        'AngularG_L': (-44, -64, 32), 'Fusiform_L': (-38, -50, -24)}
COLS = ['#1f77b4', '#aec7e8', '#d62728', '#ff7f0e',
        '#ff9896', '#ffbb78', '#2ca02c', '#9467bd']

fig = plt.figure(figsize=(11, 4.6))
plotting.plot_connectome(np.zeros((8, 8)), list(ROIS.values()), node_color=COLS,
                         node_size=80, display_mode='lyrz', figure=fig,
                         axes=[0, 0.18, 1, 0.82], annotate=True, colorbar=False)
ax = fig.add_axes([0, 0, 1, 0.16]); ax.axis('off')
for i, (n, c) in enumerate(ROIS.items()):
    x, y = (i % 4) * 0.25 + 0.03, 0.7 - 0.45 * (i // 4)
    ax.scatter(x, y, s=70, color=COLS[i])
    ax.text(x + 0.015, y, f'{n} ({c[0]}, {c[1]}, {c[2]})', va='center', fontsize=9)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
os.makedirs('results/figures', exist_ok=True)
fig.savefig('results/figures/fig1_roi_coordinates.png', dpi=200, facecolor='white')
print('saved results/figures/fig1_roi_coordinates.png')
