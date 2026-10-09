# GAE reading-connectivity subtypes (ds004765)

Analysis code for:

> Khan, A., & Khan, D. (2026). Decoding cognitive diversity in reading: Uncovering latent brain connectivity subtypes using graph autoencoders on rs-fMRI data. *Journal of Neurolinguistics, 79*, 101346. https://doi.org/10.1016/j.jneuroling.2026.101346

This repository contains the code that produced every number and figure in the revised manuscript. It replaces all earlier analyses of this article.

## Data (not included)

| Source | Where | What is used |
|---|---|---|
| OpenNeuro ds004765 (Bathelt, Taylor & Rastle, 2023) | https://doi.org/10.18112/openneuro.ds004765.v1.0.0 | `sub-*/func/*task-rest_bold.nii.gz` (raw BIDS, native space) |
| OSF archive (Bathelt, Rastle & Taylor, 2024) | https://osf.io/yhf2e | `Behavioural Measures/BehaviouralIDs_ForOSF_Final_Updated.xlsx` |

## Setup and run

```bash
pip install -r requirements.txt

# 1. Connectivity extraction, GAE training (10 seeds), embeddings, k-means (k = 3)
python src/train_gae_v1.py --bids /path/to/ds004765 --out results

# 2. Figures 2-10 and results/results_summary.json
python src/make_figures_v1.py --out results \
    --behav "/path/to/Behavioural Measures/BehaviouralIDs_ForOSF_Final_Updated.xlsx"

# 3. Figure 1 (ROI coordinates on an MNI glass brain)
python src/fig1_rois_v1.py
```

A GPU is optional. Training takes a few minutes on a CPU.

## Pipeline

1. **Time series.** Eight 6-mm spheres (`NiftiSpheresMasker`), linear detrend, 0.01-0.1 Hz band-pass, TR = 2 s.
2. **Connectivity.** Pearson r, diagonal set to 0, Fisher z (r clipped to ±0.999), then min-max rescaled to [0, 1] within each participant.
3. **GAE.** Two GCN layers (8 → 32 → 16, ReLU) on the symmetrically normalised adjacency with self-loops. Node features are the rows of the connectivity matrix. The decoder is sigmoid(ZZᵀ) and the loss is MSE. Training is full-batch Adam (lr 1e-3, weight decay 1e-5) for up to 300 epochs, with early stopping on validation MSE (patience 15) and the best checkpoint kept.
4. **Seeds.** Ten models (seeds 42-51), each with its own random 80:20 split. Every model encodes all participants. The 8 × 16 embeddings are averaged across models and flattened to 128 features.
5. **Clustering.** k-means with a pre-specified k = 3 (n_init = 100, random_state = 42). Silhouette, Davies-Bouldin and inertia are reported for k = 2-7.
6. **Graph metrics.** Absolute weights, distances = 1/w, Dijkstra shortest paths (SciPy), global efficiency and characteristic path length.
7. **Behaviour.** One-way ANOVA per measure across clusters, and four pre-selected Pearson correlations. Benjamini-Hochberg FDR q-values are written to `results_summary.json`.

## Results of the run reported in the manuscript

| Quantity | Value |
|---|---|
| Subject folders / extracted | 71 / 60 (11 failed: "spheres are empty") |
| Participants with behavioural data | 58 |
| Validation MSE, 10 seeds | 0.0845 ± 0.0073 |
| Cluster sizes (k = 3) | 14, 37, 9 |
| Best silhouette | k = 2 (0.38); k = 3 = 0.23 |
| Smallest ANOVA p | spoonerisms-last, F(2,55) = 4.35, p = .018, q = .144 |
| Correlations | \|r\| ≤ .18, all p > .18 |

Figures from that run are in `results/figures/`.

## Known limitations (please read)

These are stated in the revised manuscript and apply to all results produced by this code.

1. **ROI coordinates are MNI coordinates applied to native-space images.** The raw BOLD images are not registered to MNI space, so each sphere samples a location set by the participant's head position in the scanner, not the named anatomical region. The 11 extraction failures are spheres that fell outside the image. ROI names are labels for coordinates only.
2. **No motion correction, slice-timing correction or confound regression.**
3. **Seed-averaged embeddings.** Latent dimensions of independently trained models are not aligned, so the element-wise average has no guaranteed meaning. Embeddings include participants each model was trained on.
4. **k is fixed at 3** and is not supported by the silhouette or Davies-Bouldin criteria. Cluster stability under resampling is not assessed.
5. **Sample.** The 60 extracted participants include sub-623, who was excluded by the dataset authors, and sub-125, who is not in the behavioural file.
6. **Within-participant min-max rescaling** removes differences in overall connectivity strength. Graph metrics and cluster connectivity are therefore relative, and comparing them across clusters is circular.

A corrected analysis would use ROIs in a common space (fMRIPrep MNI derivatives, or coordinates transformed to native space), motion and confound handling, the 69 participants included by the dataset authors, data-driven k, resampling stability, and a single or aligned embedding model.

## Citation

If you use this code, please cite the article above and the dataset:

Bathelt, J., Taylor, J., & Rastle, K. (2023). *Language fMRI* (Version 1.0.0) [Data set]. OpenNeuro. https://doi.org/10.18112/openneuro.ds004765.v1.0.0

## License

MIT (code only). The data are subject to the licences of OpenNeuro and OSF.
