"""
=====================================================================
NETWORK FREQUENCY SPECTRA
=====================================================================
Compute and visualize telegrapher-implied modal frequency densities
for connectivity graphs. Shows which frequencies are naturally
supported by each network's Laplacian eigenspectrum.
"""

from pathlib import Path

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import networkx as nx
import nibabel as nib
from nilearn import surface
from lapy import TriaMesh
from lapy.solver import Solver
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

ROI = 100
tau = 0.45  # Timescale parameter for telegrapher model
FMAX = 10.0  # Maximum frequency for plotting (Hz)
NGRID = 600  # Grid points for density estimation

BANDS = {
    "Fast Delta": (2, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 80),
}
SELECT_BAND = "Fast Delta"

ROOT = Path(__file__).resolve().parent
PARC = ROOT / "parcellations" / f"Schaefer2018_{ROI}Parcels_7Networks_order.dlabel.nii"
FIG_PATH = ROOT / "figures" / f"fig_frequency_spectra_{ROI}ROI.pdf"

# ============================================================
# Utility functions
# ============================================================

def laplacian(W):
    W = np.maximum(W, 0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)
    return np.diag(W.sum(1)) - W

def telegrapher_frequencies(L, tau):
    lam = np.linalg.eigvalsh(L)[1:]
    disc = 4 * tau * lam - 1
    mask = disc > 0
    omega = np.sqrt(disc[mask]) / (2 * tau)
    return omega / (2 * np.pi)

# ============================================================
# Structural connectivity
# ============================================================

W_sc = np.load(ROOT / f"W_Sch{ROI}.npy")
L_sc = laplacian(W_sc)
L_sc_norm = np.linalg.norm(L_sc)

# ============================================================
# EDR connectivity (exponential distance rule)
# ============================================================

labels = nib.load(PARC).get_fdata().squeeze().astype(int)

coords_lh, _ = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.L.pial.surf.gii")
coords_rh, _ = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.R.pial.surf.gii")
coords = np.vstack([coords_lh, coords_rh])

centroids = np.zeros((ROI, 3))
for i, pid in enumerate(np.unique(labels)[1:]):
    centroids[i] = coords[labels == pid].mean(0)

ALPHA = 0.12  # mm^-1 decay constant
D = cdist(centroids, centroids)
np.fill_diagonal(D, np.inf)

W_edr = np.exp(-ALPHA * D)
np.fill_diagonal(W_edr, 0)
L_edr = laplacian(W_edr)
W_edr *= L_sc_norm / np.linalg.norm(L_edr)

out_path = ROOT / f"W_EDR_Sch{ROI}.npy"
np.save(out_path, W_edr)
print(f"Saved physical EDR graph to: {out_path}")

# ============================================================
# Geometric connectivity (cotangent Laplacian)
# ============================================================

coords_lh, faces_lh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.L.pial.surf.gii")
coords_rh, faces_rh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.R.pial.surf.gii")
n_lh = coords_lh.shape[0]

labels = labels[: n_lh + coords_rh.shape[0]]

def hemi_block(coords, faces, labels, roi_ids):
    K = Solver(TriaMesh(coords, faces), lump=True).stiffness.tocsr()
    Wv = (-K).maximum(0)
    W = np.zeros((ROI, ROI))
    vids = {p: np.where(labels == p)[0] for p in roi_ids}
    for p in roi_ids:
        for q in roi_ids:
            if q <= p:
                continue
            w = Wv[np.ix_(vids[p], vids[q])].sum()
            if w > 0:
                W[p-1, q-1] = W[q-1, p-1] = w
    return W

half = ROI // 2
W_geo = (
    hemi_block(coords_lh, faces_lh, labels[:n_lh], range(1, half + 1)) +
    hemi_block(coords_rh, faces_rh, labels[n_lh:], range(half + 1, ROI + 1))
)

# Add inter-hemispheric connections in nodes that are spatially close
THRESHOLD = 15.0  # mm
for i in range(half):
    for j in range(half, ROI):
        dist = np.linalg.norm(centroids[i] - centroids[j])
        if dist < THRESHOLD:
            W_geo[i, j] = W_geo[j, i] = 20.0

L_geo = laplacian(W_geo)
W_geo *= L_sc_norm / np.linalg.norm(L_geo)
out_path = ROOT / f"W_Geo_Sch{ROI}.npy"
np.save(out_path, W_geo)
print(f"Saved geometric graph to: {out_path}")

# ============================================================
# Network collection
# ============================================================

NETWORKS = {
    "Structural": W_sc,
    "EDR": W_edr,
    "Geometric": W_geo,
}

# ============================================================
# Compute modal frequencies
# ============================================================

freqs = {
    name: telegrapher_frequencies(laplacian(W), tau)
    for name, W in NETWORKS.items()
}

# ============================================================
# Plot frequency spectra
# ============================================================

# Network colors (matching c2-spectrum_compare.py)
NETWORK_COLORS = {
    "Structural": "#ff4646",
    "EDR": "#00c40a",
    "Geometric": "#7066ff",
}

fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
f_grid = np.linspace(0, FMAX, NGRID)

if SELECT_BAND:
    fmin, fmax = BANDS[SELECT_BAND]
    ax.axvspan(fmin, fmax, alpha=0.15, color="k", label=SELECT_BAND)

for name, f in freqs.items():
    if f.size < 2:
        continue
    ax.plot(f_grid, gaussian_kde(f)(f_grid), lw=1.8, label=name, color=NETWORK_COLORS[name])

ax.set_xlim(0, FMAX)
ax.set_xlabel(r"$f_k = \omega_k / (2\pi)$ (Hz)")
ax.set_ylabel("Density")
ax.legend(frameon=False)

FIG_PATH.parent.mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close()

print(f"Saved figure: {FIG_PATH}")
