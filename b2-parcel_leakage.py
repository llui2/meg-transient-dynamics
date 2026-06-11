"""
=====================================================================
PARCEL LEAKAGE ANALYSIS
=====================================================================
Visualize parcellated MEG signal leakage as a function of
topological (graph) distance on the structural connectome.
Quantifies cross-talk between parcels in beamformer reconstruction.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nibabel as nib
from nilearn import plotting, surface

warnings.filterwarnings("ignore")
logging.getLogger("nibabel").setLevel(logging.ERROR)

# ============================================================
# Configuration
# ============================================================

SUBJECT = "104012"
LOCK = "TEMG"
ROI = 100
REFERENCE_IDX = 46  # Reference parcel for visualization (100 ROI: 46, 200 ROI: 22, 400 ROI: 48)
RASTERIZED_SURFACE_DPI = 300

DATA_ROOT = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Structural connectivity matrix used to define graph distance
W_FILE = Path(f"W_EDR_Sch{ROI}.npy")

# Output figure
FIG_PATH = FIG_DIR / f"fig_parcel_leakage_{SUBJECT}_{ROI}parcels.pdf"

# ============================================================
# Load parcellated data
# ============================================================

print("[1/4] Loading parcellated resolution matrix")

npz_path = DATA_ROOT / SUBJECT / f"source_parcels_Schaefer{ROI}_{SUBJECT}_{LOCK}.npz"
if not npz_path.exists():
    raise FileNotFoundError(f"Parcellated data not found: {npz_path}")

data = np.load(npz_path)
R_roi = data["R_roi"]
labels = data["labels"]

n_parcels = R_roi.shape[0]
print(f"  Parcels: {n_parcels}")

# ============================================================
# Load connectivity and compute graph distances
# ============================================================

print("[2/4] Computing graph distances")

W = np.load(W_FILE)

# Enforce symmetry and non-negativity
W = np.maximum(W, 0)
W_sym = 0.5 * (W + W.T)

# Convert weights to distances: strong connection -> short distance
W_dist = W_sym.copy()
W_dist[W_dist == 0] = 1e-10
D_weighted = 1.0 / W_dist

# Shortest-path distances on weighted graph
D_graph = shortest_path(
    csr_matrix(D_weighted),
    directed=False,
    method="D",
)

# Replace infinities for disconnected components
finite_max = D_graph[np.isfinite(D_graph)].max()
D_graph[~np.isfinite(D_graph)] = finite_max * 2.0

# Extract off-diagonal entries
mask = ~np.eye(n_parcels, dtype=bool)
R_offdiag = np.abs(R_roi[mask])
D_offdiag = D_graph[mask]

corr, _ = stats.spearmanr(D_offdiag, R_offdiag)
print(f"  Spearman r (leakage vs graph distance) = {corr:.3f}")

# ============================================================
# Bin leakage by graph distance
# ============================================================

print("[3/4] Binning leakage by graph distance")

d_min, d_max = D_offdiag.min(), D_offdiag.max()
n_bins = 20
bin_edges = np.linspace(d_min, d_max, n_bins + 1)

bin_centers = []
bin_means = []
bin_stds = []

for i in range(len(bin_edges) - 1):
    mask_bin = (D_offdiag >= bin_edges[i]) & (D_offdiag < bin_edges[i + 1])
    if mask_bin.sum() > 10:
        bin_centers.append(0.5 * (bin_edges[i] + bin_edges[i + 1]))
        bin_means.append(R_offdiag[mask_bin].mean())
        bin_stds.append(R_offdiag[mask_bin].std())

bin_centers = np.asarray(bin_centers)
bin_means = np.asarray(bin_means)
bin_stds = np.asarray(bin_stds)

# ============================================================
# Fit exponential decay
# ============================================================

def exp_decay(x, a, beta, c):
    return a * np.exp(-x / beta) + c

try:
    popt, _ = curve_fit(
        exp_decay,
        bin_centers,
        bin_means,
        p0=[0.5, bin_centers.mean(), 0.05],
        maxfev=5000,
    )
    beta = popt[1]
    print(f"  Exponential decay β = {beta:.2f} (graph distance units)")
except Exception as e:
    print(f"  Exponential fit failed: {e}")
    popt = None
    beta = np.nan

# ============================================================
# Visualization
# ============================================================

print("[4/4] Creating figure")

fig = plt.figure(figsize=(6, 3), dpi=300)
gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

# ------------------------------------------------------------
# Left panel: leakage decay vs graph distance
# ------------------------------------------------------------

ax1 = fig.add_subplot(gs[0, 0])

ax1.text(-0.3, 0.98, "a", fontsize=18, ha="center", va="center",
         transform=ax1.transAxes, color="black", fontweight="bold",
         fontname="DejaVu Sans")

ax1.fill_between(
    bin_centers,
    bin_means - bin_stds,
    bin_means + bin_stds,
    alpha=0.2,
    color="#0040FF",
    linewidth=0,
)

ax1.plot(
    bin_centers,
    bin_means,
    color="#0040FF",
    linewidth=1.5,
    label="$\\langle |R^{\\text{ROI}}_{ab}| \\rangle_{d_{ab}=d}$",
)

if popt is not None:
    x_fit = np.linspace(bin_centers.min(), bin_centers.max(), 200)
    ax1.plot(
        x_fit,
        exp_decay(x_fit, *popt),
        "--",
        color="#FF0000",
        linewidth=1.5,
        label="$A e^{-d/\\beta}+c$",
    )

d_close = np.percentile(D_offdiag, 10)
d_nearby = np.percentile(D_offdiag, 30)

ax1.axvline(d_close, color="#FF8427", linewidth=1.2, label="$q_{0.10}(d_{ab})$")
ax1.axvline(d_nearby, color="#C93FFF", linewidth=1.2, label="$q_{0.30}(d_{ab})$")

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_xlabel("$d_{ab}$")
ax1.set_ylabel("$|R^{\mathrm{ROI}}_{ab}|$")
ax1.set_xlim(bin_centers.min() * 0.9, bin_centers.max() * 1.05)
ax1.set_ylim(0, bin_means.max() * 1.1)
ax1.legend(fontsize=7, frameon=False)

# ------------------------------------------------------------
# Right panel: parcel leakage on cortical surface
# ------------------------------------------------------------

ax2 = fig.add_subplot(gs[0, 1], projection="3d")

if REFERENCE_IDX is not None:
    ref_idx = REFERENCE_IDX
else:
    connectivity = W_sym.sum(axis=1)
    ref_idx = np.argmin(np.abs(connectivity - np.median(connectivity)))

print(f"  Reference parcel index: {ref_idx}")

leakage_ref = np.abs(R_roi[ref_idx, :])
ref_label = labels[ref_idx]

coords_lh, faces_lh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.L.pial.surf.gii")
coords_rh, faces_rh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.R.pial.surf.gii")

sulc = nib.load("plot_aux_data/fs_LR.32k.LR.sulc.dscalar.nii").get_fdata().squeeze().ravel()
n_lh = coords_lh.shape[0]
sulc_lh = sulc[:n_lh]
sulc_rh = sulc[n_lh:]

parc_path = Path("parcellations") / f"Schaefer2018_{ROI}Parcels_7Networks_order.dlabel.nii"
labels_surface = nib.load(parc_path).get_fdata().squeeze().astype(int).ravel()

leakage_lh = np.zeros(n_lh)
leakage_rh = np.zeros(coords_rh.shape[0])

for i, lab in enumerate(labels):
    leakage_lh[labels_surface[:n_lh] == lab] = leakage_ref[i]
    leakage_rh[labels_surface[n_lh:] == lab] = leakage_ref[i]

vmax = np.percentile(leakage_ref, 95)
leakage_lh = np.clip(leakage_lh, 0, vmax)
leakage_rh = np.clip(leakage_rh, 0, vmax)

ref_mask_lh = labels_surface[:n_lh] == ref_label
ref_mask_rh = labels_surface[n_lh:] == ref_label
leakage_lh[ref_mask_lh] = vmax * 1.5
leakage_rh[ref_mask_rh] = vmax * 1.5

magma = plt.cm.magma(np.linspace(0, 1, 256))
magma[-20:] = mcolors.to_rgba("#FF0000")
custom_cmap = mcolors.LinearSegmentedColormap.from_list("magma_red", magma)

plotting.plot_surf_stat_map(
    (coords_lh, faces_lh),
    stat_map=leakage_lh,
    hemi="left",
    view="dorsal",
    bg_map=sulc_lh,
    cmap=custom_cmap,
    colorbar=False,
    bg_on_data=True,
    darkness=0.4,
    vmin=0,
    vmax=vmax * 1.5,
    axes=ax2,
)

plotting.plot_surf_stat_map(
    (coords_rh, faces_rh),
    stat_map=leakage_rh,
    hemi="right",
    view="dorsal",
    bg_map=sulc_rh,
    cmap=custom_cmap,
    colorbar=False,
    bg_on_data=True,
    darkness=0.4,
    vmin=0,
    vmax=vmax * 1.5,
    axes=ax2,
)

# The cortical surface generates a very large number of vector polygons in PDF.
# Rasterize only this 3D panel at high DPI to keep appearance while reducing size.
for artist in ax2.collections:
    artist.set_rasterized(True)

ax2.view_init(elev=90, azim=-90)
ax2.set_axis_off()

ax2.text2D(-0.04, 0.98, "b", fontsize=18, ha="center", va="center",
           transform=ax2.transAxes, color="black", fontweight="bold",
           fontname="DejaVu Sans")

sm = plt.cm.ScalarMappable(
    cmap="magma",
    norm=plt.Normalize(vmin=0, vmax=vmax),
)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label("$|R^{\mathrm{ROI}}_{a_0 \, b}|$", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(
    FIG_PATH,
    bbox_inches="tight",
    facecolor="white",
    dpi=RASTERIZED_SURFACE_DPI,
)
plt.close()

print(f"Saved figure: {FIG_PATH}")

# ============================================================
# Summary statistics
# ============================================================

print("\n" + "=" * 50)
print("PARCEL LEAKAGE SUMMARY")
print("=" * 50)
print(f"Parcellation: Schaefer {ROI}")
print(f"Spearman r (leakage vs graph distance) = {corr:.3f}")

if popt is not None:
    print(f"Exponential decay β = {beta:.2f}")

print("\nLeakage at key graph distances:")
for p in [0, 10, 25, 50, 75, 90]:
    d_p = D_offdiag.min() if p == 0 else np.percentile(D_offdiag, p)
    mask_p = (D_offdiag >= 0.9 * d_p) & (D_offdiag < 1.1 * d_p)
    val = R_offdiag[mask_p].mean() if mask_p.sum() > 0 else np.nan
    print(f"  {p:3d}th percentile (d={d_p:.2f}): |R_roi| = {val:.3f}")

print("=" * 50)
