"""
=====================================================================
SOURCE LEAKAGE ANALYSIS
=====================================================================
Visualize MEG source-level signal leakage as a function of 
3D Euclidean distance. Quantifies and displays the spatial
extent of cross-talk in beamformer reconstruction.
"""

import logging
import warnings
from pathlib import Path

import h5py
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger("nibabel").setLevel(logging.ERROR)

# ============================================================
# Configuration
# ============================================================

SUBJECT = "104012"
CONDITION = "LH"
REFERENCE_IDX = 7700  # Source index for single-source visualization (None = auto)

DATA_ROOT = Path("data")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# Output figure
FIG_PATH = FIG_DIR / f"fig_source_leakage_{SUBJECT}.pdf"

# ============================================================
# Load source-level data
# ============================================================

print("[1/3] Loading source-level resolution matrix")

meta_path = DATA_ROOT / SUBJECT / "meta.mat"

with h5py.File(str(meta_path), "r") as f:
    R = np.asarray(f["R"], dtype=np.float64)
    pos = np.asarray(f["check/pos"], dtype=np.float64).T  # (n_sources, 3)

n_sources = R.shape[0]
pos_mm = pos * 10.0  # convert cm to mm

print(f"  Number of sources: {n_sources}")

# ============================================================
# Compute pairwise distances and leakage correlation
# ============================================================

print("[2/3] Computing pairwise distances")

D = squareform(pdist(pos_mm, metric="euclidean"))

offdiag_mask = ~np.eye(n_sources, dtype=bool)
R_offdiag = np.abs(R[offdiag_mask])
D_offdiag = D[offdiag_mask]

corr, _ = stats.spearmanr(D_offdiag, R_offdiag)
print(f"  Spearman correlation (distance vs |R|): r = {corr:.3f}")

# ============================================================
# Bin leakage by distance
# ============================================================

bin_width = 1.0  # mm
bin_edges = np.arange(0, 160, bin_width)

bin_centers = []
bin_means = []
bin_stds = []

for i in range(len(bin_edges) - 1):
    mask = (D_offdiag >= bin_edges[i]) & (D_offdiag < bin_edges[i + 1])
    if mask.sum() > 100:
        bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        bin_means.append(R_offdiag[mask].mean())
        bin_stds.append(R_offdiag[mask].std())

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
        p0=[0.8, 30.0, 0.1],
        maxfev=5000,
    )
    beta = popt[1]
    print(f"  Exponential decay length β = {beta:.1f} mm")
except Exception as e:
    print(f"  Exponential fit failed: {e}")
    popt = None
    beta = np.nan

# ============================================================
# Visualization
# ============================================================

print("[3/3] Creating figure")

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(6, 3), dpi=300, gridspec_kw={"width_ratios": [1.2, 1]}, tight_layout=True
)

ax1.text(-0.3, 0.98, "a", fontsize=18, ha="center", va="center",
         transform=ax1.transAxes, color="black", fontweight="bold",
         fontname="DejaVu Sans")

# ------------------------------------------------------------
# Left panel: leakage decay curve
# ------------------------------------------------------------

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
    label="$\\langle |R_{ij}| \\rangle_{d_{ij}=d}$",
)

if popt is not None:
    x_fit = np.linspace(0, 150, 200)
    ax1.plot(
        x_fit,
        exp_decay(x_fit, *popt),
        "--",
        color="#FF0000",
        linewidth=1.5,
        label=f"$A e^{{-d/\\beta}}+c$",
    )

d_close = 20
d_nearby = 50

ax1.axvline(d_close, color="#FF8427", linewidth=1.2, label=f"$d_{{ij}} = {d_close}$ mm")
ax1.axvline(d_nearby, color="#C93FFF", linewidth=1.2, label=f"$d_{{ij}} = {d_nearby}$ mm")

ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

ax1.set_xlabel("$d_{ij}$ (mm)")
ax1.set_ylabel("$|R_{ij}|$")
ax1.set_xlim(0, 150)
ax1.set_ylim(0, bin_means[0] * 1.08)
ax1.legend(fontsize=7, frameon=False)

# ------------------------------------------------------------
# Right panel: single-source leakage map
# ------------------------------------------------------------

if REFERENCE_IDX is not None:
    ref_idx = REFERENCE_IDX
else:
    center = pos_mm.mean(axis=0)
    distances = np.linalg.norm(pos_mm - center, axis=1)
    ref_idx = np.argmin(np.abs(distances - 20))

ref_pos = pos_mm[ref_idx]
leakage_ref = np.abs(R[ref_idx, :])

scatter = ax2.scatter(
    pos_mm[:, 1],
    pos_mm[:, 0],
    c=leakage_ref,
    cmap="magma",
    s=2,
    alpha=1.0,
    vmin=0,
    vmax=0.8,
    rasterized=True,
)

ax2.scatter(
    ref_pos[1],
    ref_pos[0],
    c="#FF0000",
    s=40,
    edgecolor="black",
    zorder=100,
)

theta = np.linspace(0, 2 * np.pi, 100)
for d, color in [(d_close, "#FF8427"), (d_nearby, "#C93FFF")]:
    ax2.plot(
        ref_pos[1] + d * np.cos(theta),
        ref_pos[0] + d * np.sin(theta),
        color=color,
        linewidth=1.5,
    )

ax2.set_aspect("equal")
ax2.axis("off")

ax2.text(-0.04, 0.98, "b", fontsize=18, ha="center", va="center",
         transform=ax2.transAxes, color="black", fontweight="bold",
         fontname="DejaVu Sans")

cbar = plt.colorbar(scatter, ax=ax2, shrink=0.6, pad=0.02)
cbar.set_label("$|R_{i_0 \, j}|$", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(FIG_PATH, bbox_inches="tight", facecolor="white", dpi=200)
plt.close()

print(f"Saved figure: {FIG_PATH}")

# ============================================================
# Summary statistics
# ============================================================

print("\n" + "=" * 50)
print("SOURCE LEAKAGE SUMMARY")
print("=" * 50)
print(f"Spearman r = {corr:.3f}")
print(f"Exponential decay β = {beta:.0f} mm")
print("\nLeakage at key distances:")

for d in [0, 10, 20, 50, 100]:
    if d == 0:
        val = 1.0
    else:
        mask = (D_offdiag >= d - 5) & (D_offdiag < d + 5)
        val = R_offdiag[mask].mean() if mask.sum() > 0 else np.nan
    print(f"  d = {d:3d} mm: |R| = {val:.3f}")

print("=" * 50)
