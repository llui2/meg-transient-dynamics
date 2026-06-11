"""
=====================================================================
EIGENSPECTRUM COMPARISON
=====================================================================
Compare Laplacian eigenspectra (eigenvalue vs mode index) for
different connectivity matrices. Shows how network structure
affects the spectral properties.
"""

from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update(
    {
        "font.size": 7,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
    }
)

# ============================================================
# Configuration
# ============================================================

ROI = 100

ROOT_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FIG_PATH = FIG_DIR / f"fig_eigenspectrum_{ROI}ROI.pdf"

# ============================================================
# Load connectivity matrices
# ============================================================

W_structural = np.load(ROOT_DIR / f"W_Sch{ROI}.npy")
W_edr = np.load(ROOT_DIR / f"W_EDR_Sch{ROI}.npy")
W_geo = np.load(ROOT_DIR / f"W_Geo_Sch{ROI}.npy")

N = ROI

# ============================================================
# Network colors (matching b0-modes_table.py)
# ============================================================

NETWORK_COLORS = {
    "Structural": "#ff4646",
    "EDR": "#00c40a",
    "Geometric": "#7066ff",
}

# ============================================================
# Collect networks
# ============================================================

NETWORKS = {
    "Structural": W_structural,
    "EDR": W_edr,
    "Geometric": W_geo,
}

# ============================================================
# Compute Laplacian eigenspectra
# ============================================================

spectra = {}

for name, W in NETWORKS.items():

    W = np.array(W, dtype=float)
    if W.shape != (ROI, ROI):
        raise ValueError(f"{name}: expected ({ROI},{ROI}), got {W.shape}")

    # Enforce non-negative symmetric weights
    W = np.maximum(W, 0)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0)

    # Graph Laplacian
    degree = np.diag(W.sum(axis=1))
    L = degree - W

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(L)[1:]
    spectra[name] = np.sort(eigvals)

# =====================================================================
# Plot eigenspectra
# =====================================================================
fig, ax = plt.subplots(figsize=(2.5, 2), dpi=300)

k = np.arange(1, ROI)

for name, eigvals in spectra.items():
    ax.plot(
        k,
        eigvals,
        linewidth=1.2,
        label=name,
        color=NETWORK_COLORS[name],
    )
# Plot eigenspectra
# ============================================================

ax.set_xlabel("Mode index $k$")
ax.set_ylabel("Laplacian eigenvalues $λ_k$")
ax.legend(frameon=False, loc="upper left")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(0, ROI)

plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved eigenspectrum figure: {FIG_PATH}")
