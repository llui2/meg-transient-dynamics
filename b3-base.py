"""
=====================================================================
LAPLACIAN BASIS COMPARISON
=====================================================================
Plot cumulative variance explained by Laplacian eigenmodes
for all connectivity matrices, comparing how efficiently
different network types span the signal space.
"""

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# Style
# ============================================================

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 7,
    }
)

# ============================================================
# Paths and input files
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

NETWORK_FILES = sorted(ROOT_DIR.glob("W_*Sch*.npy"))
if not NETWORK_FILES:
    raise FileNotFoundError("No network matrices matching 'W_*Sch*.npy' found.")

# ============================================================
# Containers for curves
# ============================================================

curves = []

color_map = {
    "Structural": "#ff4646",
    "EDR": "#00c40a",
    "Geometric": "#7066ff",
}

linestyle_map = {
    100: "-",
    200: "--",
    400: "-.",
}

# ============================================================
# Process each network matrix
# ============================================================

for path in NETWORK_FILES:

    print(f"Processing {path.name}")

    try:
        W = np.load(path)
    except Exception as exc:
        print(f"  Skipping (load failed): {exc}")
        continue

    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        print("  Skipping (not a square matrix)")
        continue

    # --------------------------------------------------------
    # Identify network type and ROI count from filename
    # --------------------------------------------------------

    stem = path.stem
    match = re.search(r"Sch(\d+)", stem)
    roi = int(match.group(1)) if match else None

    if "EDR" in stem:
        network_name = "EDR"
    elif "Geo" in stem:
        network_name = "Geometric"
    elif "Sch" in stem:
        network_name = "Structural"
    else:
        network_name = stem.replace("W_", "")

    label = f"{network_name} (Sch{roi})"

    color = color_map.get(network_name, "#7f8c8d")
    linestyle = linestyle_map.get(roi, "-")

    # --------------------------------------------------------
    # Compute Laplacian
    # --------------------------------------------------------

    W = np.array(W, dtype=float)
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)

    degree = np.diag(W.sum(axis=1))
    L = degree - W

    # --------------------------------------------------------
    # Laplacian spectrum
    # --------------------------------------------------------

    eigvals = np.linalg.eigvalsh(L)

    # Sort modes by decreasing squared magnitude
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]

    # --------------------------------------------------------
    # Cumulative explained variance
    # --------------------------------------------------------

    power = eigvals ** 2
    total_power = np.sum(power)

    if total_power <= 0:
        cumulative_r2 = np.linspace(0, 1, len(eigvals))
    else:
        cumulative_r2 = np.cumsum(power) / total_power

    x_axis = np.arange(1, len(cumulative_r2) + 1)

    curves.append((label, x_axis, cumulative_r2, color, linestyle))

# ============================================================
# Plot
# ============================================================

if not curves:
    raise RuntimeError("No valid Laplacian spectra were computed.")

fig, ax = plt.subplots(figsize=(4, 2.5))

for label, x, r2, color, linestyle in curves:
    ax.plot(
        x,
        r2,
        linewidth=1.1,
        label=label,
        color=color,
        linestyle=linestyle,
    )

max_len = max(len(x) for _, x, _, _, _ in curves)

ax.set_xlabel("Number of eigenmodes")
ax.set_ylabel(r"Variance of $L$ explained ($R^2$)")
ax.set_xlim(0, max_len)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle="--", alpha=0.2)

handles, labels = ax.get_legend_handles_labels()
for h in handles:
    h.set_linewidth(1.2)

ax.legend(handles, labels, fontsize=6, loc="lower right")

plt.tight_layout()
fig_path = FIG_DIR / "fig_laplacian_basis_comparison.pdf"
plt.savefig(fig_path, dpi=300)
plt.close(fig)

print(f"Saved figure: {fig_path}")
