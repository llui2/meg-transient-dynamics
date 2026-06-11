"""
=====================================================================
EFFECT SIZE PLOT
=====================================================================
Plot mean paired ΔR² (structural advantage) as a function of 
frequency band across subjects.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# Configuration
# ============================================================

ROI = 100
LOCK = "TEMG"

ROOT_DIR = Path(__file__).resolve().parent
RESULT_CSV = ROOT_DIR / "data_LF" / f"network_fit_ROI{ROI}_{LOCK}.csv"
FIG_PATH = ROOT_DIR / "figures" / f"fig_effect_size_{ROI}ROI.png"

mpl.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
    }
)

# ============================================================
# Load CSV
# ============================================================

bands = []
subjects = []

r2 = {}

with open(RESULT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["metric"] != "R2":
            continue

        band = row["band"]
        subj = row["subject"]
        model = row["model"]
        val = float(row["value"])

        if band not in bands:
            bands.append(band)
            r2[band] = {}

        if subj not in r2[band]:
            r2[band][subj] = {}

        r2[band][subj][model] = val

        if subj not in subjects:
            subjects.append(subj)

bands = list(bands)
n = len(subjects)

# ============================================================
# Compute paired effects
# ============================================================

mean_edr, sem_edr = [], []
mean_geo, sem_geo = [], []

for band in bands:
    d_edr = []
    d_geo = []

    for subj in subjects:
        d_edr.append(r2[band][subj]["Structural"] - r2[band][subj]["EDR"])
        d_geo.append(r2[band][subj]["Structural"] - r2[band][subj]["Geometric"])

    d_edr = np.array(d_edr)
    d_geo = np.array(d_geo)

    mean_edr.append(d_edr.mean())
    sem_edr.append(d_edr.std(ddof=1) / np.sqrt(n))

    mean_geo.append(d_geo.mean())
    sem_geo.append(d_geo.std(ddof=1) / np.sqrt(n))

# ============================================================
# Plot
# ============================================================

x = np.arange(len(bands))

fig, ax = plt.subplots(figsize=(3.2, 2.2), dpi=300)

ax.errorbar(
    x, mean_edr, yerr=sem_edr,
    marker=" ", linewidth=1.5, capsize=3,
    color="#00c40a", label="Structural − EDR"
)

ax.errorbar(
    x, mean_geo, yerr=sem_geo,
    marker=" ", linewidth=1.5, capsize=3,
    color="#7066ff", label="Structural − Geometric"
)

ax.axhline(0, color="k", linestyle="--", linewidth=1)

ax.set_yticks(np.arange(-0.1, 0.2, 0.1))
ax.set_xticks(x)
ax.set_xticklabels(bands)

ax.set_ylim(-0.05, 0.2)

ax.set_ylabel(r"Mean paired $\Delta R^2$")
ax.set_title("Frequency-dependent structural advantage", fontsize=8)

ax.legend(frameon=False)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")
