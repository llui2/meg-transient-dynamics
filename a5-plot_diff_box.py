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

# ============================================================
# Configuration
# ============================================================

ROI = 100
LOCK = "TEMG"

ROOT_DIR = Path(__file__).resolve().parent
RESULT_CSV = ROOT_DIR / "data" / f"network_fit_ROI{ROI}_{LOCK}.csv"
FIG_PATH = ROOT_DIR / "figures" / f"fig_effect_size_{ROI}ROI.png"

plt.rc('font', family='Helvetica', size=8)
plt.rc('mathtext', fontset='dejavusans')

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

d_edr_per_band = []
d_geo_per_band = []

for band in bands:
    d_edr = []
    d_geo = []

    for subj in subjects:
        d_edr.append(r2[band][subj]["Structural"] - r2[band][subj]["EDR"])
        d_geo.append(r2[band][subj]["Structural"] - r2[band][subj]["Geometric"])

    d_edr_per_band.append(d_edr)
    d_geo_per_band.append(d_geo)

# ============================================================
# Plot
# ============================================================

x = np.arange(len(bands))

fig, ax = plt.subplots(figsize=(3.5, 1.7), dpi=300)

# Prepare data for boxplot
data = d_edr_per_band + d_geo_per_band
positions = list(x) + list(x + 0.35)
colors = ['#00c40a'] * len(bands) + ['#7066ff'] * len(bands)

box = ax.boxplot(
    data,
    positions=positions,
    widths=0.3,
    patch_artist=True,
    showfliers=False,
    showcaps=False,  # remove horizontal bars at whisker ends
)

# Style each box group: unfilled boxes, colored lines (box/median/whiskers)
line_width = 1.2
for i, (patch, color) in enumerate(zip(box["boxes"], colors)):
    patch.set_facecolor("none")
    patch.set_edgecolor(color)
    patch.set_linewidth(line_width)
    patch.set_alpha(1.0)
    patch.set_zorder(3)

    box["medians"][i].set_color(color)
    box["medians"][i].set_linewidth(line_width)
    box["medians"][i].set_zorder(4)

    # whiskers are 2 per box
    w0, w1 = box["whiskers"][2 * i], box["whiskers"][2 * i + 1]
    for w in (w0, w1):
        w.set_color(color)
        w.set_linewidth(line_width)
        w.set_zorder(4)

# If caps exist for any reason, hide them (no horizontal bars)
for cap in box.get("caps", []):
    cap.set_visible(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.axhline(0, color="k", linestyle="-", linewidth=1, alpha=0.2)

ax.set_yticks(np.arange(-0.1, 0.2, 0.1))
ax.set_xticks(x + 0.175)
ax.set_xticklabels(bands)

ax.set_ylim(-0.08, 0.3)

ax.set_ylabel(r"Paired $\Delta R^2$")
# ax.set_title("Frequency-dependent structural advantage", fontsize=8)

# Custom legend
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#00c40a", linewidth=line_width),
    plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="#7066ff", linewidth=line_width),
]
ax.legend(handles, ['Structural − EDR', 'Structural − Geometric'], frameon=False)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")
