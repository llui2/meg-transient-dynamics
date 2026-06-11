"""
=====================================================================
NETWORK COMPARISON PLOT
=====================================================================
Plot network comparison results (R² and tau) from CSV produced
by model fitting across frequency bands.
"""

import csv
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl

# ============================================================
# Configuration
# ============================================================

ROI = 100
LOCK = "TEMG"

ROOT_DIR = Path(__file__).resolve().parent
RESULT_CSV = ROOT_DIR / "data" / f"network_fit_ROI{ROI}_{LOCK}.csv"
FIG_PATH = ROOT_DIR / "figures" / f"fig_network_comparison_{ROI}ROI.pdf"

NETWORK_NAMES = ["Structural", "EDR", "Geometric"]

mpl.rcParams.update(
    {
        "font.size": 7,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
    }
)

# ============================================================
# Load CSV and collect values
# ============================================================

if not RESULT_CSV.exists():
    raise FileNotFoundError(f"Results CSV not found: {RESULT_CSV}")

bands = []
subjects = set()

r2_values = {model: {} for model in NETWORK_NAMES}
tau_values = {model: {} for model in NETWORK_NAMES}

with open(RESULT_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:

        roi = int(row["roi"])
        lock = row["lock"]
        band = row["band"]
        metric = row["metric"]
        model = row["model"]
        value = float(row["value"])

        if band not in bands:
            bands.append(band)

        subj = row.get("subject")
        if subj is not None:
            subjects.add(subj)

        if metric == "R2":
            r2_values[model].setdefault(band, []).append(value)
        elif metric == "Tau":
            tau_values[model].setdefault(band, []).append(value)

n_subjects = len(subjects)
if n_subjects == 0:
    raise RuntimeError("No subject data found in CSV")

# ============================================================
# Compute summary statistics
# ============================================================

r2_mean = {model: [] for model in NETWORK_NAMES}
r2_std = {model: [] for model in NETWORK_NAMES}
tau_mean = {model: [] for model in NETWORK_NAMES}
tau_std = {model: [] for model in NETWORK_NAMES}

for band in bands:
    for model in NETWORK_NAMES:

        r2_arr = np.asarray(r2_values[model][band], dtype=float)
        tau_arr = np.asarray(tau_values[model][band], dtype=float)

        r2_mean[model].append(r2_arr.mean())
        r2_std[model].append(r2_arr.std())

        tau_mean[model].append(tau_arr.mean())
        tau_std[model].append(tau_arr.std())

# ============================================================
# Plot R^2 comparison
# ============================================================

x = np.arange(len(bands))
width = 0.25

colors = {
    "Structural": "#ff4646",
    "EDR": "#2fff90",
    "Geometric": "#7066ff",
}

rng = np.random.default_rng(0)

plt.rc('font', family='Helvetica', size=7)
plt.rc('mathtext', fontset='dejavusans')

fig, ax = plt.subplots(figsize=(3,1.5), dpi=300)

# Scatter subject-level points
# for i, band in enumerate(bands):
#     for offset, model in zip([-width, 0.0, width], NETWORK_NAMES):
#         vals = np.asarray(r2_values[model][band])
#         jitter = rng.normal(i + offset, width * 0.1, size=len(vals))
#         ax.scatter(
#             jitter,
#             vals,
#             facecolors="none",
#             edgecolors=colors[model],
#             linewidth=0.7,
#             s=4,
#             zorder=1,
#             alpha=0.6,
#         )

# Bar plots with error bars
error_style = dict(elinewidth=1, capsize=1, capthick=1)

ax.bar(
    x - width,
    r2_mean["Structural"],
    width,
    yerr=r2_std["Structural"],
    color=colors["Structural"],
    edgecolor="black",
    linewidth=0.8,
    alpha=1,
    error_kw=error_style,
    label="Structural",
    zorder=2,
)

ax.bar(
    x,
    r2_mean["EDR"],
    width,
    yerr=r2_std["EDR"],
    color=colors["EDR"],
    edgecolor="black",
    linewidth=0.8,
    alpha=1,
    error_kw=error_style,
    label="EDR",
    zorder=2,
)

ax.bar(
    x + width,
    r2_mean["Geometric"],
    width,
    yerr=r2_std["Geometric"],
    color=colors["Geometric"],
    edgecolor="black",
    linewidth=0.8,
    alpha=1,
    error_kw=error_style,
    label="Geometric",
    zorder=2,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_ylabel("Goodness of fit ($R^2$)")

ax.set_ylim(0, 0.7)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.set_yticklabels([0, 0.2, 0.4, 0.6])

ax.set_xticks(x)
ax.set_xticklabels(bands)

# ax.set_title(f"Network comparison (N={n_subjects})")

legend_handles = [
    Patch(facecolor=colors[name], label=name, alpha=1, edgecolor="black", linewidth=0.8)
    for name in NETWORK_NAMES
]
ax.legend(handles=legend_handles, frameon=False, loc="upper right")

os.makedirs(FIG_PATH.parent, exist_ok=True)
plt.tight_layout()
plt.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")

# ============================================================
# Print summary tables
# ============================================================

print("\n" + "=" * 100)
print(f"SUMMARY: Network Comparison ({ROI} ROIs, N={n_subjects} subjects)")
print("=" * 100)
print(f"{'Band':<12} {'R² Struc':<18} {'R² EDR':<18} {'R² Geo':<18} {'Best':<12}")
print("-" * 100)

for i, band in enumerate(bands):
    vals = [
        r2_mean["Structural"][i],
        r2_mean["EDR"][i],
        r2_mean["Geometric"][i],
    ]
    best = NETWORK_NAMES[int(np.argmax(vals))]
    print(
        f"{band:<12} "
        f"{r2_mean['Structural'][i]:.4f} ± {r2_std['Structural'][i]:.4f}   "
        f"{r2_mean['EDR'][i]:.4f} ± {r2_std['EDR'][i]:.4f}   "
        f"{r2_mean['Geometric'][i]:.4f} ± {r2_std['Geometric'][i]:.4f}   "
        f"{best:<12}"
    )

print("=" * 100)

print("\n" + "=" * 100)
print(f"TAU VALUES: Network Comparison ({ROI} ROIs, N={n_subjects} subjects)")
print("=" * 100)
print(f"{'Band':<12} {'τ Struc':<18} {'τ EDR':<18} {'τ Geo':<18}")
print("-" * 100)

for i, band in enumerate(bands):
    print(
        f"{band:<12} "
        f"{tau_mean['Structural'][i]:.4f} ± {tau_std['Structural'][i]:.4f}   "
        f"{tau_mean['EDR'][i]:.4f} ± {tau_std['EDR'][i]:.4f}   "
        f"{tau_mean['Geometric'][i]:.4f} ± {tau_std['Geometric'][i]:.4f}"
    )

print("=" * 100)
