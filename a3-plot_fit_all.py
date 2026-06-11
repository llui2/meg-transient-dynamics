"""
=====================================================================
NETWORK COMPARISON PLOT (ALL DATASETS)
=====================================================================
Plot network comparison results (R² and tau) from CSV produced
by model fitting across frequency bands, for LH/RH/LF/RF datasets.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib as mpl

# ============================================================
# Configuration
# ============================================================

ROI = 400
LOCK = "TEMG"

ROOT_DIR = Path(__file__).resolve().parent

DATASETS = [
    ("LH", "Left Hand Task", ROOT_DIR / "data_LH"),
    ("RH", "Right Hand Task", ROOT_DIR / "data_RH"),
    ("LF", "Left Foot Task", ROOT_DIR / "data_LF"),
    ("RF", "Right Foot Task", ROOT_DIR / "data_RF"),
]

FIG_PATH = ROOT_DIR / "figures" / f"fig3_network_comparison_{ROI}ROI_all.pdf"

NETWORK_NAMES = ["Structural", "EDR", "Geometric"]

mpl.rcParams.update(
    {
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
    }
)

plt.rc("mathtext", fontset="dejavusans")

COLORS = {
    "Structural": "#ff4646",
    "EDR": "#2fff90",
    "Geometric": "#7066ff",
}


def load_fit_summary(result_csv: Path) -> dict:
    if not result_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {result_csv}")

    bands = []
    subjects = set()

    r2_values = {model: {} for model in NETWORK_NAMES}
    tau_values = {model: {} for model in NETWORK_NAMES}

    with open(result_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
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
        raise RuntimeError(f"No subject data found in CSV: {result_csv}")

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

    return {
        "bands": bands,
        "n_subjects": n_subjects,
        "r2_mean": r2_mean,
        "r2_std": r2_std,
        "tau_mean": tau_mean,
        "tau_std": tau_std,
        "result_csv": result_csv,
    }


# ============================================================
# Load all datasets
# ============================================================

summaries = {}
bands_ref = None

for code, _label, data_dir in DATASETS:
    result_csv = data_dir / f"network_fit_ROI{ROI}_{LOCK}.csv"
    summary = load_fit_summary(result_csv)

    if bands_ref is None:
        bands_ref = summary["bands"]
    elif summary["bands"] != bands_ref:
        raise RuntimeError(
            f"Band list mismatch for {code}: {summary['bands']} (expected {bands_ref})"
        )

    summaries[code] = summary

# ============================================================
# Plot R^2 comparison for all datasets
# ============================================================

x = np.arange(len(bands_ref))
width = 0.25
error_style = dict(elinewidth=1, capsize=1, capthick=1)
fast_delta_label = "Fast\u00A0\u00A0\n\u00A0\u00A0-Delta"
band_labels = [fast_delta_label if band == "Fast Delta" else band for band in bands_ref]

fig, axs = plt.subplots(
    2,
    2,
    figsize=(6.5, 3.5),
    dpi=300,
    sharex=True,
    sharey=True,
)

for ax, (code, label, _data_dir) in zip(axs.ravel(), DATASETS):
    r2_mean = summaries[code]["r2_mean"]
    r2_std = summaries[code]["r2_std"]

    ax.bar(
        x - width,
        r2_mean["Structural"],
        width,
        yerr=r2_std["Structural"],
        color=COLORS["Structural"],
        edgecolor="black",
        linewidth=0.8,
        alpha=1,
        error_kw=error_style,
        zorder=2,
    )
    ax.bar(
        x,
        r2_mean["EDR"],
        width,
        yerr=r2_std["EDR"],
        color=COLORS["EDR"],
        edgecolor="black",
        linewidth=0.8,
        alpha=1,
        error_kw=error_style,
        zorder=2,
    )
    ax.bar(
        x + width,
        r2_mean["Geometric"],
        width,
        yerr=r2_std["Geometric"],
        color=COLORS["Geometric"],
        edgecolor="black",
        linewidth=0.8,
        alpha=1,
        error_kw=error_style,
        zorder=2,
    )

    ax.set_title(label)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)

# Only show x tick labels on bottom row
for ax in axs[0, :]:
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

# Only show y tick labels on left column
for ax in axs[:, 0]:
    ax.yaxis.tick_left()
    ax.yaxis.set_label_position("left")
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, labelright=False)

for ax in axs[:, 1]:
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=False, labelright=False)

for panel_label, ax in [
    ("a", axs[0, 0]),
    ("b", axs[0, 1]),
    ("c", axs[1, 0]),
    ("d", axs[1, 1]),
]:
    ax.text(
        0.85,
        0.85,
        panel_label,
        fontsize=12,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color="black",
        fontweight="bold",
        fontname="DejaVu Sans",
    )

legend_handles = [
    Patch(facecolor=COLORS[name], label=name, alpha=1, edgecolor="black", linewidth=0.8)
    for name in NETWORK_NAMES
]
fig.legend(
    handles=legend_handles,
    frameon=False,
    loc="upper center",
    ncols=len(NETWORK_NAMES),
    bbox_to_anchor=(0.5, 0.985),
)

fig.text(
    0.023,
    0.45,
    "Coefficient of determination $R^2$",
    rotation=90,
    va="center",
    ha="left",
)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=(0.06, 0.02, 0.98, 0.9))
fig.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")

# ============================================================
# Print summary tables
# ============================================================

for code, label, _data_dir in DATASETS:
    summary = summaries[code]
    n_subjects = summary["n_subjects"]
    r2_mean = summary["r2_mean"]
    r2_std = summary["r2_std"]
    tau_mean = summary["tau_mean"]
    tau_std = summary["tau_std"]

    print("\n" + "=" * 100)
    print(f"SUMMARY: {label} Network Comparison ({ROI} ROIs, N={n_subjects} subjects)")
    print("=" * 100)
    print(f"{'Band':<12} {'R² Struc':<18} {'R² EDR':<18} {'R² Geo':<18} {'Best':<12}")
    print("-" * 100)

    for i, band in enumerate(bands_ref):
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
    print(f"TAU VALUES: {label} Network Comparison ({ROI} ROIs, N={n_subjects} subjects)")
    print("=" * 100)
    print(f"{'Band':<12} {'τ Struc':<18} {'τ EDR':<18} {'τ Geo':<18}")
    print("-" * 100)

    for i, band in enumerate(bands_ref):
        print(
            f"{band:<12} "
            f"{tau_mean['Structural'][i]:.4f} ± {tau_std['Structural'][i]:.4f}   "
            f"{tau_mean['EDR'][i]:.4f} ± {tau_std['EDR'][i]:.4f}   "
            f"{tau_mean['Geometric'][i]:.4f} ± {tau_std['Geometric'][i]:.4f}"
        )

    print("=" * 100)
