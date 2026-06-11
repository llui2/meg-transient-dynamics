"""
=====================================================================
EFFECT SIZE PLOT (ALL DATASETS)
=====================================================================
Plot mean paired ΔR² (structural advantage) as a function of
frequency band across subjects, for LH/RH/LF/RF datasets.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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

FIG_PATH = ROOT_DIR / "figures" / f"fig4_effect_size_{ROI}ROI_all.pdf"

mpl.rcParams.update(
    {
        "font.size": 9,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "cm",
    }
)
plt.rc("mathtext", fontset="dejavusans")

COLOR_EDR = "#00c40a"
COLOR_GEO = "#7066ff"


def load_r2(result_csv: Path) -> dict:
    if not result_csv.exists():
        raise FileNotFoundError(f"Results CSV not found: {result_csv}")

    bands = []
    subjects = []
    r2 = {}

    with open(result_csv, newline="") as f:
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

    return {"bands": bands, "subjects": subjects, "r2": r2}


def paired_diffs_per_band(r2_pack: dict) -> tuple[list[list[float]], list[list[float]]]:
    bands = r2_pack["bands"]
    subjects = r2_pack["subjects"]
    r2 = r2_pack["r2"]

    d_edr_per_band = []
    d_geo_per_band = []

    for band in bands:
        d_edr = []
        d_geo = []
        for subj in subjects:
            models = r2[band].get(subj, {})
            if not {"Structural", "EDR", "Geometric"}.issubset(models):
                continue
            d_edr.append(models["Structural"] - models["EDR"])
            d_geo.append(models["Structural"] - models["Geometric"])

        d_edr_per_band.append(d_edr)
        d_geo_per_band.append(d_geo)

    return d_edr_per_band, d_geo_per_band


# ============================================================
# Load data
# ============================================================

packs = {}
bands_ref = None

for code, _label, data_dir in DATASETS:
    result_csv = data_dir / f"network_fit_ROI{ROI}_{LOCK}.csv"
    pack = load_r2(result_csv)
    if bands_ref is None:
        bands_ref = pack["bands"]
    elif pack["bands"] != bands_ref:
        raise RuntimeError(f"Band list mismatch for {code}: {pack['bands']} (expected {bands_ref})")
    packs[code] = pack

# ============================================================
# Plot
# ============================================================

x = np.arange(len(bands_ref))
offset = 0.35
fast_delta_label = "Fast\u00A0\u00A0\n\u00A0\u00A0-Delta"
band_labels = [fast_delta_label if band == "Fast Delta" else band for band in bands_ref]

fig, axs = plt.subplots(2, 2, figsize=(6.5, 3.5), dpi=300, sharex=True, sharey=True)

line_width = 1.2

for ax, (code, label, _data_dir) in zip(axs.ravel(), DATASETS):
    d_edr_per_band, d_geo_per_band = paired_diffs_per_band(packs[code])

    data = d_edr_per_band + d_geo_per_band
    positions = list(x) + list(x + offset)
    colors = [COLOR_EDR] * len(bands_ref) + [COLOR_GEO] * len(bands_ref)

    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.3,
        patch_artist=True,
        showfliers=False,
        showcaps=False,
    )

    for i, (patch, color) in enumerate(zip(box["boxes"], colors)):
        patch.set_facecolor("none")
        patch.set_edgecolor(color)
        patch.set_linewidth(line_width)
        patch.set_alpha(1.0)
        patch.set_zorder(3)

        box["medians"][i].set_color(color)
        box["medians"][i].set_linewidth(line_width)
        box["medians"][i].set_zorder(4)

        w0, w1 = box["whiskers"][2 * i], box["whiskers"][2 * i + 1]
        for w in (w0, w1):
            w.set_color(color)
            w.set_linewidth(line_width)
            w.set_zorder(4)

    for cap in box.get("caps", []):
        cap.set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.axhline(0, color="k", linestyle="-", linewidth=1, alpha=0.2)

    ax.set_xticks(x + offset / 2)
    ax.set_xticklabels(band_labels)
    ax.set_title(label)

    ax.set_ylim(-0.1, 0.3)
    ax.set_yticks(np.arange(-0.1, 0.31, 0.1))

# Only show x tick labels on bottom row (keep tick marks)
for ax in axs[0, :]:
    ax.tick_params(axis="x", which="both", bottom=True, labelbottom=False)

# Only show y tick labels on left column (keep tick marks everywhere on the left)
for ax in axs[:, 0]:
    ax.yaxis.tick_left()
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

handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=COLOR_EDR, linewidth=line_width),
    plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=COLOR_GEO, linewidth=line_width),
]
fig.legend(
    handles,
    ["Structural − EDR", "Structural − Geometric"],
    frameon=False,
    loc="upper center",
    ncols=2,
    bbox_to_anchor=(0.5, 0.985),
)

fig.text(0.024, 0.5, r"Paired $\Delta R^2$", rotation=90, va="center", ha="left")

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(rect=(0.06, 0.02, 0.98, 0.9))
fig.savefig(FIG_PATH)
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")
