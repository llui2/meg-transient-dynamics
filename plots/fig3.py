"""Figures 3, S3, and S4: regional fit quality and paired structural advantage.

Only the nonoverlapping sequence starting at 0.5–2 Hz is plotted. The figure
is saved only after all 51 participants have complete results for every
plotted band.
"""
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as settings
from plots.style import MODEL_COLORS, MODEL_LABELS, panel_label, apply_style, participant_boxplot, box_position, boxplot_axis, save_figure
from fit import BINS, TAU_MIN, TAU_MAX

apply_style()

order = 4
roi = int(os.getenv('ROI', '100'))
if roi not in settings.ROIS:
    raise ValueError('Expected ROI=100, 200, or 400')

root = Path(__file__).resolve().parents[1]
filename = "spectrum.csv" if roi == 100 else f"spectrum_{roi}.csv"
path = settings.RESULTS / filename
if not path.is_file():
    raise FileNotFoundError(f"{path}: run ./run.sh fit spectrum --roi {roi} first")
table = pd.read_csv(path, dtype={"subject": str})
if "roi" in table and not table.roi.eq(roi).all():
    raise ValueError(f"Wrong parcellation in {path}")
if roi != 100 and "roi" not in table:
    raise ValueError(f"Missing parcellation metadata in {path}")
table = table[table.order.eq(order)]
table = table[~(table.fmin.eq(0.5) & table.fmax.eq(4))]
if table.empty:
    raise ValueError(f"No spectrum fits for filter order {order}")
if not {"tau_min", "tau_max"}.issubset(table.columns):
    raise ValueError("Old spectrum bounds: regenerate with ./run.sh fit spectrum")
if (not np.isclose(table.tau_min, TAU_MIN).all()
        or not np.isclose(table.tau_max, TAU_MAX).all()
        or not set(zip(table.fmin, table.fmax)).issubset(BINS)):
    raise ValueError("Incompatible spectrum data: regenerate with ./run.sh fit spectrum")
keys = ["subject", "condition", "order", "fmin", "fmax", "model"]
if table.duplicated(keys).any():
    raise ValueError("Duplicate frequency-resolved fit keys")
if not np.isfinite(table.r2).all() or (table.r2 > 1 + 1e-7).any():
    raise ValueError("Invalid parcel-space R²")
if not set(table.subject).issubset(settings.SUBJECTS):
    raise ValueError("Unexpected subjects in frequency-resolved fits")

models = ("Structural", "EDR", "Geometric")
conditions = ("LH", "RH", "LF", "RF")
bands = [band for band in BINS if band != (0.5, 4)]
subjects, incomplete = [], []
for subject in settings.SUBJECTS:
    part = table[table.subject.eq(subject)]
    if part.empty:
        continue
    expected = {(subject, condition, order, float(lo), float(hi), model)
                for condition in conditions if subject in settings.subjects(condition, roi)
                for lo, hi in bands for model in models}
    actual = set(part[keys].itertuples(index=False, name=None))
    if expected <= actual:
        subjects.append(subject)
    else:
        incomplete.append(subject)
if not subjects:
    raise ValueError("No complete participants: run ./run.sh fit spectrum to fit "
                     "all nonoverlapping bands from 0.5–2 Hz across all 51 participants")
if incomplete:
    print(f"Ignoring {len(incomplete)} participants with unfinished frequency fits")

# Average conditions within participant; each point is one participant.
complete = table[table.subject.isin(subjects)]
scores = complete.pivot(index=["subject", "condition", "fmin", "fmax"],
                        columns="model", values="r2")
scores = scores.groupby(level=["subject", "fmin", "fmax"]).mean()
values = {
    model: np.array([[scores.loc[(subject, float(lo), float(hi)), model]
                      for subject in subjects] for lo, hi in bands])
    for model in models
}
rng = np.random.default_rng(20261029)


def grouped_boxes(ax, samples, position, count, color, label):
    """Use the same participant distributions as the modal comparison plots."""
    x = [box_position(i, position, count, slots=3) for i in range(len(bands))]
    for band_position, participants in zip(x, samples):
        participant_boxplot(ax, participants, band_position, color, rng)
    ax.plot([], [], color=color, linewidth=2, label=label)


# The paired structural advantage in panel (b) is the primary comparison.
fig, axes = plt.subplots(2, 1, figsize=(5.6, 3.7), sharex=True)
fig.subplots_adjust(left=0.10, right=0.995, bottom=0.13, top=0.91,
                    hspace=0.34)

for j, model in enumerate(models):
    grouped_boxes(axes[0], values[model], j, len(models), MODEL_COLORS[model], MODEL_LABELS[model])
axes[0].set_ylabel("$R^2$")
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
               ncol=3, borderaxespad=0)

for j, control in enumerate(models[1:]):
    grouped_boxes(axes[1], values["Structural"] - values[control], j, 2,
                 MODEL_COLORS[control], "Structural − " + MODEL_LABELS[control])
axes[1].axhline(0, color="0.5", linewidth=0.8, linestyle="--",
                alpha=0.65, zorder=0)
axes[1].set_ylabel(r"$\Delta R^2$")
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
               ncol=2, borderaxespad=0)

# Categorical positions represent separately fitted bands, not a continuous
# frequency axis.
labels = [f"{lo:g}–{hi:g}" for lo, hi in bands]
for j, ax in enumerate(axes):
    boxplot_axis(ax, labels, 3 if j == 0 else 2, slots=3)
    ax.yaxis.labelpad = 2
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, chr(97 + j))
axes[1].set_xlabel("Frequency (Hz)", labelpad=2)

if len(subjects) != len(settings.SUBJECTS):
    raise ValueError("Figure 3 requires complete results for all 51 participants")
target = (root / "figures/main/fig3.pdf" if roi == 100 else
          root / "figures/supp" / f"figS{3 if roi == 200 else 4}.pdf")
target.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, target, 5.6)
print(f"ROI {roi}: {len(subjects)} complete participants; saved {target}")
plt.close(fig)
