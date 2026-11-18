"""Participant-averaged modal fits and paired differences (Figures S6-S8)."""
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plots.style import (MODEL_COLORS, MODEL_LABELS, panel_label, apply_style,
                         participant_boxplot, box_position, boxplot_axis,
                         compact_boxplot_canvas, save_figure)
from plots.modal_results import BANDS, LABELS, participant_scores

apply_style()

root = Path(__file__).resolve().parents[1]
roi = int(os.getenv("ROI", "100"))
if roi not in (100, 200, 400):
    raise ValueError("Modal parcellation plots require ROI=100, 200, or 400")
scores = participant_scores(roi)
models = ("Structural", "EDR", "Geometric")
rng = np.random.default_rng(0)

fig, axes = plt.subplots(2, 1, figsize=(5.6, 3.7), sharex=True)
fig.subplots_adjust(left=0.10, right=0.995, bottom=0.13, top=0.91,
                    hspace=0.34)

for j, model in enumerate(models):
    for i, band in enumerate(BANDS):
        values = scores.xs(band, level="band")[model].to_numpy()
        participant_boxplot(axes[0], values, box_position(i, j, 3, slots=3),
                            MODEL_COLORS[model], rng)
    axes[0].plot([], [], color=MODEL_COLORS[model], linewidth=2,
                 label=MODEL_LABELS[model])
axes[0].set_ylabel(r"$R^2_{\mathrm{modal}}$", labelpad=2)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
               ncol=3, borderaxespad=0)

for j, control in enumerate(models[1:]):
    differences = scores["Structural"] - scores[control]
    for i, band in enumerate(BANDS):
        values = differences.xs(band, level="band").to_numpy()
        participant_boxplot(axes[1], values, box_position(i, j, 2, slots=3),
                            MODEL_COLORS[control], rng)
    axes[1].plot([], [], color=MODEL_COLORS[control], linewidth=2,
                 label="Structural − " + MODEL_LABELS[control])
axes[1].axhline(0, linestyle="--", color="0.5", alpha=0.65,
                linewidth=0.8, zorder=0)
axes[1].set_ylabel(r"$\Delta R^2_{\mathrm{modal}}$", labelpad=2)
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.015),
               ncol=2, borderaxespad=0)

for panel, ax in enumerate(axes):
    boxplot_axis(ax, LABELS, 3 if panel == 0 else 2, slots=3)
    panel_label(ax, chr(97 + panel))
axes[1].set_xlabel("Frequency (Hz)", labelpad=2)
compact_boxplot_canvas(fig)

number = {100: 'S6', 200: 'S7', 400: 'S8'}[roi]
final = root / "figures/supp" / f"fig{number}.pdf"
final.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, final, None)
print(f"ROI {roi}: {len(scores.index.unique('subject'))} participants; saved {final}")
plt.close(fig)
