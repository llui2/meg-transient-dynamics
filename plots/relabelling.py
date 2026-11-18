"""Plot the saved parcel-relabelling comparison (S10)."""
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import data
from plots.style import MODEL_COLORS, panel_label, apply_style, participant_boxplot, box_position, boxplot_axis, save_figure

FIG = data.ROOT / "figures/supp/figS10"
MODELS = tuple(data.NETWORKS)
BOOTSTRAP_SEED = 20261102


def plot():
    table = pd.read_csv(data.RESULTS / "relabelling_comparison.csv", dtype={"subject": str})
    result = {"Relabelled": json.loads((data.RESULTS / "relabelling_summary.json").read_text())}
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(4.5, 2.1))
    fig.subplots_adjust(left=.10, right=.995, bottom=.16, top=.90, wspace=.42)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for i, model in enumerate(MODELS):
        color = MODEL_COLORS[model]
        group = table[(table.model == model) & (table.control == "Relabelled")]
        randomized = group.pivot(index="subject", columns="network", values="r2")
        original = table[(table.model == model) & (table.control == "Original")].set_index("subject").r2
        original = original.reindex(randomized.index)
        relabelled = randomized.mean(axis=1)
        changes = relabelled - original
        assert len(original) == len(relabelled) == 51
        summary = result["Relabelled"]["models"][model]
        for ax, position, values, metric, hatch in (
                (axes[0], box_position(i, 0, 2, slots=4), original, "original_r2", None),
                (axes[0], box_position(i, 1, 2, slots=4), relabelled, "scrambled_r2", "////"),
                (axes[1], box_position(i, 0, 1, slots=4), changes, "fit_change", "////")):
            assert np.isclose(values.mean(), summary[metric]["mean"], rtol=0, atol=1e-12)
            participant_boxplot(ax, values.to_numpy(), position, color, rng,
                                hatch=hatch)
    axes[0].set(ylabel=r"$R^2$")
    axes[1].axhline(0, color="0.5", alpha=0.65, linewidth=0.8, linestyle="--", zorder=0)
    axes[1].set(ylabel=r"$\Delta R^2$")
    for ax, letter in zip(axes, "ab"):
        boxplot_axis(ax, ["Structural", "EDR", "Local"], 2 if letter == "a" else 1, slots=4)
        ax.yaxis.labelpad = 3
        panel_label(ax, letter)
    handles = [Patch(facecolor="white", edgecolor=".25", label="Original"),
               Patch(facecolor="white", edgecolor=".25", hatch="////", label="Relabelled")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.015),
               ncol=2, frameon=False, fontsize=8, handlelength=2, handleheight=1.2,
               columnspacing=1.2, borderaxespad=0, borderpad=.2)
    FIG.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, FIG.with_suffix(".pdf"), 4.5)
    plt.close(fig)
    print(f"Saved {FIG.with_suffix('.pdf')}")


if __name__ == "__main__":
    plot()
