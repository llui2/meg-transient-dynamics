"""Plot spatial coverage versus dynamical fit from saved S9 contrasts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from fit import BANDS
from plots.style import MODEL_COLORS, MODEL_LABELS, panel_label, apply_style, save_figure

MODELS = tuple(data.NETWORKS)
K = np.arange(1, 100)


def plot():
    subjects = list(data.SUBJECTS)
    table = pd.read_csv(data.RESULTS / "spatial_decomposition.csv", dtype={"subject": str})
    rng = np.random.default_rng(20261030)
    draws = rng.integers(len(subjects), size=(2500, len(subjects)))
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.11, right=0.995, bottom=0.20, top=0.88, wspace=0.12)
    for bi, (ax, band) in enumerate(zip(axes, BANDS)):
        for mi, control in enumerate(MODELS[1:], start=1):
            group = table[(table.band == band) & (table.control == control)]
            dynamic = group.pivot(index="subject", columns="modes", values="dynamic_advantage").reindex(index=subjects, columns=K).to_numpy()
            spatial = group.pivot(index="subject", columns="modes", values="spatial_advantage").reindex(index=subjects, columns=K).to_numpy()
            excess = dynamic - spatial
            mean = excess.mean(axis=0)
            low, high = np.quantile(excess[draws].mean(axis=1), [0.025, 0.975], axis=0)
            ax.plot(K, mean, linewidth=1.5, color=MODEL_COLORS[control],
                    label="Structural − " + MODEL_LABELS[control])
            ax.fill_between(K, low, high, color=MODEL_COLORS[control], alpha=0.12)
            for k in (20, 99):
                j = k - 1
                print(f"{band}, Structural − {control}, K={k}: "
                      f"observed ΔR²={dynamic[:, j].mean():+.4f}, "
                      f"spatial ΔR²={spatial[:, j].mean():+.4f}, "
                      f"excess={mean[j]:+.4f} [{low[j]:+.4f}, {high[j]:+.4f}]")
        ax.axhline(0, color="0.5", alpha=0.65, ls="--", lw=0.8, zorder=0)
        ax.set_xlim(1, 99)
        ax.set_xticks([1, 20, 40, 60, 80, 99])
        ax.set_yticks([0, 0.1, 0.2])
        ax.set_xlabel(r"Modes $K$", labelpad=3)
        ax.spines[["top", "right"]].set_visible(False)
        panel_label(ax, chr(97 + bi))
        ax.text(0.98, 0.88, "2–4 Hz" if bi == 0 else "4–8 Hz",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
    axes[0].set_ylabel(r"$D(K)$", labelpad=3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8, borderaxespad=0, borderpad=.15)

    target = data.ROOT / "figures/supp/figS9.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, target, 5.4)
    print(f"Saved {target}")
    plt.close(fig)


if __name__ == "__main__":
    plot()
