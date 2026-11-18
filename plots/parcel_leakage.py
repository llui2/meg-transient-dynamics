"""Parcel leakage versus EDR inverse-weight shortest-path distance (Figure S2)."""
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np
from scipy.optimize import curve_fit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

import data
from plots.helpers import cache_path, read_cache, surface
from plots.style import LEAKAGE_COLORS, panel_label, apply_style, save_figure

apply_style()


def decay(distance, amplitude, length, offset):
    return amplitude * np.exp(-distance / length) + offset


path = cache_path("104012")
if not path.is_file():
    raise FileNotFoundError(f"Figure S2 needs the parcel resolution cache: {path}")
_, _, _, labels, resolution = read_cache(path)
weights = np.load(data.ASSETS / "W_EDR_Sch100.npy")
weights = np.maximum(weights, 0)
weights = (weights + weights.T) / 2
weights = weights[np.ix_(labels - 1, labels - 1)]
# Use inverse edge weights as lengths, with negligible weight for missing edges.
distance = shortest_path(csr_matrix(1 / np.where(weights == 0, 1e-10, weights)),
                         directed=False, method="D")
off = ~np.eye(len(labels), dtype=bool)
x, y = distance[off], np.abs(resolution[off])
edges = np.linspace(x.min(), x.max(), 21)
centers, means, deviations = [], [], []
for low, high in zip(edges[:-1], edges[1:]):
    values = y[(x >= low) & (x < high)]
    if len(values) > 10:
        centers.append((low + high) / 2)
        means.append(values.mean())
        deviations.append(values.std())

fig = plt.figure(figsize=(5.3, 2.25))
layout = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.98,
                    wspace=0.08)
ax = fig.add_subplot(layout[0, 0])
means, deviations = np.array(means), np.array(deviations)
ax.plot(centers, means, color=LEAKAGE_COLORS["data"],
        linewidth=1.5, label="Mean")
ax.fill_between(centers, means - deviations, means + deviations,
                color=LEAKAGE_COLORS["data"], alpha=0.2,
                linewidth=0, label="±1 SD")
parameters, _ = curve_fit(decay, centers, means,
                          p0=[0.5, np.mean(centers), 0.05], maxfev=5000)
grid = np.linspace(min(centers), max(centers), 200)
ax.plot(grid, decay(grid, *parameters), color=LEAKAGE_COLORS["fit"],
        linewidth=1.5, linestyle="--", label="Fit")
for quantile, color in [(10, LEAKAGE_COLORS["guide_1"]),
                        (30, LEAKAGE_COLORS["guide_2"])]:
    ax.axvline(np.percentile(x, quantile), color=color, linewidth=1.2,
               label=rf"$q_{{{quantile/100:.2f}}}$")
ax.set(xlabel=r"Graph distance", ylabel=r"$|R^{\mathrm{ROI}}_{ab}|$",
       xlim=(min(centers) * 0.9, max(centers) * 1.05), ylim=(0, means.max() * 1.1))
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "a")
ax.legend(fontsize=8, frameon=False)
ax = fig.add_subplot(layout[0, 1], projection="3d")
reference = np.flatnonzero(labels == 47).item()  # Schaefer parcel 47.
values = np.full(100, np.nan)
values[labels - 1] = np.abs(resolution[reference])
# Mark the upper end of the cortical leakage scale in red.
magma = plt.cm.magma(np.linspace(0, 1, 256))
magma[-20:] = to_rgba(LEAKAGE_COLORS["reference"])
mapped = surface(ax, values, 100, cmap=ListedColormap(magma),
                 norm=plt.Normalize(0, values.max()), zoom=1.3)
panel_label(ax, "b")
bar = fig.colorbar(mapped, ax=ax, shrink=0.7, pad=0.02)
bar.set_label(r"$|R^{\mathrm{ROI}}_{47,b}|$", fontsize=9)
bar.ax.tick_params(labelsize=8)
output = data.ROOT / "figures/supp"
output.mkdir(parents=True, exist_ok=True)
save_figure(fig, output / "figS2.pdf", 5.4)
plt.close(fig)
