"""Network architecture and first four retained modes (Figure 5)."""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

import data
from model import graph_modes
from plots.helpers import cortex, surface
from plots.style import MODEL_COLORS, MODEL_LABELS, apply_style, save_figure

apply_style()

roi = 100
coords, faces, labels = cortex(roi)
centroids = np.array([coords[labels == i].mean(axis=0) for i in range(1, roi + 1)])
graphs = data.operators(roi, np.arange(1, roi + 1))
eigenmodes = {name: graph_modes(laplacian)[1][:, :4]
              for name, laplacian in graphs.items()}
limit = max(np.max(np.abs(modes)) for modes in eigenmodes.values())
mode_norm = plt.Normalize(-limit, limit)
fig = plt.figure(figsize=(5.6, 3.95))
grid = fig.add_gridspec(3, 5, width_ratios=[1.1, 1, 1, 1, 1],
                       wspace=-0.32, hspace=-0.09,
                       left=0.005, right=0.995, top=0.93, bottom=0.025)
for row, (name, laplacian) in enumerate(graphs.items()):
    ax = fig.add_subplot(grid[row, 0], projection="3d")
    weights = -laplacian.copy()
    np.fill_diagonal(weights, 0)
    a, b = np.triu_indices(roi, 1)
    positive = weights[a, b][weights[a, b] > 0]
    binary = np.allclose(positive, positive[0])
    threshold = 0 if binary else np.percentile(positive, [80, 85, 0][row])
    keep = weights[a, b] > (threshold if row < 2 else 0)
    ax.plot_trisurf(*coords.T, triangles=faces, color="lightgray",
                    alpha=0.1, edgecolor="none", linewidth=0, shade=False,
                    rasterized=True)
    segments = centroids[np.column_stack((a[keep], b[keep]))]
    edge_weights = weights[a[keep], b[keep]]
    relative = ((edge_weights - edge_weights.min()) / np.ptp(edge_weights)
                if np.ptp(edge_weights) > 0 else np.ones_like(edge_weights))
    colors = np.tile(to_rgba(MODEL_COLORS[name]), (len(relative), 1))
    colors[:, 3] = 0.3 + 0.7 * relative
    widths = (0.3 + 1.2 * relative) * [2, 2, 1][row]
    ax.add_collection3d(Line3DCollection(segments, colors=colors, linewidths=widths))
    radius = np.ptp(coords, axis=0).max() / 2 * 0.75
    for set_limit, midpoint in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim),
                                   (coords.max(axis=0) + coords.min(axis=0)) / 2):
        set_limit(midpoint - radius, midpoint + radius)
    ax.view_init(elev=90, azim=90)
    ax.set_axis_off()
    fig.text(0.05, 0.93 - (row + 0.5) * 0.905 / 3, MODEL_LABELS[name],
             rotation=90, ha="left", va="center", fontsize=9)
    for k in range(1, 5):
        values = eigenmodes[name][:, k - 1]
        # Eigenvector signs are arbitrary; choose a deterministic orientation.
        values = values * np.sign(values[np.argmax(np.abs(values))])
        ax = fig.add_subplot(grid[row, k], projection="3d")
        surface(ax, values, roi, cmap="seismic", norm=mode_norm, zoom=1.3, sulcal=True)
# These axes overlap deliberately; opaque patches hide neighboring brains.
for ax in fig.axes:
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
    ax.grid(False)
for k, label in enumerate(["Network", "Mode 1", "Mode 2", "Mode 3", "Mode 4"]):
    position = grid[0, k].get_position(fig)
    fig.text((position.x0 + position.x1) / 2, 0.93, label,
             ha="center", va="bottom", fontsize=9)
final = data.ROOT / 'figures/main/fig5.pdf'
final.parent.mkdir(parents=True, exist_ok=True)
save_figure(fig, final, 5.4)
print(f"Saved {final}")
plt.close(fig)
