"""LH mean response magnitude and evoked responses across magnitude deciles."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import data
from plots.helpers import cache_path, read_cache, surface
from plots.style import panel_label, apply_style, save_figure

apply_style()

paths = [cache_path(subject) for subject in data.subjects("LH", 100)]
missing = [path for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Figure 1 needs {len(missing)} missing LH parcel caches; first: {missing[0]}")

magnitudes, responses = [], []
for path in paths:
    signal, time, fs, labels, _ = read_cache(path)
    if responses and (not np.array_equal(labels, reference_labels)
                      or not np.array_equal(time, reference_time)):
        raise ValueError(f"Unaligned LH parcel cache: {path}")
    response = (time >= 0) & (time <= 0.5)
    if not response.any():
        raise ValueError(f"Missing response window: {path}")
    magnitudes.append(np.abs(signal[:, response]).mean(axis=1))
    responses.append(signal)
    reference_labels, reference_time = labels, time

magnitude = np.mean(magnitudes, axis=0)
evoked = np.mean(responses, axis=0)
# Common normalization for all parcels; avoids implying calibrated dipole units.
scale = np.max(np.abs(evoked))
if scale == 0:
    raise ValueError("All group-averaged evoked responses are zero")
# Sample one parcel from the middle of each magnitude decile (high to low).
ranked = np.argsort(magnitude)[::-1]
selected = np.array([bin_[len(bin_) // 2] for bin_ in np.array_split(ranked, 10)])
values = np.full(100, np.nan)
values[labels - 1] = magnitude

fig = plt.figure(figsize=(5.5, 2.15))
grid = fig.add_gridspec(1, 2, width_ratios=[0.72, 1], wspace=0.13)
fig.subplots_adjust(left=0.005, right=0.995, bottom=0.17, top=0.98)
ax_map = fig.add_subplot(grid[0, 0], projection="3d")
ax_time = fig.add_subplot(grid[0, 1])
# Share the magnitude colormap; the surface additionally has sulcal shading.
cmap = plt.get_cmap("Reds")
norm = plt.Normalize(magnitude.min(), np.percentile(magnitude, 95), clip=True)
surface(ax_map, values, 100, cmap=cmap, norm=norm,
        zoom=1.55, sulcal=True, darkness=0.3)
colors = cmap(norm(magnitude[selected]))

for rank, (i, color) in enumerate(zip(selected, colors)):
    ax_time.plot(time, evoked[i] / scale, linewidth=1, color=color,
                 zorder=len(selected) - rank)
ax_time.axvline(0, color="k", linestyle="--", linewidth=0.8)
ax_time.set(xlim=(-1.2, 1.2), xlabel="Time (s)", ylabel="Normalized amplitude")
ax_time.set_xticks([-1, 0, 1])
ax_time.margins(y=0.05)
ax_time.yaxis.set_major_locator(MaxNLocator(nbins=3))
ax_time.tick_params(length=3, pad=2)
ax_time.xaxis.labelpad = 2
ax_time.yaxis.labelpad = 2
panel_label(ax_map, "a")
panel_label(ax_time, "b")
output = data.ROOT / "figures/main"
output.mkdir(parents=True, exist_ok=True)
save_figure(fig, output / "fig1.pdf", 5.4)
plt.close(fig)
