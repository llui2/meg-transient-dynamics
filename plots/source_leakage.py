"""Source-resolution leakage for the illustrative 104012 left-hand response."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import data
from plots.style import LEAKAGE_COLORS, panel_label, apply_style, save_figure

apply_style()

paths = [data.RESULTS / name for name in
         ("source_leakage.csv", "source_reference.csv", "source_leakage_input.csv")]

missing = [path for path in paths if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Figure S1 needs {missing[0]}; supply the saved summaries or run ./run.sh source-leakage.")
bins, sources, provenance = (pd.read_csv(path) for path in paths)
reference = int(provenance.reference_index.iloc[0])
if (len(sources) != provenance.n_sources.iloc[0]
        or not np.array_equal(sources.source_index, np.arange(len(sources)))
        or not bins['count'].gt(100).all()
        or not np.isfinite(bins).all().all() or not np.isfinite(sources).all().all()):
    raise ValueError("Invalid source-leakage plot inputs")


def decay(distance, amplitude, length, offset):
    return amplitude * np.exp(-distance / length) + offset


parameters, _ = curve_fit(decay, bins.distance_mm, bins.mean_abs_R,
                          p0=[0.8, 30, 0.1], maxfev=5000)
fig, (ax, spatial) = plt.subplots(1, 2, figsize=(5.3, 2.25),
                                gridspec_kw={"width_ratios": [1.2, 1]})
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.18, top=0.98,
                    wspace=0.08)
ax.plot(bins.distance_mm, bins.mean_abs_R, color=LEAKAGE_COLORS["data"],
        linewidth=1.5, label="Mean")
ax.fill_between(bins.distance_mm, bins.mean_abs_R - bins.std_abs_R,
                bins.mean_abs_R + bins.std_abs_R, color=LEAKAGE_COLORS["data"],
                alpha=0.2, linewidth=0, label="±1 SD")
grid = np.linspace(0, 150, 200)
ax.plot(grid, decay(grid, *parameters), color=LEAKAGE_COLORS["fit"],
        linewidth=1.5, linestyle="--", label="Fit")
mapped = spatial.scatter(sources.y_mm, sources.x_mm, c=sources.abs_R_reference,
                          cmap="magma", s=2, vmin=0, vmax=0.8, rasterized=True)
point = sources.iloc[reference]
spatial.scatter(point.y_mm, point.x_mm, color=LEAKAGE_COLORS["reference"], s=40, edgecolor="black", zorder=100)
theta = np.linspace(0, 2 * np.pi, 100)
# Match the orange/violet distance guides in the archived source plot.
for radius, color in [(20, LEAKAGE_COLORS["guide_1"]),
                      (50, LEAKAGE_COLORS["guide_2"])]:
    ax.axvline(radius, color=color, linewidth=1.2, label=f"{radius} mm")
    spatial.plot(point.y_mm + radius * np.cos(theta), point.x_mm + radius * np.sin(theta),
                 color=color, linewidth=1.5)
ax.set(xlabel=r"$d_{ij}$ (mm)", ylabel=r"$|R_{ij}|$", xlim=(0, 150),
       ylim=(0, bins.mean_abs_R.iloc[0] * 1.08))
ax.spines[["top", "right"]].set_visible(False)
panel_label(ax, "a")
ax.legend(fontsize=8, frameon=False)
spatial.set_aspect("equal")
spatial.set_axis_off()
panel_label(spatial, "b")
bar = fig.colorbar(mapped, ax=spatial, shrink=0.7, pad=0.02)
bar.set_label(r"$|R_{j,i_0}|$", fontsize=9)
bar.ax.tick_params(labelsize=8)
output = data.ROOT / "figures/supp"
output.mkdir(parents=True, exist_ok=True)
save_figure(fig, output / "figS1.pdf", 5.4)
plt.close(fig)
print(f"Source leakage: {len(sources)} sources; exponential length β = {parameters[1]:.2f} mm")
