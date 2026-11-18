"""Small read-only helpers for cached parcels and Matplotlib cortical maps."""
from functools import lru_cache

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

import data
from data import cortex


def cache_path(subject, condition="LH"):
    return data.SIGNALS / f"{subject}_{condition}.npz"


def read_cache(path):
    with np.load(path) as pack:
        signal, time, fs, labels, resolution = (
            pack[key] for key in ("signal_100", "time", "fs", "labels_100", "resolution_100"))
    if (signal.shape != (100, len(time)) or resolution.shape != (100, 100)
            or not np.array_equal(np.sort(labels), np.arange(1, 101))
            or not np.all(np.diff(time) > 0) or float(fs) <= 0
            or not all(np.isfinite(a).all() for a in (signal, time, resolution))):
        raise ValueError(f"Invalid parcel cache: {path}")
    return signal, time, float(fs), labels, resolution




@lru_cache(maxsize=2)
def sulcal_facecolors(darkness=None):
    """Original Nilearn background: face means, normalized per hemisphere."""
    image = nib.load(data.ASSETS / "plot_aux_data" / "fs_LR.32k.LR.sulc.dscalar.nii")
    values = image.get_fdata().ravel()
    colors = []
    for name, selection, model in image.header.get_axis(1).iter_structures():
        hemi = "L" if name.endswith("LEFT") else "R"
        mesh = nib.load(data.ASSETS / "plot_aux_data" / f"fs_LR.32k.{hemi}.pial.surf.gii")
        coords, faces = (array.data for array in mesh.darrays)
        vertex = np.zeros(len(coords))
        vertex[model.vertex] = values[selection]
        background = vertex[faces].mean(axis=1)
        if background.min() < 0 or background.max() > 1:
            background = plt.Normalize(background.min(), background.max())(background)
        if darkness is not None:
            background = background * darkness
        colors.append(plt.cm.gray_r(background))
    return np.vstack(colors)


def surface(ax, values, roi, cmap="viridis", norm=None, zoom=1, sulcal=False, darkness=None):
    """Dorsal view; values are in parcel-ID order, with a gray medial wall."""
    coords, faces, labels = cortex(roi)
    vertex = np.r_[0 if sulcal else np.nan, values][labels]
    face_values = vertex[faces].mean(axis=1)
    mapped = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = mapped.to_rgba(face_values)
    if sulcal:
        # Nilearn's bg_on_data=True: 70% mode color over an opaque background.
        colors[:, :3] = 0.7 * colors[:, :3] + 0.3 * sulcal_facecolors(darkness)[:, :3]
    colors[~np.isfinite(face_values)] = (0.85, 0.85, 0.85, 1)
    mesh = ax.plot_trisurf(*coords.T, triangles=faces, shade=False,
                          linewidth=0, antialiased=False, rasterized=True)
    mesh.set_facecolors(colors)
    ax.set_box_aspect(np.ptp(coords, axis=0), zoom=zoom)
    ax.view_init(elev=90, azim=-90)
    ax.set_axis_off()
    return mapped
