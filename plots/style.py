"""Shared manuscript typography, distribution plots, and network colors.

In a Structural-minus-control difference, use the control's color.
"""
import matplotlib as mpl
from matplotlib.transforms import Bbox
import numpy as np


MODEL_COLORS = {
    "Structural": "#FF3B30",
    "EDR": "#00C853",
    "Geometric": "#7066FF",
}

# Display names are independent of the model keys stored in analysis files.
MODEL_LABELS = {"Structural": "Structural", "EDR": "EDR", "Geometric": "Local"}

# Leakage curves reuse the manuscript hues, not network identities.
# Preserve the original spatial reference and corresponding distance guides.
LEAKAGE_COLORS = {
    "data": MODEL_COLORS["Geometric"],
    "fit": MODEL_COLORS["Structural"],
    "reference": "#FF0000",
    "guide_1": "#FF8427",
    "guide_2": "#C93FFF",
}


def apply_style():
    """Apply the same readable typography and axes to every manuscript plot."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 9, "axes.titlesize": 9,
        "axes.labelpad": 3, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })



# Horizontal geometry in typographic points, independent of figure width.
BOX_WIDTH = 4.5
BOX_STEP = 7.5
BOX_GROUP_GAP = 9.0


def box_position(group, index, count, slots=None):
    """Center boxes at a fixed physical pitch within each category."""
    slots = count if slots is None else slots
    return group * (slots * BOX_STEP + BOX_GROUP_GAP) + (index-(count-1)/2)*BOX_STEP


def boxplot_axis(ax, labels, count, slots=None):
    """Register categorical groups; save_figure maintains one x unit per point."""
    centers = [box_position(i, 0, 1, slots=count if slots is None else slots)
               for i in range(len(labels))]
    half = (count-1)*BOX_STEP/2 + BOX_WIDTH/2 + 3
    ax._boxplot_extent = (centers[0]-half, centers[-1]+half)
    ax.set_xticks(centers, labels)
    ax.set_xlim(centers[0]-half-2, centers[-1]+half+2)


def _size_boxplot_axes(fig, validate=False):
    for ax in fig.axes:
        if not hasattr(ax, '_boxplot_extent'):
            continue
        lower, upper = ax._boxplot_extent
        span = ax.get_position().width * fig.get_figwidth() * 72
        if validate and span < upper-lower-1e-8:
            raise ValueError(f'Boxplot needs {(upper-lower)/72:.2f} inches; axis has {span/72:.2f}')
        midpoint = (lower+upper)/2
        ax.set_xlim(midpoint-span/2, midpoint+span/2)



def compact_boxplot_canvas(fig):
    """Trim spare canvas width while retaining aligned groups and point geometry.

    Intended for vertically stacked categorical axes with the same plot width.
    The union keeps both rows aligned when their groups contain different counts.
    """
    axes = [ax for ax in fig.axes if hasattr(ax, '_boxplot_extent')]
    positions = [ax.get_position().frozen() for ax in axes]
    if not axes:
        return
    if any(abs(p.x0-positions[0].x0)>1e-8 or abs(p.width-positions[0].width)>1e-8
           for p in positions):
        raise ValueError('Compact boxplot canvas requires aligned stacked axes')
    lower = min(ax._boxplot_extent[0] for ax in axes)
    upper = max(ax._boxplot_extent[1] for ax in axes)
    plot_width = (upper-lower)/72
    old_width = fig.get_figwidth()
    left = positions[0].x0*old_width
    right = (1-positions[0].x1)*old_width
    width = left+plot_width+right
    fig.set_size_inches(width, fig.get_figheight(), forward=False)
    for ax, position in zip(axes, positions):
        ax.set_position([left/width, position.y0, plot_width/width, position.height])


def save_figure(fig, path, width, dpi=300, pad=0.025):
    """Export at the given width in inches, preserving text/line sizes in points.

    Use width=None to export the natural content width after compacting a canvas.
    Adjust the canvas width, not the fonts, until the visible content and small
    border fill the requested width. Use an explicit final bounding box so tight
    cropping cannot change the physical width of the exported PDF.
    """
    fig.set_dpi(dpi)
    if width is None:
        _size_boxplot_axes(fig, validate=True)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        export = Bbox.from_bounds(bounds.x0-pad, bounds.y0-pad,
                                 bounds.width+2*pad, bounds.height+2*pad)
        fig.savefig(path, dpi=dpi, bbox_inches=export, pad_inches=0, facecolor="white")
        return export.width
    for _ in range(10):
        _size_boxplot_axes(fig)
        fig.canvas.draw()
        bounds = fig.get_tightbbox(fig.canvas.get_renderer())
        error = width - (bounds.width + 2 * pad)
        if abs(error) < 0.002:
            break
        fig.set_size_inches(fig.get_figwidth() + error, fig.get_figheight(), forward=False)
    else:
        raise RuntimeError(f"Could not fit {path} to {width} inches")
    _size_boxplot_axes(fig, validate=True)
    if bounds.width > width:
        raise ValueError(f"Figure content exceeds the export width: {path}")
    horizontal_pad = (width - bounds.width) / 2
    export = Bbox.from_bounds(bounds.x0 - horizontal_pad, bounds.y0 - pad,
                              width, bounds.height + 2 * pad)
    fig.savefig(path, dpi=dpi, bbox_inches=export, pad_inches=0, facecolor="white")


def participant_boxplot(ax, values, position, color, rng, width=BOX_WIDTH,
                        hatch=None):
    """Median/IQR, 1.5-IQR whiskers, and every participant (including outliers)."""
    values = np.asarray(values)
    artists = ax.boxplot(
        values, positions=[position], widths=width,
        showfliers=False, manage_ticks=False, patch_artist=True,
        boxprops={"facecolor": "none", "edgecolor": "0.25",
                  "linewidth": 0.8, "hatch": hatch},
        whiskerprops={"color": "0.25", "linewidth": 0.8},
        capprops={"color": "0.25", "linewidth": 0.8},
        medianprops={"color": color, "linewidth": 1.5})
    ax.scatter(position + rng.uniform(-BOX_WIDTH/3, BOX_WIDTH/3, len(values)), values,
               s=5, alpha=0.2, color=color, edgecolors="none", zorder=3)
    return artists


def panel_label(ax, letter):
    """Regular-weight panel markers, uniformly placed at the upper-right."""
    draw = ax.text2D if hasattr(ax, "text2D") else ax.text
    return draw(0.98, 0.98, f"({letter})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="normal",
                zorder=20)
