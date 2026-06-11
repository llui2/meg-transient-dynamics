"""
=====================================================================
NETWORK MODES TABLE
=====================================================================
Visualize connectivity networks and their Laplacian eigenmodes
in a multi-panel table layout showing network architecture and
spatial patterns of eigenmodes.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from nilearn import plotting, surface
from scipy.sparse import csgraph, csr_matrix
from scipy.sparse.linalg import eigsh

mpl.rcParams.update({
    "font.family": "sans-serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
mpl.rcParams["pdf.compression"] = 9 # maximize vector compression for smaller PDFs

# ============================================================
# LAYOUT CONFIGURATION - Easy to adjust!
# ============================================================

# Figure dimensions
fig_width = 6.1        # Total figure width in inches
fig_height = 5       # Total figure height in inches
dpi = 400             # Output resolution

# Grid spacing (adjust these to control subplot spacing)
wspace = -0.35        # Horizontal spacing between columns (negative = overlap)
hspace = -0.05         # Vertical spacing between rows

# Margins (as fraction of figure size)
left_margin = 0.01    # Left margin
right_margin = 1.0    # Right margin
top_margin = 0.92     # Top margin
bottom_margin = 0.08  # Bottom margin

# Network column (first column) width ratio
network_col_ratio = 1.1  # How much wider the network column is vs mode columns

# Labels
row_label_x = 0.05   # X position of row labels (network names)
col_label_y = 0.92    # Y position of column labels (Mode N)
label_fontsize = 11   # Font size for all labels

# Network visualization settings
zoom = 0.75           # Zoom for network brains (smaller = bigger brain)
node_size = 4         # Size of network nodes
node_color = "#777777"
brain_alpha = 0.1    # Transparency of brain surface

# Mode visualization settings
mode_cmap = "seismic" # Colormap for eigenmodes
rasterize_3d_axes = True  # Rasterize 3D axes to keep output file size small

# ============================================================
# Data Configuration
# ============================================================

ROI = 100             # Number of parcels (must match saved W matrices)
n_modes = 4           # Number of eigenmodes to show (columns 1-n_modes)

# Network files
W_structural_file = f"W_Sch{ROI}.npy"
W_edr_file = f"W_EDR_Sch{ROI}.npy"
W_geo_file = f"W_Geo_Sch{ROI}.npy"

# Surface and parcellation files
parcellation_file = f"parcellations/Schaefer2018_{ROI}Parcels_7Networks_order.dlabel.nii"
lh_surf_file = "plot_aux_data/fs_LR.32k.L.pial.surf.gii"
rh_surf_file = "plot_aux_data/fs_LR.32k.R.pial.surf.gii"
sulc_file = "plot_aux_data/fs_LR.32k.LR.sulc.dscalar.nii"

# Network display settings: (name, file, edge_color, threshold_pct, width_scale)
networks = [
    ("Structural", W_structural_file, "#ff4646", 80, 2.0),
    ("EDR", W_edr_file, "#00c40a", 85, 2.0),
    ("Geometric", W_geo_file, "#7066ff", 0, 1.0),
]

output_file = "figures/fig5_modes_table.pdf"

# ============================================================
# Load Surfaces and Parcellation
# ============================================================

print("Loading surfaces and parcellation...")

# Load surface meshes
coords_lh, faces_lh = surface.load_surf_mesh(lh_surf_file)
coords_rh, faces_rh = surface.load_surf_mesh(rh_surf_file)
n_lh = coords_lh.shape[0]
n_rh = coords_rh.shape[0]

# Combine hemispheres for 3D network plots
all_coords = np.vstack((coords_lh, coords_rh))
all_faces = np.vstack((faces_lh, faces_rh + n_lh))

# Load parcellation labels
labels = nib.load(parcellation_file).get_fdata().squeeze().astype(int)
lh_labels = labels[:n_lh] - 1  # 0-indexed for mode mapping
rh_labels = labels[n_lh:] - 1

# Load sulcal depth map
sulc = nib.load(sulc_file).get_fdata().squeeze()
lh_sulc = sulc[:n_lh]
rh_sulc = sulc[n_lh:n_lh + n_rh]

# Compute centroid for each parcel (for network visualization)
centroids = np.zeros((ROI, 3))
for i in range(ROI):
    mask = (labels == (i + 1))
    if np.sum(mask) > 0:
        centroids[i] = np.mean(all_coords[mask], axis=0)

print(f"Loaded {ROI} parcels, {n_lh} LH vertices, {n_rh} RH vertices")

# ============================================================
# Load Networks and Compute Eigenmodes
# ============================================================

print("Loading networks and computing eigenmodes...")

network_data = []  # Will store (name, W_norm, modes, color, thresh, width)

for name, filepath, color, thresh, width in networks:
    # Load network matrix
    W = np.load(filepath, allow_pickle=False).astype(float, copy=False)
    
    # Handle different matrix sizes (some have 0-indexed, some 1-indexed)
    if W.shape[0] > ROI:
        W = W[1:ROI+1, 1:ROI+1]
    
    # Compute Laplacian eigenmodes using sparse solver for speed
    Lap = csgraph.laplacian(csr_matrix(W), normed=False).asfptype()
    evals, evecs = eigsh(Lap, k=n_modes + 1, which="SM", tol=1e-6)
    order = np.argsort(evals)
    eigenvalues = evals[order]
    eigenvectors = evecs[:, order]
    
    # Skip mode 0 (constant), take modes 1 to n_modes
    modes = eigenvectors[:, 1:n_modes + 1]
    
    # Normalize W for visualization
    W_norm = W / (W.max() + 1e-10)
    
    network_data.append((name, W_norm, modes, color, thresh, width))
    print(f"  {name}: eigenvalue[1] = {eigenvalues[1]:.6f}")

# ============================================================
# Helper Functions
# ============================================================

def map_parcel_to_surface(parcel_values):
    """Map parcel-level values to vertex-level for visualization."""
    lh_vals = np.zeros(n_lh)
    rh_vals = np.zeros(n_rh)

    valid_lh = lh_labels >= 0
    valid_rh = rh_labels >= 0

    lh_vals[valid_lh] = parcel_values[lh_labels[valid_lh]]
    rh_vals[valid_rh] = parcel_values[rh_labels[valid_rh]]

    return lh_vals, rh_vals

def strip_3d_axes(ax):
    """Remove panes/background so overlapping panels stay transparent."""
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))
    if hasattr(ax, "patch"):
        ax.patch.set_alpha(0)
    for axis in (getattr(ax, "xaxis", None), getattr(ax, "yaxis", None), getattr(ax, "zaxis", None)):
        if axis is None:
            continue
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.set_visible(False)
            pane.set_edgecolor("none")
            pane.set_facecolor((0, 0, 0, 0))
    ax.grid(False)


def adjust_column_spacing(axis_grid, delta):
    """Shift columns horizontally so `wspace` directly controls the gap."""
    if abs(delta) < 1e-6:
        return
    for row_axes in axis_grid:
        for idx, ax in enumerate(row_axes[1:], start=1):
            pos = ax.get_position()
            shift = delta * idx
            pos.x0 += shift
            pos.x1 += shift
            ax.set_position(pos)


def plot_mode_brain(ax, lh_vals, rh_vals, vmin, vmax):
    """Plot eigenmode on both hemispheres."""
    plotting.plot_surf_stat_map(
        (coords_lh, faces_lh), stat_map=lh_vals, hemi="left", view="dorsal",
        bg_map=lh_sulc, cmap=mode_cmap, alpha=1, bg_on_data=True, darkness=None,
        vmin=vmin, vmax=vmax, colorbar=False, axes=ax
    )
    plotting.plot_surf_stat_map(
        (coords_rh, faces_rh), stat_map=rh_vals, hemi="right", view="dorsal",
        bg_map=rh_sulc, cmap=mode_cmap, alpha=1, bg_on_data=True, darkness=None,
        vmin=vmin, vmax=vmax, colorbar=False, axes=ax
    )


def get_edges(W, threshold_percentile):
    """Extract edges above threshold as line segments."""
    triu_idx = np.triu_indices(W.shape[0], k=1)
    weights = W[triu_idx]
    nonzero_weights = weights[weights > 0]
    
    if len(nonzero_weights) == 0:
        return [], []
    
    # Binary network detection
    is_binary = np.allclose(nonzero_weights, nonzero_weights[0])
    
    if is_binary or threshold_percentile <= 0:
        thresh = 0
    else:
        thresh = np.percentile(nonzero_weights, threshold_percentile)
    
    edges = []
    edge_weights = []
    
    for idx, (i, j) in enumerate(zip(triu_idx[0], triu_idx[1])):
        w = weights[idx]
        if w > thresh:
            edges.append([centroids[i], centroids[j]])
            edge_weights.append(w)
    
    return edges, edge_weights


def plot_network_brain(ax, W, edge_color, threshold_pct, width_scale):
    """Plot 3D brain network."""
    
    # Plot transparent brain surface
    ax.plot_trisurf(
        all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
        triangles=all_faces,
        color='lightgray',
        alpha=brain_alpha,
        edgecolor='none',
        linewidth=0,
        shade=False
    )
    
    # Get edges
    edges, weights = get_edges(W, threshold_pct)
    
    if len(edges) > 0:
        weights = np.array(weights)
        if weights.max() > weights.min():
            weights_norm = (weights - weights.min()) / (weights.max() - weights.min())
        else:
            weights_norm = np.ones_like(weights)
        
        lc = Line3DCollection(
            edges,
            colors=[(*mpl.colors.to_rgb(edge_color), 0.3 + 0.7 * w) for w in weights_norm],
            linewidths=(0.3 + 1.2 * weights_norm) * width_scale
        )
        ax.add_collection3d(lc)
    
    # Plot nodes
    ax.scatter(
        centroids[:, 0], centroids[:, 1], centroids[:, 2],
        s=node_size, c=node_color, alpha=1.0,
        edgecolors='white', linewidths=0.1
    )
    
    # Clean up axes
    ax.view_init(elev=90, azim=90)
    
    # Set equal aspect ratio with zoom
    max_range = np.array([
        all_coords[:, 0].max() - all_coords[:, 0].min(),
        all_coords[:, 1].max() - all_coords[:, 1].min(),
        all_coords[:, 2].max() - all_coords[:, 2].min()
    ]).max() / 2.0 * zoom
    
    mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
    mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
    mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    strip_3d_axes(ax)

# ============================================================
# Create Figure
# ============================================================

print("Creating table figure...")

n_rows = len(networks)
n_cols = 1 + n_modes  # Network column + mode columns

# Compute width ratios
width_ratios = [network_col_ratio] + [1] * n_modes

fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)

# Create grid
gs = fig.add_gridspec(
    n_rows, n_cols,
    width_ratios=width_ratios,
    wspace=wspace,
    hspace=hspace,
    left=left_margin,
    right=right_margin,
    top=top_margin,
    bottom=bottom_margin
)

# Compute global vmin/vmax for modes across all networks
all_modes = np.concatenate([nd[2] for nd in network_data], axis=1)
vmax = np.max(np.abs(all_modes))
vmin = -vmax

# Plot each row (network)
for row_idx, (name, W_norm, modes, color, thresh, width) in enumerate(network_data):
    print(f"  Plotting {name}...")
    
    # Column 0: Network visualization
    ax_net = fig.add_subplot(gs[row_idx, 0], projection='3d')
    if rasterize_3d_axes:
        ax_net.set_rasterized(True)
    plot_network_brain(ax_net, W_norm, color, thresh, width)
    
    # Columns 1 to n_modes: Eigenmodes
    for col_idx in range(n_modes):
        ax_mode = fig.add_subplot(gs[row_idx, col_idx + 1], projection='3d')
        if rasterize_3d_axes:
            ax_mode.set_rasterized(True)
        
        mode_vals = modes[:, col_idx]
        lh_vals, rh_vals = map_parcel_to_surface(mode_vals)
        plot_mode_brain(ax_mode, lh_vals, rh_vals, vmin, vmax)
        
        ax_mode.view_init(elev=90, azim=-90)
        ax_mode.set_axis_off()
        strip_3d_axes(ax_mode)

# Add row labels (network names)
for row_idx, (name, _, _, color, _, _) in enumerate(network_data):
    # Position in figure coordinates
    y_pos = top_margin - (row_idx + 0.5) * (top_margin - bottom_margin) / n_rows
    fig.text(
        row_label_x, y_pos,
        name,
        ha='left', va='center',
        fontsize=label_fontsize,
        rotation=90
    )

# Add column labels
col_labels = ["Network"] + [f"Mode {i+2}" for i in range(n_modes)]
for col_idx, label in enumerate(col_labels):
    # Compute x position based on width ratios
    total_width = sum(width_ratios)
    x_offset = sum(width_ratios[:col_idx]) / total_width
    col_width = width_ratios[col_idx] / total_width
    x_pos = left_margin + (right_margin - left_margin) * (x_offset + col_width / 2)
    
    fig.text(
        x_pos, col_label_y,
        label,
        ha='center', va='bottom',
        fontsize=label_fontsize,
    )

# Save figure
fig.savefig(
    output_file,
    dpi=dpi,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.02,
)
plt.close(fig)

print(f"Saved: {output_file}")
