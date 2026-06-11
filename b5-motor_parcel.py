"""
=====================================================================
MOTOR PARCEL ACTIVATION
=====================================================================
Group-level visualization of task-evoked parcel activations
for motor task. Shows spatial activation patterns and time
courses for the most activated parcels.
"""

from pathlib import Path

import h5py
import numpy as np
import nibabel as nib
from nilearn import plotting, surface
from nibabel.cifti2.cifti2_axes import LabelAxis
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================

SUBJECTS = [
    '104012','105923','106521','108323','109123','113922','116726','133019','140117','156334',
]
SUBJECTS = [
    '104012','105923','106521','108323','109123','113922','116726','133019','140117','156334',
    '162026','162935','164636','169040','175237','177746','185442','191033','191437','192641',
    '198653','204521','205119','212318','212823','255639','257845','283543','287248','293748',
    '353740','358144','406836','559053','568963','581450','599671','601127','660951','662551',
    '667056','679770','680957','706040','707749','725751','735148','783462','814649','891667', '898176'
]
ROI = 100
LOCK = "TEMG"

BASELINE_WINDOW = (-0.2, 0.0)
RESPONSE_WINDOW = (0.0, 0.5)
TIME_PLOT_WINDOW = (-1.2, 1.2)

ACTIVATION_PERCENTILE = 80
MAX_PARCELS_PLOT = 10

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FIG_PATH = FIG_DIR / f"fig8_motor_activation_{len(SUBJECTS)}subj_Sch{ROI}.png"
PARC_PATH = ROOT_DIR / "parcellations" / f"Schaefer2018_{ROI}Parcels_7Networks_order.dlabel.nii"

# ============================================================
# Load parcel label names
# ============================================================

img = nib.load(str(PARC_PATH))
id_to_name = {}

for i in range(img.ndim):
    axis = img.header.get_axis(i)
    if isinstance(axis, LabelAxis):
        id_to_name = {int(k): str(v[0]).replace("7Networks_", "")
                      for k, v in axis.label[0].items()}
        break

# ============================================================
# Group data containers
# ============================================================

labels_ref = None
t_ref = None
fs_ref = None

activation_all = []
evoked_all = []

# ============================================================
# Subject loop
# ============================================================

for subj in SUBJECTS:
    subj_dir = DATA_DIR / subj
    npz_path = subj_dir / f"source_parcels_Schaefer{ROI}_{subj}_{LOCK}.npz"
    meta_path = subj_dir / "meta.mat"

    if not npz_path.exists():
        print(f"Skipping {subj}: missing parcel data ({npz_path})")
        continue

    with np.load(npz_path, allow_pickle=True) as pack:
        X_concat = pack["ts"]
        fs = float(pack["fs"])
        trial_lengths = pack["trial_lengths"].astype(int)
        labels = pack["labels"].astype(int)

    expected = int(trial_lengths[0])
    if not np.all(trial_lengths == expected):
        raise RuntimeError(f"Trial length mismatch for subject {subj}")

    X_trials = X_concat.reshape(X_concat.shape[0], len(trial_lengths), expected)

    # Time axis
    try:
        with h5py.File(str(meta_path), "r") as f:
            t_full = np.array(f["time"]).flatten()
    except Exception:
        zero = expected // 2
        t_full = (np.arange(expected) - zero) / fs

    if t_full.shape[0] != expected:
        raise RuntimeError(f"Time axis mismatch for subject {subj}")

    # Cross-subject consistency checks
    if labels_ref is None:
        labels_ref = labels
    elif not np.array_equal(labels, labels_ref):
        raise RuntimeError("Parcel labels not aligned across subjects")

    if t_ref is None:
        t_ref = t_full
        fs_ref = fs
    elif not np.allclose(t_full, t_ref):
        raise RuntimeError("Time axis not aligned across subjects")

    # Baseline and response windows
    mask_baseline = (t_full >= BASELINE_WINDOW[0]) & (t_full <= BASELINE_WINDOW[1])
    mask_response = (t_full >= RESPONSE_WINDOW[0]) & (t_full <= RESPONSE_WINDOW[1])

    if not mask_baseline.any() or not mask_response.any():
        raise RuntimeError(f"Invalid baseline/response window for subject {subj}")

    baseline = X_trials[:, :, mask_baseline].mean(axis=(1, 2))
    response = X_trials[:, :, mask_response].mean(axis=(1, 2))

    activation_all.append(np.abs(response - baseline))
    evoked_all.append(X_trials.mean(axis=1))

# ============================================================
# Group averages
# ============================================================

if len(activation_all) == 0:
    example = (
        DATA_DIR
        / SUBJECTS[0]
        / f"source_parcels_Schaefer{ROI}_{SUBJECTS[0]}_{LOCK}.npz"
    )
    raise SystemExit(
        f"\nNo parcel data found under {DATA_DIR}.\n"
        f"Expected files like: {example}\n"
        "Generate them first (see a1-reconstruction.py) or update DATA_DIR.\n"
    )

activation = np.mean(np.vstack(activation_all), axis=0)
evoked = np.mean(np.stack(evoked_all, axis=0), axis=0)
t_full = t_ref
labels = labels_ref

# ============================================================
# Threshold activated parcels
# ============================================================

thr = np.percentile(activation, ACTIVATION_PERCENTILE)
activation_thr = np.where(activation >= thr, activation, np.nan)
vmax_act = np.nanpercentile(activation_thr, 95)

idx = np.where(activation >= thr)[0]
if idx.size == 0:
    idx = np.array([int(np.nanargmax(activation))])

idx = idx[np.argsort(activation[idx])[::-1]][:MAX_PARCELS_PLOT]
parcel_ids = labels[idx]
parcel_names = [id_to_name.get(int(pid), f"Parcel {int(pid)}") for pid in parcel_ids]

print("Plotting time courses for parcels:")
for pid, name in zip(parcel_ids, parcel_names):
    print(f"  {int(pid)}: {name}")

# ============================================================
# Surface mapping
# ============================================================

coords_lh, faces_lh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.L.pial.surf.gii")
coords_rh, faces_rh = surface.load_surf_mesh("plot_aux_data/fs_LR.32k.R.pial.surf.gii")

sulc = nib.load("plot_aux_data/fs_LR.32k.LR.sulc.dscalar.nii").get_fdata().squeeze().ravel()
n_lh = coords_lh.shape[0]
n_rh = coords_rh.shape[0]
sulc_lh = sulc[:n_lh]
sulc_rh = sulc[n_lh : n_lh + n_rh]

labels_surface = nib.load(str(PARC_PATH)).get_fdata().squeeze().astype(int).ravel()
labels_lh = labels_surface[:n_lh]
labels_rh = labels_surface[n_lh : n_lh + n_rh]

act_lh = np.full(n_lh, np.nan)
act_rh = np.full(n_rh, np.nan)

for val, lab in zip(activation_thr, labels):
    act_lh[labels_lh == lab] = val
    act_rh[labels_rh == lab] = val

# ============================================================
# Time window for plotting
# ============================================================

mask_plot = (t_full >= TIME_PLOT_WINDOW[0]) & (t_full <= TIME_PLOT_WINDOW[1])
t_plot = t_full[mask_plot]
X_plot = evoked[idx][:, mask_plot]

# ============================================================
# Figure
# ============================================================

fig = plt.figure(figsize=(8, 3.7), dpi=200)
gs = fig.add_gridspec(1, 2, width_ratios=[0.65, 1.0], wspace=0.2)

# --- Left: spatial activation ---
ax1 = fig.add_subplot(gs[0, 0], projection="3d")

plotting.plot_surf_stat_map(
    (coords_lh, faces_lh),
    stat_map=act_lh,
    hemi="left",
    view="dorsal",
    bg_map=sulc_lh,
    cmap="Reds",
    colorbar=False,
    bg_on_data=True,
    darkness=0.3,
    vmin=thr,
    vmax=vmax_act,
    axes=ax1,
)
plotting.plot_surf_stat_map(
    (coords_rh, faces_rh),
    stat_map=act_rh,
    hemi="right",
    view="dorsal",
    bg_map=sulc_rh,
    cmap="Reds",
    colorbar=False,
    bg_on_data=True,
    darkness=0.3,
    vmin=thr,
    vmax=vmax_act,
    axes=ax1,
)

ax1.view_init(elev=90, azim=-90)
ax1.set_axis_off()
ax1.set_title(f"Activation ≥ {ACTIVATION_PERCENTILE}th percentile")

# --- Right: evoked time courses ---
ax2 = fig.add_subplot(gs[0, 1])
colors = plt.cm.turbo(np.linspace(0, 1, len(idx)))

for k, (color, name) in enumerate(zip(colors, parcel_names)):
    ax2.plot(t_plot, X_plot[k], color=color, label=name, zorder=len(idx) - k)

ax2.axvline(0, color="k", linestyle="--", linewidth=0.8)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (a.u.)")
ax2.set_title("Evoked responses of top parcels")
ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7)

plt.savefig(FIG_PATH, bbox_inches="tight", facecolor="white")
plt.close()

print(f"\nSaved figure: {FIG_PATH}\n")
