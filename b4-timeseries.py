"""
=====================================================================
MODAL TIMESERIES FIT
=====================================================================
Fit telegrapher (damped wave) model to MEG parcel time series
using Laplacian eigenmodes. Visualizes modal decomposition,
model reconstruction, and per-mode goodness of fit.
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# Configuration
# ============================================================

SUBJECT = "104012" # '104012','105923','106521','108323','109123','113922','116726','133019','140117','156334'
ROI = 100
LOCK = "TEMG"

TMIN, TMAX = -0.1, 0.9
N_MODES = ROI
N_SHOW = 100

BANDS = {
    "Fast Delta": (2, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 80),
}

SELECT_BAND = "Fast Delta" # set to None for broadband

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / SUBJECT
FIG_PATH = ROOT_DIR / "figures" / f"fig_modal_timeseries_{SUBJECT}.pdf"

mpl.rcParams.update(
    {
        "font.size": 8,
        "font.family": "sans-serif",
        "mathtext.fontset": "cm",
    }
)

# ============================================================
# Load parcellated MEG data
# ============================================================

npz_path = DATA_DIR / f"source_parcels_Schaefer{ROI}_{SUBJECT}_{LOCK}.npz"
meta_path = DATA_DIR / "meta.mat"

# Choose exactly one:
W_path = ROOT_DIR / f"W_Sch{ROI}.npy"
# W_path = ROOT_DIR / f"W_EDR_Sch{ROI}.npy"
# W_path = ROOT_DIR / f"W_Geo_Sch{ROI}.npy"

with np.load(npz_path, allow_pickle=True) as pack:
    X_concat = pack["ts"]
    fs = float(pack["fs"])
    trial_lengths = pack["trial_lengths"].astype(int)
    labels = pack["labels"].astype(int)
    R_roi = pack["R_roi"]

expected = int(trial_lengths[0])

# ============================================================
# Time axis (FULL trial)
# ============================================================

try:
    import h5py

    with h5py.File(str(meta_path), "r") as f:
        t_full = np.array(f["time"]).flatten()
except Exception:
    zero = expected // 2
    t_full = (np.arange(expected) - zero) / fs

if t_full.shape[0] != expected:
    raise RuntimeError(f"time axis length {t_full.shape[0]} != expected {expected}")

# ============================================================
# Trial averaging (FULL trial)
# ============================================================

n_parcels = X_concat.shape[0]
n_trials = len(trial_lengths)

X_trials = X_concat.reshape(n_parcels, n_trials, expected)
X_full = X_trials.mean(axis=1)  # (parcels, time)

# ============================================================
# Bandpass filtering (ON FULL trial), then crop
# ============================================================

if SELECT_BAND is not None:
    fmin, fmax = BANDS[SELECT_BAND]
    nyq = 0.5 * fs
    hi = min(fmax / nyq, 0.99)
    lo = fmin / nyq
    if lo <= 0 or hi <= 0 or lo >= hi:
        raise ValueError(f"Bad band: {SELECT_BAND} ({fmin}, {fmax}) for fs={fs}")
    sos = butter(4, [lo, hi], btype="band", output="sos")
    X_full = sosfiltfilt(sos, X_full, axis=1)
    band_label = f"{SELECT_BAND} ({fmin}-{fmax} Hz)"
else:
    band_label = "Broadband"

# Now crop for fitting/plotting
time_mask = (t_full >= TMIN) & (t_full <= TMAX)
t = t_full[time_mask]
X = X_full[:, time_mask]

# ============================================================
# Load connectivity and compute Laplacian
# ============================================================

W = np.load(W_path)

# Match parcel ordering
idx = labels - 1
W = W[np.ix_(idx, idx)]

# Combinatorial Laplacian
W = np.maximum(W, 0)
W = 0.5 * (W + W.T)
np.fill_diagonal(W, 0)

D = np.diag(W.sum(axis=1))
L = D - W

# ============================================================
# Laplacian eigenspectrum
# ============================================================

eigvals, eigvecs = np.linalg.eigh(L)
lam = eigvals[1 : N_MODES + 1]
V = eigvecs[:, 1 : N_MODES + 1]

# ============================================================
# Modal coefficients (leakage-aware)
# ============================================================

Z, *_ = np.linalg.lstsq(R_roi @ V, X, rcond=None)

# ============================================================
# Global tau fit
# ============================================================

mean_var = np.mean(np.var(Z, axis=1))
var_eps = mean_var * 1e-12 if mean_var > 0 else np.finfo(float).eps


def telegrapher_basis(tvec, lam_k, tau):
    tau = max(float(tau), 1e-5)
    disc = 1.0 - 4.0 * tau * float(lam_k)
    if disc > 0:
        r1 = (-1 + np.sqrt(disc)) / (2 * tau)
        r2 = (-1 - np.sqrt(disc)) / (2 * tau)
        return np.column_stack([np.exp(r1 * tvec), np.exp(r2 * tvec)])
    elif disc < 0:
        w = np.sqrt(-disc) / (2 * tau)
        env = np.exp(-tvec / (2 * tau))
        return np.column_stack([env * np.cos(w * tvec), env * np.sin(w * tvec)])
    else:
        env = np.exp(-tvec / (2 * tau))
        return np.column_stack([env, tvec * env])


def residuals(tau_arr):
    tau = float(tau_arr[0])
    res = []
    for l, y in zip(lam, Z):
        M = telegrapher_basis(t, l, tau)
        c, *_ = np.linalg.lstsq(M, y, rcond=None)
        yhat = M @ c
        sy = np.std(y)
        if sy > np.sqrt(var_eps):
            res.append((yhat - y) / sy)
    return np.concatenate(res) if res else np.zeros(1)


sol = least_squares(residuals, x0=[0.2], bounds=(0.001, 0.5))
tau_fit = float(sol.x[0])

# ============================================================
# Per-mode fits and R^2
# ============================================================

Z_pred = np.zeros_like(Z)
r2_all = np.full(Z.shape[0], np.nan)

for k, (l, y) in enumerate(zip(lam, Z)):
    M = telegrapher_basis(t, l, tau_fit)
    c, *_ = np.linalg.lstsq(M, y, rcond=None)
    yhat = M @ c
    Z_pred[k] = yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot > var_eps:
        r2_all[k] = 1 - np.sum((y - yhat) ** 2) / ss_tot

# ============================================================
# Visualization
# ============================================================

show = min(N_SHOW, Z.shape[0])
colors = plt.cm.viridis(np.linspace(0, 1, show))

norms = np.maximum(
    np.max(np.abs(Z[:show]), axis=1),
    np.max(np.abs(Z_pred[:show]), axis=1),
)
norms = np.maximum(norms, 1e-12)

fig, axes = plt.subplots(3, 1, figsize=(4, 4), dpi=300, gridspec_kw={"hspace": 0.55})

# Original modal responses
for k in range(show):
    axes[0].plot(t, Z[k] / norms[k], lw=0.9, color=colors[k])
axes[0].axvline(0, color="k", lw=0.8)
axes[0].set_xlim(TMIN, TMAX)
axes[0].set_ylabel(r"$z_k(t)$")
axes[0].set_title(f"{SUBJECT}: {band_label} telegrapher fit ($\\tau={tau_fit:.2f}$ s)")
axes[0].set_xticklabels([])

y_limits = axes[0].get_ylim()

# Fitted responses
for k in range(show):
    axes[1].plot(t, Z_pred[k] / norms[k], lw=0.9, color=colors[k])
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlim(TMIN, TMAX)
axes[1].set_ylabel("fit")
axes[1].set_xlabel("time (s)")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(y_limits)
pos = axes[1].get_position()
axes[1].set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height])

# R^2 per mode
axes[2].bar(np.arange(1, show + 1), r2_all[:show], color=colors, width=0.6)
axes[2].set_ylabel(r"$R^2$")
axes[2].set_xlabel("mode index $k$")
axes[2].set_ylim(0, 1.05)
axes[2].set_title("variance explained per mode")
axes[2].grid(True, axis="y", alpha=0.3)

FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure: {FIG_PATH}")
