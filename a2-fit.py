"""
=====================================================================
TELEGRAPHER MODEL FITTING
=====================================================================
Fit telegrapher (damped wave) models to MEG parcel time series
and compare Structural, EDR, and Geometric connectivity models
across multiple frequency bands.
"""

import csv
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

ROI = 100
LOCK = "TEMG"

SUBJECTS = [
    '104012','105923','106521','108323','109123','113922','116726','133019','140117','156334',
    '162026','162935','164636','169040','175237','177746','185442','191033','191437','192641',
    '198653','204521','205119','212318','212823','255639','257845','283543','287248','293748',
    '353740','358144','406836','559053','568963','581450','599671','601127','660951','662551',
    '667056','679770','680957','706040','707749','725751','735148','783462','814649','891667', '898176'
]

TMIN, TMAX = -0.1, 0.9
N_MODES = ROI

BANDS = {
    "Fast Delta": (2, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 80),
}

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESULT_CSV = DATA_DIR / f"network_fit_ROI{ROI}_{LOCK}.csv"

W_STRUCT = np.load(ROOT_DIR / f"W_Sch{ROI}.npy")
W_EDR = np.load(ROOT_DIR / f"W_EDR_Sch{ROI}.npy")
W_GEO = np.load(ROOT_DIR / f"W_Geo_Sch{ROI}.npy")

# ============================================================
# Output containers
# ============================================================

metrics = {
    "R2_Structural": [],
    "R2_EDR": [],
    "R2_Geometric": [],
    "Tau_Structural": [],
    "Tau_EDR": [],
    "Tau_Geometric": [],
}

processed_subjects = []

# ============================================================
# Loop over subjects
# ============================================================

for subj in tqdm(SUBJECTS, desc="Subjects"):

    try:
        # ----------------------------------------------------
        # Load parcel data
        # ----------------------------------------------------

        npz_path = DATA_DIR / subj / f"source_parcels_Schaefer{ROI}_{subj}_{LOCK}.npz"
        meta_path = DATA_DIR / subj / "meta.mat"

        pack = np.load(npz_path)
        X_concat = pack["ts"]
        fs = float(pack["fs"])
        labels = pack["labels"].astype(int)
        R_roi = pack["R_roi"] if "R_roi" in pack else None
        trial_lengths = pack["trial_lengths"].astype(int)

        # ----------------------------------------------------
        # Time axis (FULL trial)
        # ----------------------------------------------------

        try:
            with h5py.File(str(meta_path), "r") as f:
                t_full = np.array(f["time"]).flatten()
        except Exception:
            t_full = sio.loadmat(str(meta_path), simplify_cells=True)["time"]

        expected = len(t_full)

        # ----------------------------------------------------
        # Trial averaging (FULL trial)
        # ----------------------------------------------------

        n_parcels = X_concat.shape[0]
        X_trials = X_concat.reshape(n_parcels, len(trial_lengths), expected)
        X_full = X_trials.mean(axis=1)  # (parcels, time)

        # ----------------------------------------------------
        # Connectivity alignment
        # ----------------------------------------------------

        idx = labels - 1
        W_s = W_STRUCT[np.ix_(idx, idx)]
        W_e = W_EDR[np.ix_(idx, idx)]
        W_g = W_GEO[np.ix_(idx, idx)]

        def laplacian(W):
            W = np.maximum(W, 0)
            W = 0.5 * (W + W.T)
            np.fill_diagonal(W, 0)
            return np.diag(W.sum(axis=1)) - W

        L_struct = laplacian(W_s)
        L_edr = laplacian(W_e)
        L_geo = laplacian(W_g)

        # ----------------------------------------------------
        # Per-band fitting
        # ----------------------------------------------------

        subj_results = {k: [] for k in metrics}

        for band, (fmin, fmax) in BANDS.items():

            # ===== FIX 1: filter on FULL signal =====
            nyq = 0.5 * fs
            sos = butter(
                4,
                [fmin / nyq, min(fmax / nyq, 0.99)],
                btype="band",
                output="sos",
            )
            Xb_full = sosfiltfilt(sos, X_full, axis=1)

            # ===== FIX 2: crop AFTER filtering =====
            time_mask = (t_full >= TMIN) & (t_full <= TMAX)
            t = t_full[time_mask]
            Xb = Xb_full[:, time_mask]

            def fit_model(X, L):

                eigvals, eigvecs = np.linalg.eigh(L)
                lam = eigvals[1 : N_MODES + 1]
                V = eigvecs[:, 1 : N_MODES + 1]

                if R_roi is None:
                    Z = V.T @ X
                else:
                    Z, *_ = np.linalg.lstsq(R_roi @ V, X, rcond=None)

                mean_var = np.mean(np.var(Z, axis=1))
                var_eps = mean_var * 1e-12 if mean_var > 0 else np.finfo(float).eps

                def residuals(tau_arr):
                    tau = tau_arr[0]
                    res = []
                    for l, y in zip(lam, Z):
                        disc = 1 - 4 * tau * l
                        if disc < 0:
                            w = np.sqrt(-disc) / (2 * tau)
                            env = np.exp(-t / (2 * tau))
                            M = np.column_stack([env * np.cos(w * t), env * np.sin(w * t)])
                        else:
                            r1 = (-1 + np.sqrt(disc)) / (2 * tau)
                            r2 = (-1 - np.sqrt(disc)) / (2 * tau)
                            M = np.column_stack([np.exp(r1 * t), np.exp(r2 * t)])

                        c, *_ = np.linalg.lstsq(M, y, rcond=None)
                        yhat = M @ c
                        if np.std(y) > np.sqrt(var_eps):
                            res.append((yhat - y) / np.std(y))
                    return np.concatenate(res)

                sol = least_squares(residuals, x0=[0.12], bounds=(0.01, 1.0))
                tau_opt = sol.x[0]

                r2 = []
                for l, y in zip(lam, Z):
                    disc = 1 - 4 * tau_opt * l
                    if disc < 0:
                        w = np.sqrt(-disc) / (2 * tau_opt)
                        env = np.exp(-t / (2 * tau_opt))
                        M = np.column_stack([env * np.cos(w * t), env * np.sin(w * t)])
                    else:
                        r1 = (-1 + np.sqrt(disc)) / (2 * tau_opt)
                        r2_ = (-1 - np.sqrt(disc)) / (2 * tau_opt)
                        M = np.column_stack([np.exp(r1 * t), np.exp(r2_ * t)])

                    c, *_ = np.linalg.lstsq(M, y, rcond=None)
                    yhat = M @ c
                    ss = np.sum((y - y.mean()) ** 2)
                    r2.append(0 if ss < var_eps else 1 - np.sum((y - yhat) ** 2) / ss)

                return float(np.mean(r2)), tau_opt

            r2_s, tau_s = fit_model(Xb, L_struct)
            r2_e, tau_e = fit_model(Xb, L_edr)
            r2_g, tau_g = fit_model(Xb, L_geo)

            subj_results["R2_Structural"].append(r2_s)
            subj_results["R2_EDR"].append(r2_e)
            subj_results["R2_Geometric"].append(r2_g)
            subj_results["Tau_Structural"].append(tau_s)
            subj_results["Tau_EDR"].append(tau_e)
            subj_results["Tau_Geometric"].append(tau_g)

        for k in metrics:
            metrics[k].append(subj_results[k])

        processed_subjects.append(subj)

    except Exception as e:
        print(f"Subject {subj} failed: {e}")

# ============================================================
# Save CSV
# ============================================================

with open(RESULT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["roi", "lock", "subject", "band", "metric", "model", "value"])
    for si, subj in enumerate(processed_subjects):
        for bi, band in enumerate(BANDS.keys()):
            for model in ["Structural", "EDR", "Geometric"]:
                writer.writerow([ROI, LOCK, subj, band, "R2", model, metrics[f"R2_{model}"][si][bi]])
                writer.writerow([ROI, LOCK, subj, band, "Tau", model, metrics[f"Tau_{model}"][si][bi]])

print(f"Saved results to {RESULT_CSV}")
