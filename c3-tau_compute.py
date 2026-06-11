import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, sosfiltfilt

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
SUBJECTS = [
    "104012","105923","106521","108323","109123","113922","116726","133019","140117","156334",
    "162026","162935","164636","169040","175237","177746","185442","191033","191437","192641",
    "198653","204521","205119","212318","212823","255639","257845","283543","287248","293748",
    "353740","358144","406836","559053","568963","581450","599671","601127","660951","662551",
    "667056","679770","680957","706040","707749","725751","735148","783462","814649","891667","898176"
]
ROI = 100
LOCK = "TEMG"
TMIN, TMAX = -0.1, 0.9
N_MODES = 100

USE_LOG_FREQ_BINS = True
FREQ_RANGE = (2.0, 80.0)
NUM_LOG_BINS = 15

if USE_LOG_FREQ_BINS:
    fmin, fmax = FREQ_RANGE
    edges = np.logspace(np.log10(fmin), np.log10(fmax), NUM_LOG_BINS + 1)
    band_pairs = list(zip(edges[:-1], edges[1:]))
    BANDS = {
        f"{np.sqrt(low * high):.1f} Hz": (float(low), float(high))
        for low, high in band_pairs
    }
else:
    BANDS = {
        "Fast Delta": (2, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta": (13, 30),
        "Gamma": (30, 80),
    }

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
W_STRUCTURAL = ROOT_DIR / f"W_Sch{ROI}.npy"
OUT_CSV = ROOT_DIR / "c3-tau.csv"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def bandpass_filter(X, fmin, fmax, fs):
    nyq = 0.5 * fs
    sos = butter(4, [fmin / nyq, min(fmax / nyq, 0.99)], btype="band", output="sos")
    return sosfiltfilt(sos, X, axis=1)


def compute_laplacian(W):
    W_pos = np.maximum(W, 0)
    W_sym = 0.5 * (W_pos + W_pos.T)
    degree = np.diag(W_sym.sum(axis=1))
    return degree - W_sym


def damped_oscillator_basis(t, eigenvalue, tau):
    tau = max(float(tau), 1e-5)
    disc = 1 - 4 * tau * eigenvalue
    if disc > 1e-6:
        r1 = (-1 + np.sqrt(disc)) / (2 * tau)
        r2 = (-1 - np.sqrt(disc)) / (2 * tau)
        return np.column_stack([np.exp(r1 * t), np.exp(r2 * t)])
    if disc < -1e-6:
        omega = np.sqrt(-disc) / (2 * tau)
        env = np.exp(-t / (2 * tau))
        return np.column_stack([env * np.cos(omega * t), env * np.sin(omega * t)])
    env = np.exp(-t / (2 * tau))
    return np.column_stack([env, t * env])


def get_robust_mode_coefficients(X, V, R):
    if R is None:
        return V.T @ X
    Z, *_ = np.linalg.lstsq(R @ V, X, rcond=None)
    return Z


def fit_tau(X, laplacian, n_modes, t, R_matrix=None):
    try:
        eigvals, eigvecs = np.linalg.eigh(laplacian)
    except Exception:
        return 0.0

    lam = eigvals[1 : n_modes + 1]
    V = eigvecs[:, 1 : n_modes + 1]
    Z = get_robust_mode_coefficients(X, V, R_matrix)
    mean_var = np.mean(np.var(Z, axis=1)) if Z.size else 0.0
    var_eps = mean_var * 1e-12 if mean_var > 0 else np.finfo(float).eps

    def residuals(tau_arr):
        tau = tau_arr[0]
        res = []
        for lam_val, y in zip(lam, Z):
            M = damped_oscillator_basis(t, lam_val, tau)
            coeffs, *_ = np.linalg.lstsq(M, y, rcond=None)
            y_hat = M @ coeffs
            std = np.std(y)
            if std > np.sqrt(var_eps):
                res.append((y_hat - y) / std)
        return np.concatenate(res) if res else np.zeros(1)

    sol = least_squares(residuals, x0=[0.05], bounds=(0.001, 1.0))
    return float(sol.x[0])


def load_time_vector(meta_path, expected, fs):
    try:
        with h5py.File(str(meta_path), "r") as handle:
            return np.array(handle["time"]).flatten()
    except Exception:
        t_zero = expected // 2
        return (np.arange(expected) - t_zero) / fs


def process_subject(subj, W_template):
    npz_path = DATA_DIR / subj / f"source_parcels_Schaefer{ROI}_{subj}_{LOCK}.npz"
    meta_path = npz_path.parent / "meta.mat"
    if not npz_path.exists():
        raise FileNotFoundError(f"parcel data missing ({npz_path})")

    with np.load(npz_path, allow_pickle=True) as pack:
        X_concat = pack["ts"]
        fs = float(pack["fs"])
        trial_lengths = pack["trial_lengths"].astype(int)
        labels = pack["labels"].astype(int)
        R_roi = pack["R_roi"] if "R_roi" in pack else None

    expected = int(trial_lengths[0])
    if not np.all(trial_lengths == expected):
        raise ValueError(f"Trial lengths differ for subject {subj}.")

    t_full = load_time_vector(meta_path, expected, fs)
    if t_full.shape[0] != expected:
        raise ValueError(f"Time axis mismatch for subject {subj}.")

    mask = (t_full >= TMIN) & (t_full <= TMAX)
    n_parcels = X_concat.shape[0]
    n_trials = len(trial_lengths)
    X_trials = X_concat.reshape(n_parcels, n_trials, expected)
    X_avg = X_trials.mean(axis=1)[:, mask]
    t = t_full[mask]

    if len(labels) < ROI:
        idx = labels - 1
        W_use = W_template[np.ix_(idx, idx)]
    else:
        W_use = W_template

    laplacian = compute_laplacian(W_use)

    subj_tau = {}
    for band, (fmin, fmax) in BANDS.items():
        X_band = bandpass_filter(X_avg, fmin, fmax, fs)
        subj_tau[band] = fit_tau(X_band, laplacian, N_MODES, t, R_roi)

    return subj_tau


def compute_power_law(freqs, tau_vals):
    positive = (freqs > 0) & (tau_vals > 0)
    if positive.sum() < 2:
        return 0.0
    log_f = np.log(freqs[positive])
    log_tau = np.log(tau_vals[positive])
    slope, _ = np.polyfit(log_f, log_tau, 1)
    return float(slope)


def save_tau_csv(path, bands, tau_mean, tau_std, n_subjects, tau_critical, power_exp):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "roi",
            "band",
            "fmin_hz",
            "fmax_hz",
            "freq_center_hz",
            "tau_mean_s",
            "tau_std_s",
            "n_subjects",
            "tau_critical_s",
            "power_exp",
        ])
        for (band, (fmin, fmax)), mean_tau, std_tau in zip(bands.items(), tau_mean, tau_std):
            writer.writerow([
                ROI,
                band,
                fmin,
                fmax,
                np.sqrt(fmin * fmax),
                mean_tau,
                std_tau,
                n_subjects,
                tau_critical,
                power_exp,
            ])


# ---------------------------------------------------------------------
# MAIN ANALYSIS
# ---------------------------------------------------------------------
if not W_STRUCTURAL.exists():
    raise FileNotFoundError(f"Structural matrix missing: {W_STRUCTURAL}")

W_template = np.load(W_STRUCTURAL)
L_template = compute_laplacian(W_template)
eigvals = np.linalg.eigvalsh(L_template)
lambda_2 = eigvals[1]
tau_critical = 1.0 / (4.0 * lambda_2)
print(f"Structural lambda_2 = {lambda_2:.5f}, tau_c = {tau_critical:.4f} s")

results_tau = {band: [] for band in BANDS}
processed_subjects = 0

for subj in SUBJECTS:
    try:
        subj_tau = process_subject(subj, W_template)
        for band in BANDS:
            results_tau[band].append(subj_tau[band])
        processed_subjects += 1
        print(f"Processed {subj}")
    except FileNotFoundError as exc:
        print(f"Skipping {subj}: {exc}")
    except Exception as exc:
        print(f"Subject {subj} failed: {exc}")

if processed_subjects == 0:
    raise RuntimeError("No subjects processed successfully.")

band_labels = list(BANDS.keys())
tau_mean = np.array([np.mean(results_tau[band]) for band in band_labels])
tau_std = np.array([np.std(results_tau[band]) for band in band_labels])
freq_centers = np.array([np.sqrt(fmin * fmax) for fmin, fmax in BANDS.values()])

power_exp = compute_power_law(freq_centers, tau_mean)
print(f"Power-law exponent (tau vs frequency): {power_exp:.3f}")

save_tau_csv(
    path=OUT_CSV,
    bands=BANDS,
    tau_mean=tau_mean,
    tau_std=tau_std,
    n_subjects=processed_subjects,
    tau_critical=tau_critical,
    power_exp=power_exp,
)
print(f"Saved tau summary: {OUT_CSV}")
