"""Modal, parcel-space, frequency-resolved and cumulative-mode fitting."""
import hashlib
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt

import data
from model import fit, graph_modes, coefficient_modes, temporal_basis

BANDS = {"Fast Delta": data.BANDS["Fast Delta"], "Theta": data.BANDS["Theta"]}
# The overlapping 0.5–4 Hz sensitivity band is excluded from Figure 3.
BINS = [(0.5, 4), (0.5, 2), (2, 4), (4, 6), (6, 9), (9, 13),
        (13, 18), (18, 25), (25, 35), (35, 50), (50, 65), (65, 80)]
# Match the nonoverlapping intervals plotted for the regional fits.
MODAL_BANDS = {f"{low:g}–{high:g} Hz": (low, high)
               for low, high in BINS if (low, high) != (0.5, 4)}
TAU_MIN, TAU_MAX = 0.001, 10.0
SPECTRUM_BANDS = {"2–4 Hz": (2, 4), "4–6 Hz": (4, 6),
                  "6–9 Hz": (6, 9), "9–13 Hz": (9, 13)}
SPECTRUM_OUTPUT = data.RESULTS / "modes_spectrum.csv"


def calculate(average, time, fs, labels, resolution, condition, roi, subject,
              bands=None):
    """Fit one subject/condition/ROI from its trial-averaged parcel signals."""
    graphs = data.operators(roi, labels)
    mask = (time >= -.1) & (time <= .9)
    graph_metadata = {name: graph_modes(operator)[0] for name, operator in graphs.items()}
    rows = []
    for band in MODAL_BANDS if bands is None else bands:
        low, high = MODAL_BANDS[band]
        sos = butter(4, (low, high), btype="band", fs=fs, output="sos")
        signal = sosfiltfilt(sos, average, axis=1)[:, mask]
        target_hash = hashlib.sha256(signal.tobytes()).hexdigest()
        for name, laplacian in graphs.items():
            r2, tau = fit(signal, time[mask], laplacian, resolution)
            rows.append(dict(roi=roi, condition=condition, subject=subject,
                             band=band, model=name, r2=r2, tau=tau,
                             order=4, fmin=low, fmax=high, modes=roi-1,
                             tau_min=.01, tau_max=1., target_hash=target_hash,
                             coefficient_count=len(coefficient_modes(graph_metadata[name])),
                             tau_identified=bool(np.any(graph_metadata[name] > 0))))
    return rows


def fit_parcels(signal, time, eigenvalues, mixing):
    """Optimize shared tau and modal amplitudes against the same observed X."""
    mixing = mixing / np.linalg.norm(mixing, axis=0)
    column_modes = coefficient_modes(eigenvalues)
    spatial = (mixing.T @ mixing)[np.ix_(column_modes, column_modes)]
    observed = signal / np.linalg.norm(signal)
    projected = (mixing.T @ observed)[column_modes]
    sst = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2)

    def evaluate(log_tau):
        functions = temporal_basis(time, eigenvalues, np.exp(log_tau))
        gram = spatial * (functions.T @ functions)
        rhs = np.sum(projected * functions.T, axis=1)
        try:
            factor = cholesky(gram, lower=True, check_finite=False)
            projected_fit = solve_triangular(factor, rhs, lower=True, check_finite=False)
            return 1 - (1 - projected_fit @ projected_fit) / sst
        except np.linalg.LinAlgError:
            return -np.inf

    if not np.any(np.asarray(eigenvalues) > 0):
        return float(evaluate(np.log(.01))), np.nan

    grid = np.linspace(np.log(.01), 0, 129)
    values = np.array([evaluate(t) for t in grid])
    candidates = [(values[i], grid[i]) for i in range(len(grid))]
    peaks = [i for i in range(1, len(grid) - 1)
             if values[i] >= values[i - 1] and values[i] >= values[i + 1]]
    for i in sorted(peaks, key=lambda k: values[k], reverse=True)[:4]:
        result = minimize_scalar(lambda t: -evaluate(t),
                                 bounds=(grid[i - 1], grid[i + 1]),
                                 method="bounded", options={"xatol": 1e-8})
        if not result.success:
            raise RuntimeError(result.message)
        candidates.append((-result.fun, result.x))
    r2, log_tau = max(candidates)
    if not np.isfinite(r2) or r2 > 1 + 1e-7:
        raise ValueError("Invalid parcel-space fit")
    return float(r2), float(np.exp(log_tau))


def fit_wide(signal, time, eigenvalues, mixing, tau_min=0.01, tau_max=10.0):
    """Parcel-space objective with a configurable relaxation-time interval."""
    if not 0 < tau_min < tau_max:
        raise ValueError("Require 0 < tau_min < tau_max")
    mixing = mixing / np.linalg.norm(mixing, axis=0)
    column_modes = coefficient_modes(eigenvalues)
    spatial = (mixing.T @ mixing)[np.ix_(column_modes, column_modes)]
    observed = signal / np.linalg.norm(signal)
    projected = (mixing.T @ observed)[column_modes]
    sst = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2)

    def evaluate(log_tau):
        functions = temporal_basis(time, eigenvalues, np.exp(log_tau))
        gram = spatial * (functions.T @ functions)
        rhs = np.sum(projected * functions.T, axis=1)
        try:
            factor = cholesky(gram, lower=True, check_finite=False)
            fitted = solve_triangular(factor, rhs, lower=True, check_finite=False)
            return 1 - (1 - fitted @ fitted) / sst
        except np.linalg.LinAlgError:
            return -np.inf

    if not np.any(np.asarray(eigenvalues) > 0):
        return float(evaluate(np.log(tau_min))), np.nan

    # Keep the original grid density (64 points per log decade).
    n_grid = int(round(64 * np.log10(tau_max / tau_min))) + 1
    grid = np.linspace(np.log(tau_min), np.log(tau_max), n_grid)
    scores = np.array([evaluate(t) for t in grid])
    candidates = [(scores[i], grid[i]) for i in range(len(grid))]
    peaks = [i for i in range(1, len(grid) - 1)
             if scores[i] >= scores[i - 1] and scores[i] >= scores[i + 1]]
    for i in sorted(peaks, key=lambda k: scores[k], reverse=True)[:4]:
        result = minimize_scalar(lambda t: -evaluate(t),
                                 bounds=(grid[i - 1], grid[i + 1]),
                                 method="bounded", options={"xatol": 1e-8})
        if not result.success:
            raise RuntimeError(result.message)
        candidates.append((-result.fun, result.x))
    r2, log_tau = max(candidates)
    if not np.isfinite(r2) or r2 > 1 + 1e-7:
        raise ValueError("Invalid parcel-space fit")
    return float(r2), float(np.exp(log_tau))


def fit_curve(signal, time, eigenvalues, mixing, full_tau, tau_min=.01, tau_max=1.):
    """At a fixed tau, one Cholesky fit gives the optimum at every nested K."""
    if (not 0 < tau_min < tau_max or not np.isfinite(full_tau)
            or not tau_min * (1 - 1e-12) <= full_tau <= tau_max * (1 + 1e-12)):
        raise ValueError("Require positive tau bounds containing the full-model tau")
    full_tau = float(np.clip(full_tau, tau_min, tau_max))
    n = len(eigenvalues)
    mixing = mixing / np.linalg.norm(mixing, axis=0)
    column_modes = coefficient_modes(eigenvalues)
    column_counts = np.bincount(column_modes, minlength=n)
    endpoints = np.cumsum(column_counts) - 1
    spatial = (mixing.T @ mixing)[np.ix_(column_modes, column_modes)]
    observed = signal / np.linalg.norm(signal)
    projections = (mixing.T @ observed)[column_modes]
    sst = np.sum((observed - observed.mean(axis=1, keepdims=True)) ** 2)
    memo = {}

    def evaluate(log_tau):
        log_tau = float(log_tau)
        if log_tau in memo:
            return memo[log_tau]
        temporal = temporal_basis(time, eigenvalues, np.exp(log_tau))
        gram = spatial * (temporal.T @ temporal)
        rhs = np.sum(projections * temporal.T, axis=1)
        try:
            factor = cholesky(gram, lower=True, check_finite=False)
            coefficients = solve_triangular(factor, rhs, lower=True, check_finite=False)
            explained = np.cumsum(coefficients ** 2)[endpoints]
            score = 1 - (1 - explained) / sst
        except np.linalg.LinAlgError:
            score = np.full(n, -np.inf)
        memo[log_tau] = score
        return score

    n_grid = int(round(64 * np.log10(tau_max / tau_min))) + 1
    grid = np.linspace(np.log(tau_min), np.log(tau_max), n_grid)
    scan = np.array([evaluate(x) for x in grid])
    scores = np.empty(n)
    taus = np.empty(n)
    for k in range(n):
        if not np.any(np.asarray(eigenvalues)[:k + 1] > 0):
            scores[k], taus[k] = scan[0, k], np.nan
            if not np.isfinite(scores[k]) or scores[k] > 1 + 1e-7:
                raise ValueError(f"Invalid static parcel fit at K={k + 1}")
            continue
        column = scan[:, k]
        start = int(np.argmax(column))
        candidates = [(column[start], grid[start]),
                      (evaluate(np.log(full_tau))[k], np.log(full_tau))]
        if k and np.isfinite(taus[k - 1]):
            previous = np.log(taus[k - 1])
            candidates.append((evaluate(previous)[k], previous))
        peaks = [i for i in range(1, len(grid) - 1)
                 if column[i] >= column[i - 1] and column[i] >= column[i + 1]]
        for i in sorted(peaks, key=lambda j: column[j], reverse=True)[:4]:
            result = minimize_scalar(lambda x: -evaluate(x)[k],
                                     bounds=(grid[i - 1], grid[i + 1]),
                                     method="bounded", options={"xatol": 1e-8})
            if not result.success:
                raise RuntimeError(result.message)
            candidates.append((-result.fun, result.x))
        score, tau = max(candidates)
        if not np.isfinite(score) or score > 1 + 1e-7:
            raise ValueError(f"Invalid parcel fit at K={k + 1}")
        if k and score < scores[k - 1] - 1e-8:
            raise AssertionError(f"Nested fits deteriorate at K={k + 1}")
        scores[k], taus[k] = score, np.clip(np.exp(tau), tau_min, tau_max)
    return scores, taus
