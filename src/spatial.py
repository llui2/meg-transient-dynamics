"""Calculate spatial-mode coverage and its contrast with dynamical fits (S9)."""
import hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

import data
from fit import BANDS
from model import graph_modes

MODELS = tuple(data.NETWORKS)
CONDITIONS = ("LH", "RH", "LF", "RF")
K = np.arange(1, 100)
OUTPUT = data.RESULTS / "spatial_decomposition.csv"


def spatial_scores(target, mixing):
    """Best parcel-space R² for each nested spatial subspace."""
    q, r = np.linalg.qr(mixing, mode="reduced")
    diagonal = np.abs(np.diag(r))
    if diagonal.min() < diagonal.max() * 1e-10:
        raise ValueError("Ill-conditioned mixing matrix; inspect parcel resolution")
    explained = np.cumsum(np.sum((q.T @ target) ** 2, axis=1))
    total = np.sum(target ** 2)
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2)
    if sst <= 0:
        raise ValueError("Target has zero temporal variance")
    scores = 1 - (total - explained) / sst
    if np.any(np.diff(scores) < -1e-9) or scores[-1] > 1 + 1e-8:
        raise ValueError("Invalid cumulative spatial projection")
    return scores


def run():
    subjects = list(data.SUBJECTS)
    reference = pd.read_csv(data.RESULTS / "modes.csv", dtype={"subject": str})
    reference = reference[(reference.roi == 100)
                          & reference.subject.isin(subjects)
                          & reference.band.isin(BANDS)]
    keys = ["subject", "condition", "band", "model", "modes"]
    if reference.duplicated(keys).any():
        raise ValueError("Duplicate original mode fits")
    reference = reference.set_index(keys).sort_index()

    dynamic = np.zeros((len(subjects), 2, 3, len(K)))
    spatial = np.zeros_like(dynamic)
    for si, subject in enumerate(tqdm(subjects, desc="Spatial decomposition", unit="subject")):
        counts = np.zeros(2)
        for condition in CONDITIONS:
            if subject not in data.subjects(condition, 100):
                continue
            path = data.SIGNALS / f"{subject}_{condition}.npz"
            cache = data.load_parcels(subject, condition)
            signal, time, fs = cache["signal"], cache["time"], float(cache["fs"])
            labels, resolution = cache["labels"].ravel(), cache["resolution"]
            if (signal.shape != (100, len(time)) or resolution.shape != (100, 100)
                    or not np.array_equal(np.sort(labels), np.arange(1, 101))):
                raise ValueError(f"Invalid cached parcel data: {path}")
            mask = (time >= -0.1) & (time <= 0.9)
            graphs = data.operators(100, labels)
            bases = {}
            for model, graph in graphs.items():
                _, vec = graph_modes(graph)
                bases[model] = resolution @ vec

            for bi, (band, (low, high)) in enumerate(BANDS.items()):
                sos = butter(4, [low / (fs / 2), min(high / (fs / 2), .99)],
                             btype="band", output="sos")
                target = sosfiltfilt(sos, signal, axis=1)[:, mask]
                digest = hashlib.sha256(target.tobytes()).hexdigest()
                for mi, model in enumerate(MODELS):
                    try:
                        fitted = reference.loc[(subject, condition, band, model)]
                    except KeyError as exc:
                        raise ValueError(f"Missing mode fits: {subject}/{condition}/{band}/{model}") from exc
                    if (len(fitted) != len(K) or not np.array_equal(fitted.index, K)
                            or not fitted.target_hash.eq(digest).all()):
                        raise ValueError(f"Unaligned mode fits: {subject}/{condition}/{band}/{model}")
                    unconstrained = spatial_scores(target, bases[model])
                    constrained = fitted.r2.to_numpy()
                    if np.any(constrained > unconstrained + 1e-5):
                        raise AssertionError(
                            f"Telegrapher fit exceeds unrestricted spatial fit: "
                            f"{subject}/{condition}/{band}/{model}")
                    dynamic[si, bi, mi] += constrained
                    spatial[si, bi, mi] += unconstrained
                counts[bi] += 1
        if np.any(counts == 0):
            raise ValueError(f"No valid conditions: {subject}")
        dynamic[si] /= counts[:, None, None]
        spatial[si] /= counts[:, None, None]

    # At K=99, explicit removal of only the global constant gives the same space.
    # Therefore the unrestricted spatial R² must agree across graph models.
    spread = np.max(np.ptp(spatial[:, :, :, -1], axis=2))
    if spread > 1e-6:
        raise AssertionError(f"Full-mode spatial subspaces differ: {spread:.3g}")
    print(f"Maximum K=99 spatial-score difference: {spread:.2g}")

    rows = []
    for si, subject in enumerate(subjects):
        for bi, band in enumerate(BANDS):
            for mi, control in enumerate(MODELS[1:], start=1):
                delta_dynamic = dynamic[si, bi, 0] - dynamic[si, bi, mi]
                delta_spatial = spatial[si, bi, 0] - spatial[si, bi, mi]
                structural_penalty = spatial[si, bi, 0] - dynamic[si, bi, 0]
                control_penalty = spatial[si, bi, mi] - dynamic[si, bi, mi]
                excess = control_penalty - structural_penalty
                if (np.min(structural_penalty) < -1e-5
                        or np.min(control_penalty) < -1e-5
                        or not np.allclose(excess, delta_dynamic - delta_spatial,
                                            atol=1e-10)):
                    raise AssertionError("Invalid dynamical penalty decomposition")
                for ki, k in enumerate(K):
                    rows.append((subject, band, control, int(k),
                                 delta_dynamic[ki], delta_spatial[ki],
                                 excess[ki], structural_penalty[ki], control_penalty[ki]))
    table = pd.DataFrame(rows, columns=[
        "subject", "band", "control", "modes", "dynamic_advantage",
        "spatial_advantage", "dynamical_excess",
        "structural_temporal_penalty", "control_temporal_penalty"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    run()
