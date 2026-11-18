"""Shared experimental settings: cohort, frequency bands and graph Laplacians."""
from pathlib import Path
import os
from functools import lru_cache
import nibabel as nib
import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
BANDS = {"Fast Delta": (2, 4), "Theta": (4, 8), "Alpha": (8, 13),
         "Beta": (13, 30), "Gamma": (30, 80)}
NETWORKS = {"Structural": "W_Sch", "EDR": "W_EDR_Sch",
            "Geometric": "W_Geo_Sch"}
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "data"
DATASET = "combined"
SIGNALS = OUTPUT / "signals"
RESULTS = OUTPUT / "results"
CACHE = Path(os.environ.get("MEG_CACHE", ROOT / ".cache")).resolve()
RECONSTRUCTED = CACHE / "reconstruction"
ROIS = (100, 200, 400)
SIGNAL_FIELDS = ("subject", "condition", "time", "fs", "runs", "trials_per_run", "n_trials", "protocol_id")
SUBJECTS = ("104012", "105923", "106521", "108323", "109123", "113922", "116726", "133019", "140117", "156334", "162026", "162935", "164636", "169040", "175237", "177746", "185442", "191033", "191437", "192641", "198653", "204521", "205119", "212318", "212823", "255639", "257845", "283543", "287248", "293748", "353740", "358144", "406836", "559053", "568963", "581450", "599671", "601127", "660951", "662551", "667056", "679770", "680957", "706040", "707749", "725751", "735148", "783462", "814649", "891667", "898176")


def subjects(condition, roi):
    """The same 51 participants contribute to every condition and resolution."""
    return list(SUBJECTS)


def recording_count(roi=100):
    return sum(len(subjects(c, roi)) for c in ("LH", "RH", "LF", "RF"))


def load_reconstruction(subject, condition, roi=100, dataset='combined', trials=False):
    """Load pooled means or a single run; optional trials follow run 10 then 11."""
    import json
    if dataset not in ('combined', 'run10', 'run11'):
        raise ValueError('dataset must be combined, run10 or run11')
    manifest = RECONSTRUCTED / 'complete' / f'{subject}_{condition}.json'
    if not manifest.is_file():
        raise FileNotFoundError(f'Incomplete reconstruction: {subject}/{condition}; run ./run.sh reconstruct')
    record = json.loads(manifest.read_text())
    path = RECONSTRUCTED / dataset / 'parcels' / f'{subject}_{condition}{roi}.npz'
    if roi not in record['rois'] or str(path.relative_to(RECONSTRUCTED)) not in {
            item['path'] for item in record['files']}:
        raise ValueError(f'Reconstruction is not marked complete: {path}')
    with np.load(path) as pack:
        raw = {key: pack[key] for key in pack.files if key != 'trials'}
        if str(raw['protocol_id']) != record['protocol_id']:
            raise ValueError(f'Reconstruction provenance mismatch: {path}')
        if trials and dataset != 'combined':
            raw['trials'] = pack['trials']
    if trials and dataset == 'combined':
        runs = [load_reconstruction(subject, condition, roi, f'run{run}', trials=True)
                for run in (10, 11)]
        raw['trials'] = np.concatenate([r['trials'] for r in runs])
    return raw

def operators(roi, labels):
    idx = np.asarray(labels) - 1
    result = {}
    for name, stem in NETWORKS.items():
        weights = np.load(ASSETS / f"{stem}{roi}.npy")
        weights = weights[np.ix_(idx, idx)]
        weights = np.maximum(weights, 0)
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 0)
        result[name] = np.diag(weights.sum(axis=1)) - weights
    return result


@lru_cache(maxsize=3)
def cortex(roi):
    atlas = nib.load(ASSETS / "parcellations" /
                     f"Schaefer2018_{roi}Parcels_7Networks_order.dlabel.nii")
    values = atlas.get_fdata().ravel().astype(int)
    axis = atlas.header.get_axis(1)
    points, triangles, labels = [], [], []
    for name, selection, model in axis.iter_structures():
        hemi = "L" if name.endswith("LEFT") else "R"
        mesh = nib.load(ASSETS / "plot_aux_data" / f"fs_LR.32k.{hemi}.pial.surf.gii")
        coords, faces = (array.data for array in mesh.darrays)
        expanded = np.zeros(len(coords), dtype=int)
        expanded[model.vertex] = values[selection]
        triangles.append(faces + sum(len(p) for p in points))
        points.append(coords)
        labels.append(expanded)
    return np.vstack(points), np.vstack(triangles), np.concatenate(labels)


def load_parcels(subject, condition, roi=100):
    """Read one resolution from a public pooled-response file."""
    path = SIGNALS / f"{subject}_{condition}.npz"
    with np.load(path) as pack:
        raw = {key: pack[key] for key in SIGNAL_FIELDS}
        raw.update({key: pack[f"{key}_{roi}"] for key in ("signal", "labels", "resolution")})
    return raw


def export_signal(subject, condition, rois=ROIS):
    """Export the pooled arrays once; retain their original dtypes and precision."""
    arrays = {}
    for roi in rois:
        raw = load_reconstruction(subject, condition, roi)
        if not arrays:
            arrays.update({key: raw[key] for key in SIGNAL_FIELDS})
        else:
            for key in SIGNAL_FIELDS:
                if not np.array_equal(arrays[key], raw[key]):
                    raise ValueError(f"Unaligned resolutions: {subject}/{condition}/{key}")
        arrays.update({f"{key}_{roi}": raw[key] for key in ("signal", "labels", "resolution")})
    SIGNALS.mkdir(parents=True, exist_ok=True)
    path = SIGNALS / f"{subject}_{condition}.npz"
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    return path
