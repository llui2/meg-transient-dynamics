"""
=====================================================================
MEG PARCEL RECONSTRUCTION
=====================================================================
Project MEG source-space trials to parcel space and compute
parcel-level resolution matrices for Schaefer parcellations.
"""

import os
import glob
import shutil
import tempfile

import h5py
import numpy as np
import scipy.io as sio
import nibabel as nib
from tqdm import tqdm

from neuromaps.images import dlabel_to_gifti
from neuromaps.transforms import fslr_to_fslr

# ============================================================
# Configuration
# ============================================================

SUBJECTS = [
    '104012','105923','106521','108323','109123','113922','116726','133019','140117','156334',
    '162026','162935','164636','169040','175237','177746','185442','191033','191437','192641',
    '198653','204521','205119','212318','212823','255639','257845','283543','287248','293748',
    '353740','358144','406836','559053','568963','581450','599671','601127','660951','662551',
    '667056','679770','680957','706040','707749','725751','735148','783462',
    '814649','891667', '898176'
]

ROI = 200
LOCK = "TEMG"
TASK = "LH"

PARCELLATION_FILE = (
    f"parcellations/Schaefer2018_{ROI}Parcels_7Networks_order.dlabel.nii"
)

ROOT_DIR = os.path.abspath(".")
OUTPUT_ROOT = os.path.join(ROOT_DIR, "data")

# ============================================================
# Loop over subjects
# ============================================================

for subj in SUBJECTS:

    print(f"\n=== Subject {subj} ===")

    # --------------------------------------------------------
    # Locate input and output paths
    # --------------------------------------------------------

    base_dir = f"../Downloads/{subj}/MEG/Motort/tmegpreproc"
    input_dir = os.path.join(base_dir, f"source_trials_{LOCK}_{TASK}")

    if not os.path.isdir(input_dir):
        print(f"[ERROR] Missing input directory: {input_dir}")
        continue

    output_dir = os.path.join(OUTPUT_ROOT, subj)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir,
        f"source_parcels_Schaefer{ROI}_{subj}_{LOCK}.npz"
    )

    # --------------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------------

    print("[1/6] Load metadata")

    meta_path = os.path.join(input_dir, "meta.mat")
    if not os.path.exists(meta_path):
        print(f"[ERROR] Missing meta.mat for {subj}")
        continue

    try:
        shutil.copy2(meta_path, os.path.join(output_dir, "meta.mat"))
    except Exception as e:
        print(f"[WARNING] Could not copy meta.mat: {e}")

    try:
        with h5py.File(meta_path, "r") as f:
            fs = float(f["fs"][0, 0])
            inside = np.array(f["inside"]).flatten().astype(bool)

            if "R" not in f:
                raise RuntimeError("Resolution matrix R missing")

            R_source = np.array(f["R"]).T

        n_sources = inside.sum()
        if R_source.shape != (n_sources, n_sources):
            raise RuntimeError("Resolution matrix dimension mismatch")

    except Exception as e:
        print(f"[ERROR] Failed to read metadata: {e}")
        continue

    # --------------------------------------------------------
    # 2. Load and resample parcellation
    # --------------------------------------------------------

    print("[2/6] Load and resample parcellation")

    try:
        with tempfile.TemporaryDirectory() as tmp:

            lh32, rh32 = dlabel_to_gifti(PARCELLATION_FILE)

            lh32_path = os.path.join(tmp, "lh32.gii")
            rh32_path = os.path.join(tmp, "rh32.gii")
            nib.save(lh32, lh32_path)
            nib.save(rh32, rh32_path)

            lh4k = fslr_to_fslr(lh32_path, "4k", hemi="L", method="nearest")[0]
            rh4k = fslr_to_fslr(rh32_path, "4k", hemi="R", method="nearest")[0]

            labels_full = np.concatenate(
                [lh4k.darrays[0].data, rh4k.darrays[0].data]
            ).astype(int)

        labels_inside = labels_full[inside]
        parcel_labels = np.unique(labels_inside[labels_inside > 0])
        n_parcels = len(parcel_labels)

        print(f"Detected {n_parcels} parcels")

    except Exception as e:
        print(f"[ERROR] Atlas processing failed: {e}")
        continue

    # --------------------------------------------------------
    # 3. Construct parcel projection operators
    # --------------------------------------------------------

    print("[3/6] Construct parcel projection operators")

    P = np.zeros((n_parcels, n_sources), dtype=np.float32)
    U = np.zeros((n_sources, n_parcels), dtype=np.float32)

    for j, lab in enumerate(parcel_labels):
        idx = np.where(labels_inside == lab)[0]
        if idx.size == 0:
            continue
        P[j, idx] = 1.0 / idx.size
        U[idx, j] = 1.0

    # --------------------------------------------------------
    # 4. Compute parcel-space resolution matrix
    # --------------------------------------------------------

    print("[4/6] Compute parcel resolution matrix")

    R_roi = P @ (R_source @ U)
    del R_source

    # --------------------------------------------------------
    # 5. Project trials to parcel space
    # --------------------------------------------------------

    print("[5/6] Project trials")

    trial_files = sorted(glob.glob(os.path.join(input_dir, "source_trial_*.mat")))
    parcel_trials = []
    trial_lengths = []

    for fpath in tqdm(trial_files, desc=f"Trials {subj}"):
        try:
            trial = sio.loadmat(fpath)["SourceData"]

            if trial.shape[0] != n_sources:
                if trial.shape[1] == n_sources:
                    trial = trial.T
                else:
                    raise RuntimeError("Invalid trial dimensions")

            parcel_trials.append(P @ trial)
            trial_lengths.append(trial.shape[1])

        except Exception as e:
            print(f"[ERROR] Trial failed: {fpath}: {e}")

    if parcel_trials:
        X = np.concatenate(parcel_trials, axis=1)
    else:
        X = np.empty((n_parcels, 0), dtype=np.float32)

    # --------------------------------------------------------
    # 6. Save output
    # --------------------------------------------------------

    print("[6/6] Save output")

    try:
        np.savez_compressed(
            output_file,
            ts=X,
            R_roi=R_roi,
            fs=fs,
            labels=parcel_labels,
            trial_lengths=trial_lengths,
        )
        print(f"Saved: {output_file}")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")

print("\nAll subjects processed.")
