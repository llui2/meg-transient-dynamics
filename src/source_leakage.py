"""Reproduce the pooled source-resolution illustration for Figure S1."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.io import loadmat, savemat
import data


def summarize(meta):
    paths = [data.RESULTS / name for name in
             ("source_leakage.csv", "source_reference.csv", "source_leakage_input.csv")]
    import h5py  # Only needed to read the archived MATLAB input, not to plot CSVs.

    counts = np.zeros(159, dtype=np.int64)
    totals, squares = np.zeros(159), np.zeros(159)
    with h5py.File(meta, "r") as source:
        matrix = source["R"]
        positions = np.asarray(source["check/pos"], dtype=float).T * 10
        n = len(positions)
        if (matrix.shape != (n, n) or positions.shape != (n, 3)
                or n <= 7700 or not np.isfinite(positions).all()):
            raise ValueError("Expected matching source R and check/pos (cm), including index 7700")
        reference = np.empty(n)
        # MATLAB stores compressed columns. Stream these without allocating n² distances.
        for start in range(0, n, 256):
            stop = min(start + 256, n)
            leakage = np.abs(np.asarray(matrix[:, start:stop], dtype=float))
            if not np.isfinite(leakage).all():
                raise ValueError("Nonfinite source-resolution entries")
            reference[start:stop] = leakage[7700]
            distance = cdist(positions, positions[start:stop])
            distance[np.arange(start, stop), np.arange(stop - start)] = np.inf
            keep = distance < 159  # Original bins: [0,1), …, [158,159) mm.
            bins, values = distance[keep].astype(int), leakage[keep]
            counts += np.bincount(bins, minlength=159)
            totals += np.bincount(bins, weights=values, minlength=159)
            squares += np.bincount(bins, weights=values ** 2, minlength=159)
    keep = counts > 100
    mean = totals[keep] / counts[keep]
    sd = np.sqrt(np.maximum(squares[keep] / counts[keep] - mean ** 2, 0))
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(distance_mm=(np.arange(159) + 0.5)[keep],
                      count=counts[keep], mean_abs_R=mean, std_abs_R=sd)).to_csv(paths[0], index=False)
    pd.DataFrame(dict(source_index=np.arange(n), x_mm=positions[:, 0],
                      y_mm=positions[:, 1], z_mm=positions[:, 2],
                      abs_R_reference=reference)).to_csv(paths[1], index=False)
    with meta.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    pd.DataFrame([dict(subject="104012", condition="LH", n_sources=n,
                       reference_index=7700, input_file=os.path.relpath(meta.resolve(), data.ROOT),
                       dataset=data.DATASET, runs="10,11",
                       input_sha256=digest, reference_slice="HDF5 R[7700, :]")]).to_csv(paths[2], index=False)


def main():
    if data.DATASET != 'combined':
        raise ValueError('This diagnostic uses the combined-run reconstructions')
    first = data.load_reconstruction('104012','LH',dataset='run10')
    second = data.load_reconstruction('104012','LH',dataset='run11')
    combined = data.load_reconstruction('104012','LH')
    indices = first['inside_indices'].ravel().astype(int)
    labels = loadmat(data.RECONSTRUCTED/'atlas.mat')['labels100'].ravel()[indices-1]
    projection = np.array([(labels==i)/(labels==i).sum() for i in range(1,101)])
    uniform = np.array([labels==i for i in range(1,101)],dtype=float).T
    values = {k:first[k] for k in ('inside_indices','source_orientation','leadfield_scale')}
    values.update(P=projection, U=uniform, W10=first['parcel_filter'],
                  W11=second['parcel_filter'], L10=first['parcel_leadfield'],
                  L11=second['parcel_leadfield'], counts=combined['trials_per_run'],
                  resolution=combined['resolution'])
    folder = data.CACHE/'source-leakage'
    folder.mkdir(parents=True, exist_ok=True)
    reference, output = folder/'source_reference.mat', folder/'source_resolution.mat'
    savemat(reference, values)
    source = data.SRC/'source_leakage_combined.m'
    protocol = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    reconstruction_protocol=str(combined['protocol_id']),
                    subject='104012', condition='LH', runs=[10,11])
    stamp = folder/'protocol.json'
    if output.exists() and stamp.exists() and json.loads(stamp.read_text()) != protocol:
        raise ValueError('Source diagnostic inputs changed; use a fresh output version')
    if not output.exists() or not stamp.exists():
        downloads = Path(os.getenv('MEG_DOWNLOADS',Path.home()/'Downloads'))
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        os.environ.setdefault('FIELDTRIP_HOME', str(data.ROOT/'fieldtrip-20251103'))
        expression = 'addpath('+quote(data.SRC)+'); source_leakage_combined('+','.join(map(quote,(downloads,reference,output)))+');'
        log = folder/'matlab.log'
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as handle:
            subprocess.run([os.getenv('MATLAB','matlab'),'-singleCompThread','-batch',expression],
                           cwd=data.ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
        stamp.write_text(json.dumps(protocol,indent=2)+'\n')
    summarize(output)
    subprocess.run([sys.executable,'-m','plots.source_leakage'],
                   cwd=data.ROOT,check=True)


if __name__ == '__main__':
    main()
