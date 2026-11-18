"""Prepare both motor runs and their trial-weighted mean, without model fitting.

Run ./run.sh reconstruct to resume all participants, conditions and resolutions.
Separate run archives retain individual trials; combined archives store means
and references to those trials, avoiding a second copy of the trial payload.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
import traceback

import numpy as np
import pandas as pd
from scipy.io import loadmat

import data

ROOT = data.ROOT
OUTPUT = data.RECONSTRUCTED
CONDITIONS = ('LH', 'RH', 'LF', 'RF')
ROIS = (100, 200, 400)
RUNS = (10, 11)
SCHEMA = 2


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def input_paths(subject, downloads):
    base = downloads / subject / 'MEG'
    return [base / 'Motort/tmegpreproc' /
            f'{subject}_MEG_{run}-Motort_tmegpreproc_TEMG.mat' for run in RUNS] + [
                base / 'anatomy' / f'{subject}_MEG_anatomy_{kind}.mat'
                for kind in ('headmodel', 'sourcemodel_2d')]


def identity(path):
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def input_manifest(subject, downloads):
    path = OUTPUT / 'inputs' / f'{subject}.json'
    old = json.loads(path.read_text()) if path.exists() else []
    records = []
    for source in input_paths(subject, downloads):
        record = identity(source)
        match = next((r for r in old if all(r.get(k) == v for k, v in record.items())), None)
        record['sha256'] = match['sha256'] if match else sha256(source)
        if identity(source) != {k: record[k] for k in ('path', 'bytes', 'mtime_ns')}:
            raise RuntimeError(f'Input changed while hashing: {source}')
        records.append(record)
    atomic_json(path, records)
    return records


def check_inputs(records):
    for record in records:
        if identity(Path(record['path'])) != {k: record[k] for k in ('path', 'bytes', 'mtime_ns')}:
            raise RuntimeError(f'Input changed during reconstruction: {record["path"]}')


def trial_table(subject, run, downloads):
    path = downloads / subject / 'MEG/Motort/tmegpreproc' / (
        f'{subject}_MEG_{run}-Motort_tmegpreproc_trialinfo.mat')
    info = loadmat(path, simplify_cells=True)['trlInfo']
    return info['lockTrl'][list(info['lockNames']).index('TEMG')]


def check_trial_identity(raw, table, condition):
    code = dict(LH=1, LF=2, RH=4, RF=5)[condition]
    indices = np.flatnonzero(table[:, 1] == code) + 1
    if (not np.array_equal(raw['trial_indices'], indices)
            or not np.array_equal(raw['trialinfo'], table[indices - 1], equal_nan=True)):
        raise ValueError('Reconstructed trial identities differ from HCP sidecar metadata')


def validate_run(raw, roi, run):
    n = int(raw['n_trials'])
    signal, trials, times = raw['signal'], raw['trials'], raw['time']
    if (int(raw['run']) != run or n < 1 or trials.shape != (n, roi, len(times))
            or signal.shape != (roi, len(times)) or raw['resolution'].shape != (roi, roi)
            or not np.array_equal(raw['labels'], np.arange(1, roi + 1))
            or not np.all(np.diff(times) > 0) or float(raw['fs']) <= 0
            or raw['trial_indices'].shape != (n,)
            or len(np.unique(raw['trial_indices'])) != n or raw['trialinfo'].shape[0] != n):
        raise ValueError('Invalid run identity, dimensions, labels or trial IDs')
    for key in ('signal', 'trials', 'time', 'resolution', 'source_orientation',
                'source_positions', 'parcel_filter', 'parcel_leadfield'):
        if not np.isfinite(raw[key]).all():
            raise ValueError(f'Nonfinite {key}')
    if not np.allclose(np.linalg.norm(raw['source_orientation'], axis=0), 1, atol=1e-8):
        raise ValueError('Source orientations are not unit vectors')
    np.testing.assert_allclose(signal, trials.mean(axis=0, dtype=np.float64), rtol=1e-12, atol=0)
    np.testing.assert_allclose(raw['resolution'], raw['parcel_filter'] @ raw['parcel_leadfield'],
                               rtol=1e-10, atol=1e-12)
    if not float(raw['unit_gain_error']) < 1e-5:
        raise ValueError('Unit-gain constraint failed')


def read_mat(path, roi, run):
    raw = loadmat(path)
    result = dict(
        signal=raw['X_avg'], trials=raw['X_trials'].transpose(2, 0, 1),
        resolution=raw['R_roi'], parcel_filter=raw['W_roi'], parcel_leadfield=raw['L_roi'],
        time=raw['time'].ravel(), fs=float(raw['fs'].item()),
        labels=raw['parcel_labels'].astype(int).ravel(), run=int(raw['run'].item()),
        n_trials=int(raw['n_trials'].item()), trial_indices=raw['trial_indices'].astype(int).ravel(),
        trialinfo=raw['trialinfo'], source_orientation=raw['source_orientation'],
        source_positions=raw['source_positions'], inside_indices=raw['inside_idx'].astype(int).ravel(),
        leadfield_scale=raw['leadfield_scale'].ravel(),
        channel_labels=np.array([str(x.item()) for x in raw['channel_labels'].ravel()]),
        unit_gain_error=float(raw['unit_gain_error'].item()))
    validate_run(result, roi, run)
    return result


def combine_runs(first, second):
    """Pool trials and their observation operators in a common source basis."""
    if (int(first['run']), int(second['run'])) != RUNS:
        raise ValueError('Combining requires run 10 followed by run 11')
    for key in ('labels', 'time', 'source_orientation', 'source_positions', 'inside_indices', 'leadfield_scale'):
        if not np.array_equal(first[key], second[key]):
            raise ValueError(f'Cannot combine different {key}')
    if float(first['fs']) != float(second['fs']):
        raise ValueError('Sampling rates differ')
    counts = np.array([first['n_trials'], second['n_trials']], dtype=int)
    if (counts <= 0).any():
        raise ValueError('Both runs need retained trials')
    weights = counts / counts.sum()
    result = {key: first[key] for key in ('labels', 'time', 'fs', 'source_orientation',
                                         'source_positions', 'inside_indices', 'leadfield_scale')}
    result.update(signal=weights[0] * first['signal'] + weights[1] * second['signal'],
                  resolution=weights[0] * first['resolution'] + weights[1] * second['resolution'],
                  runs=np.array(RUNS), trials_per_run=counts, n_trials=int(counts.sum()),
                  trial_runs=np.repeat(RUNS, counts),
                  trial_indices=np.concatenate([first['trial_indices'], second['trial_indices']]),
                  trialinfo=np.concatenate([first['trialinfo'], second['trialinfo']]))
    return result


def completed_path(subject, condition):
    return OUTPUT / 'complete' / f'{subject}_{condition}.json'


def complete(subject, condition, rois, protocol_id, inputs, verify=False):
    path = completed_path(subject, condition)
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if (record['protocol_id'] != protocol_id or record['inputs'] != inputs
            or not set(rois).issubset(record['rois'])):
        return False
    for item in record['files']:
        output = OUTPUT / item['path']
        if not output.exists() or output.stat().st_size != item['bytes']:
            return False
        if verify and sha256(output) != item['sha256']:
            return False
    return len(record['files']) == 3 * len(record['rois'])


def reconstruct(job):
    subject, condition, rois, atlas, downloads, matlab, protocol_id, inputs = job
    check_inputs(inputs)
    if complete(subject, condition, rois, protocol_id, inputs):
        data.export_signal(subject, condition, rois)
        return subject, condition, 'cached', 0
    if shutil.disk_usage(OUTPUT).free < 4 * 1024**3:
        raise RuntimeError('Less than 4 GiB free; refusing to start another reconstruction')
    # An interrupted replacement must never expose a mixture of old/new runs.
    completed_path(subject, condition).unlink(missing_ok=True)
    start = time.monotonic()
    logs = OUTPUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f'{subject}_{condition}.log'
    try:
        with TemporaryDirectory(prefix=f'meg_{subject}_{condition}_', dir=OUTPUT / 'temporary') as work:
            quote = lambda x: "'" + str(x).replace("'", "''") + "'"
            expression = ('addpath(' + quote(data.SRC) + '); beamform_runs('
                          + ','.join(quote(x) for x in (subject, condition, work, atlas, downloads))
                          + ',[' + ' '.join(map(str, rois)) + ']);')
            with log_path.open('w') as log:
                subprocess.run([matlab, '-singleCompThread', '-batch', expression],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            check_inputs(inputs)
            files, rows = [], []
            for roi in rois:
                runs = [read_mat(Path(work) / f'run{run}_parcel_{roi}.mat', roi, run) for run in RUNS]
                for run, raw in zip(RUNS, runs):
                    check_trial_identity(raw, trial_table(subject, run, downloads), condition)
                pooled = combine_runs(*runs)
                for dataset, raw in zip(('run10', 'run11', 'combined'), [*runs, pooled]):
                    path = OUTPUT / dataset / 'parcels' / f'{subject}_{condition}{roi}.npz'
                    raw.update(subject=subject, condition=condition, roi=roi,
                               protocol_id=protocol_id, orientation_reference_run=10)
                    if dataset == 'combined':
                        raw['trial_files'] = np.array([
                            f'run{run}/parcels/{subject}_{condition}{roi}.npz' for run in RUNS])
                    atomic_npz(path, raw)
                    files.append(dict(path=str(path.relative_to(OUTPUT)), bytes=path.stat().st_size,
                                      sha256=sha256(path)))
                    rows.append(dict(subject=subject, condition=condition, roi=roi, dataset=dataset,
                                     trials=int(raw['n_trials']), fs=float(raw['fs']),
                                     epoch_samples=len(raw['time']), path=str(path.relative_to(OUTPUT)),
                                     protocol_id=protocol_id))
                del runs, pooled, raw
            atomic_json(completed_path(subject, condition), dict(
                subject=subject, condition=condition, rois=list(rois), protocol_id=protocol_id,
                inputs=inputs, files=files, recordings=rows))
        data.export_signal(subject, condition, rois)
        return subject, condition, 'reconstructed', time.monotonic() - start
    except Exception:
        with log_path.open('a') as log:
            log.write('\n' + traceback.format_exc())
        raise


def summarize():
    # Atomic writes have .tmp.json names while other workers are completing.
    paths = sorted(p for p in (OUTPUT / 'complete').glob('*.json')
                   if not p.name.endswith('.tmp.json'))
    records = [json.loads(p.read_text()) for p in paths]
    rows = [row for record in records for row in record['recordings']]
    if rows:
        frame = pd.DataFrame(rows).sort_values(['subject', 'condition', 'roi', 'dataset'])
        temporary = OUTPUT / 'recordings.tmp.csv'
        frame.to_csv(temporary, index=False)
        temporary.replace(OUTPUT / 'recordings.csv')
    return len(records)


def audit_outputs(subjects, conditions, rois, downloads):
    """Check persisted pooled arrays against the two saved runs and HCP IDs."""
    rows = []
    for subject in subjects:
        tables = {run: trial_table(subject, run, downloads) for run in RUNS}
        for condition in conditions:
            for roi in rois:
                runs = [data.load_reconstruction(subject, condition, roi, f'run{run}') for run in RUNS]
                for run, raw in zip(RUNS, runs):
                    check_trial_identity(raw, tables[run], condition)
                    np.testing.assert_allclose(np.diff(raw['time']), 1 / float(raw['fs']),
                                               rtol=1e-10, atol=1e-12)
                expected = combine_runs(*runs)
                actual = data.load_reconstruction(subject, condition, roi)
                for key in ('signal', 'resolution'):
                    np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=0)
                for key in ('trial_runs', 'trial_indices', 'trials_per_run', 'source_orientation',
                            'leadfield_scale', 'labels', 'time'):
                    np.testing.assert_array_equal(actual[key], expected[key])
                np.testing.assert_equal(actual['trialinfo'], expected['trialinfo'])
                row = dict(subject=subject, condition=condition, roi=roi,
                           run10_trials=int(runs[0]['n_trials']), run11_trials=int(runs[1]['n_trials']),
                           combined_trials=int(actual['n_trials']),
                           run10_unit_gain_error=float(runs[0]['unit_gain_error']),
                           run11_unit_gain_error=float(runs[1]['unit_gain_error']),
                           pooled_arrays_match=True, hcp_trial_identity_match=True,
                           orientation_and_scale_match=True)
                rows.append(row)
    frame = pd.DataFrame(rows)
    temporary = OUTPUT / 'checks.tmp.csv'
    frame.to_csv(temporary, index=False)
    temporary.replace(OUTPUT / 'checks.csv')
    print(f'Checked persisted pooling and HCP trial identities in {len(rows)} recordings across resolutions.', flush=True)


def protocol(atlas, fieldtrip):
    return dict(schema=SCHEMA, runs=list(RUNS), default_dataset='combined',
                orientation='run-10 maximum-power orientation, held fixed for both runs',
                leadfield_normalization='run-10 sum-of-squares scale, held fixed for both runs',
                filters='run-specific sensors, leadfields and full-epoch covariance; 5% LCMV unit gain',
                preprocessing='EMG-locked HCP inputs; full-epoch demean/detrend; FieldTrip 0.5-100 Hz',
                pooling='retained-trial-count weighted signals and resolution matrices',
                trials='float32; dimensions trial, parcel, time; HCP row indices are 1-based',
                validation='reconstruction only; run-11 covariance uses run-11 data; no held-out fit claim',
                code_sha256={name: sha256(data.SRC / name) for name in ('beamform_runs.m', 'atlas.py')},
                fieldtrip_sha256={name: sha256(fieldtrip / name) for name in (
                    'ft_preprocessing.m', 'ft_timelockanalysis.m', 'ft_prepare_leadfield.m',
                    'ft_sourceanalysis.m', 'inverse/ft_inverse_lcmv.m')},
                atlas_sha256=sha256(atlas))


def main(args):
    subjects = [args.subject] if args.subject else list(data.SUBJECTS)
    if any(s not in data.SUBJECTS for s in subjects):
        raise ValueError('Participant is not in the study cohort')
    conditions = [args.condition] if args.condition else CONDITIONS
    rois = tuple(sorted(set(args.rois)))
    downloads = Path(os.environ.get('MEG_DOWNLOADS', Path.home() / 'Downloads')).expanduser()
    matlab = shutil.which(os.environ.get('MATLAB', 'matlab'))
    fieldtrip = Path(os.environ.get('FIELDTRIP_HOME', ROOT / 'fieldtrip-20251103'))
    os.environ.setdefault('FIELDTRIP_HOME', str(fieldtrip))
    for subject in subjects:
        for path in input_paths(subject, downloads):
            if not path.is_file():
                raise FileNotFoundError(path)
        for run in RUNS:
            trial_table(subject, run, downloads)
    if not matlab or not (fieldtrip / 'ft_defaults.m').is_file():
        raise RuntimeError('MATLAB or FieldTrip is unavailable')
    print(f'Inputs ready: {len(subjects)} participants, both runs, {len(conditions)} conditions; ROIs {rois}', flush=True)
    if args.check:
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'temporary').mkdir(exist_ok=True)
    with (OUTPUT / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atlas = OUTPUT / 'atlas.mat'
        if not atlas.exists():
            from atlas import prepare
            with (OUTPUT / 'atlas.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
                prepare(atlas, ROIS)
        settings = protocol(atlas, fieldtrip)
        protocol_id = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
        settings['protocol_id'] = protocol_id
        protocol_path = OUTPUT / 'protocol.json'
        if (protocol_path.exists() and json.loads(protocol_path.read_text()) != settings
                and any((OUTPUT / 'complete').glob('*.json'))):
            raise RuntimeError('Reconstruction protocol changed; use a new dataset directory, not old caches')
        atomic_json(protocol_path, settings)
        print('Checking and hashing input files (cached hashes are reused when file metadata matches).', flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            manifests = dict(zip(subjects, pool.map(lambda s: input_manifest(s, downloads), subjects)))
        jobs = [(s, c, rois, atlas, downloads, matlab, protocol_id, manifests[s])
                for s in subjects for c in conditions]
        if args.verify:
            invalid = [(s, c) for s in subjects for c in conditions
                       if not complete(s, c, rois, protocol_id, manifests[s], verify=True)]
            if invalid:
                raise RuntimeError(f'{len(invalid)} incomplete or invalid recordings; first: {invalid[0]}')
            summarize()
            audit_outputs(subjects, conditions, rois, downloads)
            print(f'Verified hashes and completeness of {len(jobs)} participant-condition datasets.', flush=True)
            return
        failures = []
        status_path = OUTPUT / 'status.json'
        started = time.time()
        done = 0
        atomic_json(status_path, dict(state='running', completed=0, requested=len(jobs), started=started))
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(reconstruct, job): job[:2] for job in jobs}
            for future in as_completed(futures):
                subject, condition = futures[future]
                try:
                    _, _, state, seconds = future.result()
                    done += 1
                    print(f'[{done}/{len(jobs)}] {subject} {condition}: {state} ({seconds:.0f}s)', flush=True)
                except Exception as exc:
                    failures.append(dict(subject=subject, condition=condition, error=str(exc)))
                    print(f'FAILED {subject} {condition}: {exc}; see {OUTPUT}/logs/{subject}_{condition}.log', flush=True)
                summarize()
                atomic_json(status_path, dict(state='running', completed=done, requested=len(jobs),
                                             failures=failures, started=started, updated=time.time()))
        atomic_json(status_path, dict(state='failed' if failures else 'complete', completed=done,
                                     requested=len(jobs), failures=failures, started=started, finished=time.time()))
        if failures:
            raise RuntimeError(f'{len(failures)} reconstruction jobs failed; successful jobs are resumable')
        print(f'Complete: {done} participant-condition datasets; combined means and both runs retained.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject')
    parser.add_argument('--condition', choices=CONDITIONS)
    parser.add_argument('--rois', nargs='+', type=int, choices=ROIS, default=ROIS)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    main(args)
