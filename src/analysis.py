"""Fit all manuscript models to pooled motor runs 10 and 11."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import data
from fit import calculate, fit_parcels, fit_wide, fit_curve, BINS, BANDS, SPECTRUM_BANDS, MODAL_BANDS
from model import graph_modes, coefficient_modes

ROOT = data.CACHE / 'analysis'
CONDITIONS = ('LH', 'RH', 'LF', 'RF')
STAGES = ('parcel', 'spectrum', 'modal', 'modes', 'modes-spectrum')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.csv')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def prepare():
    inputs = {}
    protocol_ids = set()
    for subject in data.SUBJECTS:
        for condition in CONDITIONS:
            path = data.SIGNALS / f'{subject}_{condition}.npz'
            inputs[path.name] = digest(path)
            for roi in data.ROIS:
                raw = data.load_parcels(subject, condition, roi)
                if (raw['signal'].shape != (roi, len(raw['time']))
                        or raw['resolution'].shape != (roi, roi)
                        or not np.array_equal(raw['labels'], np.arange(1, roi+1))):
                    raise ValueError(f'Invalid pooled response: {subject}/{condition}/{roi}')
                protocol_ids.add(str(raw['protocol_id']))
    if len(protocol_ids) != 1:
        raise ValueError('The pooled responses have different reconstruction protocols')
    protocol = dict(dataset='combined', runs=[10, 11], subjects=list(data.SUBJECTS),
                    conditions=CONDITIONS, rois=list(data.ROIS),
                    pooling='retained-trial-count weighted signal and resolution matrix',
                    independent_unit='participant; average conditions within participant',
                    reconstruction_protocol=protocol_ids.pop(), inputs=inputs)
    encoded = json.dumps(protocol, indent=2, sort_keys=True)+'\n'
    path = ROOT/'inputs.json'
    ROOT.mkdir(parents=True, exist_ok=True)
    data.RESULTS.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError('Analysis inputs changed; use a fresh local cache')
    if not path.exists():
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(encoded)
        temporary.replace(path)
    print(f'Read {len(inputs)} pooled responses at all three resolutions', flush=True)


def stage_protocol(stage):
    names = ['src/analysis.py', 'src/data.py', 'src/model.py', 'src/fit.py', 'src/spatial.py']
    value = dict(stage=stage, inputs=digest(ROOT/'inputs.json'),
                 sources={name:digest(data.ROOT/name) for name in names},
                 graphs={p.name:digest(p) for roi in (100, 200, 400)
                         for stem in data.NETWORKS.values()
                         for p in [data.ASSETS/f'{stem}{roi}.npy']})
    if stage in ('modes', 'modes-spectrum'):
        value['reference'] = digest(data.RESULTS/('parcel_fits.csv' if stage == 'modes' else 'spectrum.csv'))
    encoded = json.dumps(value, indent=2, sort_keys=True)+'\n'
    path = ROOT/'jobs'/stage/'protocol.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise ValueError(f'{stage}: code or inputs changed; do not mix cached results')
    if not path.exists():
        temporary = path.with_suffix('.tmp.json')
        temporary.write_text(encoded)
        temporary.replace(path)
    return hashlib.sha256(encoded.encode()).hexdigest()


def valid_scores(frame):
    """Undefined tau is allowed only for an explicitly static-only prefix."""
    if (not np.isfinite(frame.r2).all()
            or not frame.tau_identified.isin([True, False]).all()
            or not frame.coefficient_count.gt(0).all()):
        return False
    identified = frame.tau_identified.astype(bool)
    if not np.isfinite(frame.loc[identified, 'tau']).all():
        return False
    static = frame.loc[~identified]
    return (static.tau.isna().all() and (static.empty or
            ('modes' in static and static.coefficient_count.eq(static.modes).all())))


def fit_recording(job):
    stage, subject, condition, roi, protocol_id = job
    path = ROOT/'jobs'/stage/f'{subject}_{condition}{roi}.csv'
    expected = dict(parcel=15, spectrum=36, modal=len(MODAL_BANDS)*len(data.NETWORKS), modes=594,
                    **{'modes-spectrum':1188})[stage]
    if path.exists():
        saved = pd.read_csv(path, dtype={'subject':str})
        if (len(saved) != expected or not saved.analysis_protocol.eq(protocol_id).all()
                or not valid_scores(saved)):
            raise ValueError(f'Invalid cached results: {path}')
        return str(path)
    raw = data.load_parcels(subject, condition, roi)
    signal, times, fs = raw['signal'], raw['time'], float(raw['fs'])
    mask = (times >= -.1) & (times <= .9)
    if stage == 'modal':
        frame = pd.DataFrame(calculate(signal, times, fs, raw['labels'],
                                      raw['resolution'], condition, roi, subject))
    else:
        operators = {}
        for model, graph in data.operators(roi, raw['labels']).items():
            eig, vec = graph_modes(graph)
            operators[model] = eig, raw['resolution'] @ vec
        bands = (dict(enumerate(BINS)) if stage == 'spectrum' else
                 data.BANDS if stage == 'parcel' else
                 BANDS if stage == 'modes' else SPECTRUM_BANDS)
        reference = None
        if stage.startswith('modes'):
            reference = pd.read_csv(data.RESULTS/('parcel_fits.csv' if stage == 'modes' else 'spectrum.csv'), dtype={'subject':str})
            reference = reference[(reference.subject == subject) & (reference.condition == condition)]
        rows = []
        for band, (low, high) in bands.items():
            # Match the filter parameterization used by each existing analysis.
            sos = (butter(4, (low, high), btype='band', fs=fs, output='sos')
                   if stage in ('spectrum', 'modes-spectrum') else
                   butter(4, [low/(fs/2), min(high/(fs/2), .99)], btype='band', output='sos'))
            target = sosfiltfilt(sos, signal, axis=1)[:, mask]
            target_hash = hashlib.sha256(target.tobytes()).hexdigest()
            for model, (eig, mixing) in operators.items():
                common = dict(subject=subject, condition=condition, model=model)
                if stage == 'parcel':
                    r2, tau = fit_parcels(target, times[mask], eig, mixing)
                    rows.append(dict(**common, roi=roi, band=band, r2=r2, tau=tau,
                                     modes=roi-1, target_hash=target_hash, cache_key=protocol_id,
                                     coefficient_count=len(coefficient_modes(eig)),
                                     tau_identified=bool(np.any(eig > 0))))
                elif stage == 'spectrum':
                    r2, tau = fit_wide(target, times[mask], eig, mixing, .001, 10.)
                    rows.append(dict(**common, roi=roi, order=4, fmin=low, fmax=high,
                                     r2=r2, tau=tau, tau_min=.001, tau_max=10.,
                                     coefficient_count=len(coefficient_modes(eig)),
                                     tau_identified=bool(np.any(eig > 0))))
                else:
                    bounds = (.01, 1.) if stage == 'modes' else (.001, 10.)
                    selected = (reference[reference.band.eq(band)] if stage == 'modes'
                                else reference[reference.fmin.eq(low) & reference.fmax.eq(high) & reference.order.eq(4)])
                    full = selected[selected.model.eq(model)]
                    if len(full) != 1:
                        raise ValueError('Missing or duplicated full-model reference')
                    full = full.iloc[0]
                    if stage == 'modes' and full.target_hash != target_hash:
                        raise ValueError('Mode-curve target mismatch')
                    scores, taus = fit_curve(target, times[mask], eig, mixing, full.tau, *bounds)
                    if abs(scores[-1]-full.r2) > 1e-6:
                        raise ValueError(f'K=99 baseline mismatch: {subject}/{condition}/{band}/{model}')
                    rows.extend(dict(**common, roi=roi, order=4, band=band, fmin=low,
                                     fmax=high, modes=k, r2=r2, tau=tau,
                                     tau_min=bounds[0], tau_max=bounds[1],
                                     target_hash=target_hash, cache_key=protocol_id,
                                     coefficient_count=len(coefficient_modes(eig[:k])),
                                     tau_identified=bool(np.any(eig[:k] > 0)))
                                for k, (r2, tau) in enumerate(zip(scores, taus), 1))
        frame = pd.DataFrame(rows)
    if len(frame) != expected or not valid_scores(frame):
        raise ValueError(f'Incomplete/nonfinite {stage} results: {subject}/{condition}/{roi}')
    frame['analysis_protocol'] = protocol_id
    atomic_csv(frame, path)
    return str(path)


def run(stage, workers=16, roi=100):
    if roi != 100 and stage != 'spectrum':
        raise ValueError('Alternative ROI counts are supported for the spectrum stage only')
    protocol_id = stage_protocol(stage)
    rois = (100, 200, 400) if stage == 'modal' else (roi,)
    jobs = [(stage, s, c, roi, protocol_id) for s in data.SUBJECTS
            for c in CONDITIONS for roi in rois]
    started = time.monotonic()
    print(f'{stage}: {len(jobs)} recording/ROI jobs, {workers} workers', flush=True)
    paths = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_recording, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            paths.append(future.result())
            print(f'{stage}: {index}/{len(jobs)} complete; elapsed {time.monotonic()-started:.0f}s', flush=True)
    frame = pd.concat([pd.read_csv(p, dtype={'subject':str}) for p in sorted(paths)], ignore_index=True)
    filename = {'modal':'modal_fits.csv', 'parcel':'parcel_fits.csv', 'spectrum':'spectrum.csv',
                'modes':'modes.csv', 'modes-spectrum':'modes_spectrum.csv'}[stage]
    if stage == 'spectrum' and roi != 100:
        filename = f'spectrum_{roi}.csv'
    public = frame.drop(columns=['analysis_protocol', 'cache_key'], errors='ignore')
    atomic_csv(public, data.RESULTS/filename)
    print(f'{stage}: saved {len(frame)} fits', flush=True)
    if stage == 'modes':
        from spatial import run as spatial_run
        spatial_run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'all', *STAGES))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--roi', type=int, choices=data.ROIS, default=100,
                        help='Parcellation for spectrum fitting (default: 100)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.roi != 100:
        if args.stage != 'spectrum':
            parser.error('--roi requires the spectrum stage')
        ROOT = ROOT / f'roi{args.roi}'
    prepare()
    for stage in STAGES if args.stage == 'all' else (() if args.stage == 'prepare' else (args.stage,)):
        run(stage, args.workers, args.roi)
