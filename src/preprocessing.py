"""Fixed, full-cohort filter/window sensitivity for supplementary Figure S13.

One choice varies at a time. All seven settings are retained, regardless of
outcome. Filtering always precedes cropping; every graph refits its linear
coefficients and tau. Existing Figure 3 results are checked, never overwritten.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, sosfiltfilt

import data
from validation import BANDS, CONDITIONS, MODELS, BatchFitter, parameter_counts, recording
from model import graph_modes


@dataclass(frozen=True)
class Setting:
    name: str
    order: int
    start: float
    stop: float


SETTINGS = (
    Setting('reference', 4, -.1, .9),
    Setting('order2', 2, -.1, .9),
    Setting('order6', 6, -.1, .9),
    Setting('shorter', 4, -.1, .7),
    Setting('longer', 4, -.1, 1.1),
    Setting('onset', 4, 0., 1.),
    Setting('later', 4, .1, 1.1),
)
BY_NAME = {s.name: s for s in SETTINGS}
SEED = 20261103
BOOTSTRAPS = 10000
CACHE = data.CACHE / 'preprocessing'
PROTOCOL = CACHE / 'protocol.json'
FIT_KEYS = ['subject', 'condition', 'fmin', 'fmax', 'setting', 'model']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recordings():
    return [(s, c) for s in data.SUBJECTS for c in CONDITIONS
            if s in data.subjects(c, 100)]


def jobs():
    return [(s, c, i) for s, c in recordings() for i in range(len(BANDS))]


def prepare_protocol():
    CACHE.mkdir(parents=True, exist_ok=True)
    inputs = {f'{s}_{c}.npz': digest(data.SIGNALS/f'{s}_{c}.npz')
              for s, c in recordings()}
    protocol = dict(
        version='preprocessing-sensitivity-v2', dataset=data.DATASET,
        runs=[10,11], settings=[asdict(s) for s in SETTINGS],
        design='one factor at a time; all specified settings reported; no selection by fit or graph advantage',
        bands_hz=BANDS, subjects=data.SUBJECTS, recordings=recordings(), models=MODELS,
        roi=100, modes=99, linear_coefficients=parameter_counts(),
        nominal_parameters={m: n + 1 for m, n in parameter_counts().items()},
        zero_mode='All graphs are connected; exclude the global constant and retain 99 positive-eigenvalue modes.',
        tau_bounds_seconds=[.001, 10.], tau_grid_points=257,
        local_peaks_refined=4, log_tau_xatol=1e-8,
        filter='Butterworth SOS, forward/backward scipy.signal.sosfiltfilt with default odd padding on full epoch before cropping',
        filtering_scope='parcel-level bandpass only; upstream sensor preprocessing, source reconstruction, parcellation and leakage mapping unchanged',
        window_convention='inclusive endpoints; time remains relative to movement onset; duration panel fixes start=-0.1 s; onset panel fixes duration=1 s',
        fitting='same parcel targets and denominator for all graphs; tau and all coefficients independently refitted for every setting',
        reference='fourth order, [-0.1,0.9] s; all reference scores must reproduce spectrum.csv within 1e-6',
        checks='explicit reconstructed scores must match fitted objective within 1e-6; no missing, duplicate or nonfinite fits',
        inference='conditions averaged within each participant, then paired structural-minus-control differences; pointwise 95% participant-bootstrap intervals, not multiplicity-adjusted',
        bootstrap_samples=BOOTSTRAPS, bootstrap_seed=SEED,
        software=dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__),
        sources={name: digest(data.ROOT/name) for name in
                 ('src/preprocessing.py', 'src/validation.py', 'src/model.py', 'src/data.py')},
        inputs=inputs,
        graphs={stem: digest(data.ASSETS/f'{stem}100.npy') for stem in data.NETWORKS.values()},
        reference_sha256=digest(data.RESULTS/'spectrum.csv'))
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + '\n'
    if PROTOCOL.exists():
        if PROTOCOL.read_text() != encoded:
            raise ValueError('Protocol, implementation, or inputs changed; refusing to mix runs')
    else:
        if CACHE.exists() and any(CACHE.glob('*.csv')):
            raise ValueError('Cached fits exist without their protocol')
        PROTOCOL.write_text(encoded)
    CACHE.mkdir(exist_ok=True)
    return digest(PROTOCOL)


def window_mask(times, setting):
    if times[0] > setting.start or times[-1] < setting.stop:
        raise ValueError('Requested window exceeds available full epoch')
    mask = (times >= setting.start) & (times <= setting.stop)
    if mask.sum() < 3:
        raise ValueError('Insufficient fitting samples')
    return mask


def filtered_epoch(raw, band, order):
    sos = butter(order, BANDS[band], btype='band', fs=float(raw['fs']), output='sos')
    return sosfiltfilt(sos, raw['signal'], axis=-1)


def validate_job(frame, job, protocol_hash):
    subject, condition, band = job
    low, high = BANDS[band]
    expected = {(s.name, m) for s in SETTINGS for m in MODELS}
    required = FIT_KEYS + ['r2', 'tau', 'order', 'start', 'stop', 'samples',
                           'baseline_error', 'explicit_score_error', 'coefficient_count', 'protocol_sha256']
    if not set(required).issubset(frame.columns):
        raise ValueError('Missing sensitivity output columns')
    if len(frame) != len(expected) or frame.duplicated(['setting', 'model']).any():
        raise ValueError('Missing or duplicate sensitivity fits')
    if set(frame[['setting', 'model']].itertuples(index=False, name=None)) != expected:
        raise ValueError('Unexpected sensitivity settings/models')
    for key, value in dict(subject=subject, condition=condition, fmin=low, fmax=high,
                           protocol_sha256=protocol_hash).items():
        if not frame[key].eq(value).all():
            raise ValueError(f'Job identity/protocol mismatch: {key}')
    if not np.isfinite(frame[['r2', 'tau', 'samples', 'explicit_score_error']]).all().all():
        raise ValueError('Nonfinite sensitivity fits')
    if not frame.tau.between(.001*(1-1e-10), 10*(1+1e-10)).all() or (frame.r2 > 1+1e-7).any():
        raise ValueError('Invalid sensitivity score or tau')
    if (frame.explicit_score_error > 1e-6).any():
        raise ValueError('Explicit reconstruction disagrees with objective')
    if not frame.coefficient_count.eq(frame.model.map(parameter_counts())).all():
        raise ValueError('Incorrect linear coefficient count')
    reference = frame[frame.setting == 'reference']
    if not np.isfinite(reference.baseline_error).all() or (reference.baseline_error > 1e-6).any():
        raise ValueError('Reference does not reproduce Figure 3')
    for setting in SETTINGS:
        subset = frame[frame.setting == setting.name]
        if not (subset.order.eq(setting.order) & subset.start.eq(setting.start)
                & subset.stop.eq(setting.stop)).all():
            raise ValueError('Saved setting differs from protocol')
    return frame


def run_job(job, protocol_hash):
    subject, condition, band = job
    path = CACHE / f'{subject}_{condition}_{band}.csv'
    if path.exists():
        validate_job(pd.read_csv(path, dtype={'subject': str}), job, protocol_hash)
        return
    raw = recording(subject, condition)
    if raw['signal'].shape != (100, len(raw['time'])) or not np.isfinite(raw['signal']).all():
        raise ValueError('Invalid cached parcel signals')
    low, high = BANDS[band]
    filtered = {order: filtered_epoch(raw, band, order) for order in (2, 4, 6)}
    operators = {}
    for model, graph in data.operators(100, raw['labels']).items():
        eig, vec = graph_modes(graph)
        operators[model] = eig, raw['resolution'] @ vec
    reference = pd.read_csv(data.RESULTS/'spectrum.csv', dtype={'subject': str})
    reference = reference[(reference.subject == subject) & (reference.condition == condition)
                          & (reference.fmin == low) & (reference.fmax == high)
                          & (reference.order == 4)].set_index('model')
    if len(reference) != 3 or not reference.index.is_unique:
        raise ValueError('Missing Figure 3 reference fits')
    # Filter-order targets have identical times/designs and can share factorizations.
    # Different windows are optimized separately, preserving each graph's coefficient convention.
    groups = [SETTINGS[:3]] + [(s,) for s in SETTINGS[3:]]
    rows = []
    for group in groups:
        mask = window_mask(raw['time'], group[0])
        targets = np.stack([filtered[s.order][:, mask] for s in group])
        for model, (eig, mixing) in operators.items():
            fitter = BatchFitter(targets, raw['time'][mask], eig, mixing)
            scores, taus = fitter.fit()
            for j, (setting, r2, tau) in enumerate(zip(group, scores, taus)):
                prediction = fitter.predict(tau, j)
                y = targets[j]
                direct = 1 - np.sum((y-prediction)**2) / np.sum((y-y.mean(1, keepdims=True))**2)
                baseline_error = abs(r2-reference.loc[model, 'r2']) if setting.name == 'reference' else np.nan
                rows.append(dict(subject=subject, condition=condition, fmin=low, fmax=high,
                                 setting=setting.name, order=setting.order, start=setting.start,
                                 stop=setting.stop, samples=int(mask.sum()), model=model,
                                 r2=r2, tau=tau, coefficient_count=len(fitter.column_modes), baseline_error=baseline_error,
                                 explicit_score_error=abs(r2-direct),
                                 failed_evaluations=fitter.failed, protocol_sha256=protocol_hash))
    frame = validate_job(pd.DataFrame(rows), job, protocol_hash)
    temporary = path.with_suffix('.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def run(workers):
    protocol_hash = prepare_protocol()
    pending = jobs()
    started = time.monotonic()
    print(f'Fixed protocol: {len(pending)} jobs, {len(pending)*len(SETTINGS)*len(MODELS)} fits; {workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, job, protocol_hash): job for job in pending}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                for task in futures:
                    task.cancel()
                raise RuntimeError(f'Sensitivity job failed: {futures[future]}') from exc
            if i % 20 == 0 or i == len(pending):
                elapsed = time.monotonic()-started
                print(f'{i}/{len(pending)} jobs; {elapsed/60:.1f} min; {elapsed/i*(len(pending)-i)/60:.1f} min estimated remaining', flush=True)


def participant_scores(fits):
    """Average conditions within participants before group inference."""
    return fits.groupby(['subject', 'setting', 'fmin', 'fmax', 'model'], as_index=False).r2.mean()


def statistics(participant):
    rng = np.random.default_rng(SEED)
    subjects = sorted(participant.subject.unique())
    draws = rng.integers(len(subjects), size=(BOOTSTRAPS, len(subjects)))
    records = []
    for (low, high), band in participant.groupby(['fmin', 'fmax']):
        reference = band[band.setting == 'reference'].pivot(index='subject', columns='model', values='r2').reindex(subjects)
        for setting in SETTINGS:
            wide = band[band.setting == setting.name].pivot(index='subject', columns='model', values='r2').reindex(subjects)
            if wide.isna().any().any():
                raise ValueError('Incomplete participant summary')
            for model in MODELS:
                contrasts = [('r2', wide[model]), ('r2_change', wide[model]-reference[model])]
                if model != 'Structural':
                    delta = wide.Structural-wide[model]
                    contrasts += [('advantage', delta),
                                  ('advantage_change', delta-(reference.Structural-reference[model]))]
                for metric, series in contrasts:
                    values = series.to_numpy()
                    ci = np.quantile(values[draws].mean(1), [.025, .975])
                    records.append(dict(setting=setting.name, fmin=low, fmax=high, model=model,
                                        metric=metric, mean=values.mean(), median=np.median(values),
                                        ci_low=ci[0], ci_high=ci[1], participants=len(values),
                                        positive_participants=int(np.sum(values > 0))))
    return pd.DataFrame(records)


def summarize():
    protocol_hash = prepare_protocol()
    frames = [validate_job(pd.read_csv(CACHE/f'{s}_{c}_{b}.csv', dtype={'subject': str}),
                           (s, c, b), protocol_hash) for s, c, b in jobs()]
    fits = pd.concat(frames, ignore_index=True)
    if fits.duplicated(FIT_KEYS).any() or len(fits) != len(jobs())*len(SETTINGS)*len(MODELS):
        raise ValueError('Incomplete full-cohort sensitivity results')
    participant = participant_scores(fits)
    expected = len(data.SUBJECTS)*len(SETTINGS)*len(BANDS)*len(MODELS)
    if len(participant) != expected:
        raise ValueError('Participant summary is incomplete')
    stats = statistics(participant)
    checks = fits.assign(lower_hit=np.isclose(fits.tau, .001, rtol=1e-4, atol=0),
                         upper_hit=np.isclose(fits.tau, 10., rtol=1e-4, atol=0)).groupby(
        ['setting', 'fmin', 'fmax', 'model']).agg(
            fits=('r2', 'size'), sample_min=('samples', 'min'), sample_max=('samples', 'max'),
            lower_hits=('lower_hit', 'sum'), upper_hits=('upper_hit', 'sum'),
            max_explicit_score_error=('explicit_score_error', 'max'),
            max_baseline_error=('baseline_error', 'max'),
            failed_evaluations=('failed_evaluations', 'sum')).reset_index()
    for name, frame in [('fits', fits), ('participants', participant), ('statistics', stats), ('checks', checks)]:
        frame.drop(columns=['protocol_sha256'], errors='ignore').to_csv(data.RESULTS/f'preprocessing_{name}.csv', index=False)
    print(stats[stats.metric == 'advantage'][['setting', 'fmin', 'fmax', 'model', 'mean', 'ci_low', 'ci_high']].to_string(index=False), flush=True)
    print('Reference max error:', fits.baseline_error.max(),
          'explicit score max error:', fits.explicit_score_error.max(), flush=True)
    print('Boundary hits:', checks[['lower_hits', 'upper_hits']].sum().to_dict(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    run(args.workers)
    summarize()
