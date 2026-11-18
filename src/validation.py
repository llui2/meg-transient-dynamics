"""Spectrum-preserving signal surrogates and cross-condition tau transfer.

Run --run to fit all available recordings in the four Figure 4 bands; --summarize
validates completeness and writes participant-level summaries. Raw jobs resume.
No trial-level prediction or source reconstruction is performed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import time as clock

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.signal import butter, sosfiltfilt

import data
from model import coefficient_modes, graph_modes, temporal_basis

BANDS = ((2, 4), (4, 6), (6, 9), (9, 13))
CONDITIONS = ('LH', 'RH', 'LF', 'RF')
MODELS = tuple(data.NETWORKS)
N_SURROGATES = 25
SEED = 20261103
TAU_MIN, TAU_MAX = .001, 10.
CACHE = data.CACHE / 'validation'
PROTOCOL = CACHE / 'protocol.json'


class BatchFitter:
    """Joint parcel least squares; share factorizations across surrogate targets."""
    def __init__(self, targets, time, eigenvalues, mixing):
        self.targets = np.asarray(targets)
        self.time, self.eigenvalues = time, eigenvalues
        self.mixing = mixing / np.linalg.norm(mixing, axis=0)
        self.norm = np.linalg.norm(self.targets, axis=(1, 2))
        if np.any(self.norm == 0):
            raise ValueError('Zero target')
        self.observed = self.targets / self.norm[:, None, None]
        self.sst = np.sum((self.observed-self.observed.mean(2, keepdims=True))**2, axis=(1, 2))
        self.projected = np.einsum('im,bit->mtb', self.mixing, self.observed, optimize=True)
        self.column_modes = coefficient_modes(eigenvalues)
        self.column_projected = self.projected[self.column_modes]
        gram = self.mixing.T @ self.mixing
        self.spatial = gram[np.ix_(self.column_modes, self.column_modes)]
        self.failed = 0

    def system(self, tau, indices=None):
        functions = temporal_basis(self.time, self.eigenvalues, tau)
        gram = self.spatial * (functions.T @ functions)
        projected = self.column_projected if indices is None else self.column_projected[:, :, indices]
        rhs = np.einsum('ptb,tp->pb', projected, functions,
                        optimize=True)
        factor = cholesky(gram, lower=True, check_finite=False)
        return functions, factor, rhs

    def evaluate(self, log_tau, indices=None):
        count = len(self.targets) if indices is None else len(indices)
        try:
            _, factor, rhs = self.system(np.exp(log_tau), indices)
            fitted = solve_triangular(factor, rhs, lower=True, check_finite=False)
            sst = self.sst if indices is None else self.sst[indices]
            return 1 - (1-np.sum(fitted*fitted, axis=0))/sst
        except np.linalg.LinAlgError:
            self.failed += 1
            return np.full(count, -np.inf)

    def fit(self):
        grid = np.linspace(np.log(TAU_MIN), np.log(TAU_MAX), 257)
        scores = np.array([self.evaluate(t) for t in grid])
        best = np.argmax(scores, axis=0)
        result = scores[best, np.arange(len(best))].copy()
        log_tau = grid[best].copy()
        for j in range(len(best)):
            peaks = [i for i in range(1, len(grid)-1)
                     if scores[i,j] >= scores[i-1,j] and scores[i,j] >= scores[i+1,j]]
            for i in sorted(peaks, key=lambda i: scores[i,j], reverse=True)[:4]:
                opt = minimize_scalar(lambda t: -self.evaluate(t, [j])[0],
                                      bounds=(grid[i-1], grid[i+1]), method='bounded',
                                      options={'xatol': 1e-8})
                if not opt.success:
                    raise RuntimeError(opt.message)
                if -opt.fun > result[j]:
                    result[j], log_tau[j] = -opt.fun, opt.x
        if not np.isfinite(result).all() or np.any(result > 1+1e-7):
            raise ValueError('Invalid fit')
        return result, np.exp(log_tau)

    def predict(self, tau, index=0):
        functions, factor, rhs = self.system(tau, [index])
        coefficients = cho_solve((factor, True), rhs, check_finite=False)[:, 0]
        modes = np.zeros((len(self.eigenvalues), len(self.time)))
        np.add.at(modes, self.column_modes, functions.T * coefficients[:, None])
        return self.mixing @ modes * self.norm[index]


def surrogate_targets(filtered, seed, count=N_SURROGATES):
    """Common phase increments preserve the full-epoch multivariate spectrum.

    DC and (for even lengths) Nyquist are unchanged. Different positive
    frequencies receive independent phases; all parcels share each phase.
    """
    spectrum = np.fft.rfft(filtered, axis=-1)
    phase = np.random.default_rng(seed).uniform(0, 2*np.pi, (count, spectrum.shape[-1]))
    phase[:, 0] = 0
    if filtered.shape[-1] % 2 == 0:
        phase[:, -1] = 0
    surrogates = np.fft.irfft(spectrum[None] * np.exp(1j*phase[:, None]),
                            n=filtered.shape[-1], axis=-1)
    return np.concatenate((filtered[None], surrogates))


def recording(subject, condition):
    return data.load_parcels(subject, condition)


def jobs():
    return [(s, c, b) for s in data.SUBJECTS for c in CONDITIONS
            if s in data.subjects(c, 100) for b in range(len(BANDS))]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def parameter_counts():
    """Number of linear coefficients in each full 100-parcel graph model."""
    return {name: len(coefficient_modes(graph_modes(graph)[0]))
            for name, graph in data.operators(100, np.arange(1, 101)).items()}


def prepare_protocol():
    protocol = dict(version='common-phase-validation-v2', seed=SEED,
                    surrogate_count=N_SURROGATES, bands_hz=BANDS, modes=99,
                    linear_coefficients=parameter_counts(),
                    nominal_parameters={m: n + 1 for m, n in parameter_counts().items()},
                    zero_mode='All graphs are connected; exclude the global constant and retain 99 positive-eigenvalue modes.',
                    tau_bounds_seconds=[TAU_MIN, TAU_MAX], tau_grid_points=257,
                    local_peaks_refined=4, log_tau_xatol=1e-8,
                    fitting_window_seconds=[-.1,.9], roi=100,
                    subjects=list(data.SUBJECTS), conditions=list(CONDITIONS),
                    dataset=data.DATASET, runs=[10, 11],
                    excluded_recording=[], models=list(MODELS),
                    filter='fourth-order Butterworth, scipy sosfiltfilt, full epoch before randomization and cropping',
                    surrogate='independent uniform phase increments per frequency, shared across parcels; DC/Nyquist unchanged',
                    seed_rule='SeedSequence([20261103, integer_subject, condition_index, band_index])',
                    preserved='full-epoch auto/cross spectra, not necessarily cropped-window spectra or amplitude distributions',
                    inference='participant bootstrap conditional on 25 surrogates; descriptive Monte Carlo tail fractions, no multiplicity-adjusted significance claims',
                    transfer='each ordered pair of available conditions; source tau fixed; target coefficients refitted; not held-out waveform prediction',
                    example='LH recording with median mean observed parcel variance across the four bands; parcel nearest median variance within each band; no fit scores used',
                    sources={name:digest(data.ROOT/name) for name in ('src/validation.py','src/model.py','src/data.py')},
                    inputs={p.name:digest(p) for p in sorted(data.SIGNALS.glob('*.npz'))},
                    graphs={stem:digest(data.ASSETS/f'{stem}100.npy') for stem in data.NETWORKS.values()},
                    spectrum_sha256=digest(data.RESULTS/'spectrum.csv'))
    encoded=json.dumps(protocol, indent=2, sort_keys=True)+'\n'
    if PROTOCOL.exists() and PROTOCOL.read_text() != encoded:
        raise ValueError('Protocol or inputs changed; do not mix cached runs')
    CACHE.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(encoded)


def run_job(job):
    subject, condition, band = job
    path = CACHE / f'{subject}_{condition}_{band}.csv'
    if path.exists():
        saved = pd.read_csv(path)
        if len(saved) != 3*(N_SURROGATES+1) or saved.duplicated(['model','surrogate']).any():
            raise ValueError(f'Invalid cached job {path}')
        return str(path)
    raw=recording(subject, condition)
    low, high=BANDS[band]
    filtered=sosfiltfilt(butter(4,(low,high),btype='band',fs=float(raw['fs']),output='sos'), raw['signal'])
    seed=np.random.SeedSequence([SEED,int(subject),CONDITIONS.index(condition),band])
    full=surrogate_targets(filtered, seed)
    # Audit the actual generated full-epoch spectra before any fits.
    original_fft=np.fft.rfft(filtered, axis=-1)
    generated_fft=np.fft.rfft(full[1:],axis=-1)
    power_error=float(np.max(np.abs(abs(generated_fft)**2-abs(original_fft)**2))/np.max(abs(original_fft)**2))
    # All parcel pairs checked at every Fourier bin for one surrogate; identical
    # shared-phase construction is used for every realization.
    original_cross=np.einsum('if,jf->ijf',original_fft,original_fft.conj())
    generated_cross=np.einsum('if,jf->ijf',generated_fft[0],generated_fft[0].conj())
    cross_error=float(np.max(abs(generated_cross-original_cross))/np.max(abs(original_cross)))
    if max(power_error,cross_error)>1e-11:
        raise ValueError('Surrogate spectrum mismatch')
    mask=(raw['time']>=-.1)&(raw['time']<=.9)
    targets=full[:,:,mask]
    rows=[]
    reference=pd.read_csv(data.RESULTS/'spectrum.csv', dtype={'subject':str})
    reference=reference[(reference.subject==subject)&(reference.condition==condition)&(reference.fmin==low)&(reference.fmax==high)&(reference.order==4)].set_index('model')
    for model, graph in data.operators(100,raw['labels']).items():
        eig, vec=graph_modes(graph)
        fitter=BatchFitter(targets,raw['time'][mask],eig,raw['resolution']@vec)
        scores,taus=fitter.fit()
        difference=abs(scores[0]-reference.loc[model,'r2'])
        if difference>1e-6:
            raise ValueError(f'Figure 3 baseline mismatch: {job} {model}: {difference}')
        for i,(r2,tau) in enumerate(zip(scores,taus)):
            rows.append(dict(subject=subject,condition=condition,fmin=low,fmax=high,
                             model=model,surrogate=i,r2=r2,tau=tau,modes=99,
                             coefficient_count=len(fitter.column_modes),
                             baseline_error=difference,power_error=power_error,cross_error=cross_error,
                             failed_evaluations=fitter.failed))
    temp=path.with_suffix('.tmp')
    pd.DataFrame(rows).to_csv(temp,index=False)
    temp.replace(path)
    return str(path)


def run(workers):
    prepare_protocol()
    started=clock.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_job,job):job for job in jobs()}
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%10==0 or i==len(futures):
                print(f'{i}/{len(futures)} jobs; {(clock.monotonic()-started)/60:.1f} min',flush=True)


def summarize():
    frames=[]
    for s,c,b in jobs():
        frames.append(pd.read_csv(CACHE/f'{s}_{c}_{b}.csv',dtype={'subject':str}))
    fits=pd.concat(frames,ignore_index=True)
    fits.to_csv(data.RESULTS/'validation_fits.csv',index=False)
    keys=['subject','condition','fmin','fmax','model','surrogate']
    assert len(fits)==data.recording_count()*4*3*26 and not fits.duplicated(keys).any()
    assert np.isfinite(fits[['r2','tau']]).all().all()
    assert fits.groupby(['subject','condition','fmin','fmax','model']).surrogate.apply(lambda x:set(x)==set(range(26))).all()
    participant=fits.groupby(['subject','fmin','fmax','model','surrogate'],as_index=False)[['r2','tau']].mean()
    participant.to_csv(data.RESULTS/'validation_surrogates.csv',index=False)
    rng=np.random.default_rng(SEED)
    draws=rng.integers(51,size=(10000,51))
    rows=[]
    for (low,high),group in participant.groupby(['fmin','fmax']):
        wide=group.pivot(index=['subject','surrogate'],columns='model',values='r2')
        for control in MODELS[1:]:
            advantage=(wide.Structural-wide[control]).unstack('surrogate').reindex(data.SUBJECTS)
            original=advantage[0].to_numpy()
            null=advantage.loc[:,1:].to_numpy()
            for metric,values in [('observed',original),('surrogate_mean',null.mean(1)),('observed_minus_surrogate',original-null.mean(1))]:
                ci=np.quantile(values[draws].mean(1),[.025,.975])
                rows.append(dict(fmin=low,fmax=high,control=control,metric=metric,mean=values.mean(),ci_low=ci[0],ci_high=ci[1],participants=51,surrogates=25,
                                 monte_carlo_upper_tail=(1+np.count_nonzero(null.mean(0)>=original.mean()))/26))
    pd.DataFrame(rows).to_csv(data.RESULTS/'validation_statistics.csv',index=False)
    diagnostic=fits.assign(at_lower=np.isclose(fits.tau,TAU_MIN,rtol=1e-4,atol=0),at_upper=np.isclose(fits.tau,TAU_MAX,rtol=1e-4,atol=0),kind=np.where(fits.surrogate==0,'Observed','Surrogate')).groupby(['fmin','fmax','model','kind']).agg(fits=('r2','size'),lower_hits=('at_lower','sum'),upper_hits=('at_upper','sum'),max_baseline_error=('baseline_error','max'),max_power_error=('power_error','max'),max_cross_error=('cross_error','max')).reset_index()
    diagnostic.to_csv(data.RESULTS/'validation_checks.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


def transfer():
    reference=pd.read_csv(data.RESULTS/'spectrum.csv',dtype={'subject':str})
    rows=[]
    for subject in data.SUBJECTS:
        available=[c for c in CONDITIONS if subject in data.subjects(c,100)]
        for target in available:
            raw=recording(subject,target)
            mask=(raw['time']>=-.1)&(raw['time']<=.9)
            operators={m:graph_modes(g) for m,g in data.operators(100,raw['labels']).items()}
            for low,high in BANDS:
                signal=sosfiltfilt(butter(4,(low,high),btype='band',fs=float(raw['fs']),output='sos'),raw['signal'])[:,mask]
                subset=reference[(reference.subject==subject)&(reference.fmin==low)&(reference.fmax==high)&(reference.order==4)].set_index(['condition','model'])
                for model,(eig,vec) in operators.items():
                    fitter=BatchFitter(signal[None],raw['time'][mask],eig,raw['resolution']@vec)
                    own=subset.loc[(target,model)]
                    own_score=fitter.evaluate(np.log(own.tau))[0]
                    if abs(own_score-own.r2)>1e-6:
                        raise ValueError('Transfer baseline mismatch')
                    for source in available:
                        if source==target:
                            continue
                        tau=float(subset.loc[(source,model),'tau'])
                        score=fitter.evaluate(np.log(tau))[0]
                        if not np.isfinite(score):
                            raise ValueError('Nonfinite transfer score')
                        rows.append(dict(subject=subject,source=source,target=target,fmin=low,fmax=high,model=model,tau_source=tau,r2=score,r2_refitted=own_score))
    result=pd.DataFrame(rows)
    assert len(result)==sum(len([c for c in CONDITIONS if s in data.subjects(c,100)]) * (len([c for c in CONDITIONS if s in data.subjects(c,100)])-1) for s in data.SUBJECTS)*len(BANDS)*len(MODELS)
    result.to_csv(data.RESULTS/'validation_transfer.csv',index=False)
    print(f'Saved {len(result)} ordered-condition transfer fits',flush=True)


def transfer_statistics():
    frame=pd.read_csv(data.RESULTS/'validation_transfer.csv',dtype={'subject':str})
    participant=frame.groupby(['subject','fmin','fmax','model'])[['r2','r2_refitted']].mean().reset_index()
    draws=np.random.default_rng(SEED).integers(51,size=(10000,51))
    rows=[]
    for low,high in BANDS:
        group=participant[(participant.fmin==low)&(participant.fmax==high)]
        for model in MODELS:
            g=group[group.model==model].set_index('subject').reindex(data.SUBJECTS)
            penalty=(g.r2-g.r2_refitted).to_numpy()
            ci=np.quantile(penalty[draws].mean(1),[.025,.975])
            rows.append(dict(fmin=low,fmax=high,model=model,metric='transfer_minus_refitted',mean=penalty.mean(),ci_low=ci[0],ci_high=ci[1]))
        for control in MODELS[1:]:
            wide=group.pivot(index='subject',columns='model',values='r2').reindex(data.SUBJECTS)
            delta=(wide.Structural-wide[control]).to_numpy()
            ci=np.quantile(delta[draws].mean(1),[.025,.975])
            rows.append(dict(fmin=low,fmax=high,model=control,metric='transferred_structural_advantage',mean=delta.mean(),ci_low=ci[0],ci_high=ci[1]))
    pd.DataFrame(rows).to_csv(data.RESULTS/'validation_transfer_statistics.csv',index=False)


def prepare_examples():
    candidates=[]
    for subject in data.SUBJECTS:
        raw=recording(subject,'LH')
        mask=(raw['time']>=-.1)&(raw['time']<=.9)
        variance=[]
        for low,high in BANDS:
            x=sosfiltfilt(butter(4,(low,high),btype='band',fs=float(raw['fs']),output='sos'),raw['signal'])[:,mask]
            variance.append(np.var(x,axis=1).mean())
        candidates.append((np.mean(variance),subject))
    _,subject=sorted(candidates)[len(candidates)//2]
    raw=recording(subject,'LH'); mask=(raw['time']>=-.1)&(raw['time']<=.9)
    reference=pd.read_csv(data.RESULTS/'spectrum.csv',dtype={'subject':str})
    reference=reference[(reference.subject==subject)&(reference.condition=='LH')&(reference.order==4)]
    rows=[]; metadata=[]
    for low,high in BANDS:
        x=sosfiltfilt(butter(4,(low,high),btype='band',fs=float(raw['fs']),output='sos'),raw['signal'])[:,mask]
        variance=np.var(x,axis=1)
        parcel=int(np.argsort(variance,kind='stable')[len(variance)//2])
        scale=float(np.max(np.abs(x[parcel])))
        columns=dict(subject=subject,condition='LH',fmin=low,fmax=high,parcel=int(raw['labels'][parcel]),time=raw['time'][mask],observed=x[parcel]/scale)
        for model,graph in data.operators(100,raw['labels']).items():
            eig,vec=graph_modes(graph)
            fitter=BatchFitter(x[None],raw['time'][mask],eig,raw['resolution']@vec)
            ref=reference[(reference.fmin==low)&(reference.fmax==high)&(reference.model==model)].iloc[0]
            prediction=fitter.predict(ref.tau)
            actual=1-np.sum((x-prediction)**2)/np.sum((x-x.mean(1,keepdims=True))**2)
            if abs(actual-ref.r2)>1e-6:
                raise ValueError('Explicit trajectory baseline mismatch')
            columns[model]=prediction[parcel]/scale
            metadata.append(dict(subject=subject,condition='LH',fmin=low,fmax=high,parcel=int(raw['labels'][parcel]),model=model,tau=ref.tau,whole_parcel_r2=actual,parcel_r2=1-np.sum((x[parcel]-prediction[parcel])**2)/np.sum((x[parcel]-x[parcel].mean())**2),display_scale=scale))
        rows.append(pd.DataFrame(columns))
    pd.concat(rows,ignore_index=True).to_csv(data.RESULTS/'validation_trajectories.csv',index=False)
    pd.DataFrame(metadata).to_csv(data.RESULTS/'validation_trajectory_selection.csv',index=False)
    print('Example subject:',subject,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    run(args.workers)
    summarize()
    transfer()
    transfer_statistics()
    prepare_examples()
