"""Fit the 25 fixed-spectrum parcel permutations used in Figure S10.

Run ./run.sh relabel. Completed graph ensembles and recording fits are reused
only when their input and numerical-code fingerprints still match.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import inspect
import json
import time

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

import data
from data import cortex
from fit import fit_parcels
from model import coefficient_modes, graph_modes

OUT = data.CACHE / "relabelling"
K = 20
SEED = 20461102
CONDITIONS = ("LH", "RH", "LF", "RF")
MODELS = tuple(data.NETWORKS)


def relabel(original, distance, seed):
    """Randomize parcel assignment while preserving the complete graph spectrum.

    w_new[i,j] = w_original[p[i],p[j]]. Observations and resolution matrices
    remain in their original anatomical order. This tests regional alignment,
    not arbitrary orthogonal eigenvectors or a change of unlabelled topology.
    """
    permutation = np.random.default_rng(seed).permutation(len(original))
    w = original[np.ix_(permutation, permutation)]
    old_l, new_l = laplacian(original), laplacian(w)
    old_e, old_v = np.linalg.eigh(old_l)
    new_e = np.linalg.eigvalsh(new_l)
    upper = np.triu_indices(len(w), 1)
    support, reference = w > 0, original > 0
    assert np.array_equal(w, w.T) and not np.diag(w).any()
    assert np.array_equal(np.sort(w[upper]), np.sort(original[upper]))
    assert np.array_equal(support.sum(1), reference.sum(1)[permutation])
    assert np.allclose(w.sum(1), original.sum(1)[permutation], rtol=1e-14, atol=1e-12)
    components = connected_components(reference, directed=False, return_labels=False)
    assert connected_components(support, directed=False, return_labels=False) == components
    assert np.allclose(new_l, old_l[np.ix_(permutation, permutation)], rtol=1e-14, atol=1e-12)
    assert np.allclose(new_e, old_e, rtol=1e-12, atol=1e-10)
    residual = np.linalg.norm(new_l @ old_v[permutation] - old_v[permutation] * old_e) / np.linalg.norm(old_l)
    assert residual < 1e-12
    old_retained, old_basis = graph_modes(old_l)
    new_retained, new_basis = graph_modes(new_l)
    positive = old_retained[:K] > 0
    assert np.count_nonzero(new_retained == 0) == components - 1
    # The component-constant subspace must move with the permuted graph.
    zeros = components - 1
    if zeros:
        expected = old_basis[permutation, :zeros]
        actual = new_basis[:, :zeros]
        assert np.allclose(expected @ expected.T, actual @ actual.T, atol=1e-12)
    checks = dict(permutation=json.dumps(permutation.tolist()),
                  components=int(components), retained_zero_modes=int(zeros),
                  moved_parcel_fraction=float(np.mean(permutation != np.arange(len(w)))),
                  spectrum_max_absolute_error=float(np.max(abs(new_e-old_e))),
                  eigenpair_relative_residual=float(residual),
                  edges=int(reference[upper].sum()),
                  replaced_fraction=float(1-(support[upper]&reference[upper]).sum()/reference[upper].sum()),
                  adjacency_relative_difference=float(np.linalg.norm(w-original)/np.linalg.norm(original)),
                  laplacian_relative_difference=float(np.linalg.norm(new_l-old_l)/np.linalg.norm(old_l)),
                  laplacian_norm_change_pct=float(100*(np.linalg.norm(new_l)/np.linalg.norm(old_l)-1)),
                  strength_relative_max=float(np.max(abs(w.sum(1)/original.sum(1)-1))),
                  leading_spectrum_relative_rms=float(np.sqrt(np.mean(((new_retained[:K][positive]-old_retained[:K][positive])/old_retained[:K][positive])**2))),
                  length_ks=float(ks_2samp(distance[upper][support[upper]],distance[upper][reference[upper]]).statistic))
    assert checks["adjacency_relative_difference"] > 1e-6
    return w, checks, []


def laplacian(w):
    return np.diag(w.sum(axis=1)) - w


def inputs():
    graphs = []
    for stem in data.NETWORKS.values():
        w = np.load(data.ASSETS / f"{stem}100.npy")
        w = np.maximum(w, 0)
        w = (w + w.T) / 2
        np.fill_diagonal(w, 0)
        graphs.append(w)
    xyz, _, labels = cortex(100)
    centroids = np.array([xyz[labels == i].mean(0) for i in range(1, 101)])
    return np.stack(graphs), cdist(centroids, centroids)


def generate_graph(job):
    model, network, original, distance, digest = job
    seed = SEED + 1000 * MODELS.index(model) + network
    path = OUT / "chains" / f"{model}_{network:02d}.npz"
    if path.exists():
        with np.load(path) as pack:
            if str(pack["digest"]) != digest:
                raise ValueError(f"Stale graph cache: {path}")
            return model, network, pack["weights"], json.loads(str(pack["checks"])), json.loads(str(pack["history"]))
    start = time.perf_counter()
    w, checks, history = relabel(original, distance, seed)
    np.savez_compressed(path, weights=w, checks=json.dumps(checks), history=json.dumps(history), digest=digest)
    print(f"{model} {network}/25: replaced {checks['replaced_fraction']:.1%}, "
          f"weighted difference {checks['adjacency_relative_difference']:.1%}, "
          f"max strength error {checks['strength_relative_max']:.2%}, "
          f"{time.perf_counter()-start:.1f}s", flush=True)
    return model, network, w, checks, history


def prepare_graphs(count, workers):
    original, distance = inputs()
    # Plot-only edits must not invalidate computed networks or fits.
    functions = (inputs, laplacian, relabel)
    recipe = "".join(inspect.getsource(f) for f in functions)
    recipe += (data.SRC / "model.py").read_text()
    # Keep the signature used by the saved graph ensembles.
    digest = hashlib.sha256(original.tobytes() + distance.tobytes() + recipe.encode()
                            + f"{SEED}/10.0/0.01/10/{MODELS}".encode()).hexdigest()
    file = OUT / "networks.npz"
    if file.exists():
        pack = np.load(file)
        if str(pack["digest"]) != digest or len(pack["weights"]) != len(MODELS) * (count + 1):
            raise ValueError("Existing network cache differs; use a fresh output directory")
        return digest
    graphs, diagnostics, histories = {}, [], []
    jobs = [(model, n, original[i], distance, digest)
            for i, model in enumerate(MODELS) for n in range(1, count+1)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(generate_graph, jobs))
    for model, n, w, checks, history in results:
        graphs[model, n] = w
        diagnostics.append(dict(model=model, network=n, seed=SEED + 1000 * MODELS.index(model) + n, **checks))
        histories.extend(dict(model=model, network=n, **h) for h in history)
    # Separate chains must yield distinct graphs, not copies of one another.
    rows = []
    upper = np.triu_indices(100, 1)
    for model in MODELS:
        for i in range(1, count+1):
            for j in range(i + 1, count+1):
                a, b = graphs[model, i][upper], graphs[model, j][upper]
                difference = np.linalg.norm(a-b) / np.linalg.norm(a)
                assert difference > 1e-6
                rows.append(dict(model=model, network_i=i, network_j=j, weighted_difference=difference))
    weights, models, networks = [], [], []
    for i, model in enumerate(MODELS):
        for n in range(count+1):
            weights.append(original[i] if n == 0 else graphs[model, n])
            models.append(model)
            networks.append(n)
    np.savez_compressed(file, weights=np.stack(weights), models=models, networks=networks,
                        distance=distance, digest=digest)
    pd.DataFrame(diagnostics).to_csv(OUT / "graph_checks.csv", index=False)
    pd.DataFrame(histories).to_csv(OUT / "mixing.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "pairwise_graph_checks.csv", index=False)
    return digest


def fit_recording(job):
    subject, condition, digest = job
    source = data.SIGNALS / f"{subject}_{condition}.npz"
    output = OUT / "fits" / f"{subject}_{condition}.csv"
    provenance = hashlib.sha256(source.read_bytes() + digest.encode() +
                                inspect.getsource(fit_recording).encode() +
                                (data.SRC / "fit.py").read_bytes() +
                                (data.SRC / "model.py").read_bytes()).hexdigest()
    if output.exists():
        old = pd.read_csv(output)
        if old.provenance.eq(provenance).all():
            return str(output)
        raise ValueError(f"Stale fit cache: {output}")
    raw = data.load_parcels(subject, condition)
    signal, time_axis, fs = raw["signal"], raw["time"], float(raw["fs"])
    labels, resolution = raw["labels"].ravel().astype(int), raw["resolution"]
    assert signal.shape == (100, len(time_axis)) and resolution.shape == (100, 100)
    assert np.array_equal(np.sort(labels), np.arange(1, 101))
    mask = (time_axis >= -.1) & (time_axis <= .9)
    target = sosfiltfilt(butter(4, [4, 8], fs=fs, btype="band", output="sos"), signal, axis=1)[:, mask]
    target_hash = hashlib.sha256(target.tobytes()).hexdigest()
    sst = np.sum((target - target.mean(axis=1, keepdims=True)) ** 2)
    total = np.sum(target ** 2)
    pack = np.load(OUT / "networks.npz")
    rows = []
    idx = labels - 1
    for model, network, w in zip(pack["models"], pack["networks"], pack["weights"]):
        values, vectors = graph_modes(laplacian(w[np.ix_(idx, idx)]))
        values, vectors = values[:K], vectors[:, :K]
        mixing = resolution @ vectors
        q, r = np.linalg.qr(mixing, mode="reduced")
        assert np.min(abs(np.diag(r))) > np.max(abs(np.diag(r))) * 1e-10
        spatial = 1 - (total - np.sum((q.T @ target) ** 2)) / sst
        if network == 0:
            reference_q = q
        singular = np.clip(np.linalg.svd(reference_q.T @ q, compute_uv=False), 0, 1)
        r2, tau = fit_parcels(target, time_axis[mask], values, mixing)
        assert r2 <= spatial + 1e-7
        rows.append(dict(subject=subject, condition=condition, model=model, network=network,
                         r2=r2, tau=tau, coefficient_count=len(coefficient_modes(values)), spatial_r2=spatial,
                         subspace_overlap=np.mean(singular**2),
                         target_hash=target_hash, provenance=provenance))
    pd.DataFrame(rows).to_csv(output, index=False)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--graphs-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for folder in (OUT, OUT / "fits", OUT / "chains"):
        folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(version="allgraphs-parcel-relabel-v2", models=list(MODELS),
                    networks=25, seed_base=SEED, seed_model_offset=1000,
                    kernel="Independent uniform permutations p of 100 parcels: W_new = W_original[p,p]; no rescaling.",
                    constraints="Exact eigenvalues, weights, unlabelled graph topology, degree and strength multisets. Anatomical assignments and spatial embedding randomized.",
                    observation="MEG targets, parcel labels and resolution matrices remain fixed; only graph parcel assignment changes.",
                    roi=100, band_hz=[4, 8], modes=K, tau_bounds_seconds=[.01, 1],
                    linear_coefficients={m: len(coefficient_modes(graph_modes(g)[0][:K]))
                                         for m, g in data.operators(100, np.arange(1, 101)).items()},
                    zero_mode="All graphs are connected; exclude the global constant before selecting the first 20 positive-eigenvalue modes.",
                    selection="All 75 graph checks precede fitting; no graph selection by MEG scores.")
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("Protocol changed; use a fresh output directory")
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    digest = prepare_graphs(25, args.workers)
    if args.graphs_only:
        print("Graph checks complete.")
        return
    jobs = [(s, c, digest) for s in data.SUBJECTS for c in CONDITIONS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fit_recording, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            future.result()
            if n % 20 == 0 or n == len(jobs):
                print(f"Checked or fitted {n}/{len(jobs)} recordings", flush=True)
    table = pd.concat([pd.read_csv(p, dtype={"subject": str})
                       for p in sorted((OUT / "fits").glob("*.csv"))], ignore_index=True)
    assert len(table) == data.recording_count() * 26 * len(MODELS)
    assert not table.duplicated(["subject", "condition", "model", "network"]).any()
    table.drop(columns=["provenance"]).to_csv(data.RESULTS / "relabelling_fits.csv", index=False)
    refresh()
    summarize_results()


SOURCE = OUT
SOURCES = {"Relabelled": SOURCE}
TABLE = data.RESULTS / "relabelling_comparison.csv"
SPECTRA = data.RESULTS / "relabelling_spectra.csv"
CHECKS = data.RESULTS / "relabelling_checks.csv"
STATISTICS = data.RESULTS / "relabelling_statistics.csv"
CONTROL_MODELS = {name: MODELS for name in SOURCES}
BOOTSTRAP_SEED = 20261102
BOOTSTRAP_SAMPLES = 5000

def refresh_one(source):
    models = json.loads((source / "protocol.json").read_text())["models"]
    fits = pd.read_csv(data.RESULTS / "relabelling_fits.csv", dtype={"subject": str})
    assert len(fits) == data.recording_count() * 26 * len(models) and set(fits.model) == set(models)
    assert not fits.duplicated(["subject", "condition", "model", "network"]).any()
    assert fits.groupby(["subject", "condition"]).target_hash.nunique().eq(1).all()
    # Independently generated manuscript fits must reproduce all three baselines.
    old = pd.read_csv(data.RESULTS / "modes.csv", dtype={"subject": str})
    old = old[(old.roi == 100) & (old.band == "Theta") & (old.modes == K)]
    baseline = fits[fits.network == 0].merge(old, on=["subject", "condition", "model"],
                                           suffixes=("_new", "_old"), validate="one_to_one")
    assert len(baseline) == data.recording_count() * len(models)
    assert baseline.target_hash_new.eq(baseline.target_hash_old).all()
    assert abs(baseline.r2_new-baseline.r2_old).max() < 1e-6
    participant = fits.groupby(["subject", "model", "network"]).agg(
        overlap=("subspace_overlap", "mean"), r2=("r2", "mean"), tau=("tau", "mean"),
        conditions=("condition", "size")).reset_index()
    pack = np.load(source / "networks.npz")
    spectra = []
    for model, network, w in zip(pack["models"], pack["networks"], pack["weights"]):
        operator = np.diag(w.sum(1)) - w
        if network == 0:
            assert np.allclose(operator, data.operators(100, np.arange(1, 101))[model])
        values = graph_modes(operator)[0][:K]
        spectra.extend(dict(model=model, network=network, mode=i, eigenvalue=v)
                       for i, v in enumerate(values, 1))
    return participant, pd.DataFrame(spectra), pd.read_csv(source / "graph_checks.csv")


def refresh():
    tables, spectra, checks = [], [], []
    for control, source in SOURCES.items():
        t, s, c = refresh_one(source)
        t["control"] = np.where(t.network == 0, "Original", control)
        s["control"] = np.where(s.network == 0, "Original", control)
        c["control"] = control
        if tables:
            a = tables[0].query("network == 0")
            a = a[a.model.isin(CONTROL_MODELS[control])].sort_values(["subject", "model"])
            b = t.query("network == 0").sort_values(["subject", "model"])
            assert np.allclose(a[["r2", "tau"]], b[["r2", "tau"]], rtol=0, atol=1e-10)
            t, s = t[t.network > 0], s[s.network > 0]
        tables.append(t)
        spectra.append(s)
        checks.append(c)
    pd.concat(tables, ignore_index=True).to_csv(TABLE, index=False)
    pd.concat(spectra, ignore_index=True).to_csv(SPECTRA, index=False)
    pd.concat(checks, ignore_index=True).to_csv(CHECKS, index=False)


def read_inputs():
    table = pd.read_csv(TABLE, dtype={"subject": str})
    spectra, checks = pd.read_csv(SPECTRA), pd.read_csv(CHECKS)
    protocols = {
        "Relabelled": json.loads((OUT / "protocol.json").read_text()),
    }
    assert protocols["Relabelled"]["version"] == "allgraphs-parcel-relabel-v2"
    for control, protocol in protocols.items():
        assert protocol["modes"] == K and protocol["networks"] == 25
        assert protocol["models"] == list(CONTROL_MODELS[control])
        assert protocol["roi"] == 100 and protocol["band_hz"] == [4, 8]
        assert protocol["tau_bounds_seconds"] == [.01, 1]
        group = checks[checks.control == control]
        assert len(group) == 25 * len(CONTROL_MODELS[control])
        assert (group.spectrum_max_absolute_error < 1e-10).all()
        assert (group.eigenpair_relative_residual < 1e-12).all()
        assert group.retained_zero_modes.eq(group.components - 1).all()
        for encoded in group.permutation:
            assert sorted(json.loads(encoded)) == list(range(100))
        assert (group.adjacency_relative_difference > 1e-6).all()
        for model in CONTROL_MODELS[control]:
            assert set(group[group.model == model].network) == set(range(1, 26))
    assert set(checks.control) == set(SOURCES)
    assert set(table.control) == set(spectra.control) == {"Original", *SOURCES}
    assert len(checks) == 25 * len(MODELS) and not checks.duplicated(["control", "model", "network"]).any()
    assert len(table) == 51 * 26 * len(MODELS) and not table.duplicated(["control", "subject", "model", "network"]).any()
    assert set(table.subject) == set(data.SUBJECTS) and set(table.model) == set(MODELS)
    expected = {s: sum(s in data.subjects(c, 100) for c in ("LH", "RH", "LF", "RF")) for s in data.SUBJECTS}
    assert table.conditions.eq(table.subject.map(expected)).all()
    for control in ("Original", *SOURCES):
        ids = {0} if control == "Original" else set(range(1, 26))
        models = MODELS if control == "Original" else CONTROL_MODELS[control]
        for model in models:
            g = table[(table.control == control) & (table.model == model)]
            assert set(g.network) == ids and g.groupby("subject").size().eq(len(ids)).all()
            assert set(spectra[(spectra.control == control) & (spectra.model == model)].network) == ids
    assert np.allclose(table[table.network == 0].overlap, 1)
    assert len(spectra) == 26 * len(MODELS) * K
    assert not spectra.duplicated(["control", "model", "network", "mode"]).any()
    for _, group in spectra.groupby(["control", "model", "network"]):
        assert set(group["mode"]) == set(range(1, K+1))
        values = group.sort_values("mode").eigenvalue.to_numpy()
        assert np.isfinite(values).all() and (values >= 0).all() and (np.diff(values) >= 0).all()
        model = group.model.iloc[0]
        expected_zeros = int(checks[checks.model == model].retained_zero_modes.iloc[0])
        assert np.count_nonzero(values == 0) == expected_zeros
    for model in MODELS:
        original_values = spectra[(spectra.model == model) & (spectra.network == 0)].sort_values("mode").eigenvalue.to_numpy()
        for _, group in spectra[(spectra.control == "Relabelled") & (spectra.model == model)].groupby("network"):
            assert np.allclose(group.sort_values("mode").eigenvalue, original_values, rtol=1e-12, atol=1e-10)
    assert np.isfinite(table[["overlap", "r2", "tau"]]).all().all()
    assert table.overlap.between(0, 1+1e-10).all() and (table.r2 <= 1+1e-10).all()
    assert table.tau.between(.01-1e-12, 1+1e-12).all()
    return table, spectra, checks


def summarize_one(table, spectra, models_to_use=MODELS):
    subjects = sorted(data.SUBJECTS)
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(len(subjects), size=(BOOTSTRAP_SAMPLES, len(subjects)))

    def interval(values):
        values = np.asarray(values)
        assert values.shape == (len(subjects),)
        return dict(mean=float(values.mean()), ci=np.quantile(values[draws].mean(1), [.025, .975]).tolist())

    models, originals, scrambled = {}, {}, {}
    for model in models_to_use:
        group = table[table.model == model]
        original = group[group.network == 0].set_index("subject").loc[subjects]
        randomized = group[group.network > 0].groupby("subject")[["r2", "overlap"]].mean().loc[subjects]
        originals[model], scrambled[model] = original.r2, randomized.r2
        individual = group[group.network > 0].pivot(index="subject", columns="network", values="r2").loc[subjects]
        delta = randomized.r2 - original.r2
        models[model] = dict(original_r2=interval(original.r2), scrambled_r2=interval(randomized.r2),
                             fit_change=interval(delta), overlap=interval(randomized.overlap),
                             participants_improved=int((delta > 0).sum()),
                             networks_improved=int((individual.mean(axis=0) > original.r2.mean()).sum()))
    for model in models_to_use:
        models[model]["original_minus_original_structural"] = interval(originals[model]-originals["Structural"])
        models[model]["scrambled_minus_original_structural"] = interval(scrambled[model]-originals["Structural"])
    contrasts = {}
    for model in models_to_use[1:]:
        contrasts[f"Original Structural minus {model}"] = interval(originals["Structural"]-originals[model])
        contrasts[f"Scrambled Structural minus {model}"] = interval(scrambled["Structural"]-scrambled[model])
    distances = []
    for model in models_to_use:
        group = spectra[spectra.model == model]
        reference = group[group.network == 0].sort_values("mode").eigenvalue.to_numpy()
        for network, spectrum in group[group.network > 0].groupby("network"):
            values = spectrum.sort_values("mode").eigenvalue.to_numpy()
            positive = reference > 0
            assert np.array_equal(values == 0, reference == 0)
            distances.append(dict(model=model, network=int(network),
                                  zero_modes=int(np.sum(~positive)),
                                  relative_rms=float(np.sqrt(np.mean(((values[positive]-reference[positive])/reference[positive])**2)))))
    return dict(models=models, paired_family_contrasts=contrasts, spectrum_distance=distances)


def summarize(table, spectra):
    return {control: summarize_one(table[table.control.isin(["Original", control])],
                                   spectra[spectra.control.isin(["Original", control])], CONTROL_MODELS[control])
            for control in SOURCES}


def save_statistics(result):
    rows = []
    for control, summary in result.items():
        for model, metrics in summary["models"].items():
            for metric in ("original_r2", "scrambled_r2", "fit_change", "overlap",
                           "original_minus_original_structural", "scrambled_minus_original_structural"):
                rows.append(dict(control=control, metric=metric, contrast=model, **metrics[metric]))
        for name, item in summary["paired_family_contrasts"].items():
            rows.append(dict(control=control, metric="paired_r2_difference", contrast=name, **item))
    for row in rows:
        row["ci_low"], row["ci_high"] = row.pop("ci")
        row["participants"] = 51
        row["bootstrap_samples"] = BOOTSTRAP_SAMPLES
    pd.DataFrame(rows).to_csv(STATISTICS, index=False)


def summarize_results():
    table, spectra, checks = read_inputs()
    result = summarize(table, spectra)
    save_statistics(result)
    for control, source in SOURCES.items():
        summary = result[control]
        summary["controls"] = {model: {key: group[key].agg(["min", "max"]).tolist()
                                       for key in ("replaced_fraction", "adjacency_relative_difference",
                                                   "laplacian_relative_difference", "strength_relative_max",
                                                   "laplacian_norm_change_pct")}
                                for model, group in checks[checks.control == control].groupby("model")}
        summary["scope"] = dict(roi=100, band_hz=[4, 8], modes=K, networks_per_family=25,
                                 bootstrap_unit="participant; conditions and fixed ensemble averaged first",
                                 intervals="pointwise exploratory intervals conditional on fixed ensembles",
                                 fit_change="relabelled minus own original, positive favors relabelling",
                                 panel_a="absolute original and relabelled reconstruction R2",
                                 panel_b="relabelled minus own family original, paired within participants",
                                 figure_distribution="51 participant values; conditions and 25 permutations averaged within participant; boxes median/IQR, whiskers 1.5 IQR; dots include all participants",
                                 similarity="observed spatial subspace versus own original; not dynamic equivalence")
        (data.RESULTS / "relabelling_summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
