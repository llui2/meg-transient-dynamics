"""Figure 4: structural parcel-space advantage as the number of modes increases."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data
from plots.style import MODEL_COLORS, MODEL_LABELS, panel_label, apply_style, save_figure
from fit import SPECTRUM_BANDS, SPECTRUM_OUTPUT
from fit import TAU_MIN, TAU_MAX

apply_style()

root = Path(__file__).resolve().parents[1]
bands = list(SPECTRUM_BANDS)
models = ["Structural", "EDR", "Geometric"]
conditions = ["LH", "RH", "LF", "RF"]
modes = np.arange(1, 100)

df = pd.read_csv(SPECTRUM_OUTPUT, dtype={"subject": str, "fmin": float, "fmax": float})
if ("tau_identified" not in df or not df.tau_identified.all()
        or not df.coefficient_count.eq(2 * df.modes).all()):
    raise ValueError("Connected graphs require two temporal coefficients per retained mode")
if (not df.roi.eq(100).all() or not df.order.eq(4).all()
        or not df.tau_min.eq(TAU_MIN).all() or not df.tau_max.eq(TAU_MAX).all()
        or not df.tau.between(TAU_MIN, TAU_MAX).all()
        or not np.isfinite(df.r2).all()
        or not np.isfinite(df.tau).all()
        or not df.band.isin(bands).all()):
    raise ValueError("Figure 4 requires four-band parcel fits with Figure 3 tau bounds")
for band, (low, high) in SPECTRUM_BANDS.items():
    part = df[df.band.eq(band)]
    if not part.fmin.eq(low).all() or not part.fmax.eq(high).all():
        raise ValueError(f"Mismatched frequency bounds for {band}")
keys = ["subject", "condition", "band", "model", "modes"]
if df.duplicated(keys).any():
    raise ValueError("Duplicate cumulative fits")
if not np.isfinite(df.r2).all() or (df.r2 > 1 + 1e-7).any():
    raise ValueError("Invalid cumulative parcel-space R²")
# The K=99 endpoints must reproduce the frequency-resolved Figure 3 scores.
reference = pd.read_csv(data.RESULTS / "spectrum.csv", dtype={"subject": str})
endpoint_keys = ["subject", "condition", "order", "fmin", "fmax", "model"]
endpoints = df[df.modes.eq(99)].merge(
    reference[endpoint_keys + ["r2", "tau_min", "tau_max"]], on=endpoint_keys,
    how="left", validate="one_to_one", suffixes=("", "_reference"))
if (endpoints.r2_reference.isna().any()
        or not endpoints.tau_min_reference.eq(TAU_MIN).all()
        or not endpoints.tau_max_reference.eq(TAU_MAX).all()
        or not np.allclose(endpoints.r2, endpoints.r2_reference, rtol=0, atol=1e-6)):
    raise ValueError("Figure 4 K=99 endpoints do not match Figure 3")
available = set(df[keys].itertuples(index=False, name=None))
subjects = []
for subject in data.SUBJECTS:
    required = {(subject, condition, band, model, int(k))
                for condition in conditions if subject in data.subjects(condition, 100)
                for band in bands for model in models for k in modes}
    if required <= available:
        subjects.append(subject)
if not subjects:
    raise ValueError("No complete participants: run ./run.sh fit modes-spectrum")

index = pd.MultiIndex.from_tuples(
    [(subject, condition) for subject in subjects for condition in conditions
     if subject in data.subjects(condition, 100)],
    names=["subject", "condition"])
columns = pd.MultiIndex.from_product([bands, models, modes],
                                     names=["band", "model", "modes"])
scores = df.pivot(index=["subject", "condition"],
                  columns=["band", "model", "modes"], values="r2")
scores = scores.reindex(index=index, columns=columns)
if not np.isfinite(scores.to_numpy()).all():
    raise ValueError("Incomplete participant mode fits")

# Each participant is one statistical unit, even with multiple motor conditions.
rng = np.random.default_rng(20261025)
weights = rng.multinomial(len(subjects), [1 / len(subjects)] * len(subjects),
                          size=10000) / len(subjects)
fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.9), sharex=True, sharey=True)
fig.subplots_adjust(left=0.10, right=0.995, bottom=0.11, top=0.90,
                    wspace=0.14, hspace=0.28)
# Nested mode sets define continuous curves with participant-bootstrap intervals.
for j, (band, ax) in enumerate(zip(bands, axes.flat)):
    for m, control in enumerate(models[1:], start=1):
        difference = scores[band]["Structural"] - scores[band][control]
        paired = difference.groupby(level="subject").mean().reindex(subjects).to_numpy()
        mean = paired.mean(axis=0)
        lo, hi = np.quantile(weights @ paired, [0.025, 0.975], axis=0)
        ax.plot(modes, mean, color=MODEL_COLORS[control], linewidth=1.5,
                linestyle="-",
                label="Structural − " + MODEL_LABELS[control])
        ax.fill_between(modes, lo, hi, color=MODEL_COLORS[control], alpha=0.12)
    ax.axhline(0, color="0.5", alpha=0.65, linewidth=0.8, linestyle="--", zorder=0)
    ax.text(0.98, 0.88, band,
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.set_xlim(1, 99)
    ax.set_xticks([1, 20, 40, 60, 80, 99])
    ax.yaxis.labelpad = 2
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, chr(97 + j))
    if j % 2 == 0:
        ax.set_ylabel(r"$\Delta R^2_K$")
for ax in axes[-1]:
    ax.set_xlabel(r"Modes $K$", labelpad=2)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.985),
           ncol=2, frameon=False, fontsize=8, borderaxespad=0, borderpad=.15)
full_cohort = len(subjects) == len(data.SUBJECTS)
if full_cohort:
    output = root / "figures/main"
    output.mkdir(parents=True, exist_ok=True)
    target = output / "fig4.pdf"
    save_figure(fig, target, 6.2)
    print(f"Figure 4: {len(subjects)} complete participants; saved {target}")
else:
    print(f"Figure 4: {len(subjects)} complete participants; "
          "no manuscript figure saved for an incomplete cohort.")
plt.close(fig)
