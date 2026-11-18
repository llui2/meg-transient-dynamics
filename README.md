# Linking connectome structure to transient brain dynamics

This repository provides the data and code to reproduce the analyses and figures
reported in this study. Using motor-task MEG recordings from 51 Human Connectome
Project participants, we test how well models based on the structural connectome
explain transient cortical responses compared with models based on control graphs.
We evaluate model fits at three spatial resolutions (100, 200 and 400 cortical parcels).

Regenerate the five main and thirteen supplementary figures as PDFs in `figures/`
using the included results.

```bash
./run.sh plots
```

`data/signals/` contains reconstructed signals and resolution matrices at
100, 200 and 400 parcels for each participant and motor condition.
`assets/` holds graph inputs and `data/results/` holds saved fits.
In the tables, `Geometric` denotes the paper’s local graph. Scores are
in-sample R² and relaxation times are in seconds.

To refit the models from the reconstructed signals and redraw the figures, run

```bash
./run.sh fit
./run.sh fit spectrum --roi 200
./run.sh fit spectrum --roi 400
./run.sh relabel
./run.sh validation
./run.sh preprocessing
./run.sh plots
```

Fits resume from `.cache/`. After changing graph inputs, fitting code or the
frequency grid, export `MEG_CACHE` to a fresh directory for all stages.
Set `WORKERS` to change the default of four workers. Run `./run.sh help`
for individual stages and graph checks.

Optional reconstruction from raw HCP data requires MATLAB, FieldTrip, Connectome Workbench and the HCP
files for the cohort in `src/data.py`. Under `MEG_DOWNLOADS`, preserve
`<subject>/MEG/Motort/tmegpreproc/` and `<subject>/MEG/anatomy/`.

```bash
export MEG_DOWNLOADS=/path/to/HCP
export FIELDTRIP_HOME=/path/to/fieldtrip
./run.sh reconstruct
./run.sh source-leakage
```

HCP-derived data are subject to the [HCP data-use terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
The earlier implementation is tagged `v1.0.0`.
