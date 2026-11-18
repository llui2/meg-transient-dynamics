#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"

help() {
    cat <<'USAGE'
Usage: ./run.sh COMMAND

  plots                 Redraw all 18 figures from saved data
  fit [stage] [--roi N]  Fit all models, or parcel/spectrum/modal/modes/modes-spectrum
  relabel               Fit the parcel permutations used in Fig. S10
  validation            Run signal surrogates and timescale transfer (Figs. S11-S12)
  preprocessing         Run filter/window sensitivity (S13)
  graphs [options]      Verify graph construction; --output DIR saves rebuilt controls
  reconstruct [options] Reconstruct HCP motor runs 10 and 11 (requires MATLAB)
  source-leakage         Rebuild the S1 source-resolution inputs (requires MATLAB)

Set WORKERS to control parallel analyses (default 4).
See README.md for data and setup.
USAGE
}

command=${1:-help}
if (( $# )); then shift; fi
case "$command" in
    help|-h|--help) help; exit 0 ;;
    plots|fit|relabel|validation|preprocessing|graphs|reconstruct|source-leakage) ;;
    *) printf 'Unknown command: %s\n' "$command" >&2; help >&2; exit 2 ;;
esac
if [[ $command != fit && $command != reconstruct && $command != graphs && $# -ne 0 ]]; then
    help >&2
    exit 2
fi
workers=${WORKERS:-4}
if [[ ! $workers =~ ^[1-9][0-9]*$ ]]; then
    echo 'WORKERS must be a positive integer.' >&2
    exit 2
fi
if [[ -n ${PYTHON:-} ]]; then
    python=$PYTHON
elif [[ -x .venv/bin/python ]]; then
    python="$PWD/.venv/bin/python"
else
    python=python3
fi
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MPLBACKEND=Agg

run() { "$python" -m "$@"; }

case "$command" in
    plots)
        for module in fig1 fig2 fig3 fig4 fig5 source_leakage parcel_leakage; do
            run "plots.$module"
        done
        for roi in 100 200 400; do
            ROI="$roi" run plots.modal_fit
        done
        for module in spatial relabelling validation preprocessing; do
            run "plots.$module"
        done
        for roi in 200 400; do
            ROI="$roi" run plots.fig3
        done
        echo 'All 18 manuscript figures have been regenerated.' ;;
    fit)
        if (( $# == 0 )); then set -- all; fi
        run analysis "$@" --workers "$workers" ;;
    relabel)
        run relabelling --workers "$workers"
        run plots.relabelling ;;
    validation)
        run validation --workers "$workers"
        run plots.validation ;;
    preprocessing)
        run preprocessing --workers "$workers"
        run plots.preprocessing ;;
    graphs)
        run graphs "$@" ;;
    reconstruct)
        run reconstruction --workers "$workers" "$@" ;;
    source-leakage)
        run source_leakage ;;
esac
