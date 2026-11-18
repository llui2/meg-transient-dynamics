"""Full-cohort modal fits and participant averages for Figures S6-S8."""
import numpy as np
import pandas as pd

import data
from fit import MODAL_BANDS

BANDS = list(MODAL_BANDS)
LABELS = [f"{low:g}–{high:g}" for low, high in MODAL_BANDS.values()]
KEYS = ["condition", "subject", "band", "model"]


def load(roi):
    table = pd.read_csv(data.RESULTS / "modal_fits.csv", dtype={"subject": str})
    table = table[table.roi.eq(roi)]
    required = {"fmin", "fmax", "order", "tau_min", "tau_max", "modes", "target_hash"}
    if not required.issubset(table.columns):
        raise ValueError("Old modal frequency grid: rerun ./run.sh fit modal with a fresh cache")
    if (not table.order.eq(4).all() or not table.modes.eq(roi-1).all()
            or not table.tau_min.eq(.01).all() or not table.tau_max.eq(1.).all()):
        raise ValueError("Incompatible modal fitting settings")
    for band, (low, high) in MODAL_BANDS.items():
        part = table[table.band.eq(band)]
        if not part.fmin.eq(low).all() or not part.fmax.eq(high).all():
            raise ValueError(f"Wrong frequency bounds for {band}")
    if table.duplicated(KEYS).any():
        raise ValueError(f"Duplicate modal fit keys at ROI{roi}")
    if not np.isfinite(table[["r2", "tau"]]).all().all():
        raise ValueError(f"Nonfinite modal fits at ROI{roi}")
    expected = {(condition, subject, band, model)
                for condition in ("LH", "RH", "LF", "RF")
                for subject in data.subjects(condition, roi)
                for band in BANDS for model in data.NETWORKS}
    actual = set(table[KEYS].itertuples(index=False, name=None))
    if actual - expected:
        raise ValueError(f"Unexpected modal fit keys at ROI{roi}")
    if expected - actual:
        raise ValueError(f"ROI{roi}: {len(expected - actual)} modal fits missing; "
                         "run ./run.sh fit modal first.")
    return table


def participant_scores(roi):
    """Average the four separately fitted task scores within each participant."""
    table = load(roi)
    grouped = table.groupby(["subject", "band", "model"])
    if not grouped.size().eq(4).all():
        raise ValueError("Modal participant averages require all four motor conditions")
    scores = grouped.r2.mean().unstack("model")
    expected = pd.MultiIndex.from_product([data.SUBJECTS, BANDS],
                                         names=["subject", "band"])
    scores = scores.reindex(index=expected, columns=data.NETWORKS)
    if not np.isfinite(scores.to_numpy()).all():
        raise ValueError("Incomplete participant-level modal scores")
    return scores
