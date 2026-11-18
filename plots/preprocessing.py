"""Figure S13: paired structural advantages across fixed preprocessing settings."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import data
from plots.style import (apply_style, MODEL_COLORS, MODEL_LABELS,
                         participant_boxplot, box_position, boxplot_axis, save_figure)
from preprocessing import SETTINGS, BANDS, MODELS, SEED


PANELS = (
    ('Filter order', ('order2', 'reference', 'order6'), ('2', '4', '6')),
    ('Window duration (s)', ('shorter', 'reference', 'longer'), ('0.8', '1.0', '1.2')),
    ('Window start (s)', ('reference', 'onset', 'later'), ('−0.1', '0', '0.1')),
)


def plot():
    participant = pd.read_csv(data.RESULTS/'preprocessing_participants.csv',
                              dtype={'subject': str})
    keys = ['subject', 'setting', 'fmin', 'fmax', 'model']
    expected = {(subject, s.name, low, high, m) for subject in data.SUBJECTS
                for s in SETTINGS for low, high in BANDS for m in MODELS}
    if (set(participant[keys].itertuples(index=False, name=None)) != expected
            or participant.duplicated(keys).any()):
        raise ValueError('Figure requires complete sensitivity results for all 51 participants')
    if not np.isfinite(participant.r2).all():
        raise ValueError('Nonfinite sensitivity results')
    wide = participant.pivot(index=keys[:-1], columns='model', values='r2')
    advantages = wide[list(MODELS[1:])].rsub(wide.Structural, axis=0).reset_index()
    apply_style()
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(4, 3, figsize=(6.2, 5.4), sharey='row')
    fig.subplots_adjust(left=.13, right=.985, bottom=.10, top=.93, hspace=.28, wspace=.20)
    for b, (low, high) in enumerate(BANDS):
        band = advantages[(advantages.fmin == low) & (advantages.fmax == high)]
        values = band[list(MODELS[1:])].to_numpy()
        ymin = min(0., values.min())
        ymax = max(0., values.max())
        margin = max((ymax-ymin)*.08, .003)
        for col, (title, settings, labels) in enumerate(PANELS):
            ax = axes[b, col]
            ax.axhline(0, color='.5', alpha=.65, lw=.8, ls='--', zorder=0)
            for group, setting in enumerate(settings):
                selected = band[band.setting == setting].set_index('subject').loc[list(data.SUBJECTS)]
                for j, model in enumerate(MODELS[1:]):
                    participant_boxplot(ax, selected[model].to_numpy(),
                                        box_position(group, j, 2), MODEL_COLORS[model], rng)
            boxplot_axis(ax, labels, 2)
            ax.set_ylim(ymin-margin, ymax+margin)
            ax.tick_params(axis='both', labelsize=8)
            if b == len(BANDS)-1:
                ax.set_xlabel(title, labelpad=5)
            if col == 0:
                ax.set_ylabel(f'{low}–{high} Hz  '+r'$\Delta R^2$')
    # Match the other compact box plots: one x unit per typographic point,
    # with axis limits ending just beyond the outer boxes.
    old_width = fig.get_figwidth()
    positions = [[ax.get_position().frozen() for ax in row] for row in axes]
    left = positions[0][0].x0 * old_width
    right = (1-positions[0][-1].x1) * old_width
    lower, upper = axes[0, 0]._boxplot_extent
    plot_width = (upper-lower)/72
    gap = .32  # Keep the single-line bottom labels clear of one another.
    width = left + 3*plot_width + 2*gap + right
    fig.set_size_inches(width, fig.get_figheight(), forward=False)
    for row, row_positions in zip(axes, positions):
        for col, (ax, position) in enumerate(zip(row, row_positions)):
            ax.set_position([(left+col*(plot_width+gap))/width,
                             position.y0, plot_width/width, position.height])
    fig.legend(handles=[Line2D([], [], color=MODEL_COLORS[m],
                              label=f'Structural − {MODEL_LABELS[m]}') for m in MODELS[1:]],
               loc='upper center', ncol=2, bbox_to_anchor=(.55,.99))
    exported = save_figure(fig, data.ROOT/'figures/supp/figS13.pdf', None)
    print(f'Figure S13 compact export width: {exported:.6f} inches')
    plt.close(fig)
    print('Created Figure S13 from all seven predefined settings.')


if __name__ == '__main__':
    plot()
