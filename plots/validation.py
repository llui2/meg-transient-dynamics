"""Render supplementary Figures S5, S11, and S12 from the validation results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import data
from plots.style import apply_style,MODEL_COLORS,MODEL_LABELS,panel_label,participant_boxplot,box_position,boxplot_axis,compact_boxplot_canvas,save_figure
from validation import BANDS,MODELS,SEED

FIGURES=data.ROOT/'figures/supp'
LABELS=[f'{a}–{b}' for a,b in BANDS]


def finish(fig,name):
    width=6.2
    if name in ('figS11','figS12'):
        compact_boxplot_canvas(fig)
        width=None
    exported=save_figure(fig,FIGURES/f'{name}.pdf',width)
    if exported is not None:
        print(f'{name}: compact export width {exported:.6f} inches')
    plt.close(fig)


def surrogate_figure():
    participant=pd.read_csv(data.RESULTS/'validation_surrogates.csv',dtype={'subject':str})
    fig,axes=plt.subplots(2,1,figsize=(3.6,3.7),sharex=True)
    fig.subplots_adjust(left=.17,right=.99,bottom=.13,top=.85,hspace=.27)
    rng=np.random.default_rng(SEED)
    for b,(low,high) in enumerate(BANDS):
        subset=participant[(participant.fmin==low)&(participant.fmax==high)]
        wide=subset.pivot(index=['subject','surrogate'],columns='model',values='r2')
        for j,control in enumerate(MODELS[1:]):
            delta=(wide.Structural-wide[control]).unstack('surrogate').reindex(data.SUBJECTS)
            observed=delta[0].to_numpy(); null=delta.loc[:,1:].mean(axis=1).to_numpy()
            for k,(values,hatch) in enumerate(((observed,None),(null,'////'))):
                participant_boxplot(axes[0],values,box_position(b,2*j+k,4),MODEL_COLORS[control],rng,hatch=hatch)
            participant_boxplot(axes[1],observed-null,box_position(b,j,2,slots=4),MODEL_COLORS[control],rng)
    for ax,letter in zip(axes,'ab'):
        ax.axhline(0,color='.5',alpha=.65,lw=.8,ls='--')
        boxplot_axis(ax,LABELS,4 if letter=='a' else 2,slots=4)
        panel_label(ax,letter)
    axes[0].set_ylabel(r'$\Delta R^2$')
    axes[1].set_ylabel(r'Excess $\Delta R^2$')
    axes[1].set_xlabel('Frequency (Hz)')
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[c],label=f'vs {MODEL_LABELS[c]}') for c in MODELS[1:]]+[Patch(facecolor='none',edgecolor='.25',label='Observed'),Patch(facecolor='none',edgecolor='.25',hatch='////',label='Surrogates')],loc='upper center',ncol=2,bbox_to_anchor=(.53,1.01))
    finish(fig,'figS11')




def trajectory_figure():
    values=pd.read_csv(data.RESULTS/'validation_trajectories.csv',dtype={'subject':str})
    fig,axes=plt.subplots(4,2,figsize=(6.2,4.8),sharex=True)
    fig.subplots_adjust(left=.10,right=.99,bottom=.11,top=.84,hspace=.28,wspace=.20)
    for b,(low,high) in enumerate(BANDS):
        group=values[(values.fmin==low)&(values.fmax==high)]
        axes[b,0].plot(group.time,group.observed,color='.15',lw=1.6,label='Observed')
        for model in MODELS:
            axes[b,0].plot(group.time,group[model],color=MODEL_COLORS[model],lw=1.1,label=MODEL_LABELS[model])
            axes[b,1].plot(group.time,group.observed-group[model],color=MODEL_COLORS[model],lw=1.1)
        for col in range(2):
            ax=axes[b,col]
            ax.axhline(0,color='.5',alpha=.65,lw=.8,ls='--')
            ax.axvline(0,color='.7',lw=.6)
        extent=max(float(np.abs(group[['observed', *MODELS]].to_numpy()).max()),
                   max(float(np.abs(group.observed-group[m]).max()) for m in MODELS)) * 1.08
        for ax in axes[b]: ax.set_ylim(-extent,extent)
        axes[b,0].set_title(f'{low}–{high} Hz',loc='left',pad=3,fontsize=8)
    for ax,title in zip(axes[0],('Fits','Residuals')):
        position=ax.get_position()
        fig.text((position.x0+position.x1)/2,.89,title,ha='center',fontsize=9)
    fig.supylabel('Amplitude',x=.015,fontsize=9)
    fig.supxlabel('Time (s)',y=.015,fontsize=9)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.53,.98))
    finish(fig,'figS5')


def transfer_figure():
    frame=pd.read_csv(data.RESULTS/'validation_transfer.csv',dtype={'subject':str})
    # Equal target-condition weights: every target has the same number of source
    # conditions within a participant (three sources for each pooled target).
    participant=frame.groupby(['subject','fmin','fmax','model'])[['r2','r2_refitted']].mean().reset_index()
    rng=np.random.default_rng(SEED)
    fig,axes=plt.subplots(2,1,figsize=(4.4,3.7),sharex=True)
    fig.subplots_adjust(left=.14,right=.99,bottom=.13,top=.85,hspace=.27)
    for b,(low,high) in enumerate(BANDS):
        group=participant[(participant.fmin==low)&(participant.fmax==high)]
        for j,model in enumerate(MODELS):
            g=group[group.model==model].set_index('subject').reindex(data.SUBJECTS)
            for k,(key,hatch) in enumerate((('r2_refitted',None),('r2','////'))):
                participant_boxplot(axes[0],g[key],box_position(b,2*j+k,6),MODEL_COLORS[model],rng,hatch=hatch)
        for j,control in enumerate(MODELS[1:]):
            wide=group.pivot(index='subject',columns='model',values='r2').reindex(data.SUBJECTS)
            delta=(wide.Structural-wide[control]).to_numpy()
            participant_boxplot(axes[1],delta,box_position(b,j,2,slots=6),MODEL_COLORS[control],rng)
    for ax,letter in zip(axes,'ab'):
        ax.axhline(0,color='.5',alpha=.65,lw=.8,ls='--')
        boxplot_axis(ax,LABELS,6 if letter=='a' else 2,slots=6); panel_label(ax,letter)
    axes[0].set_ylabel(r'$R^2$')
    axes[1].set_ylabel(r'$\Delta R^2$')
    axes[1].set_xlabel('Frequency (Hz)')
    fig.legend(handles=[Line2D([],[],color=MODEL_COLORS[m],label=MODEL_LABELS[m]) for m in MODELS],loc='upper center',ncol=3,bbox_to_anchor=(.53,1.01))
    fig.legend(handles=[Patch(facecolor='none',edgecolor='.25',label=r'Refitted $\tau$'),Patch(facecolor='none',edgecolor='.25',hatch='////',label=r'Transferred $\tau$')],loc='upper center',ncol=2,bbox_to_anchor=(.53,.95))
    finish(fig,'figS12')


if __name__=='__main__':
    apply_style()
    trajectory_figure()
    surrogate_figure()
    transfer_figure()
