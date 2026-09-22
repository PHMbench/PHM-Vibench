#!/usr/bin/env python3
"""Regenerate two explanatory diagrams. No data, inference or result values.

Problem figure: same observation and frozen backbone, changed information block,
measured outcome. Method figure: interpretable programs, additive logit correction,
known convex projection, source guard. Dashed boxes denote inherited components.
"""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                     'svg.fonttype':'none', 'pdf.fonttype':42})


def box(ax, gid, x, y, w, h, title, lines, inherited=False):
    patch=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10,rounding_size=0.1',
                        fill=False,linewidth=1.2,linestyle='--' if inherited else '-')
    patch.set_gid(gid); ax.add_patch(patch)
    text=ax.text(x+w/2,y+h-.27,title,ha='center',va='top',fontweight='bold',fontsize=11)
    text.set_gid(gid+'-title')
    text=ax.text(x+w/2,y+h/2-.16,'\n'.join(lines),ha='center',va='center',fontsize=10.5,linespacing=1.5)
    text.set_gid(gid+'-body')


def arrow(ax,gid,start,end,label=None):
    patch=FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,linewidth=1.0)
    patch.set_gid(gid); ax.add_patch(patch)
    if label:
        ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.14,label,ha='center',fontsize=10)


def save(fig,path):
    for extension in ('svg','pdf','png'):
        fig.savefig(path.with_suffix('.'+extension),dpi=300,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)


def problem(output):
    fig,ax=plt.subplots(figsize=(14.8,6.6));ax.set(xlim=(0,15),ylim=(0,6.7));ax.axis('off')
    ax.text(.15,6.5,'Problem: operator value beyond a fixed interpretable predictor',fontsize=16,fontweight='bold')
    box(ax,'fixed',.25,4.8,14.4,.95,'Fixed conditions',
        ['Observation X and sampling rate  •  backbone θ  •  physical-unit roles  •  fitting budget'],True)
    box(ax,'observation',.35,2.45,2.45,1.65,'Same observation', ['Finite vibration window X','No additional sensor','No test information'],True)
    box(ax,'baseline',3.65,2.45,3.15,1.65,'Frozen signal backbone', ['TSPN logits z₀(X)','Native features h₀(X)','Source accuracy floor: 0.80'],True)
    box(ax,'intervention',7.65,2.05,3.2,2.45,'Changed information Z', ['Intercept only','Native features (F terms)','Named signal programs','Same correction family'])
    box(ax,'outcomes',11.65,2.45,2.9,1.65,'Measured outcomes', ['Brier difference ΔR','Accuracy difference ΔA','Exact correction trace Ξ'])
    arrow(ax,'x-to-base',(2.9,3.25),(3.55,3.25))
    arrow(ax,'base-to-intervention',(6.9,3.25),(7.55,3.25))
    arrow(ax,'intervention-to-outcomes',(10.95,3.25),(11.55,3.25))
    ax.text(7.5,1.1,'Question: does operator information improve prediction beyond calibration or added capacity?',ha='center',fontsize=12)
    ax.text(7.5,.55,'An 80% source qualification floor is not an 80% target guarantee.  Windows are not independent units.',ha='center',fontsize=10.5)
    save(fig,output/'problem_setting')


def method(output):
    fig,ax=plt.subplots(figsize=(15.8,8.1));ax.set(xlim=(0,16),ylim=(0,8.1));ax.axis('off')
    ax.text(.1,7.9,'Method: baseline-retaining signal-program correction',fontsize=16,fontweight='bold')
    box(ax,'signal',.25,3.4,2.0,1.25,'Input X', ['Same native window','Known sampling rate'],True)
    box(ax,'tspn',3.1,5.75,3.4,1.25,'Inherited TSPN', ['Train on fit; tune checkpoint','Freeze θ, z₀ and h₀'],True)
    box(ax,'programs',3.1,2.1,3.4,2.4,'Named signal programs', ['Hz-band filtering → envelope','Modulation spectrum','Valid-lag autocorrelation','Signed Teager and statistics'])
    box(ax,'offset',7.3,2.75,3.0,1.8,'Additive correction', ['r(X) = W φ(X) + b','Fit W, b on grow units','Each feature has a logit term'])
    box(ax,'softmax',7.3,5.75,3.0,1.25,'Corrected predictor', ['q = softmax(z₀ / T + r)','Baseline: p₀ = softmax(z₀ / T)'])
    box(ax,'projection',11.25,5.45,4.2,1.8,'Known convex projection', ['pρ = (1 − ρ)p₀ + ρq','ρ fitted by grow Brier','ρ = 0 preserves the baseline'],True)
    box(ax,'guard',11.25,2.5,4.2,1.8,'Independent source guard', ['Freeze candidate before select','Compare Brier and class error','Return pρ or retain p₀'],True)
    arrow(ax,'input-tspn',(1.25,4.75),(3.0,6.25))
    arrow(ax,'input-programs',(2.35,3.9),(3.0,3.9))
    arrow(ax,'phi-offset',(6.6,3.4),(7.2,3.4),'φ(X)')
    arrow(ax,'tspn-logits',(6.6,6.4),(7.2,6.4))
    arrow(ax,'offset-logits',(8.8,4.65),(8.8,5.65))
    arrow(ax,'softmax-projection',(10.4,6.35),(11.15,6.35))
    arrow(ax,'projection-guard',(13.35,5.35),(13.35,4.4))
    arrow(ax,'guard-output',(13.35,2.4),(13.35,1.8))
    ax.text(13.35,1.55,'Frozen output p̂(X)',ha='center',fontsize=12,fontweight='bold')
    ax.text(5.2,1.25,'Matched controls: intercept-only and equal-dimensional native features\nNo extra TSPN training across correction controls',ha='center',fontsize=10.5,linespacing=1.5)
    ax.text(8,.42,'Dashed: inherited foundation/adaptation.  Solid: studied operator-information intervention.\nExact attribution is to the logit correction; target improvement still requires held-out evidence.',ha='center',fontsize=10.5,linespacing=1.4)
    save(fig,output/'method_overview')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    problem(args.output);method(args.output)
