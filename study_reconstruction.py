"""Reproducible bounded paper audit, no parameter fitting and no seed search."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grindcae380.core import Case, cutting_efficiency, grain_force, predict
from grindcae380.workflow import run_case


def main():
    root=Path(__file__).resolve().parent
    sources=['explicit_depths','stochastic_eq40']
    aliases=['explicit_depths','paper_stochastic']
    for source,name in zip(sources,aliases,strict=True):
        c=Case.paper_example(source=source)
        case_path=root/'examples'/f'{name}.json'
        with case_path.open('x',encoding='utf-8') as f:json.dump(c.to_dict(),f,indent=2,ensure_ascii=False,allow_nan=False)
        run_case(c,root/'outputs'/name)
        printed=replace(c,formula_convention='printed')
        printed_path=root/'examples'/f'{name}_printed.json'
        with printed_path.open('x',encoding='utf-8') as f:json.dump(printed.to_dict(),f,indent=2,ensure_ascii=False,allow_nan=False)
        run_case(printed,root/'outputs'/f'{name}_printed')
    destination=root/'outputs'/'reconstruction_study'
    destination.mkdir(exist_ok=False)
    c=Case.paper_example()
    study=[]
    # Fig15 label comparisons: omitted wheel details remain a comparison limitation.
    for depth,published in [(10,40),(20,201),(30,252),(40,277)]:
        p=predict(replace(c,depth_m=depth*1e-6))
        study.append(dict(depth_um=depth,paper_figure15_dynamic_count=published,
            reconstruction_dynamic_count=p['counts']['dynamic'],
            total_t_N=p['total_t_N'],total_n_N=p['total_n_N']))
    baseline=predict(c)
    units=[]
    for value,label in [(.01836,'18.36 N mm hypothesis'),(18.36,'18.36 N m hypothesis')]:
        p=predict(replace(c,wear_coefficient_N_m=value))
        units.append(dict(interpretation=label,K1_N_m=value,Fn_N=p['total_n_N'],Ft_N=p['total_t_N'],wear_n_N=p['components']['wear_n_N']))
    report=dict(seed=c.seed,vibration_steps=c.vibration_steps,base_counts=baseline['counts'],
        base_diagnostics=baseline['wheel_diagnostics'],figure15_count_comparison=study,K1_unit_sensitivity=units,
        paper_figure18_dry_printed_prediction=dict(Fn_N=65.53,Ft_N=39.99),
        paper_figure18_nmql_printed_prediction=dict(Fn_N=65.01,Ft_N=28.21),
        paper_text_nmql_experiment=dict(Fn_N=65.06,Ft_N=28.48),
        figure18_replication='not_achieved',
        reason='The published wheel-generation and finite-entry rules do not uniquely specify the author implementation. No fit/tuning performed.',
        beta_at_join=cutting_efficiency(3.8e-6),beta_right_limit=cutting_efficiency(3.80000000001e-6))
    (destination/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(11.5,8))
    x=np.linspace(0,7,500)
    axes[0,0].plot(x,[cutting_efficiency(float(v)*1e-6) for v in x],color='#286f91')
    for v in [1.18,2.85,3.8]:axes[0,0].axvline(v,ls=':',color='#777777',lw=.8)
    axes[0,0].set(xlabel='Effective grain depth (um)',ylabel='Cutting efficiency beta',title='Eq.2 + Table 2 (unmodified)')
    for mode,style in [('continuous_projection','-'),('printed','--')]:
        values=[grain_force(float(v)*1e-6,225e6,540e6,1.047,.607,mode).total_n_N for v in x]
        axes[0,1].plot(x,values,style,label=mode)
    axes[0,1].set(xlabel='Effective grain depth (um)',ylabel='Single-grain normal force (N)',title='Explicit equation conventions')
    axes[0,1].legend(fontsize=8)
    axes[1,0].plot([s['depth_um'] for s in study],[s['paper_figure15_dynamic_count'] for s in study],'o-',label='Paper Fig.15 labels')
    axes[1,0].plot([s['depth_um'] for s in study],[s['reconstruction_dynamic_count'] for s in study],'s--',label='Literal Eq.40 reconstruction')
    axes[1,0].set(xlabel='Grinding depth (um)',ylabel='Dynamic grain count',title='Wheel reconstruction gap')
    axes[1,0].legend(fontsize=8)
    heights=np.array([r['protrusion_m'] for r in baseline['grains']])
    axes[1,1].hist(heights*1e3,bins=35,color='#69a899')
    axes[1,1].set(xlabel='Protrusion height (mm)',ylabel='Sampled grain count',title='Eq.40 unbounded sum: 800 increments')
    for ax in axes.flat:ax.grid(alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('GrindCAE 3.8.0 development candidate | reproduction audit')
    fig.text(.5,.012,'Fixed seed 2017; no curve fitting. Paper figure replication NOT achieved.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.035,1,.95));fig.savefig(destination/'audit.png',dpi=170);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
