"""Reproducible model study. Creates a new directory; never overwrites results.

This is a numerical research/example run, not an automated test suite.
"""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from grindcae380.reconstruction import ReconstructionCase, calibrate_normal, predict
from grindcae380.workflow import run_case

SEEDS = tuple(range(2017,2033))
SOURCE = 'Zhang 2017 Fig.18 dry printed model prediction Fn=65.53 N at Table 6 conditions; secondary reconstruction target, not a raw experimental datum.'


def ensemble(case, seeds=SEEDS):
    samples=[]
    for seed in seeds:
        r=predict(replace(case, reference_case=replace(case.reference_case,seed=seed)))
        samples.append(dict(seed=seed, normal_N=r['total_n_N'], tangential_N=r['total_t_N'],
                            dynamic=r['counts']['dynamic'], static=r['counts']['static'],
                            wear_normal_N=r['components']['wear_n_N'],
                            first_entry_normal_N=r['wheel_diagnostics']['first_entry_nonwear_normal_N'],
                            beta_extrapolation_count=r['wheel_diagnostics']['beta_extrapolation_count']))
    summary={}
    for key in samples[0]:
        if key=='seed':continue
        values=np.array([s[key] for s in samples],dtype=float)
        summary[key]=dict(mean=float(values.mean()),sd=float(values.std(ddof=1)),
                          min=float(values.min()),max=float(values.max()))
    return dict(seeds=list(seeds),statistics=summary,samples=samples)


def write_json(path, data):
    with path.open('x',encoding='utf-8') as f:
        json.dump(data,f,indent=2,ensure_ascii=False,allow_nan=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    dest=args.output_dir.resolve()
    dest.mkdir(parents=True,exist_ok=False)
    base=ReconstructionCase.paper_example()
    fitted,calibration=calibrate_normal(base,target_normal_N=65.53,source=SOURCE,seeds=SEEDS)
    periodic=replace(base,boundary_rule='periodic_envelope')
    periodic_fitted,periodic_calibration=calibrate_normal(periodic,target_normal_N=65.53,source=SOURCE,seeds=SEEDS)
    for name,case in [('bounded_unfitted',base),('bounded_reference_calibrated',fitted),
                      ('periodic_reference_calibrated',periodic_fitted)]:
        write_json(dest/f'{name}.json',case.to_dict())
        run_case(case,dest/name)
    base_stats=ensemble(base)
    fitted_stats=ensemble(fitted)
    new_seed_stats=ensemble(fitted,tuple(range(2033,2049)))
    comparison=[]
    for name,mu,fn,ft in [('Dry',.607,65.53,39.99),('Flood',.446,65.08,29.50),
                          ('MQL',.505,65.39,32.65),('NMQL',.417,65.01,28.21)]:
        case=replace(fitted,reference_case=replace(fitted.reference_case,friction_coefficient=mu))
        e=ensemble(case)
        comparison.append(dict(condition=name,friction_coefficient=mu,paper_normal_N=fn,paper_tangential_N=ft,
                               normal_used_for_fit=name=='Dry',tangential_used_for_fit=False,
                               normal_relative_difference_percent=100*(e['statistics']['normal_N']['mean']/fn-1),
                               tangential_relative_difference_percent=100*(e['statistics']['tangential_N']['mean']/ft-1),
                               ensemble=e))
    variations=[('depth_um','depth_m',1e-6,[10,15,20,30,40]),
                ('feed_m_min','feed_speed_m_s',1/60,[1,2,3,4]),
                ('wheel_speed_m_s','wheel_speed_m_s',1,[10,20,30,40]),
                ('width_mm','width_m',1e-3,[20,50]),
                ('relief_um','relief_depth_m',1e-6,[45,90,135])]
    sweeps=[]
    for axis,field,scale,values in variations:
        for value in values:
            case=replace(fitted,relief_depth_m=value*scale) if field=='relief_depth_m' else replace(
                fitted,reference_case=replace(fitted.reference_case,**{field:value*scale}))
            sweeps.append(dict(axis=axis,value=value,coefficient_refitted=False,ensemble=ensemble(case)))
    boundary_fixed=ensemble(replace(fitted,boundary_rule='periodic_envelope'))
    periodic_stats=ensemble(periodic_fitted)
    printed=ensemble(replace(fitted,reference_case=replace(fitted.reference_case,formula_convention='printed')))
    report=dict(study='Declared physical/statistical closure; fixed geometry assumptions and seed sets',
                calibration=calibration,baseline_unfitted=base_stats,baseline_fitted=fitted_stats,
                new_seed_ensemble=new_seed_stats,paper_figure18_comparison=comparison,
                parameter_sweeps=sweeps,periodic_with_finite_entry_coefficient=boundary_fixed,
                periodic_separate_calibration=periodic_calibration,periodic_refitted=periodic_stats,
                printed_with_same_coefficient=printed,
                evidence_class='Numerical research run; not automated regression or independent physical validation',
                limits=[
                    'Only dry mean normal force is fitted. All paper comparison values are published model predictions.',
                    'New seeds measure sampling variation of the same assumed distribution, not experimental generalization.',
                    'Periodic refit is an identifiability demonstration, not selection of the closest curve.',
                    'Sparse track geometry, virgin first grains, beta extrapolation and unmeasured recess support remain material uncertainty.',
                    'Parameter sweeps keep K1 fixed. Their curves are model outputs and are not experimentally verified.',
                ])
    write_json(dest/'study.json',report)
    records=[]
    for row in sweeps:
        e=row['ensemble']['statistics']
        records.append(dict(axis=row['axis'],value=row['value'],normal_mean_N=e['normal_N']['mean'],
                            normal_sd_N=e['normal_N']['sd'],tangential_mean_N=e['tangential_N']['mean'],
                            tangential_sd_N=e['tangential_N']['sd'],dynamic_mean=e['dynamic']['mean']))
    with (dest/'parameter_sweeps.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    names=[r['condition'] for r in comparison]
    xx=np.arange(len(names))
    for ax,key,paper,title in [(axes[0,0],'normal_N','paper_normal_N','Normal force: dry reference fitted'),
                                (axes[0,1],'tangential_N','paper_tangential_N','Tangential force: no fitting')]:
        means=[r['ensemble']['statistics'][key]['mean'] for r in comparison]
        sd=[r['ensemble']['statistics'][key]['sd'] for r in comparison]
        ax.errorbar(xx,means,yerr=sd,fmt='o-',capsize=4,label='Reconstruction mean +/- seed SD')
        ax.plot(xx,[r[paper] for r in comparison],'s--',label='Paper Fig.18 printed predictions')
        ax.set(xticks=xx,xticklabels=names,ylabel='Force (N)',title=title)
        ax.legend(fontsize=7)
    for ax,axis,label in [(axes[0,2],'depth_um','Depth (um)'),(axes[1,0],'feed_m_min','Feed (m/min)'),
                           (axes[1,1],'wheel_speed_m_s','Wheel speed (m/s)')]:
        rr=[r for r in records if r['axis']==axis]
        for name,color in [('normal','#306a93'),('tangential','#c27d48')]:
            ax.errorbar([r['value'] for r in rr],[r[name+'_mean_N'] for r in rr],
                        yerr=[r[name+'_sd_N'] for r in rr],fmt='o-',capsize=3,color=color,label=name)
        ax.set(xlabel=label,ylabel='Force (N)',title='Fixed K1; mean +/- seed SD');ax.legend(fontsize=8)
    ax=axes[1,2]
    groups=[fitted_stats,boundary_fixed,periodic_stats]
    xx=np.arange(3)
    for offset,key,color in [(-.17,'normal_N','#306a93'),(.17,'tangential_N','#c27d48')]:
        ax.bar(xx+offset,[g['statistics'][key]['mean'] for g in groups],width=.34,color=color,label=key)
    ax.set(xticks=xx,xticklabels=['Finite entry','Periodic\nsame K1','Periodic\nrefitted K1'],ylabel='Force (N)',title='Boundary / coefficient ambiguity')
    ax.legend(fontsize=8)
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('GrindCAE 3.8.0 | literature-based reconstruction with declared closure')
    fig.text(.5,.012,'16 fixed seeds; reference calibration is not independent validation. Periodic refit is sensitivity only.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.95));fig.savefig(dest/'study.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(output_directory=str(dest),calibration=calibration,
                          baseline=fitted_stats['statistics'],new_seeds=new_seed_stats['statistics'],
                          comparison=[{k:v for k,v in r.items() if k!='ensemble'} for r in comparison]),indent=2))


if __name__=='__main__':main()
