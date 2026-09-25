"""Independent integrals and known geometries catch wrong factors/units/branches."""
import importlib.util
import json
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad


def api():
    assert importlib.util.find_spec('grindcae380.core') is not None, 'literature kernel is missing'
    from grindcae380 import core
    return core


def test_ploughing_hand_solution_and_zero_contact():
    k = api()
    # a=1 um, theta=pi/4: Fs_t=225e6*1e-12; Fs_n=pi/2*Fs_t.
    r = k.grain_force(1e-6, 225e6, 540e6, math.pi/4, 0.5, 'continuous_projection')
    assert r.plastic_t_N == pytest.approx(0.000225, rel=1e-12)
    assert r.plastic_n_N == pytest.approx(0.0003534291735288517, rel=1e-12)
    assert r.rake_t_N == pytest.approx(0.00017671458676442586, rel=1e-12)
    assert r.rake_n_N == pytest.approx(r.rake_t_N, rel=1e-12)
    assert r.removal_t_N == r.removal_n_N == 0
    z = k.grain_force(0, 225e6, 540e6, 1.047, 0.607, 'printed')
    assert z.total_t_N == z.total_n_N == 0


@pytest.mark.parametrize('mode',['printed','continuous_projection'])
@pytest.mark.parametrize('depth',[1.18e-6,2e-6,2.85e-6,3.8e-6,5e-6])
def test_closed_force_matches_independent_surface_integrals(mode, depth):
    k=api(); theta=1.047; sy=225e6; sb=540e6; mu=.417
    # Coefficients transcribed independently from Table 2; do not call beta helper.
    a=depth*1e6
    params=[(.908,4.594,2.258),(.187,1.891,.887),(.377,2.484,1.178),(.118,1.864,.205),(.307,1.481,.206),(.142,.849,.537)]
    beta=sum(A*math.exp(-((a-B)/C)**2) for A,B,C in params) if a<=3.8 else .922*math.exp(-5.211*math.exp(-2.004*a))
    alpha=math.acos(math.sqrt(beta)); t=math.tan(theta)
    removal=math.pi*t*sb/(2*(1-math.sin(alpha)))
    endstress=sy+removal*(2 if mode=='printed' else 1)
    stress=lambda x: endstress*x/alpha if x<alpha else sy+removal
    ft=quad(lambda x: stress(x)*depth**2*t*math.cos(x),0,math.pi/2,points=[alpha],epsabs=1e-14)[0]
    fn=quad(lambda x: stress(x)*depth**2*t*t,0,math.pi/2,points=[alpha],epsabs=1e-14)[0]
    friction=quad(lambda x: mu*stress(x)*depth**2*t/math.cos(theta),0,math.pi/2,points=[alpha],epsabs=1e-14)[0]
    r=k.grain_force(depth,sy,sb,theta,mu,mode)
    assert r.plastic_t_N+r.removal_t_N==pytest.approx(ft,rel=1e-10)
    assert r.plastic_n_N+r.removal_n_N==pytest.approx(fn,rel=1e-10)
    assert r.rake_t_N==pytest.approx(friction*(1 if mode=='printed' else math.sin(theta)),rel=1e-10)
    assert r.rake_n_N==pytest.approx(friction*math.cos(theta),rel=1e-10)


def test_stage_boundaries_and_unmodified_beta_join():
    k=api()
    assert [k.grain_stage(x*1e-6) for x in [0,1,1.18,2.84,2.85,4]]==['inactive','ploughing','transition','transition','cutting','cutting']
    assert k.cutting_efficiency(0)>0 # Table2 fit not silently clamped to zero.
    assert abs(k.cutting_efficiency(3.8e-6)-k.cutting_efficiency(3.80000001e-6))>1e-4
    r=k.grain_force(3e-6,225e6,540e6,1.047,0,'continuous_projection')
    assert r.rake_t_N==r.rake_n_N==0


def test_shadowing_uses_previous_dynamic_grain_not_previous_static_grain():
    k=api()
    # beta_kinematic=0.02; pitch=100 um => 2 um step gain.
    depths,previous=k.dynamic_depths(np.array([[10e-6],[5e-6],[9e-6]]),1e-4,.02)
    assert depths[:,0]==pytest.approx([5e-6,0,1.5e-6],rel=1e-12)
    assert previous[:,0].tolist()==[-1,0,0]


def test_wear_units_aggregate_and_fingerprint():
    k=api(); c=k.Case.paper_example(source='explicit_depths')
    c=replace(c,depths_m=(1e-6,2e-6,0.0),wear_coefficient_N_m=.01836)
    p=k.predict(c)
    # 4*K1*Nd*(2/60)/20/0.3, Nd=2.
    assert p['components']['wear_n_N']==pytest.approx(.000816)
    assert p['components']['wear_t_N']==pytest.approx(.000495312)
    assert p['counts']=={'sampled':3,'static':2,'dynamic':2,'ploughing':1,'transition':1,'cutting':0}
    assert p['total_t_N']==pytest.approx(sum(row['total_t_N'] for row in p['grains'])+.000495312)
    assert p['input_fingerprint']!=k.predict(replace(c,formula_convention='printed'))['input_fingerprint']
    assert p['input_fingerprint']==k.predict(k.Case.from_mapping(c.to_dict()))['input_fingerprint']


@pytest.mark.parametrize('field,value',[('wheel_speed_m_s',True),('feed_speed_m_s','2'),('depth_m',float('nan')),('width_m',0),('cone_half_angle_rad',math.pi/2),('seed',1.2),('schema_version',True),('formula_convention','auto'),('wear_provenance',''),('depths_m',[-1e-6]),('vibration_steps',0)])
def test_strict_inputs_reject_invalid_values(field,value):
    k=api(); d=k.Case.paper_example(source='explicit_depths').to_dict(); d[field]=value
    with pytest.raises(ValueError): k.Case.from_mapping(d)


def test_unknown_fields_and_unsupported_material_fit():
    k=api(); d=k.Case.paper_example().to_dict(); d['force_N']=100
    with pytest.raises(ValueError): k.Case.from_mapping(d)
    d=k.Case.paper_example().to_dict(); d['material_id']='17CrNi2MoVNb'
    with pytest.raises(ValueError): k.Case.from_mapping(d)


def test_seeded_wheel_and_zero_feed_depth():
    k=api(); c=k.Case.paper_example()
    p=k.predict(c); q=k.predict(c)
    assert p==q
    assert p['counts']['dynamic']<=p['counts']['static']<=p['counts']['sampled']
    assert p['counts']['sampled']>0
    assert p['wheel_diagnostics']['vibration_method']=='unbounded_sum_uniform_eq40'
    assert k.predict(replace(c,seed=c.seed+1))['grains']!=p['grains']
    for z in [replace(c,depth_m=0),replace(c,feed_speed_m_s=0)]:
        r=k.predict(z)
        assert r['total_t_N']==r['total_n_N']==0


def test_publication_reread_rejects_tampering_and_existing_directory(tmp_path):
    k=api()
    from grindcae380.workflow import run_case,read_result
    c=k.Case.paper_example(source='explicit_depths'); target=tmp_path/'result'
    p=run_case(c,target)
    assert read_result(target)['total_n_N']==p['total_n_N']
    before={f.name:f.read_bytes() for f in target.iterdir()}
    with pytest.raises(FileExistsError):run_case(c,target)
    assert before=={f.name:f.read_bytes() for f in target.iterdir()}
    (target/'grains.csv').write_text('tampered',encoding='utf-8')
    with pytest.raises(ValueError):read_result(target)


def test_failed_render_does_not_publish(tmp_path,monkeypatch):
    k=api()
    from grindcae380 import workflow
    def fail(*args,**kwargs):raise RuntimeError('render failure')
    monkeypatch.setattr(workflow,'plot_result',fail)
    with pytest.raises(RuntimeError):workflow.run_case(k.Case.paper_example(),tmp_path/'failed')
    assert not (tmp_path/'failed').exists()


def test_cli_uses_independent_package(tmp_path):
    k=api()
    import os,subprocess,sys
    from pathlib import Path
    c=tmp_path/'case.json'; c.write_text(json.dumps(k.Case.paper_example(source='explicit_depths').to_dict()),encoding='utf-8')
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).parents[1]/'src'),MPLBACKEND='Agg')
    p=subprocess.run([sys.executable,'-m','grindcae380',str(c),'--output-dir',str(tmp_path/'out')],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert (tmp_path/'out'/'summary.json').is_file()
    assert 'not_verified' in p.stdout


def test_inverse_wear_coefficient_uses_fixed_population_and_both_force_components():
    k=api(); c=k.Case.paper_example(source='explicit_depths')
    c=replace(c,depths_m=(1e-6,),cone_half_angle_rad=math.pi/4,friction_coefficient=.5,wear_coefficient_N_m=0)
    # Hand force at theta45, plus K1=0.9 N m gives wear Fn=.02 N, Ft=.01 N.
    fit=k.calibrate_wear(c, measured_normal_N=.020530143760293276,
                        measured_tangential_N=.010401714586764426, source='synthetic analytical fixture')
    assert fit['normal_estimate_N_m']==pytest.approx(.9,rel=1e-12)
    assert fit['tangential_estimate_N_m']==pytest.approx(.9,rel=1e-12)
    assert fit['relative_consistency_error']<1e-12
    assert fit['independent_validation']=='not_verified'
    with pytest.raises(ValueError):
        k.calibrate_wear(c,measured_normal_N=0,measured_tangential_N=0,source='incompatible')


def test_dimensional_scaling_and_distinct_friction_projection():
    k=api()
    p=k.grain_force(4e-6,225e6,540e6,1.047,.607,'continuous_projection')
    q=k.grain_force(4e-6,450e6,1080e6,1.047,.607,'continuous_projection')
    assert q.total_t_N==pytest.approx(2*p.total_t_N,rel=1e-12)
    assert q.total_n_N==pytest.approx(2*p.total_n_N,rel=1e-12)
    assert p.rake_t_N/p.rake_n_N==pytest.approx(math.tan(1.047),rel=1e-12)
    c=k.Case.paper_example(source='explicit_depths')
    r=k.predict(c)
    # Supplied population already accounts for width; must not multiply forces again.
    s=k.predict(replace(c,width_m=c.width_m*2))
    assert s['total_t_N']==r['total_t_N']


def test_summary_tampering_cannot_become_valid_by_rehashing_artifacts(tmp_path):
    k=api()
    from grindcae380.workflow import run_case,read_result
    target=tmp_path/'r';run_case(k.Case.paper_example(source='explicit_depths'),target)
    summary=target/'summary.json'; d=json.loads(summary.read_text(encoding='utf-8'))
    d['total_t_N']+=1
    summary.write_text(json.dumps(d),encoding='utf-8')
    with pytest.raises(ValueError):read_result(target)
