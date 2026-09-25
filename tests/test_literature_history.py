import copy
from dataclasses import replace
from types import SimpleNamespace as NS
import json
import math
import numpy as np
import pytest

from grindcae.literature_history import MODE, LiteratureHistoryCase
from grindcae.literature_history.projection import StripTemplate, build_template, project_load
from grindcae380.dispatch import predict
from literature_history_fixture import built_case, project_and_settings


def test_compose_roundtrip_and_source_changes():
    from grindcae.gui.analysis import AnalysisInputBuilder
    p,s=project_and_settings(); build=AnalysisInputBuilder()
    a=build.build(p,MODE,s).case
    assert LiteratureHistoryCase.from_mapping(a.to_dict())==a
    p.process['specific_grinding_energy_J_per_m3']*=2
    b=build.build(p,MODE,s).case
    assert a.input_fingerprint==b.input_fingerprint
    p.process['wheel_surface_speed_m_per_s']*=2
    c=build.build(p,MODE,s).case
    assert c.input_fingerprint!=a.input_fingerprint
    assert predict(c.literature)['total_n_N']!=predict(a.literature)['total_n_N']


@pytest.mark.parametrize('change', ['schema','extra','width','yield','placeholder','nan'])
def test_strict_composed_input_rejects_invalid(change):
    d=built_case().to_dict()
    if change=='schema':d['literature_history_schema_version']=True
    elif change=='extra':d['extra']=0
    elif change=='width':d['literature']['reference_case']['width_m']*=2
    elif change=='yield':d['literature']['reference_case']['yield_stress_Pa']*=2
    elif change=='placeholder':d['history_template']['moving_load_pass']['reference_case']['pass_load']['force_model']['calibration']['normal_to_tangential_force_ratio']=2
    else:d['literature']['reference_case']['friction_coefficient']=float('nan')
    with pytest.raises(ValueError):LiteratureHistoryCase.from_mapping(d)


def synthetic_position(direction='positive_x', interval=(0.,1.)):
    prepared=NS(total_degrees_of_freedom=6,component_dofs=np.array([[0,2,4],[1,3,5]]))
    left,right=interval; ratio=right-left
    facets=[NS(overlap_x_start_m=max(left,a),overlap_x_end_m=min(right,b),facet_x_start_m=a,facet_dx_m=b-a,
               node_start_id=i,node_end_id=i+1) for i,(a,b) in enumerate(((0.,.5),(.5,1.)))]
    trajectory=NS(case=NS(workpiece=NS(length_m=1.),single_pass=NS(relative_feed_direction=direction)),exact_arc_projected_length_m=1.)
    position=NS(pass_load=NS(effective_contact_interval_m=interval,contact_ratio=ratio,
        current_Fx_N=(1 if direction=='positive_x' else -1)*4*ratio,current_Fy_N=-8*ratio,trajectory_result=trajectory),
        position=NS(motion_coordinate_m=0.),load_mapping=NS(facets=facets))
    forces={f'{k}_{d}_N':np.array([1.,0.])*(1 if d=='t' else 2) for k in ('plastic','removal','rake','wear') for d in ('t','n')}
    return prepared,position,StripTemplate(np.array([0.,.5,1.]),forces,{k:float(v.sum()) for k,v in forces.items()})


@pytest.mark.parametrize('direction',['positive_x','negative_x'])
@pytest.mark.parametrize('interval',[(0.,1.),(.1,.7),(.5,1.),(0.,.2)])
def test_projection_components_force_and_first_moment(direction,interval):
    prepared,pos,t=synthetic_position(direction,interval)
    vector,records=project_load(prepared,pos,t)
    assert np.allclose([vector[::2].sum(),vector[1::2].sum()],[pos.pass_load.current_Fx_N,pos.pass_load.current_Fy_N],atol=1e-12)
    assert all(abs(r['residual_N'])<1e-12 for r in records)
    support=(0.,.5) if direction=='positive_x' else (.5,1.)
    a,b=max(interval[0],support[0]),min(interval[1],support[1])
    centroid=(a+b)/2 if b>a else sum(interval)/2
    assert np.dot(vector[::2],[0,.5,1])==pytest.approx(pos.pass_load.current_Fx_N*centroid,abs=1e-12)
    assert all(r['uniform_fallback']==(b<=a) for r in records)


def test_zero_contact_and_zero_component():
    prepared,pos,t=synthetic_position()
    pos.pass_load.effective_contact_interval_m=None;pos.pass_load.contact_ratio=0
    pos.pass_load.current_Fx_N=pos.pass_load.current_Fy_N=0
    v,r=project_load(prepared,pos,t);assert not v.any();assert len(r)==8


def test_real_population_strip_sum_and_no_second_width():
    result=predict(built_case().literature);template=build_template(result)
    for key,value in template.forces_N.items():assert value.sum()==pytest.approx(result['components'][key],rel=1e-12)
    assert sum(v.sum() for k,v in template.forces_N.items() if k.endswith('_t_N'))==pytest.approx(result['total_t_N'])


@pytest.fixture(scope='module')
def published_history(tmp_path_factory):
    from grindcae.literature_history import run_literature_history,read_result
    output=tmp_path_factory.mktemp('history381')/'result'
    result=run_literature_history(built_case(),output)
    return output,result


def test_small_history_publication(published_history):
    from grindcae.literature_history import run_literature_history,read_result
    output,result=published_history
    assert read_result(output)==result
    assert result['comparison']['maximum_component_projection_residual_N']<1e-10
    for route in ('baseline','spatial'):
        assert result['routes'][route]['final_unloaded']['external_force_norm_N']==0
    with pytest.raises(FileExistsError):run_literature_history(built_case(),output)


def test_catalog_origin_project_archive_and_stale(published_history,tmp_path):
    from grindcae.gui.analysis import AnalysisInputBuilder
    from grindcae.gui.analysis_workspace import AnalysisWorkspace,analysis_presentation_state
    from grindcae.gui.analysis_results import AnalysisResultAdapter
    from grindcae.gui.workbench import WorkbenchSession
    from grindcae.gui.project import save_project,load_project
    from grindcae.gui.project_archive import export_project_archive,import_project_archive
    from grindcae.gui.result_catalog import ResultCatalog
    from grindcae.gui.origin_export import export_origin_data
    from grindcae.gui.project_history import import_result_directory,RecentResultStore
    output,result=published_history;p,s=project_and_settings()
    b=AnalysisInputBuilder().build(p,MODE,s)
    AnalysisWorkspace(p,MODE,s,str(output)).save_configuration()
    session=WorkbenchSession(p)
    session.begin_analysis(mode=MODE,output_directory=output,input_fingerprint=b.input_fingerprint,input_snapshot=b.normalized_input)
    pub=AnalysisResultAdapter().read(MODE,output)
    session.register_published_result(mode=MODE,result_format=pub.result_format,output_directory=output,summary_path=pub.summary_path,
        artifact_paths={'representative_image':pub.representative_image},created_at='2026-09-25T22:00:00+08:00',input_fingerprint=b.input_fingerprint)
    path=save_project(p,tmp_path/'project.gcae');restored=load_project(path)
    catalog=ResultCatalog().read(restored.latest_analysis_result)
    assert len(catalog.category('fields').items)==2
    assert len(export_origin_data(catalog,tmp_path/'origin').table_paths)==8
    archive=export_project_archive(path,tmp_path/'complete.gcae-archive')
    imported=import_project_archive(archive,tmp_path/'restored')
    assert ResultCatalog().read(load_project(imported.project_path).latest_analysis_result).published.mode==MODE
    assert import_result_directory(output,store=RecentResultStore(tmp_path/'recent.json')).analysis_type==MODE
    changed=replace(s,literature_case=replace(b.case.literature,reference_case=replace(b.case.literature.reference_case,friction_coefficient=.4)).to_dict())
    state=AnalysisWorkspace(restored,MODE,changed).evaluate()
    assert analysis_presentation_state(restored,state).status=='stale'


@pytest.mark.parametrize('damage',['hash','missing','extra','unsafe','force','final','projection','png'])
def test_corrupt_result_is_rejected(published_history,tmp_path,damage):
    import shutil
    from grindcae.literature_history import read_result
    output,_=published_history; dest=tmp_path/'damaged';shutil.copytree(output,dest)
    path=dest/'summary.json';s=json.loads(path.read_text(encoding='utf-8'))
    if damage=='hash':(dest/'grains.csv').write_text('bad',encoding='utf-8')
    elif damage=='missing':(dest/'strip_template.json').unlink()
    elif damage=='extra':(dest/'extra.txt').write_text('extra')
    elif damage=='unsafe':s['artifacts']['input.json']='../input.json'
    elif damage=='force':s['literature_force']['Fn_N']+=1
    elif damage=='final':s['routes']['spatial']['final_unloaded']['external_force_norm_N']=1
    elif damage=='projection':s['comparison']['maximum_component_projection_residual_N']=1
    elif damage=='png':(dest/'literature_history_comparison.png').write_bytes(b'bad')
    path.write_text(json.dumps(s),encoding='utf-8')
    with pytest.raises((ValueError,OSError,RuntimeError)):read_result(dest)


def test_failed_solve_does_not_publish_or_change_input(tmp_path,monkeypatch):
    import grindcae.literature_history.workflow as workflow
    case=built_case();before=case.to_dict()
    def fail(*args,**kwargs):raise RuntimeError('injected history failure')
    monkeypatch.setattr(workflow,'advance_prepared_history',fail)
    with pytest.raises(RuntimeError,match='injected'):workflow.run_literature_history(case,tmp_path/'result')
    assert not (tmp_path/'result').exists();assert case.to_dict()==before
    assert not list(tmp_path.glob('.literature-history-*'))


def test_publication_failure_preserves_other_results(published_history,tmp_path,monkeypatch):
    import grindcae.literature_history.workflow as workflow
    output,result=published_history
    before=(output/'summary.json').read_bytes()
    real_rename=workflow.os.rename
    def fail_final(source,target):
        if str(source).endswith('publication'):raise OSError('injected final rename')
        return real_rename(source,target)
    monkeypatch.setattr(workflow.os,'rename',fail_final)
    with pytest.raises(OSError,match='injected final'):workflow.run_literature_history(built_case(),tmp_path/'new_result')
    assert not (tmp_path/'new_result').exists()
    assert (output/'summary.json').read_bytes()==before
