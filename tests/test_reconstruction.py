"""Coverage for previously untested closure, version dispatch and host handoff."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from grindcae380.reconstruction import ReconstructionCase, periodic_depths, predict, calibrate_normal
from grindcae380.dispatch import read_case
from grindcae380.workflow import run_case, read_result


def test_periodic_envelope_hand_geometry():
    # Two equal grains: preceding grain gain = slope*pitch = 2 um, average 1 um.
    depths,previous=periodic_depths([[10e-6],[10e-6]],1e-4,.02)
    assert depths[:,0]==pytest.approx([1e-6,1e-6])
    assert previous[:,0].tolist()==[1,0]
    # Taller predecessor suppresses lower grain; same taller grain recurs after 4 um gain.
    depths,_=periodic_depths([[10e-6],[5e-6]],1e-4,.02)
    assert depths[:,0]==pytest.approx([2e-6,0])


@pytest.mark.parametrize('boundary',['finite_entry','periodic_envelope'])
def test_bounded_population_and_no_feed_zero_depth(boundary):
    c=replace(ReconstructionCase.paper_example(),boundary_rule=boundary)
    result=predict(c)
    assert result==predict(c)
    assert all(-90e-6<=r['protrusion_m']<=0 for r in result['grains'])
    assert all(.158e-3<=r['diameter_m']<=.202e-3 for r in result['grains'])
    assert result['counts']['sampled']==1544
    for field in ['depth_m','feed_speed_m_s']:
        zero=predict(replace(c,reference_case=replace(c.reference_case,**{field:0})))
        assert zero['total_t_N']==zero['total_n_N']==0
        assert zero['counts']['dynamic']==0


def test_schema_rejects_ambiguous_fields():
    c=ReconstructionCase.paper_example()
    with pytest.raises(ValueError):read_case(dict(c.to_dict(),extra=1))
    with pytest.raises(ValueError):read_case(dict(c.to_dict(),schema_version=True))
    with pytest.raises(ValueError):replace(c,relief_depth_m=float('nan'))
    with pytest.raises(ValueError):replace(c,relief_depth_m=.001)


def test_reference_calibration_does_not_mutate_baseline():
    c=ReconstructionCase.paper_example()
    calibrated,report=calibrate_normal(c,target_normal_N=65.53,source='published model reference',seeds=(2017,2018))
    results=[predict(replace(calibrated,reference_case=replace(calibrated.reference_case,seed=s))) for s in (2017,2018)]
    assert np.mean([r['total_n_N'] for r in results])==pytest.approx(65.53,rel=1e-12)
    assert c.reference_case.wear_coefficient_N_m==0
    assert report['independent_validation']=='not_verified'
    with pytest.raises(ValueError):calibrate_normal(c,target_normal_N=.0001,source='too small')


def test_v2_result_round_trip_and_tamper_rejection(tmp_path):
    result=run_case(ReconstructionCase.paper_example(),tmp_path/'output')
    assert read_result(tmp_path/'output')==result
    path=tmp_path/'output'/'summary.json'
    payload=json.loads(path.read_text(encoding='utf-8'));payload['total_n_N']+=1
    path.write_text(json.dumps(payload),encoding='utf-8')
    with pytest.raises(ValueError):read_result(tmp_path/'output')


def test_host_force_and_history_adapter_preserve_template(tmp_path):
    host=Path('D:/CODEX/project-Grinding.CAE')
    sys.path.insert(0,str(host/'src'))
    from grindcae380.legacy_bridge import export_bridge
    source=tmp_path/'source'
    run_case(ReconstructionCase.paper_example(),source)
    template=host/'examples/fixed_mesh_elastoplastic_history_pass.json'
    original=template.read_bytes()
    result=export_bridge(source,tmp_path/'bridge',history_template=template,material_provenance='Explicit uncalibrated host demonstration')
    assert result['force_residuals_N']['normal']==pytest.approx(0,abs=1e-12)
    assert result['force_residuals_N']['tangential']==pytest.approx(0,abs=1e-12)
    history=json.loads((tmp_path/'bridge/history_case.json').read_text(encoding='utf-8'))
    ref=history['moving_load_pass']['reference_case']
    assert ref['analysis']['thickness']==.05
    assert ref['material']['yield_strength']==225e6
    assert ref['material']['tangent_modulus']==2e9
    assert template.read_bytes()==original
