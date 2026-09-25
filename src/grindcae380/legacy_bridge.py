"""One-operating-point total-force handoff to existing GrindCAE contracts.

The specific energy/ratio are an algebraic serialization bridge, never a new
material calibration valid over other operating points. Optional history input
retains explicit host constitutive assumptions. This module does not run FE.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .core import canonical
from .dispatch import read_case
from .workflow import digest, read_result


def force_input_from_result(result):
    case=read_case(result['normalized_input'])
    c=getattr(case,'reference_case',case)
    if c.source=='explicit_depths':
        raise ValueError('A wheel-population result is required for host force handoff')
    q=c.feed_speed_m_s*c.depth_m*c.width_m
    ft,fn=result['total_t_N'],result['total_n_N']
    if q<=0 or ft<=0 or fn<=0:
        raise ValueError('Legacy force Schema 1 requires positive process/forces; zero-load result cannot be encoded')
    source='Zhang2017 reconstructed force at fingerprint '+result['input_fingerprint']
    return dict(force_model_schema_version=1,unit_system='SI',model_type='specific_grinding_energy_force_ratio',
        process=dict(wheel_surface_speed_m_per_s=c.wheel_speed_m_s,
                     workpiece_feed_speed_m_per_s=c.feed_speed_m_s,depth_of_cut_m=c.depth_m,
                     grinding_width_m=c.width_m,wheel_diameter_m=c.wheel_diameter_m),
        calibration=dict(specific_grinding_energy_J_per_m3=ft*c.wheel_speed_m_s/q,
                         normal_to_tangential_force_ratio=fn/ft,
                         calibration_id='literature380_point_bridge_'+result['input_fingerprint'][:16],
                         specific_grinding_energy_source=source+'; us=Ft*Vs/(Vw*ap*b), algebraic point conversion.',
                         force_ratio_source=source+'; ratio=Fn/Ft, algebraic point conversion.',
                         applicability_notes='Valid only at this exact saved operating point. Recompute literature kernel and regenerate bridge after any process change. '
                         'Not an empirical calibration, true-contact force prescription, or independent validation. '
                         'Read the adjacent bridge_manifest.json and source_result for reconstruction/wear provenance.'))


def build_history_input(template, result, force_input, material_provenance):
    """Set geometry, process, width and yield stress; preserve host E/nu/Et.

    Caller explicitly supplies provenance for retained host elastic/hardening
    parameters because the paper does not identify a complete J2 material.
    """
    if not isinstance(material_provenance,str) or not material_provenance.strip():
        raise ValueError('Explicit provenance for retained host E, nu and tangent_modulus is required')
    from grindcae.history_pass.models import FixedMeshElastoplasticHistoryPassCase
    FixedMeshElastoplasticHistoryPassCase.from_mapping(template)
    data=deepcopy(template)
    c=read_case(result['normalized_input'])
    c=getattr(c,'reference_case',c)
    reference=data['moving_load_pass']['reference_case']
    reference['pass_load']['force_model']=deepcopy(force_input)
    trajectory=reference['pass_load']['trajectory']
    trajectory['wheel']['diameter_m']=c.wheel_diameter_m
    trajectory['single_pass']['depth_of_cut_m']=c.depth_m
    reference['analysis']['thickness']=c.width_m
    reference['material']['yield_strength']=c.yield_stress_Pa
    parsed=FixedMeshElastoplasticHistoryPassCase.from_mapping(data)
    changes=[]
    def diff(a,b,path=''):
        if isinstance(a,dict) and isinstance(b,dict):
            for key in sorted(set(a)|set(b)):diff(a.get(key),b.get(key),path+'/'+key)
        elif a!=b:changes.append(dict(path=path,template_value=a,bridge_value=b))
    diff(template,data)
    return parsed.to_dict(),dict(material_provenance=material_provenance,
        retained_host_material={k:reference['material'][k] for k in ('E','nu','tangent_modulus')},
        yield_source='Paper Table 6 yield stress; fracture stress does not define J2 hardening or damage.',
        changes=changes,
        field_interpretation='Prescribed macro load response of the supplied host material and fixed mesh; no grain contact or chip separation.')


def export_bridge(source_directory, output_directory, *, history_template=None, material_provenance=None):
    from grindcae.force_model import GrindingForceCase, predict_grinding_force
    import grindcae
    target=Path(output_directory).resolve()
    if target.exists():raise FileExistsError(f'Use a new output directory: {target}')
    result=read_result(source_directory)
    force_payload=force_input_from_result(result)
    force_case=GrindingForceCase.from_dict(force_payload)
    host_prediction=predict_grinding_force(force_case)
    expected=(result['total_t_N'],result['total_n_N'])
    actual=(host_prediction.tangential_force_magnitude_N,host_prediction.normal_force_magnitude_N)
    residuals=[a-b for a,b in zip(actual,expected)]
    if not all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(actual,expected)):
        raise ValueError('Host algebraic handoff does not conserve literature force')
    host_root=Path(grindcae.__file__).resolve().parent
    source_files=['force_model/core.py','force_model/models.py','history_pass/models.py',
                  'history_pass/loading.py','history_pass/workflow.py','elastoplastic/j2.py']
    manifest=dict(bridge_format='grindcae380_legacy_point_bridge_v1',
                  source_result_directory=str(Path(source_directory).resolve()),
                  source_input_fingerprint=result['input_fingerprint'],
                  host_version=grindcae.__version__,host_package_path=str(host_root),
                  host_source_hashes={p:digest(host_root/p) for p in source_files},
                  force_residuals_N=dict(tangential=residuals[0],normal=residuals[1]),
                  operating_point_lock=force_payload['process'],
                  operating_point_lock_enforcement='Bridge generation locks the point; old GUI/CLI does not enforce lock after manual edits. Regenerate after edits.',
                  force_semantics='Total force across width; host thickness equals grinding width for history handoff. No second width multiplication.',
                  historical_material_caveat='Paper plastic-flow/fracture stresses do not specify a complete FE constitutive law.',
                  execution_status='Host force API executed; optional history input parsed only. No FE solve, GUI acceptance or regression tests run.',
                  source_result=result)
    payloads={'force_case.json':force_payload,'host_force_prediction.json':host_prediction.to_dict()}
    if history_template is not None:
        template=json.loads(Path(history_template).read_text(encoding='utf-8-sig'))
        history,meta=build_history_input(template,result,force_payload,material_provenance)
        payloads['history_case.json']=history
        manifest['history_adapter']=dict(template_path=str(Path(history_template).resolve()),
                                          template_sha256=digest(Path(history_template)),**meta)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.grindcae380-bridge-',dir=target.parent) as temp:
        folder=Path(temp)
        for name,payload in payloads.items():
            (folder/name).write_text(json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        manifest['artifacts']={name:digest(folder/name) for name in payloads}
        (folder/'bridge_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        os.rename(folder,target)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_result',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--history-template',type=Path)
    parser.add_argument('--material-provenance')
    args=parser.parse_args()
    try:
        r=export_bridge(args.source_result,args.output_dir,history_template=args.history_template,
                        material_provenance=args.material_provenance)
    except (ValueError,OSError,ImportError) as exc:parser.exit(2,f'Bridge rejected: {exc}\n')
    print(json.dumps({k:r[k] for k in ('bridge_format','host_version','force_residuals_N','execution_status')},indent=2))


if __name__=='__main__':main()
