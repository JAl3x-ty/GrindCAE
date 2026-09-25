"""One-way literature loads and two independent histories on one shared mesh.

Development implementation for 3.8.1. No Abaqus backend is involved.
Publication uses an exclusive new directory and validates staged artifacts.
"""
from dataclasses import fields
import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from grindcae.elastoplastic_fem import prepare_elastoplastic_mesh_with_facet_tractions
from grindcae.history_pass import assemble_fixed_top_mapping_load_vector, FixedMeshElastoplasticHistoryPassCase
from grindcae.history_pass.workflow import advance_prepared_history, FixedMeshElastoplasticHistoryPassResult
from grindcae.history_pass.__main__ import export_result, artifact_paths, validate_artifacts
from grindcae.moving_load_pass import solve_fixed_mesh_moving_load_pass
from grindcae380.legacy_bridge import force_input_from_result
from grindcae380.workflow import run_case, read_result as read_force_result, digest

from .models import LiteratureHistoryCase, RESULT_FORMAT, canonical
from .projection import build_template, project_load

ROUTES = ('baseline', 'spatial')
PROJECTION_COLUMNS = ('position_id','component','target_N','assembled_N','residual_N','uniform_fallback')
EXPORTED_ROUTE_FILES = ('pass_history.csv','pass_residual_state.png','final_results.vtu','final_nodes.csv',
                        'final_elements.csv','final_surface_profile.png')


def resolved_history(case, prediction):
    data = case.history_template.to_dict()
    data['moving_load_pass']['reference_case']['pass_load']['force_model'] = force_input_from_result(prediction)
    return FixedMeshElastoplasticHistoryPassCase.from_mapping(data)


def _wrap(history, moving, prepared, shared):
    return FixedMeshElastoplasticHistoryPassResult(history, moving, prepared,
        **{field.name:getattr(shared,field.name) for field in fields(shared)})


def _dump(path, data):
    path.write_text(json.dumps(data,ensure_ascii=True,allow_nan=False,indent=2),encoding='utf-8')


def _plot_comparison(results, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3,1,figsize=(10,9),constrained_layout=True)
    for name,result in results.items():
        rows = result.history_rows
        x = np.arange(len(rows))
        axes[0].plot(x,[r.maximum_displacement_m*1e6 for r in rows],label=name)
        axes[1].plot(x,[r.maximum_equivalent_plastic_strain for r in rows],label=name)
        axes[2].plot(x,[r.accumulated_plastic_dissipation_J for r in rows],label=name)
    for ax,ylabel in zip(axes,('Maximum displacement [um]','Maximum eq. plastic strain [-]','Plastic dissipation [J]')):
        ax.set_ylabel(ylabel); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel('Committed target row, including final unload')
    fig.suptitle('Literature-force history | uniform vs circumferential strips\nPrescribed macro loads; material assumptions recorded')
    fig.savefig(path,dpi=150); plt.close(fig)


def run_literature_history(case, output_directory, progress=None):
    if not isinstance(case,LiteratureHistoryCase):
        raise TypeError('validated LiteratureHistoryCase required')
    output = Path(output_directory).expanduser().resolve()
    if output.exists():
        raise FileExistsError('Use a new result directory: '+str(output))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.literature-history-',dir=output.parent) as temporary:
        work = Path(temporary)
        stage = work/'publication'; stage.mkdir()
        prediction = run_case(case.literature,stage/'force')
        history = resolved_history(case,prediction)
        moving = solve_fixed_mesh_moving_load_pass(history.moving_load_pass,work/'moving')
        fem = history.reference_case.to_elastoplastic_fem_case(peak_force_x_N=0.,peak_force_y_N=0.)
        prepared = prepare_elastoplastic_mesh_with_facet_tractions(fem,moving.reference_mesh,
            np.empty(0,dtype=np.int32),peak_force_x_N=0.,peak_force_y_N=0.,
            facet_line_load_x_N_per_m=np.empty(0),facet_line_load_y_N_per_m=np.empty(0))
        template = build_template(prediction)
        uniform = tuple(assemble_fixed_top_mapping_load_vector(prepared,p.load_mapping) for p in moving.positions)
        spatial=[]; records=[]
        for position,uniform_vector in zip(moving.positions,uniform,strict=True):
            vector,rows = project_load(prepared,position,template)
            spatial.append(vector)
            records.extend(dict(position_id=position.position.position_id,**row) for row in rows)
            for axis in (0,1):
                dofs=prepared.component_dofs[axis]
                if not np.isclose(uniform_vector[dofs].sum(),vector[dofs].sum(),rtol=1e-10,atol=1e-10):
                    raise ValueError('Uniform and spatial forces differ')
        for targets in (uniform,spatial):
            if np.any(targets[0]) or np.any(targets[-1]):
                raise ValueError('History must start and finish at zero load')
        count=len(moving.positions); results={}; summaries={}
        for route_index,(name,targets) in enumerate(zip(ROUTES,(uniform,tuple(spatial)),strict=True)):
            def update(current,total,message,offset=route_index*count,label=name):
                if progress is not None:
                    progress(offset+current,2*count,f'{label}：{message}')
            shared=advance_prepared_history(history,moving,prepared,targets,progress=update)
            result=_wrap(history,moving,prepared,shared)
            results[name]=result
            _,summary=export_result(result,stage/name)
            # Keep the existing numerical fields; amend the external-load provenance.
            summary['literature_load_route']=name
            summary['limitations']=['Prescribed literature macro load; fixed-mesh small-strain J2 response.',
                'Partial-contact force scales by host contact ratio; spatial shape normalizes separately.',
                'No material separation, chips, thermal coupling, roughness or independent validation.']
            summary['artifacts']={k:p.name for k,p in artifact_paths(stage/name).items()}
            _dump(stage/name/'pass_summary.json',summary)
            summaries[name]=summary
            for filename in EXPORTED_ROUTE_FILES:
                shutil.copy2(stage/name/filename,stage/(name+'_'+filename))
        _dump(stage/'input.json',case.to_dict())
        _dump(stage/'resolved_history_input.json',history.to_dict())
        shutil.copy2(stage/'force'/'grains.csv',stage/'grains.csv')
        shutil.copy2(stage/'force'/'force_components.png',stage/'force_components.png')
        shutil.copy2(stage/'baseline'/'reference_mesh.msh',stage/'reference_mesh.msh')
        with (stage/'projection_history.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=PROJECTION_COLUMNS); writer.writeheader(); writer.writerows(records)
        _dump(stage/'strip_template.json',dict(boundaries_u=template.boundaries_u.tolist(),
            forces_N={k:v.tolist() for k,v in template.forces_N.items()},totals_N=template.totals_N))
        _plot_comparison(results,stage/'literature_history_comparison.png')
        differences=[float(np.linalg.norm(a-b)) for a,b in zip(uniform,spatial,strict=True)]
        summary=dict(result_format=RESULT_FORMAT,normalized_input=case.to_dict(),input_fingerprint=case.input_fingerprint,
            unit_system='SI',package_version=__import__('grindcae').__version__,
            literature_force=dict(Ft_N=prediction['total_t_N'],Fn_N=prediction['total_n_N'],
                power_W=prediction['total_t_N']*case.literature.reference_case.wheel_speed_m_s,
                components=prediction['components'],counts=prediction['counts']),
            routes={name:{k:summaries[name][k] for k in ('pass','maximum_response','final_unloaded','mesh_reuse')} for name in ROUTES},
            comparison=dict(total_force_history_identical=True,maximum_nodal_load_difference_N=max(differences),
                maximum_component_projection_residual_N=max(abs(r['residual_N']) for r in records),
                uniform_fallback_component_count=sum(r['uniform_fallback'] for r in records)),
            material_provenance=case.material_provenance,projection=case.projection,
            assumptions=['Circumferential constant-force strips aggregate axial grains; no cross-track geometry.',
                'u=0 is lower motion-coordinate contact boundary; shallow wheel coordinates map to exact host arc projection.',
                'Wear is allocated equally among dynamic grains; width is already included in the total force.',
                'Partial-contact component totals use contact ratio once; clipped shapes normalize independently.',
                'A component with no clipped support uses a recorded uniform shape at the same target force.',
                'Engineering process/yield controls the coupled case; beta/fracture/friction/grit remain literature-page assumptions.',
                'J2 E/nu/Et come from the engineering material; transfer beyond 440C requires calibration.'],
            experimental_validation='not_verified', development_status='3.8.1 release candidate; independent experimental validation pending')
        files=sorted(p for p in stage.rglob('*') if p.is_file())
        summary['artifacts']={p.relative_to(stage).as_posix():p.relative_to(stage).as_posix() for p in files}
        summary['sha256']={name:digest(stage/name) for name in summary['artifacts']}
        _dump(stage/'summary.json',summary)
        read_result(stage)
        os.rename(stage,output)
    return summary


def read_result(directory):
    """Read-only integrity/provenance checks; never execute a finite-element solve."""
    root=Path(directory).resolve()
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    if summary.get('result_format') != RESULT_FORMAT:
        raise ValueError('Invalid literature history result format')
    canonical(summary)
    case=LiteratureHistoryCase.from_mapping(summary['normalized_input'])
    if summary['input_fingerprint'] != case.input_fingerprint:
        raise ValueError('Literature history fingerprint mismatch')
    if json.loads((root/'input.json').read_text(encoding='utf-8')) != case.to_dict():
        raise ValueError('Saved input differs from result')
    registered=summary['artifacts']
    if not isinstance(registered,dict) or set(registered)!=set(summary['sha256']):
        raise ValueError('Invalid artifact hashes')
    expected_files={'input.json','resolved_history_input.json','grains.csv','force_components.png',
                    'reference_mesh.msh','projection_history.csv','strip_template.json','literature_history_comparison.png'}
    expected_files.update('force/'+name for name in ('input.json','summary.json','grains.csv','force_components.png'))
    for name in ROUTES:
        expected_files.update(name+'/'+p.name for p in artifact_paths(root/name).values())
        expected_files.update(name+'_'+filename for filename in EXPORTED_ROUTE_FILES)
    if set(registered)!=expected_files:
        raise ValueError('Literature history managed artifact contract mismatch')
    for key,rel in registered.items():
        path=root/rel
        if key != rel or Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Unsafe result artifact')
        if not path.resolve().is_relative_to(root) or digest(path)!=summary['sha256'][key]:
            raise ValueError('Result artifact hash or containment mismatch: '+rel)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Symlink in result bundle')
    if actual != set(registered)|{'summary.json'}:
        raise ValueError('Missing or unregistered result files')
    force=read_force_result(root/'force')
    if force['normalized_input'] != case.literature.to_dict():
        raise ValueError('Force result belongs to another input')
    for output_key,force_key in (('Ft_N','total_t_N'),('Fn_N','total_n_N')):
        if summary['literature_force'][output_key] != force[force_key]:
            raise ValueError('Force summary mismatch')
    if summary['literature_force']['components']!=force['components'] or summary['literature_force']['counts']!=force['counts']:
        raise ValueError('Force component or population mismatch')
    if summary['literature_force']['power_W']!=force['total_t_N']*case.literature.reference_case.wheel_speed_m_s:
        raise ValueError('Force power mismatch')
    expected=resolved_history(case,force).to_dict()
    for filename in ('grains.csv','force_components.png'):
        if digest(root/filename)!=digest(root/'force'/filename):
            raise ValueError('Displayed force artifact differs from source')
    if json.loads((root/'resolved_history_input.json').read_text(encoding='utf-8')) != expected:
        raise ValueError('Resolved history input mismatch')
    history_rows={}
    for name in ROUTES:
        validate_artifacts(artifact_paths(root/name))
        child=json.loads((root/name/'pass_summary.json').read_text(encoding='utf-8'))
        if child['normalized_input']!=expected or child['literature_load_route']!=name:
            raise ValueError('History provenance mismatch')
        for filename in EXPORTED_ROUTE_FILES:
            if digest(root/(name+'_'+filename))!=digest(root/name/filename):
                raise ValueError('Displayed route artifact differs from source')
        for key in ('pass','maximum_response','final_unloaded','mesh_reuse'):
            if summary['routes'][name][key]!=child[key]:
                raise ValueError('Route summary mismatch')
        if child['final_unloaded']['external_force_norm_N'] != 0:
            raise ValueError('History final target is not unloaded')
        with (root/name/'pass_history.csv').open(encoding='utf-8',newline='') as stream:
            rows=list(csv.DictReader(stream)); history_rows[name]=rows
        for row in rows:
            for key,value in row.items():
                if key not in ('pass_state','state_committed') and not math.isfinite(float(value)):
                    raise ValueError('Nonfinite history value')
            for axis in ('x','y'):
                if not math.isclose(float(row['F'+axis+'_N']),float(row['assembled_F'+axis+'_N']),rel_tol=1e-10,abs_tol=1e-10):
                    raise ValueError('History force conservation mismatch')
    if len(history_rows['baseline'])!=len(history_rows['spatial']):
        raise ValueError('Route history length mismatch')
    for a,b in zip(history_rows['baseline'],history_rows['spatial'],strict=True):
        for key in ('position_id','motion_coordinate_m','pass_state','Fx_N','Fy_N'):
            if a[key]!=b[key]:
                raise ValueError('Comparison routes differ in targets')
    if digest(root/'baseline'/'reference_mesh.msh')!=digest(root/'spatial'/'reference_mesh.msh') or digest(root/'reference_mesh.msh')!=digest(root/'baseline'/'reference_mesh.msh'):
        raise ValueError('Routes do not use identical reference mesh')
    with (root/'projection_history.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream); rows=list(reader)
        if tuple(reader.fieldnames or ())!=PROJECTION_COLUMNS:
            raise ValueError('Projection CSV columns differ')
    if len(rows)!=8*summary['routes']['spatial']['pass']['target_position_count']:
        raise ValueError('Projection record count mismatch')
    expected_keys={(str(r['position_id']),key) for r in history_rows['spatial'][:-1] for key in force['components']}
    if len({(r['position_id'],r['component']) for r in rows})!=len(rows) or {(r['position_id'],r['component']) for r in rows}!=expected_keys:
        raise ValueError('Projection identity mismatch')
    for row in rows:
        a,b,r=(float(row[k]) for k in ('target_N','assembled_N','residual_N'))
        if not all(math.isfinite(v) for v in (a,b,r)) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10) or not math.isclose(b-a,r,abs_tol=1e-12):
            raise ValueError('Invalid component conservation record')
        if row['uniform_fallback'] not in ('True','False'):
            raise ValueError('Invalid fallback flag')
    for history_row in history_rows['spatial'][:-1]:
        selected=[r for r in rows if r['position_id']==history_row['position_id']]
        for suffix,force_key in (('_t_N','Fx_N'),('_n_N','Fy_N')):
            total=math.fsum(float(r['target_N']) for r in selected if r['component'].endswith(suffix))
            if not math.isclose(total,float(history_row[force_key]),rel_tol=1e-10,abs_tol=1e-10):
                raise ValueError('Projection components differ from history force')
    template=build_template(force)
    if json.loads((root/'strip_template.json').read_text(encoding='utf-8')) != dict(boundaries_u=template.boundaries_u.tolist(),
            forces_N={k:v.tolist() for k,v in template.forces_N.items()},totals_N=template.totals_N):
        raise ValueError('Strip template differs from saved force population')
    if summary['comparison']['total_force_history_identical'] is not True or summary['comparison']['maximum_component_projection_residual_N'] != max(abs(float(r['residual_N'])) for r in rows):
        raise ValueError('Comparison summary mismatch')
    if summary['comparison']['uniform_fallback_component_count'] != sum(r['uniform_fallback']=='True' for r in rows):
        raise ValueError('Fallback count mismatch')
    import meshio
    for name in ROUTES:
        mesh=meshio.read(root/name/'final_results.vtu')
        arrays=[mesh.points,*mesh.point_data.values()]
        arrays.extend(a for group in mesh.cell_data.values() for a in group)
        if any(not np.isfinite(a).all() for a in arrays if np.issubdtype(np.asarray(a).dtype,np.number)):
            raise ValueError('Nonfinite finite-element field')
    from PIL import Image
    for relative in registered:
        if relative.endswith('.png'):
            with Image.open(root/relative) as picture:
                if picture.format!='PNG' or min(picture.size)<1:
                    raise ValueError('Invalid result image')
                picture.verify()
    return summary
