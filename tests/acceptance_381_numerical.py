"""Bounded physical/numerical exercises; synthetic material is labelled explicitly."""
from pathlib import Path
import sys,json,time,csv
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from literature_history_fixture import built_case
from grindcae.literature_history import run_literature_history,read_result

target=root/'outputs'/('numerical_381_'+str(int(time.time())));target.mkdir()
report={}
for name,direction,yield_mpa in [('forward','positive_x','225'),('reverse','negative_x','225'),('synthetic_plastic','positive_x','0.5')]:
    case=built_case(direction,yield_mpa)
    started=time.monotonic()
    result=run_literature_history(case,target/name)
    assert read_result(target/name)==result
    for route in ('baseline','spatial'):
        final=result['routes'][route]['final_unloaded']
        assert final['external_force_norm_N']==0
        assert final['balance_residual_N']<1e-6
        if name=='synthetic_plastic':
            assert final['maximum_equivalent_plastic_strain']>0
            assert final['accumulated_plastic_dissipation_J']>0
        with (target/name/route/'pass_history.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        strain=np.array([float(r['maximum_equivalent_plastic_strain']) for r in rows])
        assert np.all(np.diff(strain)>=-1e-14)
    report[name]=dict(output=str(target/name),elapsed_s=time.monotonic()-started,
                     force=result['literature_force'],routes=result['routes'],comparison=result['comparison'])
    print(name,report[name]['elapsed_s'],flush=True)
    (root/'evidence/381_numerical_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
for route in ('baseline','spatial'):
    a=report['forward']['routes'][route]['maximum_response']['maximum_displacement_m']
    b=report['reverse']['routes'][route]['maximum_response']['maximum_displacement_m']
    # Mesh triangulation need not mirror; this is a coarse directional diagnostic.
    report.setdefault('direction_diagnostic',{})[route]=dict(forward_m=a,reverse_m=b,relative_difference=abs(a-b)/max(a,b))
(root/'evidence/381_numerical_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(target,flush=True)
