"""Exclusive-directory transactional publication and validated rereading."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .core import canonical
from .dispatch import RESULT_FORMATS, read_case, predict


def plot_result(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    c = result['components']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ['Plastic flow', 'Fracture stress', 'Rake friction', 'Wear flat']
    for axis, suffix, title in zip(ax, ['t', 'n'], ['Tangential force', 'Normal force']):
        axis.bar(names, [c[f'{key}_{suffix}_N'] for key in ['plastic','removal','rake','wear']], color=['#407e9c','#cb8154','#69a899','#9a88b9'])
        axis.set_ylabel('Force (N)'); axis.set_title(title)
        axis.tick_params(axis='x', labelrotation=20)
        axis.grid(axis='y', alpha=.2); axis.set_axisbelow(True)
    fig.suptitle('Zhang 2017 analytical force reconstruction | '+result['formula_convention'])
    fig.text(.5,.015,'Declared assumptions; experimental agreement not verified',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93)); fig.savefig(path,dpi=160); plt.close(fig)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_case(case, output_directory):
    target = Path(output_directory).resolve()
    if target.exists():
        raise FileExistsError(f'Use a new output directory: {target}')
    result = predict(case)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.grindcae380-', dir=target.parent) as temp:
        folder = Path(temp)
        (folder/'input.json').write_text(json.dumps(case.to_dict(), indent=2, ensure_ascii=False, allow_nan=False),encoding='utf-8')
        with (folder/'grains.csv').open('w', encoding='utf-8-sig', newline='') as f:
            cols = list(result['grains'][0]) if result['grains'] else ['grain_id','depth_m','stage','total_t_N','total_n_N']
            writer=csv.DictWriter(f,fieldnames=cols); writer.writeheader(); writer.writerows(result['grains'])
        plot_result(result,folder/'force_components.png')
        result['artifacts']={name:digest(folder/name) for name in ['input.json','grains.csv','force_components.png']}
        (folder/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        read_result(folder)
        # On Windows rename to an existing directory fails. No managed file is overwritten.
        os.rename(folder,target)
    return result


def read_result(directory):
    from PIL import Image
    folder=Path(directory).resolve()
    required={'summary.json','input.json','grains.csv','force_components.png'}
    if {p.name for p in folder.iterdir()} != required:
        raise ValueError('unexpected or missing result artifacts')
    if any(p.is_symlink() or p.resolve().parent!=folder or not p.is_file() for p in folder.iterdir()):
        raise ValueError('unsafe result artifact')
    r=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if r.get('result_format') not in RESULT_FORMATS or set(r.get('artifacts',{})) != required-{'summary.json'}:
        raise ValueError('wrong result contract')
    for name, sha in r['artifacts'].items():
        if digest(folder/name)!=sha:
            raise ValueError(f'artifact hash mismatch: {name}')
    c=read_case(json.loads((folder/'input.json').read_text(encoding='utf-8')))
    expected=predict(c)
    # Recompute deterministic result; package version may differ on a later reader.
    for key in expected:
        if key!='kernel_version' and r.get(key)!=expected[key]:
            raise ValueError(f'result does not agree with saved input: {key}')
    with (folder/'grains.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=len(r['grains']):
        raise ValueError('CSV grain count mismatch')
    for row, expected_row in zip(rows,r['grains'],strict=True):
        if set(row)!=set(expected_row):
            raise ValueError('CSV fields mismatch')
        for key,v in expected_row.items():
            if isinstance(v,str):
                if row[key]!=v:raise ValueError('CSV text mismatch')
            elif not math.isfinite(float(row[key])) or float(row[key])!=float(v):
                raise ValueError('CSV numeric mismatch')
    with Image.open(folder/'force_components.png') as img:
        if img.format!='PNG' or min(img.size)<100:raise ValueError('invalid result PNG')
        img.verify()
    canonical(r)
    return r
