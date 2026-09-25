"""python -m grindcae380 CASE.json --output-dir NEW_DIRECTORY."""
import argparse
import json
from pathlib import Path

from .dispatch import read_case
from .workflow import run_case


def main():
    parser=argparse.ArgumentParser(description='GrindCAE 3.8.0 independent Zhang 2017 force kernel')
    parser.add_argument('case',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    try:
        case=read_case(json.loads(args.case.read_text(encoding='utf-8-sig')))
        result=run_case(case,args.output_dir)
    except (ValueError,OSError) as exc:
        parser.exit(2,f'Calculation rejected: {exc}\n')
    print(json.dumps({k:result[k] for k in ['kernel_version','total_t_N','total_n_N','counts','formula_convention','experimental_validation']},indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
