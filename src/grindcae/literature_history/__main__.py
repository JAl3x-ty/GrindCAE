import argparse
import json
from pathlib import Path
from .models import LiteratureHistoryCase
from .workflow import run_literature_history


def main():
    parser=argparse.ArgumentParser(description='Literature-force uniform/strip J2 history; development route')
    parser.add_argument('case',type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    case=LiteratureHistoryCase.from_mapping(json.loads(args.case.read_text(encoding='utf-8-sig')))
    run_literature_history(case,args.output_dir)


if __name__=='__main__':
    main()
