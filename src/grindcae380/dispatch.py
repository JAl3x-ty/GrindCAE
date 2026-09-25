"""Versioned input/result dispatch; legacy computation remains unchanged."""
from .core import Case, RESULT_FORMAT as V1, predict as predict_v1
from .reconstruction import ReconstructionCase, RESULT_FORMAT as V2, predict as predict_v2

RESULT_FORMATS = (V1,V2)


def read_case(data):
    if not isinstance(data,dict):
        raise ValueError('case must be a JSON object')
    if type(data.get('schema_version')) is not int:
        raise ValueError('schema_version must be an integer')
    if data['schema_version']==1:
        return Case.from_mapping(data)
    if data['schema_version']==2:
        return ReconstructionCase.from_mapping(data)
    raise ValueError('unsupported schema_version')


def predict(case):
    if isinstance(case,Case):
        return predict_v1(case)
    if isinstance(case,ReconstructionCase):
        return predict_v2(case)
    raise ValueError('validated versioned case required')
