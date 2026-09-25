"""Experimental Abaqus batch boundary. A zero exit code is insufficient evidence."""
from pathlib import Path
import re


def classify_job(directory, name, returncode, log):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name):
        raise ValueError('Invalid job name')
    directory=Path(directory)
    lower=log.lower()
    status='incomplete'
    if 'license' in lower and ('not available' in lower or 'error' in lower or 'unable' in lower):
        status='license_unavailable'
    elif returncode!=0 or 'abaqus error' in lower or 'exited with error' in lower:
        status='failed'
    else:
        sta=directory/(name+'.sta')
        odb=directory/(name+'.odb')
        if (sta.is_file() and odb.is_file() and odb.stat().st_size>0 and
            'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in sta.read_text(errors='replace').upper()):
            status='solver_completed'
    return {'status':status, 'solver_completed':status=='solver_completed',
            'physical_acceptance':'pending_odb_validation' if status=='solver_completed' else 'unavailable'}
