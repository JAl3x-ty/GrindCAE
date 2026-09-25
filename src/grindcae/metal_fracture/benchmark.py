"""Independent physical acceptance of the controlled interface benchmark."""
import math


def assess_interface_cycle(rows, *, strength_Pa, fracture_energy_J_m2, height_m,
                           bulk_count_initial, bulk_count_final):
    if not rows or any(not math.isfinite(float(r[k])) for r in rows
                       for k in ('gap_m', 'traction_Pa', 'damage')):
        raise ValueError('Missing or nonfinite physical history')
    for x in (strength_Pa, fracture_energy_J_m2, height_m):
        if not math.isfinite(x) or x <= 0:
            raise ValueError('Positive finite benchmark parameters required')
    opening = [r for r in rows if r['phase'] == 'open']
    closing = [r for r in rows if r['phase'] == 'close']
    reopening = [r for r in rows if r['phase'] == 'reopen']
    work = sum(.5*(a['traction_Pa']+b['traction_Pa'])*(b['gap_m']-a['gap_m'])
               for a,b in zip(opening, opening[1:]))
    peak = max((r['traction_Pa'] for r in opening), default=0.)
    checks = {
        'complete_cycle': bool(opening and closing and reopening),
        'peak_traction': abs(peak / strength_Pa - 1.) <= .02,
        'fracture_work': abs(work / fracture_energy_J_m2 - 1.) <= .02,
        'fully_failed': bool(opening) and opening[-1]['damage'] >= 1.-1e-6,
        'compressive_recontact': any(r['traction_Pa'] < -strength_Pa*1e-3 for r in closing),
        'penetration': min(r['gap_m'] for r in rows) >= -height_m*1e-6,
        'no_rebonding': bool(reopening) and all(r['damage'] >= 1.-1e-6 and
                         r['traction_Pa'] <= strength_Pa*1e-6 for r in reopening),
        'irreversibility': all(b['damage'] >= a['damage']-1e-6 for a,b in zip(rows, rows[1:])),
        'bulk_retained': bulk_count_initial > 0 and bulk_count_initial == bulk_count_final,
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'opening_work_J_m2': work, 'peak_traction_Pa': peak,
            'minimum_gap_m': min(r['gap_m'] for r in rows)}
