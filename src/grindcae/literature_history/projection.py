"""Four-component conservative strip loads, integrated against P1 edge shapes.

Axial tracks collapse into circumferential strips. This is a recorded macro
closure, not a measured pressure field or resolved grain contact. Partial
contact uses the retained contact-ratio total; clipping only changes shape.
"""
from dataclasses import dataclass
import math
import numpy as np

from grindcae.trajectory import motion_coordinate_from_physical_x

COMPONENTS = ('plastic', 'removal', 'rake', 'wear')


@dataclass(frozen=True)
class StripTemplate:
    boundaries_u: np.ndarray
    forces_N: dict[str, np.ndarray]
    totals_N: dict[str, float]


def build_template(prediction):
    diag = prediction['wheel_diagnostics']
    nx = diag['nx']
    length = diag['contact_length_m']
    if nx < 1 or length <= 0:
        raise ValueError('No circumferential strips available for FE projection')
    boundaries = np.arange(nx + 1, dtype=float) * diag['pitch_m'] / length
    boundaries[-1] = 1.0
    forces = {f'{k}_{d}_N': np.zeros(nx) for k in COMPONENTS for d in ('t', 'n')}
    dynamic = prediction['counts']['dynamic']
    for row in prediction['grains']:
        i = row['circumferential_index']
        for d in ('t', 'n'):
            for k in COMPONENTS[:3]:
                forces[f'{k}_{d}_N'][i] += row[f'{k}_{d}_N']
            if row['depth_m'] > 0 and dynamic:
                forces[f'wear_{d}_N'][i] += prediction['components'][f'wear_{d}_N'] / dynamic
    for key, values in forces.items():
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid strip force')
        if not math.isclose(float(values.sum()), prediction['components'][key], rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError('Strip force does not conserve literature component '+key)
    return StripTemplate(boundaries, forces, dict(prediction['components']))


def project_load(prepared, position, template):
    vector = np.zeros(prepared.total_degrees_of_freedom)
    records = []
    interval = position.pass_load.effective_contact_interval_m
    ratio = position.pass_load.contact_ratio
    if interval is None:
        if ratio != 0 or position.pass_load.current_Fx_N != 0 or position.pass_load.current_Fy_N != 0:
            raise ValueError('Absent contact interval with nonzero target force')
        for key in template.forces_N:
            records.append(dict(component=key, target_N=0., assembled_N=0., residual_N=0., uniform_fallback=False))
        return vector, records
    trajectory = position.pass_load.trajectory_result
    width = trajectory.case.workpiece.length_m
    arc = trajectory.exact_arc_projected_length_m
    direction = trajectory.case.single_pass.relative_feed_direction
    s = position.position.motion_coordinate_m
    left, right = interval
    def u_at(x):
        return (motion_coordinate_from_physical_x(x, width, direction)-s)/arc
    ua, ub = sorted((u_at(left), u_at(right)))
    if ua < -1e-9 or ub > 1+1e-9 or right <= left or arc <= 0:
        raise ValueError('Contact interval outside normalized projection strip domain')
    edges = template.boundaries_u
    sign_t = 1.0 if position.pass_load.current_Fx_N >= 0 else -1.0
    for key, values in template.forces_N.items():
        axis = 0 if key.endswith('_t_N') else 1
        sign = sign_t if axis == 0 else -1.0
        target = ratio * template.totals_N[key]
        density = values/np.diff(edges)
        overlap = np.maximum(0., np.minimum(edges[1:], ub)-np.maximum(edges[:-1], ua))
        integral = float(np.dot(density, overlap))
        fallback = target > 0 and integral == 0
        scale = target/(right-left) if fallback else (target/(arc*integral) if integral > 0 else 0.)
        component_vector = np.zeros_like(vector)
        for facet in position.load_mapping.facets:
            a, b = max(left, facet.overlap_x_start_m), min(right, facet.overlap_x_end_m)
            if b <= a:
                continue
            u0, u1 = u_at(a), u_at(b)
            cuts = [a, b]
            for edge in edges[1:-1]:
                if min(u0,u1) < edge < max(u0,u1):
                    cuts.append(a+(edge-u0)*(b-a)/(u1-u0))
            cuts.sort()
            for x0,x1 in zip(cuts,cuts[1:]):
                midpoint = .5*(x0+x1)
                strip = min(len(density)-1, max(0,int(np.searchsorted(edges,u_at(midpoint),side='right')-1)))
                q = scale if fallback else scale*density[strip]
                # Constant q times linear P1 is integrated exactly at midpoint.
                local = (midpoint-facet.facet_x_start_m)/facet.facet_dx_m
                if not -1e-10 <= local <= 1+1e-10:
                    raise ValueError('Invalid P1 facet coordinate')
                force = sign*q*(x1-x0)
                component_vector[prepared.component_dofs[axis,facet.node_start_id]] += force*(1-local)
                component_vector[prepared.component_dofs[axis,facet.node_end_id]] += force*local
        assembled = float(component_vector[prepared.component_dofs[axis]].sum())
        residual = assembled-sign*target
        if not math.isclose(assembled,sign*target,rel_tol=1e-10,abs_tol=1e-10):
            raise ValueError('Component projection failed force conservation: '+key)
        vector += component_vector
        records.append(dict(component=key,target_N=sign*target,assembled_N=assembled,
                            residual_N=residual,uniform_fallback=fallback))
    sums = [float(vector[prepared.component_dofs[a]].sum()) for a in (0,1)]
    if not np.allclose(sums,[position.pass_load.current_Fx_N,position.pass_load.current_Fy_N],rtol=1e-10,atol=1e-10):
        raise ValueError('Literature projection differs from host force history')
    return vector,records
