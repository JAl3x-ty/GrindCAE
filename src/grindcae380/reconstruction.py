"""Explicit bounded-wheel closure of gaps in Zhang 2017; not author code.

Schema 2 wraps the unchanged schema-1 material/process parameter record.
The legacy source, vibration_steps and depths fields select no wheel algorithm
here: bounded_uniform_recess_v1 is the only schema-2 population algorithm.
"""
from dataclasses import dataclass, fields, replace
import hashlib
import math
from statistics import NormalDist

import numpy as np

from .core import Case, canonical, dynamic_depths, number, predict as predict_v1

RESULT_FORMAT = 'grindcae_zhang2017_reconstruction_v2'


@dataclass(frozen=True)
class ReconstructionCase:
    schema_version: int
    reference_case: Case
    relief_depth_m: float
    boundary_rule: str
    reconstruction_provenance: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError('reconstruction schema_version must be integer 2')
        if not isinstance(self.reference_case, Case):
            raise ValueError('reference_case requires a validated schema-1 parameter record')
        if self.reference_case.source != 'stochastic_eq40':
            raise ValueError('reference_case source must be stochastic_eq40, used only as parameter record')
        depth = number(self.relief_depth_m, 'relief_depth_m', strict=True)
        if depth > self.reference_case.grain_min_m:
            raise ValueError('relief_depth_m must not exceed minimum grain diameter')
        if self.boundary_rule not in ('finite_entry', 'periodic_envelope'):
            raise ValueError('boundary_rule must be finite_entry or periodic_envelope')
        if not isinstance(self.reconstruction_provenance, str) or not self.reconstruction_provenance.strip():
            raise ValueError('reconstruction_provenance required')

    def to_dict(self):
        return dict(schema_version=2, reference_case=self.reference_case.to_dict(),
                    relief_depth_m=self.relief_depth_m, boundary_rule=self.boundary_rule,
                    reconstruction_provenance=self.reconstruction_provenance)

    @classmethod
    def from_mapping(cls, data):
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError('schema 2 requires exactly the registered fields')
        return cls(**dict(data, reference_case=Case.from_mapping(data['reference_case'])))

    @classmethod
    def paper_example(cls):
        base = replace(Case.paper_example(), wear_coefficient_N_m=0.,
                       wear_provenance='Wear disabled until explicit reference calibration; no guessed unit conversion.')
        return cls(2, base, (base.grain_min_m+base.grain_max_m)/4,
                   'finite_entry',
                   'bounded_uniform_recess_v1: fixed dressing plane z=0; recess uniform over half the mean grain diameter. '
                   'Maximum-entropy closure given this bounded support; unmeasured geometry assumption, not a paper datum. '
                   'Truncated Table 6 grain diameters; independent axial tracks; boundary selected explicitly by boundary_rule.')


def periodic_depths(static_depth_m, pitch_m, kinematic_slope):
    """Exact lower envelope of all preceding grains in a repeated contact patch.

    h_i=max(0,min(d_i, min_k(s*k*p+d_i-d_(i-k)))), k=1..nx.
    Including the preceding copy of self bounds the gap by one patch advance.
    This is a periodic RVE approximation, not a full-wheel trajectory.
    """
    d = np.asarray(static_depth_m, dtype=float)
    if d.ndim != 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('static depths must be a finite nonnegative 2D matrix')
    pitch = number(pitch_m, 'pitch_m', strict=True)
    slope = number(kinematic_slope, 'kinematic_slope')
    h = d.copy()
    previous = np.full(d.shape, -1, dtype=np.int64)
    indices = np.arange(d.shape[0])[:, None]
    for lag in range(1, d.shape[0]+1):
        candidate = slope*lag*pitch+d-np.roll(d, lag, axis=0)
        selected = candidate < h
        previous = np.where(selected, (indices-lag) % d.shape[0], previous)
        h = np.minimum(h, candidate)
    return np.maximum(h, 0.)/2, previous


def bounded_population(case):
    c = case.reference_case
    mean = (c.grain_min_m+c.grain_max_m)/2
    std = (c.grain_max_m-c.grain_min_m)/6
    vg = 2*(31-c.organization_number)/100
    pitch = mean*math.sqrt(math.pi/(4*vg))
    length = math.sqrt(c.wheel_diameter_m*c.depth_m)
    nx, ny = math.floor(length/pitch), math.floor(c.width_m/pitch)
    count = nx*ny
    if count > 200000 or (case.boundary_rule == 'periodic_envelope' and count*nx > 20000000):
        raise ValueError('bounded reconstruction workload exceeded')
    size_seed, height_seed = np.random.SeedSequence(c.seed).spawn(2)
    size_rng = np.random.Generator(np.random.PCG64(size_seed))
    height_rng = np.random.Generator(np.random.PCG64(height_seed))
    if std > 0:
        normal = NormalDist()
        lo, hi = normal.cdf(-3), normal.cdf(3)
        u = size_rng.uniform(lo, hi, count)
        sizes = np.array([mean+std*normal.inv_cdf(float(v)) for v in u]).reshape(nx, ny)
    else:
        sizes = np.full((nx, ny), mean)
    recess = height_rng.uniform(0., case.relief_depth_m, size=(nx, ny))
    static = np.maximum(c.depth_m-recess, 0.)
    slope = 2*c.feed_speed_m_s/c.wheel_speed_m_s*math.sqrt(c.depth_m/c.wheel_diameter_m)
    operation = dynamic_depths if case.boundary_rule == 'finite_entry' else periodic_depths
    ag, previous = operation(static, pitch, slope)
    if c.feed_speed_m_s == 0:
        ag[:] = 0.
    rows = [dict(grain_id=i*ny+j, circumferential_index=i, axial_index=j,
                 x_m=(i+.5)*pitch, y_m=(j+.5)*pitch, diameter_m=float(sizes[i,j]),
                 protrusion_m=-float(recess[i,j]), static_depth_m=float(static[i,j]),
                 previous_dynamic_row=int(previous[i,j]), depth_m=float(ag[i,j]))
            for i in range(nx) for j in range(ny)]
    diag = dict(algorithm='bounded_uniform_recess_v1', nx=nx, ny=ny, pitch_m=pitch,
                contact_length_m=length, represented_length_m=nx*pitch, represented_width_m=ny*pitch,
                dressing_plane_m=0., recess_support_m=[0.,case.relief_depth_m],
                protrusion_range_m=float(np.ptp(recess)) if count else 0.,
                corundum_fraction=vg, mean_diameter_m=mean, diameter_std_before_truncation_m=std,
                diameter_distribution='normal_conditioned_on_table6_range',
                boundary_rule=case.boundary_rule, integer_rule='floor',
                rng='numpy.PCG64; SeedSequence.spawn diameter,height', numpy_version=np.__version__,
                reference_fields_unused=['source','vibration_steps','depths_m'],
                kinematic_slope=slope,
                first_entry_dynamic_count=int(sum(r['depth_m']>0 and r['previous_dynamic_row']==-1 for r in rows)),
                beta_extrapolation_count=int(sum(r['depth_m']>4.5e-6 for r in rows)))
    return rows, diag


def predict(case):
    if not isinstance(case, ReconstructionCase):
        raise ValueError('validated ReconstructionCase required')
    rows, diag = bounded_population(case)
    base = case.reference_case
    force_case = replace(base, source='explicit_depths', depths_m=tuple(r['depth_m'] for r in rows) or (0.,))
    result = predict_v1(force_case)
    for row, force in zip(rows, result['grains']):
        row.update({key:value for key,value in force.items() if key not in ('grain_id','static_depth_m')})
    result['grains'] = rows
    result['counts']['sampled'] = len(rows)
    result['counts']['static'] = sum(r['static_depth_m']>0 for r in rows)
    diag['first_entry_nonwear_normal_N'] = math.fsum(r['total_n_N'] for r in rows if r['previous_dynamic_row']==-1)
    diag['first_entry_nonwear_tangential_N'] = math.fsum(r['total_t_N'] for r in rows if r['previous_dynamic_row']==-1)
    result.update(result_format=RESULT_FORMAT, normalized_input=case.to_dict(),
                  input_fingerprint=hashlib.sha256(canonical(case.to_dict()).encode('utf-8')).hexdigest(),
                  wheel_diagnostics=diag,
                  reproduction_status='literature_based_bounded_reconstruction_with_declared_closure',
                  reconstruction_provenance=case.reconstruction_provenance,
                  wear_provenance=base.wear_provenance)
    result['limitations'] = [
        'Bounded uniform recess is an unmeasured dressing/embedding closure; half mean diameter is an explicit default assumption.',
        'Independent axial tracks omit cross-track conical groove overlap and measured spatial correlations.',
        'Finite-entry response includes virgin-material first grains; periodic_envelope repeats the contact patch, not the full circumference.',
        'Table 6 width 50 mm is retained; Table 7 lists 20 mm. Width and height scale require physical measurement.',
        'Sampled grain diameter sets population pitch statistically; local force uses a common cone angle and effective depth.',
        'K1 is supplied in N m with recorded provenance; a reference-point fit does not recover the undocumented printed unit.',
        'Continuous_projection enforces stated stress continuity and friction projection; printed convention remains available.',
        'Table 2 beta is retained including its 3.8 um discontinuity; depths above 4.5 um extrapolate the scratch evidence.',
        'No FE field, chip geometry, thermal evolution, measured roughness or independent experimental validation.',
    ]
    canonical(result)
    return result


def calibrate_normal(case, *, target_normal_N, source, seeds=tuple(range(2017,2033))):
    """Fit K1 only to ensemble mean Fn; keep Ft as an unfitted comparison.

    No units are inferred from the paper's undocumented numeric K1=18.36.
    A nonnegative solution must exist for the declared geometry and boundary.
    """
    target = number(target_normal_N, 'target_normal_N', strict=True)
    if not isinstance(source,str) or not source.strip():
        raise ValueError('calibration source required')
    if not isinstance(seeds,(tuple,list)) or not seeds or len(seeds)>128:
        raise ValueError('provide 1..128 unique calibration seeds')
    if any(type(s) is not int or not 0<=s<2**63 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('seeds must be unique nonnegative bounded integers')
    c = case.reference_case
    results = [predict(replace(case, reference_case=replace(c,seed=s,wear_coefficient_N_m=0.))) for s in seeds]
    baseline = math.fsum(r['total_n_N'] for r in results)/len(results)
    mean_count = math.fsum(r['counts']['dynamic'] for r in results)/len(results)
    scale = 4*mean_count*c.feed_speed_m_s/c.wheel_speed_m_s/c.wheel_diameter_m
    if scale <= 0 or target < baseline:
        raise ValueError('no nonnegative identifiable K1 for this population and target')
    k1 = (target-baseline)/scale
    report = dict(method='ensemble_normal_only_linear_inverse', source=source,
                  target_normal_N=target, seeds=list(seeds), coefficient_N_m=k1,
                  mean_nonwear_normal_N=baseline, mean_dynamic_count=mean_count,
                  mean_tangential_N=math.fsum(r['total_t_N'] for r in results)/len(results)+c.friction_coefficient*(target-baseline),
                  calibration_case_fingerprint=hashlib.sha256(canonical(case.to_dict()).encode('utf-8')).hexdigest(),
                  independent_validation='not_verified',
                  identifiability='Conditional on the specified wheel population, boundary, material and formula; not a unique physical wear-area measurement.')
    provenance = canonical(report)
    calibrated = replace(case, reference_case=replace(c,wear_coefficient_N_m=k1,wear_provenance=provenance))
    return calibrated, report
