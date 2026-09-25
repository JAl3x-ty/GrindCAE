"""Zhang et al. (2017), DOI 10.1016/j.ijmachtools.2017.06.002.

All public lengths are m, stresses Pa, speeds m/s and forces N. The Table 2
fit alone evaluates depth in micrometres. See docs/PAPER_AUDIT.md before
interpreting the stochastic reconstruction as experimental prediction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math

import numpy as np

DOI = '10.1016/j.ijmachtools.2017.06.002'
RESULT_FORMAT = 'grindcae_zhang2017_force_v1'
CONVENTIONS = ('printed', 'continuous_projection')


def number(value, name, *, minimum=0.0, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name}: finite numeric value required (not bool/text)')
    if value < minimum or (strict and value == minimum):
        raise ValueError(f'{name}: outside physical range')
    return float(value)


def canonical(data):
    return json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class Case:
    schema_version: int
    unit_system: str
    source: str
    material_id: str
    yield_stress_Pa: float
    fracture_stress_Pa: float
    cone_half_angle_rad: float
    friction_coefficient: float
    formula_convention: str
    wheel_diameter_m: float
    width_m: float
    depth_m: float
    wheel_speed_m_s: float
    feed_speed_m_s: float
    grain_min_m: float
    grain_max_m: float
    organization_number: int
    seed: int
    vibration_steps: int
    wear_coefficient_N_m: float
    wear_provenance: str
    depths_m: tuple[float, ...]

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError('schema_version must be integer 1')
        if self.unit_system != 'SI':
            raise ValueError('unit_system must be SI')
        if self.source not in ('explicit_depths', 'stochastic_eq40'):
            raise ValueError('source must select explicit depths or equation 40 reconstruction')
        if self.material_id != '440C_zhang2017_literature':
            raise ValueError('Table 2 beta fit is available only for the named 440C literature material')
        if self.formula_convention not in CONVENTIONS:
            raise ValueError('formula_convention must be explicit')
        for name in ('yield_stress_Pa', 'fracture_stress_Pa', 'wheel_diameter_m',
                     'width_m', 'wheel_speed_m_s', 'grain_min_m', 'grain_max_m'):
            number(getattr(self, name), name, strict=True)
        for name in ('depth_m', 'feed_speed_m_s', 'friction_coefficient', 'wear_coefficient_N_m'):
            number(getattr(self, name), name)
        angle = number(self.cone_half_angle_rad, 'cone_half_angle_rad', strict=True)
        if angle >= math.pi / 2:
            raise ValueError('cone_half_angle_rad must be below pi/2')
        if self.depth_m >= self.wheel_diameter_m / 2:
            raise ValueError('depth_m must be below wheel radius')
        if self.grain_min_m > self.grain_max_m:
            raise ValueError('grain_min_m must not exceed grain_max_m')
        for name, low, high in (('seed', 0, 2**63-1), ('vibration_steps', 800, 100000), ('organization_number', 0, 30)):
            v = getattr(self, name)
            if type(v) is not int or not low <= v <= high:
                raise ValueError(f'{name}: integer in [{low}, {high}] required')
        if not isinstance(self.wear_provenance, str) or not self.wear_provenance.strip():
            raise ValueError('wear_provenance required, including unit assumption or calibration source')
        if not isinstance(self.depths_m, (list, tuple)):
            raise ValueError('depths_m must be an array')
        for d in self.depths_m:
            number(d, 'depths_m element')
        object.__setattr__(self, 'depths_m', tuple(self.depths_m))
        if self.source == 'explicit_depths' and not self.depths_m:
            raise ValueError('explicit_depths requires at least one depth')
        if self.source == 'stochastic_eq40' and self.depths_m:
            raise ValueError('stochastic source cannot also provide depths')

    def to_dict(self):
        d = asdict(self)
        d['depths_m'] = list(self.depths_m)
        return d

    @classmethod
    def from_mapping(cls, d):
        keys = {f.name for f in fields(cls)}
        if not isinstance(d, dict) or set(d) != keys:
            raise ValueError('input must contain exactly the registered schema fields')
        return cls(**d)

    @classmethod
    def paper_example(cls, source='stochastic_eq40'):
        return cls(1, 'SI', source, '440C_zhang2017_literature', 225e6, 540e6,
                   1.047, .607, 'continuous_projection', .3, .05, 15e-6,
                   20., 2/60, .158e-3, .202e-3, 12, 2017, 800, .01836,
                   'Unverified unit hypothesis: paper K1=18.36 interpreted as N mm; SI value 0.01836 N m. Not calibrated.',
                   (0., .5e-6, 1e-6, 1.18e-6, 2e-6, 2.85e-6, 3.8e-6, 5e-6)
                   if source == 'explicit_depths' else ())


def cutting_efficiency(depth_m):
    a = number(depth_m, 'depth_m') * 1e6
    if a <= 3.8:
        p = ((.908, 4.594, 2.258), (.187, 1.891, .887), (.377, 2.484, 1.178),
             (.118, 1.864, .205), (.307, 1.481, .206), (.142, .849, .537))
        beta = math.fsum(A * math.exp(-((a-B)/C)**2) for A, B, C in p)
    else:
        beta = .922 * math.exp(-5.211 * math.exp(-2.004*a))
    if not 0 < beta < 1:
        raise ValueError('Table 2 beta fit outside its admissible angular domain')
    return beta


def grain_stage(depth_m):
    a = number(depth_m, 'depth_m')
    if a == 0:
        return 'inactive'
    if a < 1.18e-6:
        return 'ploughing'
    return 'transition' if a < 2.85e-6 else 'cutting'


@dataclass(frozen=True)
class GrainForce:
    depth_m: float
    stage: str
    beta: float
    alpha1_rad: float
    plastic_t_N: float
    plastic_n_N: float
    removal_t_N: float
    removal_n_N: float
    rake_t_N: float
    rake_n_N: float

    @property
    def total_t_N(self):
        return math.fsum((self.plastic_t_N, self.removal_t_N, self.rake_t_N))

    @property
    def total_n_N(self):
        return math.fsum((self.plastic_n_N, self.removal_n_N, self.rake_n_N))

    def to_dict(self):
        return dict(asdict(self), total_t_N=self.total_t_N, total_n_N=self.total_n_N)


def grain_force(depth_m, yield_stress_Pa, fracture_stress_Pa, cone_half_angle_rad,
                friction_coefficient, formula_convention):
    a = number(depth_m, 'depth_m')
    sy = number(yield_stress_Pa, 'yield_stress_Pa', strict=True)
    sb = number(fracture_stress_Pa, 'fracture_stress_Pa', strict=True)
    theta = number(cone_half_angle_rad, 'cone_half_angle_rad', strict=True)
    mu = number(friction_coefficient, 'friction_coefficient')
    if theta >= math.pi/2 or formula_convention not in CONVENTIONS:
        raise ValueError('unsupported angle or formula convention')
    stage = grain_stage(a); beta = cutting_efficiency(a)
    alpha = math.acos(math.sqrt(beta)); t = math.tan(theta); q = a*a
    if stage in ('inactive', 'ploughing'):
        pt = sy*q*t; pn = math.pi/2*sy*q*t*t
        rt = rn = 0.
        friction_scalar = math.pi/2*sy
        ft = mu*q*t*t*friction_scalar
        fn = mu*q*t*friction_scalar
    else:
        # Eq.7; algebraic rationalization avoids cancellation as beta -> 0.
        denom = beta / (1 + math.sqrt(1-beta))
        ds = math.pi*t*sb/(2*denom)
        factor = 2. if formula_convention == 'printed' else 1.
        # Independent analytic primitives of alpha*cos(alpha) and alpha.
        flow_t = math.sin(alpha)+(math.cos(alpha)-1)/alpha
        chip_t = 1-math.sin(alpha)
        chip_n = math.pi/2-alpha
        pt = sy*q*t*(flow_t+chip_t)
        pn = sy*q*t*t*(alpha/2+chip_n)
        # This split groups the fracture-stress terms in both regions together.
        rt = ds*q*t*(factor*flow_t+chip_t)
        rn = ds*q*t*t*(factor*alpha/2+chip_n)
        friction_scalar = (sy+factor*ds)*alpha/2+(sy+ds)*chip_n
        total_friction = mu*q*t/math.cos(theta)*friction_scalar
        ft = total_friction*(1. if formula_convention == 'printed' else math.sin(theta))
        fn = total_friction*math.cos(theta)
    r = GrainForce(a, stage, beta, alpha, pt, pn, rt, rn, ft, fn)
    if any(not math.isfinite(v) or v < 0 for v in (pt, pn, rt, rn, ft, fn)):
        raise ValueError('nonfinite or negative force; check physical input scales')
    return r


def dynamic_depths(static_depth_m, pitch_m, kinematic_slope):
    """Finite-entry Eq.44 reconstruction, columns are independent axial tracks.

    The first active grain cuts virgin material at its static depth. Subsequent
    grains are compared with the last dynamic grain, capped by virgin material.
    This boundary rule is explicit reconstruction, absent from the paper.
    """
    d = np.asarray(static_depth_m, dtype=float)
    if d.ndim != 2 or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError('static depths must be a finite nonnegative 2D matrix')
    pitch = number(pitch_m, 'pitch_m', strict=True)
    slope = number(kinematic_slope, 'kinematic_slope')
    depths = np.zeros_like(d); previous = np.full(d.shape, -1, dtype=np.int64)
    for col in range(d.shape[1]):
        prev = -1
        for row in range(d.shape[0]):
            if d[row, col] <= 0:
                continue
            previous[row, col] = prev
            hmax = d[row, col] if prev < 0 else min(d[row, col],
                slope*(row-prev)*pitch+d[row, col]-d[prev, col])
            if hmax > 0:
                depths[row, col] = hmax/2
                prev = row
    return depths, previous


def wheel_population(c):
    mean = (c.grain_max_m+c.grain_min_m)/2
    std = (c.grain_max_m-c.grain_min_m)/6
    vg = 2*(31-c.organization_number)/100
    pitch = mean*math.sqrt(math.pi/(4*vg)); gap = pitch-mean
    length = math.sqrt(c.wheel_diameter_m*c.depth_m)
    nx = math.floor(length/pitch); ny = math.floor(c.width_m/pitch)
    if nx*ny > 200000 or nx*ny*c.vibration_steps > 100000000:
        raise ValueError('requested wheel exceeds bounded reconstruction workload')
    rng = np.random.Generator(np.random.PCG64(c.seed))
    # Eq.33 is the untruncated normal. Report out-of-table-range samples.
    sizes = rng.normal(mean, std, size=(nx, ny))
    if (sizes <= 0).any():
        raise ValueError('untruncated normal generated nonpositive grain size')
    zc = np.zeros((nx, ny))
    for _ in range(c.vibration_steps):
        zc += rng.uniform(-gap, gap, size=(nx, ny))
    protrusion = sizes/2+zc
    static = np.maximum(protrusion-protrusion.max()+c.depth_m, 0.) if protrusion.size else protrusion.copy()
    ag, previous = dynamic_depths(static, pitch, 2*c.feed_speed_m_s/c.wheel_speed_m_s*math.sqrt(c.depth_m/c.wheel_diameter_m))
    if c.feed_speed_m_s == 0:
        ag[:] = 0 # steady grinding model; no stationary indentation response.
    rows = []
    for i in range(nx):
        for j in range(ny):
            rows.append(dict(grain_id=i*ny+j, circumferential_index=i, axial_index=j,
                x_m=(i+.5)*pitch, y_m=(j+.5)*pitch, diameter_m=float(sizes[i,j]),
                protrusion_m=float(protrusion[i,j]), static_depth_m=float(static[i,j]),
                previous_dynamic_row=int(previous[i,j]), depth_m=float(ag[i,j])))
    diag = dict(nx=nx, ny=ny, contact_length_m=length, pitch_m=pitch, gap_m=gap,
        corundum_fraction=vg, mean_diameter_m=mean, diameter_std_m=std,
        protrusion_range_m=float(np.ptp(protrusion)) if protrusion.size else 0.,
        outside_nominal_diameter_range=int(((sizes<c.grain_min_m)|(sizes>c.grain_max_m)).sum()),
        vibration_method='unbounded_sum_uniform_eq40', rng='numpy.PCG64', numpy_version=np.__version__,
        finite_entry_rule='first_static_depth_then_previous_dynamic_capped_by_static',
        integer_rule='floor', normal_distribution='untruncated_eq33')
    return rows, diag


def predict(case):
    if not isinstance(case, Case):
        raise ValueError('predict requires validated Case')
    c = case
    if c.source == 'stochastic_eq40':
        rows, diagnostics = wheel_population(c)
    else:
        rows = [dict(grain_id=i, depth_m=float(d), static_depth_m=float(d)) for i,d in enumerate(c.depths_m)]
        diagnostics = {'source': 'explicit_depths', 'meaning': 'supplied effective grain depths, not wheel reconstruction'}
    counts = dict(sampled=len(rows), static=0, dynamic=0, ploughing=0, transition=0, cutting=0)
    for row in rows:
        counts['static'] += int(row['static_depth_m'] > 0)
        force = grain_force(row['depth_m'], c.yield_stress_Pa, c.fracture_stress_Pa,
                            c.cone_half_angle_rad, c.friction_coefficient, c.formula_convention)
        row.update(force.to_dict())
        if force.stage != 'inactive':
            counts['dynamic'] += 1
            counts[force.stage] += 1
    names = ('plastic_t_N', 'plastic_n_N', 'removal_t_N', 'removal_n_N', 'rake_t_N', 'rake_n_N')
    components = {name: math.fsum(row[name] for row in rows) for name in names}
    components['wear_n_N'] = 4*c.wear_coefficient_N_m*counts['dynamic']*c.feed_speed_m_s/c.wheel_speed_m_s/c.wheel_diameter_m
    components['wear_t_N'] = c.friction_coefficient*components['wear_n_N']
    total_t = math.fsum(v for name,v in components.items() if name.endswith('_t_N'))
    total_n = math.fsum(v for name,v in components.items() if name.endswith('_n_N'))
    data = c.to_dict()
    from . import __version__
    result = dict(result_format=RESULT_FORMAT, kernel_version=__version__, source_doi=DOI,
        normalized_input=data, input_fingerprint=hashlib.sha256(canonical(data).encode('utf-8')).hexdigest(),
        formula_convention=c.formula_convention, counts=counts, components=components,
        total_t_N=total_t, total_n_N=total_n, resultant_N=math.hypot(total_t,total_n),
        grains=rows, wheel_diagnostics=diagnostics,
        reproduction_status='formula_implementation_with_declared_reconstruction_assumptions',
        experimental_validation='not_verified', paper_figure18_replication='not_verified',
        limitations=[
            '440C Table 2 beta fit only; stresses are literature inputs, not a new material calibration.',
            'Eq9/10 and Eq21 are inconsistent in the paper; selected convention is recorded.',
            'K1=18.36 has no published unit; wear_provenance records the explicit user assumption.',
            'Eq40 and dynamic-grain finite-entry boundary rules do not fully specify the author program.',
            'No displacement, FE stress, fractured chips, thermal field or final surface roughness is predicted.',
            'Paper reported 4.19/4.31 percent errors are not this implementation validation results.',
        ])
    canonical(result) # Reject every nonfinite persisted intermediate.
    return result


def calibrate_wear(case, *, measured_normal_N, measured_tangential_N, source):
    """Eq.28/29 inverse estimates; report disagreement rather than silently fit.

    These estimates use one fixed population and one fixed formula convention.
    A calibration point is never counted as independent validation.
    """
    from dataclasses import replace
    fn = number(measured_normal_N, 'measured_normal_N')
    ft = number(measured_tangential_N, 'measured_tangential_N')
    if not isinstance(source, str) or not source.strip():
        raise ValueError('calibration source required')
    baseline = predict(replace(case, wear_coefficient_N_m=0.))
    scale = 4*baseline['counts']['dynamic']*case.feed_speed_m_s/case.wheel_speed_m_s/case.wheel_diameter_m
    if scale <= 0 or case.friction_coefficient <= 0:
        raise ValueError('both-component wear calibration needs nonzero active population, feed and friction')
    kn = (fn-baseline['total_n_N'])/scale
    kt = (ft-baseline['total_t_N'])/(case.friction_coefficient*scale)
    if min(kn,kt) < 0:
        raise ValueError('measured force below non-wear baseline; no nonnegative K1 solution')
    result=dict(normal_estimate_N_m=kn,tangential_estimate_N_m=kt,
        relative_consistency_error=abs(kn-kt)/max(kn,kt) if max(kn,kt)>0 else 0.,
        source=source, input_fingerprint=baseline['input_fingerprint'],
        independent_validation='not_verified',
        selection='Report both estimates; no automatic averaging or parameter replacement.')
    canonical(result)
    return result
