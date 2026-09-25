"""Explicit Schema3 energetic damage route and independent material probes."""
from dataclasses import dataclass
import numpy as np
from grindcae.elastoplastic import update_plane_strain_j2
from .thermodynamic_energy import stored_energy,fracture_resistance,damage_surface


@dataclass(frozen=True)
class EnergeticDamageHistory:
    damage: float
    initial_driving_energy: float
    fracture_energy_density: float

    def __post_init__(self):
        if (not np.isfinite(self.damage) or not 0.<=self.damage<=1.
            or not np.isfinite(self.initial_driving_energy) or self.initial_driving_energy<=0.
            or not np.isfinite(self.fracture_energy_density) or self.fracture_energy_density<=0.):
            raise ValueError('invalid energetic damage history')


@dataclass(frozen=True)
class EnergeticDamageResponse:
    history: EnergeticDamageHistory
    material_update: object
    stress: np.ndarray
    tangent: np.ndarray
    damage_dissipation_density: float


@dataclass(frozen=True)
class ConstrainedEnergeticDamageResponse(EnergeticDamageResponse):
    constraint_residual: float
    constraint_strain_gradient: np.ndarray
    constraint_damage_derivative: float
    stress_damage_derivative: np.ndarray
    onset: object | None = None


def constrained_update(material,committed_material,history,strain_increment,damage,*,active_side_direction=None):
    """Trial joint (strain,D) response; the caller must solve Y-R(D)=0.

    No event or history is committed here. In particular D=1 and zero nominal
    force do not imply the energetic constraint has been satisfied.
    """
    if not np.isfinite(damage) or not history.damage<=damage<=1.:
        raise ValueError('joint trial damage must be irreversible and at most one')
    effective=update_plane_strain_j2(material,committed_material,strain_increment,
                                   active_side_direction=active_side_direction)
    stress=np.asarray(effective.state.stress_tensor_Pa)
    alpha=effective.state.equivalent_plastic_strain
    H=material.internal_hardening_modulus
    energy=stored_energy(stress,alpha,damage,young_modulus=material.E,
                         poisson_ratio=material.nu,hardening_modulus=H)
    curve=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                             fracture_energy_density=history.fracture_energy_density)
    elastic=(1+material.nu)/material.E*stress-material.nu/material.E*np.trace(stress)*np.eye(3)
    gradient=np.array([np.sum(elastic*np.asarray(column)) for column in effective.stress_tensor_tangent_Pa])
    gradient+=H*alpha*np.asarray(effective.equivalent_plastic_strain_gradient)
    nominal=(1-damage)*np.asarray(effective.in_plane_stress_Pa)
    resistance=curve.resistance(damage)
    return ConstrainedEnergeticDamageResponse(
        EnergeticDamageHistory(damage,history.initial_driving_energy,history.fracture_energy_density),
        effective,nominal,(1-damage)*np.asarray(effective.algorithmic_tangent_Pa),
        curve.cumulative(damage),energy['effective_total']-resistance,gradient,
        -curve.derivative(damage),-np.asarray(effective.in_plane_stress_Pa))


def constrained_onset_update(material,committed_material,committed_damage,damage_input,
        strain_increment,damage,*,fracture_energy_density,active_side_direction=None):
    """Joint trial when energetic onset occurs inside the current increment.

    The onset energy remains the existing plastic-trigger event evaluated at
    the current trial. Its chain derivative is included in Y-R(D,Y0)=0; no
    second softening law or fitted event energy is introduced.
    """
    onset=locate_plastic_initiation(material,committed_material,damage_input,
        np.asarray(strain_increment,dtype=float),omega_before=committed_damage.omega)
    if onset is None:
        raise ValueError('joint onset coordinate requires a localized initiation event')
    G=float(fracture_energy_density)
    if not np.isfinite(G) or G<=0.:
        raise ValueError('positive fracture energy density required')
    damage=float(damage)
    if not 0.<=damage<=1.:
        raise ValueError('joint onset damage must be within [0,1]')
    effective=update_plane_strain_j2(material,committed_material,
        np.asarray(strain_increment,dtype=float),active_side_direction=active_side_direction)
    stress=np.asarray(effective.state.stress_tensor_Pa)
    alpha=effective.state.equivalent_plastic_strain
    H=material.internal_hardening_modulus
    energy=stored_energy(stress,alpha,damage,young_modulus=material.E,
        poisson_ratio=material.nu,hardening_modulus=H)
    history=EnergeticDamageHistory(damage,onset['Y0'],G)
    curve=fracture_resistance(initial_driving_energy=onset['Y0'],
        fracture_energy_density=G)
    elastic=(1+material.nu)/material.E*stress-material.nu/material.E*np.trace(stress)*np.eye(3)
    gradient=np.array([np.sum(elastic*np.asarray(column))
        for column in effective.stress_tensor_tangent_Pa])
    gradient+=H*alpha*np.asarray(effective.equivalent_plastic_strain_gradient)
    q=1.-damage+damage*onset['Y0']/G
    dR_dY0=(1.-damage-damage*onset['Y0']/G)/q**3
    constraint_gradient=gradient-dR_dY0*np.asarray(onset['Y0_gradient'])
    nominal=(1.-damage)*np.asarray(effective.in_plane_stress_Pa)
    return ConstrainedEnergeticDamageResponse(
        history,effective,nominal,(1.-damage)*np.asarray(effective.algorithmic_tangent_Pa),
        curve.cumulative(damage),energy['effective_total']-curve.resistance(damage),
        constraint_gradient,-curve.derivative(damage),
        -np.asarray(effective.in_plane_stress_Pa),onset)


def update(material,committed_material,history,strain_increment,*,active_side_direction=None):
    if history.damage==1.:
        from dataclasses import replace
        retained=update_plane_strain_j2(material,committed_material,np.zeros(3))
        retained=replace(retained,state=committed_material)
        return EnergeticDamageResponse(history,retained,np.zeros(3),np.zeros((3,3)),history.fracture_energy_density)
    effective=update_plane_strain_j2(material,committed_material,strain_increment,
                                   active_side_direction=active_side_direction)
    stress=np.asarray(effective.state.stress_tensor_Pa)
    alpha=effective.state.equivalent_plastic_strain
    H=material.internal_hardening_modulus
    energy=stored_energy(stress,alpha,history.damage,young_modulus=material.E,
                         poisson_ratio=material.nu,hardening_modulus=H)
    curve=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                             fracture_energy_density=history.fracture_energy_density)
    elastic=(1+material.nu)/material.E*stress-material.nu/material.E*np.trace(stress)*np.eye(3)
    gradient=np.array([np.sum(elastic*np.asarray(column)) for column in effective.stress_tensor_tangent_Pa])
    gradient+=H*alpha*np.asarray(effective.equivalent_plastic_strain_gradient)
    unloading_side=False
    loading_side=False
    if active_side_direction is not None:
        direction=np.asarray(active_side_direction,dtype=float)
        if direction.shape!=(3,) or not np.all(np.isfinite(direction)):
            raise ValueError('finite plane-strain active-side direction required')
        resistance=curve.resistance(history.damage)
        roundoff=64*np.finfo(float).eps*max(resistance,energy['effective_total'])
        unloading_side=(abs(energy['effective_total']-resistance)<=roundoff
                        and float(gradient@direction)<0.)
        loading_side=(abs(energy['effective_total']-resistance)<=roundoff
                      and float(gradient@direction)>0.)
    if unloading_side:
        from .thermodynamic_energy import DamageSurfaceState
        surface=DamageSurfaceState(history.damage,0.,False)
    elif loading_side and curve.curvature>0 and history.damage<1.:
        from .thermodynamic_energy import DamageSurfaceState
        # Current-side derivative at an already committed surface; roundoff
        # below R must not select an elastic tangent on a loading direction.
        # Keep D and Gamma unchanged at this zero-increment state.
        surface=DamageSurfaceState(history.damage,1./curve.derivative(history.damage),True)
    else:
        surface=damage_surface(curve,energy['effective_total'],committed_damage=history.damage)
    nominal=(1-surface.damage)*np.asarray(effective.in_plane_stress_Pa)
    tangent=(1-surface.damage)*np.asarray(effective.algorithmic_tangent_Pa)-np.outer(
        np.asarray(effective.in_plane_stress_Pa),surface.derivative_per_energy*gradient)
    return EnergeticDamageResponse(EnergeticDamageHistory(surface.damage,history.initial_driving_energy,
        history.fracture_energy_density),effective,nominal,tangent,curve.cumulative(surface.damage))


def constrained_proportional_state(material,committed_material,history,direction,damage):
    """Material-point path prototype: solve Y(a)-R(D)=0 with D prescribed.

    This is not a global equilibrium solver or permission to commit a folded path.
    """
    from scipy.optimize import brentq
    direction=np.asarray(direction,dtype=float)
    if direction.shape!=(3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction)==0:
        raise ValueError('finite nonzero plane-strain direction required')
    if damage<history.damage: raise ValueError('damage cannot decrease')
    curve=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                             fracture_energy_density=history.fracture_energy_density)
    target=curve.resistance(damage)
    def evaluate(amplitude):
        result=update_plane_strain_j2(material,committed_material,amplitude*direction)
        energy=stored_energy(result.state.stress_tensor_Pa,result.state.equivalent_plastic_strain,damage,
            young_modulus=material.E,poisson_ratio=material.nu,hardening_modulus=material.internal_hardening_modulus)
        return result,energy
    high=1e-6
    while evaluate(high)[1]['effective_total']<target:
        high*=2
        if high>1: raise ValueError('no proportional root in small-strain diagnostic range')
    amplitude=brentq(lambda a:evaluate(a)[1]['effective_total']-target,0.,high,xtol=1e-16)
    result,energy=evaluate(amplitude)
    stress=np.asarray(result.state.stress_tensor_Pa)
    elastic=(1+material.nu)/material.E*stress-material.nu/material.E*np.trace(stress)*np.eye(3)
    gradient=np.array([np.sum(elastic*np.asarray(c)) for c in result.stress_tensor_tangent_Pa])
    gradient+=material.internal_hardening_modulus*result.state.equivalent_plastic_strain*np.asarray(result.equivalent_plastic_strain_gradient)
    matrix=np.array([[gradient@direction,-curve.derivative(damage)],[0.,1.]])
    return dict(amplitude=amplitude,damage=damage,constraint_residual=energy['effective_total']-target,
        full_augmented_rank=int(np.linalg.matrix_rank(matrix)),
        nominal_stress=(1-damage)*np.asarray(result.in_plane_stress_Pa),augmented_matrix=matrix)


def locate_plastic_initiation(material,committed_material,damage_input,strain_increment,*,omega_before,
                              _endpoint_response=None):
    """Locate existing plastic trigger before freezing the energetic onset scale."""
    from scipy.optimize import brentq
    from grindcae.elastoplastic.j2 import equivalent_von_mises_stress
    increment=np.asarray(strain_increment,dtype=float)
    if not 0<=omega_before<1: raise ValueError('initiation requires an uninitiated history')
    def evaluate(fraction):
        result=(_endpoint_response if fraction == 1. and _endpoint_response is not None
                else update_plane_strain_j2(material,committed_material,fraction*increment))
        tensor=np.asarray(result.state.stress_tensor_Pa)
        q=equivalent_von_mises_stress(tensor)
        eta=np.trace(tensor)/(3*q) if q>0 else -np.inf
        delta=max(0.,result.state.equivalent_plastic_strain-committed_material.equivalent_plastic_strain)
        omega=omega_before+(delta/damage_input.failure_strain_at(eta) if eta>=damage_input.triaxiality_cutoff else 0.)
        return result,omega
    if evaluate(1.)[1]<1.: return None
    fraction=brentq(lambda f:evaluate(f)[1]-1.,0.,1.,xtol=1e-14)
    result,_=evaluate(fraction)
    energy=stored_energy(result.state.stress_tensor_Pa,result.state.equivalent_plastic_strain,0.,
        young_modulus=material.E,poisson_ratio=material.nu,hardening_modulus=material.internal_hardening_modulus)
    stress=np.asarray(result.state.stress_tensor_Pa)
    columns=np.asarray(result.stress_tensor_tangent_Pa)
    q=equivalent_von_mises_stress(stress)
    deviator=stress-np.trace(stress)*np.eye(3)/3.
    q_gradient=np.array([1.5*np.sum(deviator*column)/q for column in columns])
    mean=np.trace(stress)/3.
    eta=mean/q
    eta_gradient=np.array([np.trace(column)/3. for column in columns])/q-mean*q_gradient/q**2
    failure=damage_input.failure_strain_at(eta)
    slope=0.
    for (left,first),(right,second) in zip(damage_input.failure_strain_table[:-1],damage_input.failure_strain_table[1:]):
        if left<eta<right:
            slope=(second-first)/(right-left)
            break
    alpha=result.state.equivalent_plastic_strain
    delta=alpha-committed_material.equivalent_plastic_strain
    alpha_gradient=np.asarray(result.equivalent_plastic_strain_gradient)
    omega_gradient=alpha_gradient/failure-delta*slope*eta_gradient/failure**2
    denominator=float(omega_gradient@increment)
    if denominator<=0. or not np.isfinite(denominator):
        raise ValueError('initiation event has no regular increasing local crossing')
    fraction_gradient=-fraction*omega_gradient/denominator
    elastic=(1+material.nu)/material.E*stress-material.nu/material.E*np.trace(stress)*np.eye(3)
    Y_gradient=np.array([np.sum(elastic*column) for column in columns])
    Y_gradient+=material.internal_hardening_modulus*alpha*alpha_gradient
    onset_gradient=fraction*Y_gradient+float(Y_gradient@increment)*fraction_gradient
    return dict(fraction=fraction,material_state=result.state,Y0=energy['effective_total'],
                Y0_gradient=onset_gradient,fraction_gradient=fraction_gradient)
