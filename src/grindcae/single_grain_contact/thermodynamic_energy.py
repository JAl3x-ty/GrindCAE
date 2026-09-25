"""Schema3 potential and work identities; no evolution law or residual fitting."""
import math
import numpy as np
from dataclasses import dataclass


def stored_energy(stress,alpha,damage,*,young_modulus,poisson_ratio,hardening_modulus):
    stress=np.asarray(stress,dtype=float)
    if (stress.shape!=(3,3) or not np.all(np.isfinite(stress))
        or not all(math.isfinite(v) for v in (alpha,damage,young_modulus,poisson_ratio,hardening_modulus))
        or alpha<0 or not 0<=damage<=1 or young_modulus<=0 or not -1<poisson_ratio<.5 or hardening_modulus<0):
        raise ValueError('invalid thermodynamic state or elastic constants')
    if not np.allclose(stress,stress.T,rtol=0.,atol=64*np.finfo(float).eps*max(1.,np.max(abs(stress)))):
        raise ValueError('stress must be symmetric')
    elastic_strain=(1+poisson_ratio)/young_modulus*stress-poisson_ratio/young_modulus*np.trace(stress)*np.eye(3)
    elastic=.5*float(np.sum(stress*elastic_strain))
    hardening=.5*hardening_modulus*alpha**2
    return dict(effective_elastic=elastic,effective_hardening=hardening,effective_total=elastic+hardening,
                stored_elastic=(1-damage)*elastic,stored_hardening=(1-damage)*hardening,
                stored_total=(1-damage)*(elastic+hardening),damage_driving_energy=elastic+hardening)


def plastic_work_partition(alpha_before,alpha_after,damage,*,yield_stress,hardening_modulus):
    """Exact fixed-D scalar J2 work, with energetic isotropic hardening."""
    if (not all(math.isfinite(v) for v in (alpha_before,alpha_after,damage,yield_stress,hardening_modulus))
        or alpha_before<0 or alpha_after<alpha_before or not 0<=damage<=1 or yield_stress<=0 or hardening_modulus<0):
        raise ValueError('invalid irreversible plastic increment')
    dissipated=(1-damage)*yield_stress*(alpha_after-alpha_before)
    storage=(1-damage)*.5*hardening_modulus*(alpha_after**2-alpha_before**2)
    return dict(plastic_work=dissipated+storage,hardening_storage_increment=storage,plastic_dissipation=dissipated)


@dataclass(frozen=True)
class FractureResistance:
    initial: float
    curvature: float
    total: float | None = None

    def resistance(self,damage):
        if not math.isfinite(damage) or not 0<=damage<=1:
            raise ValueError('damage outside [0,1]')
        return self.initial/(1.-self.curvature*damage)**2

    def derivative(self,damage):
        return 2.*self.curvature*self.resistance(damage)/(1.-self.curvature*damage)

    def cumulative(self,damage):
        self.resistance(damage)
        # Preserve the prescribed integral endpoint, rather than reconstructing
        # G through 1-(1-Y0/G) and accumulating a signed roundoff remainder.
        if damage==1. and self.total is not None:return self.total
        return self.initial*damage/(1.-self.curvature*damage)


def fracture_resistance(*,initial_driving_energy,fracture_energy_density):
    if not all(math.isfinite(v) and v>0 for v in (initial_driving_energy,fracture_energy_density)):
        raise ValueError('positive driving and fracture energies required')
    curvature=1.-initial_driving_energy/fracture_energy_density
    if not math.isfinite(curvature) or curvature>=1.:
        raise ValueError('fracture energy ratio exceeds representable resistance range')
    return FractureResistance(initial_driving_energy,curvature,fracture_energy_density)


class DamagePathConstraintRequired(ValueError):
    """The active material surface needs joint strain/damage continuation."""


@dataclass(frozen=True)
class DamageSurfaceState:
    damage: float
    derivative_per_energy: float
    active: bool


def damage_surface(curve,driving_energy,*,committed_damage):
    """Local stable return only; decreasing/flat resistance is never a hidden jump."""
    resistance=curve.resistance(committed_damage)
    if not math.isfinite(driving_energy) or driving_energy<0:
        raise ValueError('nonnegative finite damage driving energy required')
    if committed_damage==1. or driving_energy<resistance:
        return DamageSurfaceState(committed_damage,0.,False)
    if curve.curvature<=0:
        raise DamagePathConstraintRequired('flat/decreasing resistance requires joint path constraint')
    damage=min(1.,max(committed_damage,(1.-math.sqrt(curve.initial/driving_energy))/curve.curvature))
    derivative=math.sqrt(curve.initial/driving_energy)/(2.*curve.curvature*driving_energy)
    return DamageSurfaceState(damage,0. if damage==1. else derivative,True)
