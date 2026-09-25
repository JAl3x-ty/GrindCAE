"""Fixed active normal-constraint equations; no history or topology commits.

Multipliers are total nodal normal forces in N, not a material energy source.
This endpoint constraint prototype is not the clipped-edge integration route.
"""
from dataclasses import dataclass
from dataclasses import replace
import numpy as np


@dataclass(frozen=True)
class CircleEdgeGap:
    gap_m: float
    fraction: float
    gradient: np.ndarray
    hessian: np.ndarray


def circle_edge_gap(endpoints,center,radius):
    """Minimum circle gap on a straight P1 edge and current-side derivatives.

    This is geometry only, not an activated contact or a new traction law.
    Interior closest coordinates are statically condensed: H=Hxx-Hxs Hss^-1 Hsx.
    At an endpoint the closest coordinate is frozen on the endpoint side.
    """
    x=np.asarray(endpoints,dtype=float);c=np.asarray(center,dtype=float)
    if (x.shape!=(2,2) or c.shape!=(2,) or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(c)) or not np.isfinite(radius) or radius<=0):
        raise ValueError('invalid circle edge geometry')
    edge=x[1]-x[0];length2=float(edge@edge)
    if length2<=0:raise ValueError('degenerate circle edge')
    raw=float((c-x[0])@edge/length2);s=float(np.clip(raw,0.,1.))
    radial=x[0]+s*edge-c;distance=float(np.linalg.norm(radial))
    if distance<=64*np.finfo(float).eps*radius:
        raise ValueError('nonunique circle edge normal')
    normal=radial/distance
    shape=np.hstack(((1-s)*np.eye(2),s*np.eye(2)))
    curvature=(np.eye(2)-np.outer(normal,normal))/distance
    gradient=shape.T@normal;hessian=shape.T@curvature@shape
    if 0.<raw<1.:
        mixed=np.r_[-normal,normal]+shape.T@curvature@edge
        denominator=float(edge@curvature@edge)
        if denominator<=0:raise ValueError('nonunique closest edge coordinate')
        hessian-=np.outer(mixed,mixed)/denominator
    gradient.setflags(write=False);hessian.setflags(write=False)
    return CircleEdgeGap(distance-radius,s,gradient,hessian)


@dataclass(frozen=True)
class NormalConstraintAssembly:
    force: np.ndarray
    gaps: np.ndarray
    all_gaps: np.ndarray
    force_displacement: np.ndarray
    force_multiplier: np.ndarray
    force_marker: np.ndarray
    gap_displacement: np.ndarray
    gap_marker: np.ndarray


def circle_edge_normal_constraints(coordinates,dofs,edges,displacement,center,radius,active,forces,motion):
    """Explicit frictionless edge-minimum KKT prototype, no history migration.

    Rigid nonpenetration is imposed on an entire straight edge via its minimum
    gap. Its total multiplier generates F*grad(g), including closest-point
    motion in the tangent. Shared endpoint constraints require rank handling
    by the caller; they are never regularized here.
    """
    x=np.asarray(coordinates,dtype=float);ids=np.asarray(dofs,dtype=int)
    edges=np.asarray(edges,dtype=int);u=np.asarray(displacement,dtype=float)
    active=np.asarray(active,dtype=int);forces=np.asarray(forces,dtype=float)
    motion=np.asarray(motion,dtype=float)
    if (x.ndim!=2 or x.shape[1]!=2 or ids.shape!=x.shape or u.ndim!=1
        or edges.ndim!=2 or edges.shape[1]!=2 or np.any(edges<0) or np.any(edges>=len(x))
        or np.any(ids<0) or np.any(ids>=len(u)) or active.ndim!=1 or forces.shape!=active.shape
        or len(set(active))!=len(active) or np.any(active<0) or np.any(active>=len(edges))
        or motion.shape!=(2,) or not all(np.all(np.isfinite(v)) for v in (x,u,forces,motion))):
        raise ValueError('invalid edge normal constraint identities')
    geometry=[circle_edge_gap((x+u[ids])[edge],center,radius) for edge in edges]
    gaps=np.array([g.gap_m for g in geometry]);size=len(u);count=len(active)
    force=np.zeros(size);K=np.zeros((size,size));L=np.zeros((size,count));M=np.zeros(size)
    G=np.zeros((count,size));gm=np.zeros(count);translation=np.tile(motion,2)
    for j,i in enumerate(active):
        local_ids=ids[edges[i]].ravel();g=geometry[i]
        if len(set(local_ids))!=4:raise ValueError('degenerate edge constraint DOFs')
        local=forces[j]*g.hessian
        force[local_ids]+=forces[j]*g.gradient
        K[np.ix_(local_ids,local_ids)]+=local
        L[local_ids,j]=g.gradient;M[local_ids]-=local@translation
        G[j,local_ids]=g.gradient;gm[j]=-g.gradient@translation
    return NormalConstraintAssembly(force,gaps[active],gaps,K,L,M,G,gm)


@dataclass(frozen=True)
class NormalMaterialPathAssembly:
    residual: np.ndarray
    matrix: np.ndarray
    marker_column: np.ndarray
    body: object
    contact: NormalConstraintAssembly
    displacement: np.ndarray
    energy_increment: object = None
    energy_budget: object = None
    contact_events: tuple = ()


@dataclass(frozen=True)
class NormalMaterialEnergyIncrement:
    contact_work_J: float
    material_storage_change_J: float
    plastic_dissipation_J: float
    damage_dissipation_J: float
    penalty_storage_change_J: float
    multiplier_potential_change_J: float

    @property
    def physical_residual_J(self):
        return (self.contact_work_J-self.material_storage_change_J
                -self.plastic_dissipation_J-self.damage_dissipation_J
                -self.penalty_storage_change_J)

    @property
    def algorithmic_residual_J(self):
        return self.physical_residual_J-self.multiplier_potential_change_J


@dataclass(frozen=True)
class IdealEdgeInertialIncrement:
    time_s: float
    grain_increment_m: float
    displacement: np.ndarray
    velocity_m_per_s: np.ndarray
    normal_forces_N: np.ndarray
    body: object
    contact: NormalConstraintAssembly
    residual_norm_N: float
    residual_tolerance_N: float
    correction_norm_m: float
    kinetic_energy_J: float
    external_contact_work_J: float
    material_storage_change_J: float
    plastic_dissipation_J: float
    damage_dissipation_J: float
    energy_residual_J: float
    maximum_energy_scale_J: float


@dataclass(frozen=True)
class _InertialDamageCoordinateAssembly:
    body: object
    constrained_background_ids: tuple[int, ...]
    constrained_element_ids: tuple[int, ...]
    damage_values: np.ndarray
    internal_force_damage_derivatives_N: np.ndarray
    constraint_residuals: np.ndarray
    constraint_displacement_derivatives: np.ndarray
    constraint_damage_derivatives: np.ndarray


def _assemble_inertial_damage_coordinates(
        case,prepared,committed,damage_input,displacement,damage_by_background,*,
        active_side_direction=None,terminal_background_ids=()):
    """Assemble ordinary and joint energetic elements in one fixed topology."""
    from scipy.sparse import coo_matrix
    from .damage_coupling import (
        constrained_damage_onset_element_trial_response,
        constrained_damage_element_trial_response,
        damage_element_trial_response,
    )
    from .solver import _DamageBodyAssembly

    joint_ids=tuple(sorted(int(value) for value in damage_by_background))
    terminal_ids=set(int(value) for value in terminal_background_ids)
    joint_columns={value:index for index,value in enumerate(joint_ids)}
    total=prepared.total_degrees_of_freedom
    displacement=np.asarray(displacement,dtype=float)
    increment=displacement-np.asarray(committed.displacement_vector_m,dtype=float)
    internal=np.zeros(total);columns=np.zeros((total,len(joint_ids)))
    constraints=np.zeros(len(joint_ids));constraint_u=np.zeros((len(joint_ids),total))
    constraint_d=np.zeros((len(joint_ids),len(joint_ids)))
    rows=[];cols=[];values=[];materials=[];damages=[];strains=[];stresses=[];sigma_zz=[]
    branches=[];lengths=[];element_forces=[];element_ids=[]
    for element_id,dofs in enumerate(prepared.structural.element_dofs):
        background=int(committed.background_element_ids[element_id])
        kwargs=dict(
            thickness_m=case.analysis.thickness,
            displacement_increment=increment[dofs],
            active_side_direction=(None if active_side_direction is None
                else np.asarray(active_side_direction,dtype=float)[dofs]),
        )
        if background in joint_columns:
            joint_function=(constrained_damage_element_trial_response
                if getattr(committed.damage_states[element_id],'energetic_history',None) is not None
                else constrained_damage_onset_element_trial_response)
            response=joint_function(case.material,damage_input,
                committed.element_states[element_id],committed.damage_states[element_id],
                prepared.structural.element_B_matrices_per_m[element_id],
                prepared.structural.element_areas_m2[element_id],
                damage=damage_by_background[background],**kwargs)
            column=joint_columns[background]
            columns[dofs,column]+=response.internal_force_damage_derivative_N
            constraints[column]=response.constraint_residual
            constraint_u[column,dofs]=response.constraint_displacement_derivative
            constraint_d[column,column]=response.constraint_damage_derivative
            if background in terminal_ids:
                constraints[column]=float(damage_by_background[background]-1.)
                constraint_u[column,dofs]=0.
                constraint_d[column,column]=1.
            element_ids.append(element_id)
        else:
            response=damage_element_trial_response(
                case.material,damage_input,committed.element_states[element_id],
                committed.damage_states[element_id],
                prepared.structural.element_B_matrices_per_m[element_id],
                prepared.structural.element_areas_m2[element_id],**kwargs)
        update=response.material_update
        internal[dofs]+=response.internal_force_N
        local=np.asarray(response.tangent_stiffness_N_per_m,dtype=float)
        rows.extend(np.repeat(dofs,6).tolist());cols.extend(np.tile(dofs,6).tolist())
        values.extend(local.reshape(-1).tolist())
        materials.append(update.material_update.state);damages.append(update.damage_state)
        strains.append(response.total_strain_increment+
            prepared.structural.element_B_matrices_per_m[element_id]@
            np.asarray(committed.displacement_vector_m)[dofs])
        stresses.append(update.in_plane_stress_Pa);sigma_zz.append(update.sigma_zz_Pa)
        branches.append(update.tangent_branch);lengths.append(response.characteristic_length_m)
        element_forces.append(response.internal_force_N)
    tangent=coo_matrix((values,(rows,cols)),shape=(total,total)).tocsr()
    body=_DamageBodyAssembly(
        internal_force_N=internal,tangent_stiffness_N_per_m=tangent,
        element_states=tuple(materials),damage_states=tuple(damages),
        element_total_strain=np.asarray(strains),element_stress_Pa=np.asarray(stresses),
        element_sigma_zz_Pa=np.asarray(sigma_zz),tangent_branches=tuple(branches),
        finite_difference_branch_crossing_count=0,
        characteristic_lengths_m=np.asarray(lengths),
        element_internal_force_vectors_N=np.asarray(element_forces))
    return _InertialDamageCoordinateAssembly(
        body,joint_ids,tuple(element_ids),
        np.asarray([damage_by_background[value] for value in joint_ids]),columns,
        constraints,constraint_u,constraint_d)


def _inertial_material_energy_increment(case, prepared, committed, body, *,
        evolution='energetic_rational_fracture'):
    """Measure material storage and physical dissipation for the momentum step."""

    from .thermodynamic_energy import stored_energy

    weights = np.asarray(prepared.structural.element_areas_m2) * case.analysis.thickness
    storage = plastic = damage = 0.0
    for weight, old, new, old_damage, new_damage in zip(
        weights,
        committed.element_states,
        body.element_states,
        committed.damage_states,
        body.damage_states,
        strict=True,
    ):
        energetic=(evolution=='energetic_rational_fracture'
            or getattr(new_damage,'energetic_history',None) is not None)
        def energy(material, damage_state):
            return stored_energy(
                material.stress_tensor_Pa,
                material.equivalent_plastic_strain,
                damage_state.damage,
                young_modulus=case.material.E,
                poisson_ratio=case.material.nu,
                hardening_modulus=case.material.internal_hardening_modulus,
            )["stored_total" if energetic else "stored_elastic"]

        storage += weight * (energy(new, new_damage) - energy(old, old_damage))
        delta_alpha = new.equivalent_plastic_strain - old.equivalent_plastic_strain
        if energetic:
            plastic += (weight*case.material.yield_strength*.5
                *(2.-old_damage.damage-new_damage.damage)*delta_alpha)
        else:
            # The retained linear-softening contract counts hardening work
            # with pre-initiation plastic work, then books its softening work
            # in the damage ledger. Do not reinterpret this saved evolution.
            if new_damage.initiation_equivalent_plastic_strain is not None:
                delta_alpha=max(0.,min(new.equivalent_plastic_strain,
                    new_damage.initiation_equivalent_plastic_strain)-old.equivalent_plastic_strain)
            plastic += weight*.5*(old.current_yield_strength_Pa+new.current_yield_strength_Pa)*delta_alpha
        damage += weight * (
            new_damage.damage_dissipation_density_J_per_m3
            - old_damage.damage_dissipation_density_J_per_m3
        )
    return float(storage), float(plastic), float(damage)


def advance_ideal_edge_inertial_step(case,prepared,committed,damage_input,*,
        displacement_velocity_m_per_s,center_m,radius_m,grain_motion_direction,
        grain_speed_m_per_s,time_step_s,active_edges,normal_forces_N,constraint_edges,
        density_kg_per_m3):
    """One fixed-active-set real-mass midpoint step on the ideal edge KKT model.

    This is the saved-instability adapter.  It preserves the contact
    representation used by the snapshot and performs no topology commit.
    """
    from .inertial_transition import assemble_consistent_mass_matrix
    from .solver import _assemble_damage_trial
    from .damage_coupling import damage_element_trial_response
    dt=float(time_step_s);speed=float(grain_speed_m_per_s)
    direction=np.asarray(grain_motion_direction,dtype=float)
    center0=np.asarray(center_m,dtype=float);velocity0=np.asarray(displacement_velocity_m_per_s,dtype=float)
    active=np.asarray(active_edges,dtype=int);forces0=np.asarray(normal_forces_N,dtype=float)
    if (not np.isfinite(dt) or dt<=0 or not np.isfinite(speed) or speed<=0
        or direction.shape!=(2,) or not np.all(np.isfinite(direction))
        or abs(np.linalg.norm(direction)-1.)>1e-12 or center0.shape!=(2,)
        or velocity0.shape!=(prepared.total_degrees_of_freedom,)
        or forces0.shape!=active.shape or np.any(forces0<0)):
        raise ValueError('invalid ideal-edge inertial input')
    free=prepared.structural.free_dofs;fixed=prepared.structural.fixed_dofs
    mass=assemble_consistent_mass_matrix(prepared.structural,
        density_kg_per_m3=density_kg_per_m3,thickness_m=case.analysis.thickness).toarray()
    u0=np.asarray(committed.displacement_vector_m,dtype=float)
    velocity0=velocity0.copy();velocity0[fixed]=0.
    center1=center0+dt*speed*direction
    displacement1=u0+dt*velocity0
    forces=forces0.copy();correction_norm=np.inf;residual_norm=np.inf
    displacement_tolerance=(case.newton.displacement_absolute_tolerance_m
        +case.newton.displacement_relative_tolerance*max(np.linalg.norm(u0[free]),1.))
    force_scale=max(np.linalg.norm(forces0),case.newton.residual_absolute_tolerance_N)
    body0=_assemble_damage_trial(case,prepared,committed,u0,damage_input)
    contact0=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
        prepared.candidate_component_dofs,constraint_edges,u0,center0,radius_m,
        active,forces0,direction)
    committed_damage_by_background=dict(
        zip(committed.background_element_ids,committed.damage_states,strict=True))
    constrained_background_ids=()
    damage_values=np.empty(0);damage_scales=np.empty(0);inactive_damage_ids=set()
    terminal_damage_ids=set()
    constrained_committed_states={}
    for iteration in range(case.newton.maximum_iterations+1):
        side=displacement1-u0
        from .thermodynamic_energy import DamagePathConstraintRequired
        current_ids=set(constrained_background_ids)
        trial_ids=[]
        for element_id,(background,state,dofs) in enumerate(zip(
                committed.background_element_ids,committed.damage_states,
                prepared.structural.element_dofs,strict=True)):
            background=int(background)
            if (state.damage>=1. or background in current_ids
                    or background in inactive_damage_ids):
                continue
            try:
                damage_element_trial_response(
                    case.material,damage_input,committed.element_states[element_id],state,
                    prepared.structural.element_B_matrices_per_m[element_id],
                    prepared.structural.element_areas_m2[element_id],
                    thickness_m=case.analysis.thickness,
                    displacement_increment=(displacement1-u0)[dofs],
                    active_side_direction=side[dofs])
            except DamagePathConstraintRequired:
                trial_ids.append(background)
        if trial_ids:
            previous_damage=dict(zip(constrained_background_ids,damage_values,strict=True))
            constrained_background_ids=tuple(sorted(current_ids|set(trial_ids)))
            for background in trial_ids:
                state=committed_damage_by_background[background]
                if getattr(state,'energetic_history',None) is None:
                    element_id=int(np.flatnonzero(
                        np.asarray(committed.background_element_ids)==background)[0])
                    dofs=prepared.structural.element_dofs[element_id]
                    from .energetic_damage import locate_plastic_initiation, EnergeticDamageHistory
                    onset=locate_plastic_initiation(
                        case.material,committed.element_states[element_id],damage_input,
                        prepared.structural.element_B_matrices_per_m[element_id]@
                        (displacement1-u0)[dofs],omega_before=state.omega)
                    if onset is None:
                        raise RuntimeError(
                            'ideal-edge inertial onset event could not be localized')
                    state=replace(state,omega=1.,damage=0.,status='DAMAGED',
                        initiation_equivalent_plastic_strain=(
                            onset['material_state'].equivalent_plastic_strain),
                        initiation_equivalent_stress_Pa=case.material.yield_strength,
                        energetic_history=EnergeticDamageHistory(
                            0.,onset['Y0'],damage_input.fracture_energy_J_per_m2/
                            np.sqrt(4.*prepared.structural.element_areas_m2[element_id]/np.pi)))
                constrained_committed_states[background]=state
            damage_values=np.asarray([
                previous_damage.get(
                    background,
                    constrained_committed_states.get(
                        background,committed_damage_by_background[background]).damage)
                for background in constrained_background_ids],dtype=float)
            damage_scales=np.asarray([
                constrained_committed_states.get(
                    background,committed_damage_by_background[background]
                ).energetic_history.initial_driving_energy
                for background in constrained_background_ids],dtype=float)
        for _activation_pass in range(len(committed.damage_states)+1):
            try:
                if constrained_background_ids:
                    assembled=_assemble_inertial_damage_coordinates(
                        case,prepared,
                        (committed if not constrained_committed_states else replace(
                            committed,damage_states=tuple(
                                constrained_committed_states.get(int(background),state)
                                for background,state in zip(
                                    committed.background_element_ids,committed.damage_states,strict=True)))),
                        damage_input,displacement1,
                        dict(zip(constrained_background_ids,damage_values,strict=True)),
                        active_side_direction=side,
                        terminal_background_ids=terminal_damage_ids)
                    body=assembled.body
                else:
                    assembled=None
                    body=_assemble_damage_trial(case,prepared,committed,displacement1,damage_input,
                                                active_side_direction=side)
                break
            except DamagePathConstraintRequired as activation_error:
                discovered=[]
                existing=set(constrained_background_ids)
                trial_state=(committed if not constrained_committed_states else replace(
                    committed,damage_states=tuple(
                        constrained_committed_states.get(int(background),state)
                        for background,state in zip(
                            committed.background_element_ids,committed.damage_states,strict=True))))
                for element_id,(background,state,dofs) in enumerate(zip(
                        trial_state.background_element_ids,trial_state.damage_states,
                        prepared.structural.element_dofs,strict=True)):
                    background=int(background)
                    if state.damage>=1. or background in existing:
                        continue
                    try:
                        damage_element_trial_response(
                            case.material,damage_input,committed.element_states[element_id],state,
                            prepared.structural.element_B_matrices_per_m[element_id],
                            prepared.structural.element_areas_m2[element_id],
                            thickness_m=case.analysis.thickness,
                            displacement_increment=(displacement1-u0)[dofs],
                            active_side_direction=side[dofs])
                    except DamagePathConstraintRequired:
                        discovered.append(background)
                if not discovered and constrained_background_ids:
                    # A formerly joint coordinate can return to its admissible
                    # frozen side after the displacement correction.  Drop it
                    # from the joint set and reassemble the ordinary branch.
                    inactive_damage_ids.update(constrained_background_ids)
                    constrained_background_ids=()
                    constrained_committed_states={}
                    damage_values=np.empty(0);damage_scales=np.empty(0)
                    terminal_damage_ids.clear()
                    continue
                if not discovered:
                    raise RuntimeError(
                        "ideal-edge inertial step requires constrained damage coordinates"
                    ) from activation_error
                previous=dict(zip(constrained_background_ids,damage_values,strict=True))
                constrained_background_ids=tuple(sorted(
                    set(constrained_background_ids)|set(discovered)))
                for background in discovered:
                    state=committed_damage_by_background[background]
                    if getattr(state,'energetic_history',None) is None:
                        element_id=int(np.flatnonzero(
                            np.asarray(committed.background_element_ids)==background)[0])
                        dofs=prepared.structural.element_dofs[element_id]
                        from .energetic_damage import locate_plastic_initiation, EnergeticDamageHistory
                        onset=locate_plastic_initiation(
                            case.material,committed.element_states[element_id],damage_input,
                            prepared.structural.element_B_matrices_per_m[element_id]@
                            (displacement1-u0)[dofs],omega_before=state.omega)
                        if onset is None:
                            raise RuntimeError(
                                'ideal-edge inertial onset event could not be localized')
                        state=replace(state,omega=1.,damage=0.,status='DAMAGED',
                            initiation_equivalent_plastic_strain=(
                                onset['material_state'].equivalent_plastic_strain),
                            initiation_equivalent_stress_Pa=case.material.yield_strength,
                            energetic_history=EnergeticDamageHistory(
                                0.,onset['Y0'],damage_input.fracture_energy_J_per_m2/
                                np.sqrt(4.*prepared.structural.element_areas_m2[element_id]/np.pi)))
                    constrained_committed_states[background]=state
                damage_values=np.asarray([
                    previous.get(background,
                        constrained_committed_states.get(
                            background,committed_damage_by_background[background]).damage)
                    for background in constrained_background_ids],dtype=float)
                damage_scales=np.asarray([
                    constrained_committed_states.get(
                        background,committed_damage_by_background[background]
                    ).energetic_history.initial_driving_energy
                    for background in constrained_background_ids],dtype=float)
        else:
            raise RuntimeError(
                "ideal-edge inertial step requires constrained damage coordinates"
            )
        contact=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
            prepared.candidate_component_dofs,constraint_edges,displacement1,center1,radius_m,
            active,forces,direction)
        # Closest points can meet at a shared endpoint during Newton, even when
        # the accepted state's rows were independent. Eliminate only identical
        # algebraic rows at this iterate; keep all background edge identities so
        # they can separate again at the next geometry evaluation.
        independent_contact = []
        for local in range(len(active)):
            duplicate = next((other for other in independent_contact
                if contact.gaps[local] == contact.gaps[other]
                and np.array_equal(contact.gap_displacement[local],
                                   contact.gap_displacement[other])), None)
            if duplicate is None:
                independent_contact.append(local)
            else:
                forces[duplicate] += forces[local]
                forces[local] = 0.0
        if len(independent_contact) != len(active):
            contact=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
                prepared.candidate_component_dofs,constraint_edges,displacement1,center1,radius_m,
                active,forces,direction)
        velocity1=2.*(displacement1-u0)/dt-velocity0
        displacement_increment = displacement1 - u0
        average_internal = 0.5 * (body0.internal_force_N + body.internal_force_N)
        # The registered trapezoidal momentum equation uses constitutive nodal
        # forces. Measure its energy error after convergence; do not feed that
        # measured error back as an additional force.
        momentum=(mass@(velocity1-velocity0)/dt
            +average_internal-.5*(contact0.force+contact.force))[free]
        residual_norm=float(np.linalg.norm(momentum))
        gap_norm=float(np.linalg.norm(contact.gaps))
        damage_norm=(0. if assembled is None else float(np.linalg.norm(
            assembled.constraint_residuals/damage_scales)))
        residual_tolerance=(case.newton.residual_absolute_tolerance_N
            +case.newton.residual_relative_tolerance*force_scale)
        if (residual_norm<=residual_tolerance and gap_norm<=1e-15
            and damage_norm<=1e-10
            and correction_norm<=displacement_tolerance):break
        if iteration>=case.newton.maximum_iterations:
            raise RuntimeError('ideal-edge inertial midpoint did not converge')
        tangent=(2.*mass/dt**2+.5*(body.tangent_stiffness_N_per_m.toarray()
                 -contact.force_displacement))
        if assembled is None:
            damage_columns=np.zeros((len(free),0));damage_u=np.zeros((0,len(free)))
            damage_d=np.zeros((0,0));damage_residual=np.empty(0)
        else:
            damage_columns=.5*assembled.internal_force_damage_derivatives_N[free]
            damage_u=assembled.constraint_displacement_derivatives[:,free]
            damage_d=assembled.constraint_damage_derivatives
            damage_residual=assembled.constraint_residuals/damage_scales
        matrix=np.block([
            [tangent[np.ix_(free,free)],-.5*contact.force_multiplier[free],damage_columns],
            [contact.gap_displacement[:,free],np.zeros((len(active),len(active))),
             np.zeros((len(active),len(damage_values)))],
            [damage_u,np.zeros((len(damage_values),len(active))),damage_d]])
        values=np.r_[momentum/force_scale,contact.gaps/max(dt*speed,1e-15),damage_residual]
        scaled=matrix.copy();scaled[:len(free)]/=force_scale
        scaled[len(free):len(free)+len(active)]/=max(dt*speed,1e-15)
        if len(damage_values):
            scaled[len(free)+len(active):]/=damage_scales[:,None]
        independent = np.r_[np.arange(len(free)),
            len(free) + np.asarray(independent_contact, dtype=int),
            np.arange(len(free) + len(active), len(values))]
        try:
            correction = np.zeros(len(values))
            correction[independent] = np.linalg.solve(
                scaled[np.ix_(independent, independent)], -values[independent])
        except np.linalg.LinAlgError as exc:
            raise RuntimeError('ideal-edge inertial KKT system is singular') from exc
        if not np.all(np.isfinite(correction)):
            raise RuntimeError('ideal-edge inertial correction is non-finite')
        displacement1=displacement1.copy();displacement1[free]+=correction[:len(free)];displacement1[fixed]=u0[fixed]
        forces=forces+correction[len(free):len(free)+len(active)]
        negative=np.flatnonzero(forces<-case.newton.residual_absolute_tolerance_N)
        if negative.size:
            identities=','.join(str(int(active[index])) for index in negative)
            raise RuntimeError(f'ideal-edge inertial step requires contact release: {identities}')
        forces=np.maximum(forces,0.)
        if len(damage_values):
            minimum=np.asarray([
                constrained_committed_states.get(
                    background,committed_damage_by_background[background]).damage
                for background in constrained_background_ids])
            candidate=damage_values+correction[len(free)+len(active):]
            above=np.flatnonzero(candidate>1.)
            if above.size:
                terminal_damage_ids.update(
                    constrained_background_ids[index] for index in above)
                candidate[above]=1.
            if np.any(candidate+64*np.finfo(float).eps<minimum):
                falling=set(
                    constrained_background_ids[index]
                    for index in np.flatnonzero(
                        candidate+64*np.finfo(float).eps<minimum))
                inactive_damage_ids.update(falling)
                retained=tuple(
                    background for background in constrained_background_ids
                    if background not in falling)
                previous=dict(zip(constrained_background_ids,damage_values,strict=True))
                constrained_background_ids=retained
                damage_values=np.asarray(
                    [previous[background] for background in retained],dtype=float)
                damage_scales=np.asarray([
                    constrained_committed_states.get(
                        background,committed_damage_by_background[background]
                    ).energetic_history.initial_driving_energy
                    for background in retained],dtype=float)
                terminal_damage_ids.intersection_update(retained)
            else:
                damage_values=np.clip(candidate,minimum,1.)
        correction_norm=float(np.linalg.norm(correction[:len(free)]))
    u1=displacement1;velocity1=2.*(u1-u0)/dt-velocity0;velocity1[fixed]=0.
    final_body=body
    final_contact=contact
    storage,plastic,damage=_inertial_material_energy_increment(
        case,prepared,committed,final_body)
    kinetic0=.5*float(velocity0@mass@velocity0);kinetic1=.5*float(velocity1@mass@velocity1)
    work=float(.5*(contact0.force+final_contact.force)@(u1-u0));residual=(kinetic1-kinetic0)+storage+plastic+damage-work
    scale=max(abs(kinetic1-kinetic0),abs(storage),abs(plastic),abs(damage),abs(work),np.finfo(float).tiny)
    return IdealEdgeInertialIncrement(dt,dt*speed,u1,velocity1,forces,final_body,final_contact,
        residual_norm,residual_tolerance,correction_norm,kinetic1,work,storage,plastic,damage,residual,scale)


def measure_normal_material_increment(case, prepared, before, after, contact_before,
        contact_after, *, grain_increment_m, accepted_multipliers_N=None,
        penalty_stiffness_N_per_m=None):
    """Actual energetic-material/normal-contact increment, no acceptance or fill.

    Trapezoid work and plastic heat retain their integration errors. The AL
    potential is reported separately and never spends the physical budget.
    Both contacts must use identical active identities and fixed AL multipliers.
    """
    from .thermodynamic_energy import stored_energy
    dx=np.asarray(grain_increment_m,dtype=float)
    weights=np.asarray(prepared.structural.element_areas_m2)*case.analysis.thickness
    if dx.shape!=(2,) or not np.all(np.isfinite(dx)):
        raise ValueError('invalid grain energy increment')
    storage=plastic=damage=0.
    for weight,old,new,od,nd in zip(weights,before.element_states,after.element_states,
            before.damage_states,after.damage_states,strict=True):
        da=new.equivalent_plastic_strain-old.equivalent_plastic_strain
        dg=nd.damage_dissipation_density_J_per_m3-od.damage_dissipation_density_J_per_m3
        if da<0 or nd.damage<od.damage or dg<0:
            raise ValueError('joint energy requires irreversible material history')
        def energy(m,d):
            return stored_energy(m.stress_tensor_Pa,m.equivalent_plastic_strain,d.damage,
                young_modulus=case.material.E,poisson_ratio=case.material.nu,
                hardening_modulus=case.material.internal_hardening_modulus)['stored_total']
        storage+=weight*(energy(new,nd)-energy(old,od))
        plastic+=weight*case.material.yield_strength*.5*(2.-od.damage-nd.damage)*da
        damage+=weight*dg
    force0=np.asarray(contact_before.force,dtype=float)
    force1=np.asarray(contact_after.force,dtype=float)
    if force0.ndim!=1 or force0.shape!=force1.shape or force0.size%2:
        raise ValueError('inconsistent contact force coordinates')
    work=float(.5*(force0+force1).reshape(-1,2).sum(axis=0)@dx)
    penalty=multiplier=0.
    if (accepted_multipliers_N is None)!=(penalty_stiffness_N_per_m is None):
        raise ValueError('AL multiplier and penalty arrays must be supplied together')
    if accepted_multipliers_N is not None:
        g0=np.asarray(contact_before.gaps);g1=np.asarray(contact_after.gaps)
        lam=np.asarray(accepted_multipliers_N);k=np.asarray(penalty_stiffness_N_per_m)
        active_augmented_constraint(g0,lam-k*g0,lam,k)
        active_augmented_constraint(g1,lam-k*g1,lam,k)
        penalty=float(.5*np.sum(k*(g1*g1-g0*g0)))
        multiplier=float(-lam@(g1-g0))
    values=(work,storage,plastic,damage,penalty,multiplier)
    if not all(np.isfinite(value) for value in values):
        raise ValueError('nonfinite joint energy measurement')
    return NormalMaterialEnergyIncrement(*map(float,values))


def assemble_normal_material_path(case,prepared,committed,damage_input,state,distance,*,
        center,radius,motion,active,force_scale_N_per_m,accepted_multipliers_N=None,
        penalty_stiffness_N_per_m=None,active_side_direction=None,constraint_edges=None):
    """Existing material assembly coupled to independent endpoint normal forces.

    This explicit prototype never enters v1/v2 dispatch. Ideal and fixed-AL
    equations are deliberately explicit; neither changes accepted histories.
    """
    from .solver import _assemble_damage_trial
    if not np.isfinite(force_scale_N_per_m) or force_scale_N_per_m<=0:
        raise ValueError('positive N/m force coordinate scale required')
    free=prepared.structural.free_dofs;n=len(free);active=np.asarray(active,dtype=int)
    state=np.asarray(state,dtype=float)
    if state.shape!=(n+len(active),) or not np.all(np.isfinite(state)) or not np.isfinite(distance):
        raise ValueError('invalid normal material path coordinate')
    if (accepted_multipliers_N is None)!=(penalty_stiffness_N_per_m is None):
        raise ValueError('AL multiplier and penalty arrays must be supplied together')
    displacement=np.asarray(committed.displacement_vector_m).copy()
    displacement[free]=state[:n]
    body=_assemble_damage_trial(case,prepared,committed,displacement,damage_input,
                                active_side_direction=active_side_direction)
    if constraint_edges is None:
        contact=circle_normal_constraints(prepared.candidate_reference_coordinates_m,
            prepared.candidate_component_dofs,displacement,np.asarray(center)+distance*np.asarray(motion),
            radius,active,state[n:]*force_scale_N_per_m,motion)
    else:
        if accepted_multipliers_N is not None:
            raise ValueError('endpoint AL history cannot be mapped implicitly to edge constraints')
        contact=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
            prepared.candidate_component_dofs,constraint_edges,displacement,
            np.asarray(center)+distance*np.asarray(motion),radius,active,state[n:]*force_scale_N_per_m,motion)
    if accepted_multipliers_N is None:
        constraint=contact.gaps;derivative=np.zeros(len(active))
    else:
        constraint,derivative,_=active_augmented_constraint(contact.gaps,state[n:]*force_scale_N_per_m,
            accepted_multipliers_N,penalty_stiffness_N_per_m)
    tangent=contact.force_displacement-body.tangent_stiffness_N_per_m.toarray()
    matrix=np.block([[tangent[np.ix_(free,free)],contact.force_multiplier[free]*force_scale_N_per_m],
        [contact.gap_displacement[:,free],np.diag(derivative*force_scale_N_per_m)]])
    return NormalMaterialPathAssembly(np.r_[(contact.force-body.internal_force_N)[free],constraint],
        matrix,np.r_[contact.force_marker[free],contact.gap_marker],body,contact,displacement)


def trace_normal_material_target(case,prepared,committed,damage_input,*,center,radius,motion,
        active,normal_forces_N,history_direction,target_distance_m,arc_length_m,
        force_tolerance_N,gap_tolerance_m,path_tolerance_m,correction_tolerance_m,maximum_steps,
        accepted_multipliers_N=None,penalty_stiffness_N_per_m=None,
        energy_budget=None,cumulative_work_J=0.,constraint_edges=None):
    """Connected fixed-topology trial path with the existing physical budget.

    No topology/material-status changes. Contact-set changes and folds return
    explicit failures with retained converged trial prefix, never silent skips.
    """
    from dataclasses import replace
    from .path_continuation import PathContinuationError
    from .workflow import EventEnergyBudget
    from .solver import ContactSolverError
    if (not np.isfinite(target_distance_m) or target_distance_m<=0
        or not np.isfinite(arc_length_m) or arc_length_m<=0
        or type(maximum_steps) is not int or maximum_steps<1):
        raise ValueError('invalid connected path target or budget')
    free=prepared.structural.free_dofs;n=len(free);scale=case.material.E*case.analysis.thickness
    current=committed;distance=0.;x=np.r_[committed.displacement_vector_m[free],np.asarray(normal_forces_N)/scale]
    reference=np.asarray(history_direction,dtype=float);prefix=[]
    budget=EventEnergyBudget() if energy_budget is None else energy_budget
    work=float(cumulative_work_J)
    if not np.isfinite(work):raise ValueError('nonfinite initial path work')
    tolerances=dict(force_rows=n,force_tolerance_N=force_tolerance_N,gap_tolerance_m=gap_tolerance_m,
        path_tolerance_m=path_tolerance_m,correction_tolerance_m=correction_tolerance_m,
        maximum_corrections=case.newton.maximum_iterations)
    def evaluate(state,marker,direction):
        side=np.zeros(prepared.total_degrees_of_freedom);side[free]=direction[:n]
        return assemble_normal_material_path(case,prepared,current,damage_input,state,marker,
            center=center,radius=radius,motion=motion,active=active,force_scale_N_per_m=scale,
            accepted_multipliers_N=accepted_multipliers_N,penalty_stiffness_N_per_m=penalty_stiffness_N_per_m,
            active_side_direction=side,constraint_edges=constraint_edges)
    stage='entry_assembly';point=None
    try:
        previous_contact=evaluate(x,distance,reference).contact
        for _ in range(maximum_steps):
            stage='current_side_initialization'
            def matrix(direction):
                value=evaluate(x,distance,direction)
                return np.column_stack([value.matrix,value.marker_column])
            def admissible(direction):
                if direction[-1]<=0.:return False
                v=evaluate(x,distance,direction)
                for j,identity in enumerate(active):
                    if abs(x[n+j])<=gap_tolerance_m and direction[n+j]<0.:return False
                if constraint_edges is not None:
                    inactive=np.setdiff1d(np.arange(len(constraint_edges)),active)
                    for identity in inactive:
                        if abs(v.contact.all_gaps[identity])>gap_tolerance_m:continue
                        g=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
                            prepared.candidate_component_dofs,constraint_edges,v.displacement,
                            np.asarray(center)+distance*np.asarray(motion),radius,np.array([identity]),np.zeros(1),motion)
                        if float(g.gap_displacement[0,free]@direction[:n]+g.gap_marker[0]*direction[-1])<0.:return False
                return True
            tangent=initialize_joint_normal_direction(matrix,reference,maximum_side_states=2*prepared.element_count+1,
                                                       primal_coordinate_count=n,direction_is_admissible=admissible)
            if tangent[-1]<=0:
                stage='marker_fold'
                raise PathContinuationError('connected path reached a marker fold')
            def equations(state,marker):
                value=evaluate(state,marker,tangent)
                return value.residual,value.matrix,value.marker_column
            predicted=np.r_[x,distance]+arc_length_m*tangent
            stage='joint_correction'
            point=correct_joint_normal_path(predicted,tangent,equations,**tolerances)
            if point.marker<=distance:
                stage='marker_fold'
                raise PathContinuationError('connected path correction folded marker')
            if point.marker>=target_distance_m:
                fraction=(target_distance_m-distance)/(point.marker-distance)
                target_seed=np.r_[x+fraction*(point.state-x),target_distance_m]
                target_normal=np.zeros_like(target_seed);target_normal[-1]=1.
                point=correct_joint_normal_path(target_seed,target_normal,equations,**tolerances)
            value=evaluate(point.state,point.marker,tangent)
            stage='contact_material_qualification'
            inactive=np.setdiff1d(np.arange(len(value.contact.all_gaps)),active)
            events=()
            if (np.any(point.state[n:]<0) or np.any(value.contact.all_gaps[inactive]<-gap_tolerance_m)):
                if constraint_edges is None:
                    raise PathContinuationError('endpoint contact event requires explicit geometry adapter')
                stage='contact_event_localization'
                candidates=[]
                changes=[('release',int(i)) for i in np.asarray(active)[point.state[n:]<0]]
                changes += [('activation',int(i)) for i in inactive[value.contact.all_gaps[inactive]<-gap_tolerance_m]]
                for kind,identity in changes:
                    def event(state,marker):
                        if kind=='release':
                            column=n+list(active).index(identity);row=np.zeros(len(state));row[column]=1.
                            return state[column],row,0.
                        v=evaluate(state,marker,tangent)
                        g=circle_edge_normal_constraints(prepared.candidate_reference_coordinates_m,
                            prepared.candidate_component_dofs,constraint_edges,v.displacement,
                            np.asarray(center)+marker*np.asarray(motion),radius,np.array([identity]),np.zeros(1),motion)
                        return g.gaps[0],np.r_[g.gap_displacement[0,free],np.zeros(len(active))],g.gap_marker[0]
                    located=locate_normal_contact_section(np.r_[point.state,point.marker],
                        lambda state,marker:evaluate(state,marker,tangent),event,**tolerances)
                    marker=located.state[-1]
                    if not distance-path_tolerance_m<=marker<=point.marker+path_tolerance_m:
                        raise PathContinuationError('contact section lies outside connected bracket')
                    candidates.append((marker,kind,identity,located))
                candidates.sort(key=lambda item:item[0])
                marker,kind,identity,located=candidates[0]
                from .path_continuation import PathState
                point=PathState(located.state[:-1],float(marker),tangent,located.residual_norm,
                    0.,located.correction_count,located.correction_norm)
                value=evaluate(point.state,point.marker,tangent)
                events=tuple((k,i) for m,k,i,_ in candidates if abs(m-marker)<=path_tolerance_m)
            if any(new.equivalent_plastic_strain<old.equivalent_plastic_strain for new,old in zip(value.body.element_states,current.element_states)):
                raise PathContinuationError('connected path plastic history decreased')
            if any(new.damage<old.damage for new,old in zip(value.body.damage_states,current.damage_states)):
                raise PathContinuationError('connected path damage history decreased')
            stage='accepted_prefix_energy'
            increment=measure_normal_material_increment(case,prepared,current,value.body,
                previous_contact,value.contact,grain_increment_m=(point.marker-distance)*np.asarray(motion),
                accepted_multipliers_N=accepted_multipliers_N,
                penalty_stiffness_N_per_m=penalty_stiffness_N_per_m)
            try:
                next_budget=budget.accept(work+increment.contact_work_J,
                                          increment.physical_residual_J,event=bool(events))
            except ContactSolverError as cause:
                error=PathContinuationError('connected normal path physical energy budget exceeded')
                error.energy_increment=increment
                error.failure_stage='accepted_prefix_energy'
                raise error from cause
            budget=next_budget;work+=increment.contact_work_J
            value=replace(value,energy_increment=increment,energy_budget=budget,contact_events=events)
            prefix.append((point,value))
            if abs(point.marker-target_distance_m)<=path_tolerance_m:return tuple(prefix)
            reference=point.tangent;x=point.state;distance=point.marker
            previous_contact=value.contact
            if events:
                forces=dict(zip(active,x[n:],strict=True))
                for kind,identity in events:
                    if kind=='release':forces.pop(identity,None)
                    else:forces[identity]=0.
                active=np.array(sorted(forces),dtype=int)
                x=np.r_[x[:n],[forces[i] for i in active]]
                reference=np.r_[reference[:n],np.zeros(len(active)),reference[-1]]
            current=replace(current,displacement_vector_m=value.displacement,element_states=value.body.element_states,
                            damage_states=value.body.damage_states,separating_states=value.body.separating_states)
        raise PathContinuationError('connected normal path exceeded maximum_steps')
    except PathContinuationError as error:
        error.converged_trial_prefix=tuple(prefix)
        error.failure_stage=stage
        error.last_converged_trial=point
        raise


def active_augmented_constraint(gaps,forces,accepted_multipliers,penalty_stiffness):
    """Active AL equation in total-force units; multiplier potential is numerical.

    Eliminating F from g+(F-Lambda)/kA=0 recovers F=Lambda-kA*g.
    The fixed-Lambda potential is only an algorithmic energy audit. It must
    never be added to material dissipation or treated as an external source.
    Negative trial F is permitted here; active-set acceptance must reject it.
    """
    g,F,lam,k=(np.asarray(value,dtype=float) for value in
               (gaps,forces,accepted_multipliers,penalty_stiffness))
    if (g.ndim!=1 or any(v.shape!=g.shape for v in (F,lam,k))
        or not all(np.all(np.isfinite(v)) for v in (g,F,lam,k))
        or np.any(k<=0) or np.any(lam<0)):
        raise ValueError('invalid active augmented normal constraint')
    return g+(F-lam)/k,1./k,-lam*g+.5*k*g*g


def correct_normal_active_set_target(seed,active,assembly,*,constraint_count,force_rows,
        force_tolerance_N,gap_tolerance_m,path_tolerance_m,correction_tolerance_m,maximum_corrections):
    """Bounded frictionless KKT side updates; every assembly freezes history.

    Negative multipliers and penetrating inactive constraints route the
    algorithm only. No material event or state is committed by this routine.
    """
    from .path_continuation import PathContinuationError
    x=np.asarray(seed,dtype=float).copy();active=np.asarray(active,dtype=int).copy()
    if (type(constraint_count) is not int or constraint_count<0
        or x.shape!=(force_rows+len(active)+1,) or not np.all(np.isfinite(x))
        or len(set(active))!=len(active) or np.any(active<0) or np.any(active>=constraint_count)):
        raise ValueError('invalid normal active-set seed')
    seen=set();last_failure=None
    for _ in range(2*constraint_count+1):
        value=assembly(x[:-1],x[-1],active)
        inactive=np.setdiff1d(np.arange(constraint_count),active)
        negative=active[x[force_rows:-1]<0.]
        penetrating=inactive[value.contact.all_gaps[inactive]<-gap_tolerance_m]
        updated=np.union1d(np.setdiff1d(active,negative),penetrating).astype(int)
        if not np.array_equal(updated,active):
            forces=dict(zip(active,x[force_rows:-1],strict=True))
            x=np.r_[x[:force_rows],[max(0.,forces.get(i,0.)) for i in updated],x[-1]]
            active=updated
        value=assembly(x[:-1],x[-1],active)
        unique=[];multipliers=[]
        for j in range(len(active)):
            duplicate=next((k for k,index in enumerate(unique)
                if np.array_equal(value.contact.gap_displacement[j],value.contact.gap_displacement[index])
                and value.contact.gaps[j]==value.contact.gaps[index]),None)
            if duplicate is None:
                unique.append(j);multipliers.append(x[force_rows+j])
            else:
                multipliers[duplicate]+=x[force_rows+j]
        if len(unique)!=len(active):
            active=active[unique]
            x=np.r_[x[:force_rows],multipliers,x[-1]]
        key=tuple(active)
        if key in seen:
            raise PathContinuationError('normal complementarity active-set cycle') from last_failure
        seen.add(key)
        normal=np.zeros_like(x);normal[-1]=1.
        def equations(state,marker):
            v=assembly(state,marker,active)
            return v.residual,v.matrix,v.marker_column
        try:
            point=correct_joint_normal_path(x,normal,equations,force_rows=force_rows,
                force_tolerance_N=force_tolerance_N,gap_tolerance_m=gap_tolerance_m,
                path_tolerance_m=path_tolerance_m,correction_tolerance_m=correction_tolerance_m,
                maximum_corrections=maximum_corrections)
        except PathContinuationError as error:
            error.constraint_active_ids=tuple(map(int,active))
            diagnostic=getattr(error,'correction_diagnostics',None)
            if diagnostic is None:raise
            trial=np.asarray(diagnostic['combined'])
            v=assembly(trial[:-1],trial[-1],active)
            inactive=np.setdiff1d(np.arange(constraint_count),active)
            if not (np.any(trial[force_rows:-1]<0.)
                    or np.any(v.contact.all_gaps[inactive]<-gap_tolerance_m)):
                raise
            # Fixed marker stays exactly the caller's target, even if a
            # failed corrector carries a tiny constraint roundoff.
            x=np.r_[trial[:-1],x[-1]];last_failure=error
            continue
        value=assembly(point.state,point.marker,active)
        inactive=np.setdiff1d(np.arange(constraint_count),active)
        if np.all(point.state[force_rows:]>=0.) and np.all(value.contact.all_gaps[inactive]>=-gap_tolerance_m):
            return point,value,tuple(map(int,active))
        x=np.r_[point.state,x[-1]]
    raise PathContinuationError('normal complementarity side budget exhausted') from last_failure


def initialize_joint_normal_direction(matrix_for_direction,history_direction,*,maximum_side_states,
                                      primal_coordinate_count=None,direction_is_admissible=None):
    """Close one-dimensional nullspace against current sides and accepted history.

    Both signs are tested; marker sign alone never selects a material side.
    Repeated matrices are not iterated indefinitely. A failure means this
    bounded closure found no qualified direction, not that all paths fail.
    """
    from .path_continuation import PathContinuationError
    reference=np.asarray(history_direction,dtype=float)
    if (reference.ndim!=1 or len(reference)<2 or not np.all(np.isfinite(reference))
        or np.linalg.norm(reference)==0 or type(maximum_side_states) is not int or maximum_side_states<1):
        raise ValueError('invalid joint history direction or side budget')
    if primal_coordinate_count is not None:
        if type(primal_coordinate_count) is not int or not 0<primal_coordinate_count<len(reference):
            raise ValueError('invalid primal coordinate count')
        reference=reference.copy()
        reference[primal_coordinate_count:-1]=0.
        if np.linalg.norm(reference)==0:
            raise PathContinuationError('joint history has no primal orientation')
    reference=reference/np.linalg.norm(reference);pending=[reference,-reference];seen=set();valid=[]
    while pending and len(seen)<maximum_side_states:
        side=pending.pop(0);matrix=np.asarray(matrix_for_direction(side),dtype=float)
        if matrix.shape!=(len(reference)-1,len(reference)) or not np.all(np.isfinite(matrix)):
            raise PathContinuationError('invalid joint direction matrix')
        key=matrix.tobytes()
        if key in seen:continue
        seen.add(key)
        scale=np.max(np.abs(matrix),axis=1)
        if np.any(scale==0):raise PathContinuationError('joint direction has zero equilibrium row')
        scaled=matrix/scale[:,None]
        if np.linalg.matrix_rank(scaled)!=len(reference)-1:
            raise PathContinuationError('joint direction nullspace is not one-dimensional')
        _,_,vh=np.linalg.svd(scaled,full_matrices=True)
        for sign in (1.,-1.):
            direction=sign*vh[-1]
            checked=np.asarray(matrix_for_direction(direction),dtype=float)
            if checked.shape!=matrix.shape or not np.all(np.isfinite(checked)):
                raise PathContinuationError('invalid checked joint matrix')
            # Force and gap rows carry different units. A global matrix norm
            # can conceal an incompatible constraint side behind E*t.
            comparison_scale=np.maximum(np.max(np.abs(matrix),axis=1),
                                        np.max(np.abs(checked),axis=1))
            change=np.max(np.abs(checked-matrix)/comparison_scale[:,None])
            if change<=64*np.finfo(float).eps:
                score=float(direction@reference)
                if score>64*np.finfo(float).eps and (direction_is_admissible is None or direction_is_admissible(direction)):
                    valid.append((score,direction.copy()))
            else:pending.append(direction)
    if not valid:raise PathContinuationError('joint current side has no history-consistent direction')
    distinct=[]
    for score,direction in valid:
        if not any(np.linalg.norm(direction-other)<=1e-10 for other in distinct):
            distinct.append(direction)
    if primal_coordinate_count is not None and len(distinct)>1:
        error=PathContinuationError('multiple current-side branches require physical selection')
        error.candidate_directions=tuple(distinct)
        raise error
    valid.sort(key=lambda item:item[0],reverse=True)
    return valid[0][1]


def locate_normal_contact_section(seed,assembly,event,**tolerances):
    """Solve equilibrium plus one contact section, with frozen caller history.

    Event supplies (g, g_x, g_marker) for entering contact, or the scaled
    multiplier and its derivative for leaving contact. No damage is forced.
    Returned state contains the physical unknowns followed by physical marker;
    the returned dummy marker is fixed to zero and has no trajectory meaning.
    """
    seed=np.asarray(seed,dtype=float)
    def equations(state,dummy):
        value=assembly(state[:-1],state[-1])
        residual,row,column=event(state[:-1],state[-1])
        matrix=np.vstack([np.column_stack([value.matrix,value.marker_column]),np.r_[row,column]])
        return np.r_[value.residual,residual],matrix,np.zeros(len(state))
    predicted=np.r_[seed,0.];normal=np.zeros_like(predicted);normal[-1]=1.
    return correct_joint_normal_path(predicted,normal,equations,**tolerances)


def correct_joint_normal_path(predicted,normal,equations,*,force_rows,force_tolerance_N,
        gap_tolerance_m,path_tolerance_m,correction_tolerance_m,maximum_corrections):
    """Joint Newton correction with separate physical acceptance norms.

    Equations return force rows in N and contact constraints in m. Force
    unknowns must already use caller-declared length scaling. No trial history
    is committed here; active-set qualification remains the caller's job.
    """
    from .path_continuation import PathContinuationError,PathState,_oriented_equilibrium_tangent
    predicted=np.asarray(predicted,dtype=float);normal=np.asarray(normal,dtype=float)
    tolerances=(force_tolerance_N,gap_tolerance_m,path_tolerance_m,correction_tolerance_m)
    if (predicted.ndim!=1 or normal.shape!=predicted.shape or len(predicted)<2
        or not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(normal))
        or np.linalg.norm(normal)==0 or not all(np.isfinite(t) and t>0 for t in tolerances)
        or type(force_rows) is not int or not 0<force_rows<len(predicted)
        or type(maximum_corrections) is not int or maximum_corrections<1):
        raise ValueError('invalid joint path settings')
    normal=normal/np.linalg.norm(normal);combined=predicted.copy();correction_norm=0.
    for iteration in range(maximum_corrections+1):
        value,J,column=equations(combined[:-1],combined[-1])
        value=np.asarray(value);J=np.asarray(J);column=np.asarray(column)
        size=len(combined)-1
        if (value.shape!=(size,) or J.shape!=(size,size) or column.shape!=(size,)
            or not all(np.all(np.isfinite(v)) for v in (value,J,column))):
            raise PathContinuationError('invalid joint normal system')
        force_norm=float(np.linalg.norm(value[:force_rows]))
        gap_norm=float(np.max(np.abs(value[force_rows:]),initial=0.))
        constraint=float((combined-predicted)@normal)
        if (force_norm<=force_tolerance_N and gap_norm<=gap_tolerance_m
            and abs(constraint)<=path_tolerance_m and correction_norm<=correction_tolerance_m):
            tangent_scale=np.max(np.abs(np.column_stack([J,column])),axis=1)
            if np.any(tangent_scale==0):
                raise PathContinuationError('joint normal tangent has zero equilibrium row')
            return PathState(combined[:-1],float(combined[-1]),
                _oriented_equilibrium_tangent(J/tangent_scale[:,None],column/tangent_scale,normal),
                force_norm,constraint,iteration,correction_norm)
        if iteration==maximum_corrections:break
        matrix=np.vstack([np.column_stack([J,column]),normal])
        rhs=-np.r_[value,constraint]
        scale=np.max(np.abs(matrix),axis=1)
        if np.any(scale==0):raise PathContinuationError('joint normal correction system is singular')
        try:delta=np.linalg.solve(matrix/scale[:,None],rhs/scale)
        except np.linalg.LinAlgError as cause:
            raise PathContinuationError('joint normal correction system is singular') from cause
        combined+=delta;correction_norm=float(np.linalg.norm(delta))
    error=PathContinuationError('joint normal correction exceeded maximum_corrections')
    frozen=combined.copy();frozen.setflags(write=False)
    error.correction_diagnostics=dict(combined=frozen,force_residual_N=force_norm,
        contact_constraint_m=gap_norm,path_constraint_m=constraint,correction_m=correction_norm,
        iterations=iteration)
    raise error


def circle_normal_constraints(coordinates,dofs,displacement,center,radius,active,forces,motion):
    coordinates=np.asarray(coordinates,dtype=float);dofs=np.asarray(dofs,dtype=int)
    displacement=np.asarray(displacement,dtype=float);active=np.asarray(active,dtype=int)
    forces=np.asarray(forces,dtype=float);center=np.asarray(center,dtype=float)
    motion=np.asarray(motion,dtype=float)
    if (coordinates.ndim!=2 or coordinates.shape[1]!=2 or dofs.shape!=coordinates.shape
        or displacement.ndim!=1 or active.ndim!=1 or forces.shape!=active.shape
        or center.shape!=(2,) or motion.shape!=(2,) or radius<=0
        or len(set(active))!=len(active) or np.any(active<0) or np.any(active>=len(coordinates))
        or np.any(dofs<0) or np.any(dofs>=len(displacement))
        or not all(np.all(np.isfinite(v)) for v in (coordinates,displacement,center,motion,forces,radius))):
        raise ValueError('invalid normal constraint coordinates or identities')
    delta=coordinates+displacement[dofs]-center
    distance=np.linalg.norm(delta,axis=1)
    if np.any(distance==0):raise ValueError('normal constraint at circle center is undefined')
    normals=delta/distance[:,None];all_gaps=distance-radius
    size=len(displacement);count=len(active)
    force=np.zeros(size);K=np.zeros((size,size));L=np.zeros((size,count))
    M=np.zeros(size);G=np.zeros((count,size));gm=np.zeros(count)
    for j,i in enumerate(active):
        ids=dofs[i];normal=normals[i]
        local=forces[j]/distance[i]*(np.eye(2)-np.outer(normal,normal))
        force[ids]+=forces[j]*normal
        K[np.ix_(ids,ids)]+=local;L[ids,j]=normal;M[ids]-=local@motion
        G[j,ids]=normal;gm[j]=-normal@motion
    return NormalConstraintAssembly(force,all_gaps[active],all_gaps,K,L,M,G,gm)
