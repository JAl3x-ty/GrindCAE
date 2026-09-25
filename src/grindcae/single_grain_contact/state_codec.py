"""Versioned accepted-input checkpoint; topology is validated, never regenerated."""
from dataclasses import asdict
import json
import numpy as np
from grindcae.elastoplastic.j2 import MaterialPointState
from .contact_law import ContactPointState
from .damage_material import DamageState
from .energetic_damage import EnergeticDamageHistory
from .continuous_separation import SeparatingState,SolveMode,SolveFailure
from .material_topology import MaterialState
from .path_continuation import AcceptedPathReference,PathState


def validate_material_compatibility(prepared,state,material):
    """Check effective elastic/plastic history against the retained P1 field."""
    for point,damage,B,dofs in zip(state.element_states,state.damage_states,
            prepared.structural.element_B_matrices_per_m,prepared.structural.element_dofs,strict=True):
        # Fully failed points freeze their effective history; their zero
        # nominal stress no longer constrains subsequent displacement.
        if damage.damage==1.:
            history=damage.energetic_history
            if history is not None:
                from .thermodynamic_energy import stored_energy,fracture_resistance
                driving=stored_energy(point.stress_tensor_Pa,point.equivalent_plastic_strain,1.,
                    young_modulus=material.E,poisson_ratio=material.nu,
                    hardening_modulus=material.internal_hardening_modulus)['effective_total']
                resistance=fracture_resistance(initial_driving_energy=history.initial_driving_energy,
                    fracture_energy_density=history.fracture_energy_density).resistance(1.)
                if abs(driving-resistance)>1e-10*history.initial_driving_energy:
                    raise ValueError('checkpoint terminal damage surface is not localized')
            continue
        strain=B@state.displacement_vector_m[dofs]
        total=np.array([[strain[0],.5*strain[2],0.],
            [.5*strain[2],strain[1],0.],[0.,0.,0.]])
        stress=np.asarray(point.stress_tensor_Pa)
        elastic=((1+material.nu)*stress-material.nu*np.trace(stress)*np.eye(3))/material.E
        reconstructed=elastic+np.asarray(point.plastic_strain_tensor)
        tolerance=512*np.finfo(float).eps*max(1.,np.linalg.norm(total),np.linalg.norm(reconstructed))
        if np.linalg.norm(total-reconstructed)>tolerance:
            raise ValueError('checkpoint material compatibility with displacement is violated')


def validate_endpoint_momentum(prepared,state,case,contact_force_N):
    """Independently check the stored dynamic endpoint, without material trials."""
    if state.solve_mode is not SolveMode.INERTIAL_TRANSITION:
        return
    from .inertial_transition import assemble_consistent_mass_matrix
    mass=assemble_consistent_mass_matrix(prepared.structural,
        density_kg_per_m3=case.material.density_kg_per_m3,thickness_m=case.analysis.thickness)
    if state.acceleration_vector_m_per_s2.shape!=(prepared.total_degrees_of_freedom,):
        raise ValueError('checkpoint endpoint momentum lacks acceleration')
    internal=np.zeros(prepared.total_degrees_of_freedom)
    for point,damage,B,dofs,area in zip(state.element_states,state.damage_states,
            prepared.structural.element_B_matrices_per_m,prepared.structural.element_dofs,
            prepared.structural.element_areas_m2,strict=True):
        stress=(1.-damage.damage)*np.asarray(point.stress_tensor_Pa)
        internal[dofs]+=B.T@np.array([stress[0,0],stress[1,1],stress[0,1]])*float(area)*case.analysis.thickness
    residual=mass@state.acceleration_vector_m_per_s2+internal-contact_force_N
    free=prepared.structural.free_dofs
    tolerance=case.newton.residual_absolute_tolerance_N+case.newton.residual_relative_tolerance*np.linalg.norm(contact_force_N[free])
    if not np.all(np.isfinite(residual)) or np.linalg.norm(residual[free])>tolerance:
        raise ValueError('checkpoint endpoint momentum disagrees with physical forces')


def encode_committed_state(prepared,state):
    def path(value):
        if value is None: return None
        return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(value).items()}
    data=dict(format='grindcae_committed_contact_v1',
        background_node_ids=prepared.active_to_background_node_ids.tolist(),
        background_element_ids=prepared.active_to_background_element_ids.tolist(),
        reference_coordinates_m=prepared.structural.imported_mesh.mesh.p.T.tolist(),
        connectivity=prepared.structural.imported_mesh.mesh.t.T.tolist(),
        circle_edge_quadrature_order=getattr(prepared,'circle_edge_quadrature_order',0),
        marker=state.marker,displacement_vector_m=state.displacement_vector_m.tolist(),
        element_states=[asdict(v) for v in state.element_states],damage_states=[asdict(v) for v in state.damage_states],
        contact_states=[asdict(v) for v in state.contact_states],candidate_keys=state.candidate_keys,
        edge_contact_states=[dict(key=k,state=asdict(v)) for k,v in state.edge_contact_states],
        archived_edge_contact_states=[dict(key=k,state=asdict(v)) for k,v in state.archived_edge_contact_states],
        grain_reference_m=state.grain_reference_m,grain_center_m=state.grain_center_m,
        path_state=path(state.path_state),accepted_path_reference=path(state.accepted_path_reference),
        solve_mode=state.solve_mode.value,solve_failure=state.solve_failure.value if state.solve_failure else None,
        separating_states=[dict(background_element_id=k,state=asdict(v)) for k,v in state.separating_states],
        physical_time_s=float(getattr(state,'physical_time_s',0.)),
        velocity_vector_m_per_s=np.asarray(getattr(state,'velocity_vector_m_per_s',())).tolist(),
        acceleration_vector_m_per_s2=np.asarray(getattr(state,'acceleration_vector_m_per_s2',())).tolist(),
        kinetic_energy_J=float(getattr(state,'kinetic_energy_J',0.)),
        transient_absolute_energy_error_J=float(getattr(state,'transient_absolute_energy_error_J',0.)),
        transient_maximum_energy_scale_J=float(getattr(state,'transient_maximum_energy_scale_J',0.)),
        transient_stable_step_count=int(getattr(state,'transient_stable_step_count',0)),
        transient_step_count=int(getattr(state,'transient_step_count',0)),
        transient_last_time_step_s=float(getattr(state,'transient_last_time_step_s',0.)),
        exported_kinetic_energy_J=float(getattr(state,'exported_kinetic_energy_J',0.)),
        exported_momentum_kg_m_per_s=list(getattr(state,'exported_momentum_kg_m_per_s',(0.,0.))))
    return json.loads(json.dumps(data,allow_nan=False))


def decode_committed_state(prepared,data,*,density_kg_per_m3=None,thickness_m=None):
    from .solver import CommittedContactStructureState
    if not isinstance(data,dict) or data.get('format')!='grindcae_committed_contact_v1':
        raise ValueError('invalid checkpoint format')
    json.dumps(data,allow_nan=False)
    for key,expected in (
        ('background_node_ids',prepared.active_to_background_node_ids.tolist()),
        ('background_element_ids',prepared.active_to_background_element_ids.tolist()),
        ('reference_coordinates_m',prepared.structural.imported_mesh.mesh.p.T.tolist()),
        ('connectivity',prepared.structural.imported_mesh.mesh.t.T.tolist()),
        ('circle_edge_quadrature_order',getattr(prepared,'circle_edge_quadrature_order',0))):
        if data.get(key)!=expected: raise ValueError('checkpoint topology/integration identity mismatch')
    displacement=np.asarray(data['displacement_vector_m'],dtype=float)
    if displacement.shape!=(prepared.total_degrees_of_freedom,) or not np.all(np.isfinite(displacement)):
        raise ValueError('invalid checkpoint displacement')
    if np.any(displacement[prepared.structural.fixed_dofs]!=0.):
        raise ValueError('checkpoint violates fixed support')
    velocity=np.asarray(data.get('velocity_vector_m_per_s',()),dtype=float)
    acceleration=np.asarray(data.get('acceleration_vector_m_per_s2',()),dtype=float)
    for name,value in (('velocity',velocity),('acceleration',acceleration)):
        if value.size and (value.shape!=(prepared.total_degrees_of_freedom,) or not np.all(np.isfinite(value))):
            raise ValueError(f'invalid checkpoint {name}')
        if value.size and np.any(value[prepared.structural.fixed_dofs]!=0.):
            raise ValueError(f'checkpoint {name} violates fixed support')
    physical_time=float(data.get('physical_time_s',0.));kinetic=float(data.get('kinetic_energy_J',0.))
    exported=float(data.get('exported_kinetic_energy_J',0.))
    if (density_kg_per_m3 is None) != (thickness_m is None):
        raise ValueError('checkpoint mass audit requires density and thickness together')
    if density_kg_per_m3 is not None:
        from .inertial_transition import assemble_consistent_mass_matrix
        mass=assemble_consistent_mass_matrix(prepared.structural,
            density_kg_per_m3=density_kg_per_m3,thickness_m=thickness_m)
        physical_kinetic=0.5*float(velocity@mass@velocity) if velocity.size else 0.
        if not np.isfinite(kinetic) or abs(physical_kinetic-kinetic)>64*np.finfo(float).eps*max(
                abs(kinetic),abs(physical_kinetic),np.finfo(float).tiny):
            raise ValueError('checkpoint kinetic energy disagrees with physical mass and velocity')
    transient_error=float(data.get('transient_absolute_energy_error_J',0.))
    transient_scale=float(data.get('transient_maximum_energy_scale_J',0.))
    stable_count=data.get('transient_stable_step_count',0)
    step_count=data.get('transient_step_count',0)
    last_dt=float(data.get('transient_last_time_step_s',0.))
    if not np.isfinite(last_dt) or last_dt<0.:
        raise ValueError('invalid checkpoint last physical time step')
    if type(step_count) is not int or step_count<0:
        raise ValueError('invalid checkpoint transient step count')
    if type(stable_count) is not int or not 0<=stable_count<=8:
        raise ValueError('invalid checkpoint transient stable step count')
    if (not np.isfinite(transient_error) or not np.isfinite(transient_scale)
        or transient_error<0. or transient_scale<0.
        or transient_error>max(1e-10,1e-5*transient_scale)):
        raise ValueError('invalid checkpoint transient energy budget')
    momentum=tuple(float(v) for v in data.get('exported_momentum_kg_m_per_s',(0.,0.)))
    if (not all(np.isfinite(v) for v in (physical_time,kinetic,exported,*momentum))
        or physical_time<0 or kinetic<0 or exported<0 or len(momentum)!=2):
        raise ValueError('invalid checkpoint dynamic energy/history')
    materials=tuple(MaterialPointState(**v) for v in data['element_states'])
    damages=[]
    for raw in data['damage_states']:
        value=dict(raw)
        history=value.get('energetic_history')
        if history is not None: value['energetic_history']=EnergeticDamageHistory(**history)
        if (not 0<=value['damage']<=1 or value['omega']<0
            or value['status'] not in ('ACTIVE','DAMAGED','SEPARATING','REMOVED')
            or value['damage_dissipation_density_J_per_m3']<0):
            raise ValueError('invalid checkpoint damage')
        if value['status']=='REMOVED':
            raise ValueError('checkpoint active topology contains REMOVED material')
        if history is not None and history['damage']!=value['damage']:
            raise ValueError('checkpoint damage history disagrees')
        damages.append(DamageState(**value))
    if len(materials)!=prepared.element_count or len(damages)!=prepared.element_count:
        raise ValueError('checkpoint material count mismatch')
    for material,damage in zip(materials,damages,strict=True):
        if material.equivalent_plastic_strain!=damage.equivalent_plastic_strain:
            raise ValueError('checkpoint material plastic history disagrees with damage')
        if damage.energetic_history is not None:
            from .thermodynamic_energy import fracture_resistance
            h=damage.energetic_history
            expected=fracture_resistance(initial_driving_energy=h.initial_driving_energy,
                fracture_energy_density=h.fracture_energy_density).cumulative(h.damage)
            tolerance=64*np.finfo(float).eps*max(abs(expected),abs(damage.damage_dissipation_density_J_per_m3))
            if abs(expected-damage.damage_dissipation_density_J_per_m3)>tolerance:
                raise ValueError('checkpoint material fracture energy disagrees with damage')
    keys=tuple((int(n),tuple(edges)) for n,edges in data['candidate_keys'])
    if keys!=prepared.candidate_keys: raise ValueError('checkpoint candidate identity mismatch')
    contacts=tuple(ContactPointState(**v) for v in data['contact_states'])
    if len(contacts)!=len(keys): raise ValueError('checkpoint contact count mismatch')
    def owned(name):
        result=tuple((tuple(row['key']),ContactPointState(**row['state'])) for row in data[name])
        if len({k for k,_ in result})!=len(result): raise ValueError('duplicate checkpoint edge identity')
        return result
    active=owned('edge_contact_states');archived=owned('archived_edge_contact_states')
    if set(k for k,_ in active)!=set(prepared.edge_endpoint_keys):
        raise ValueError('checkpoint edge identity mismatch')
    if set(k for k,_ in active)&set(k for k,_ in archived): raise ValueError('checkpoint active archived overlap')
    reference=data['accepted_path_reference']
    if reference is not None:
        reference=dict(reference)
        for name in ('background_node_ids','source_markers'): reference[name]=tuple(reference[name])
        reference['displacement_direction_m']=tuple(tuple(v) for v in reference['displacement_direction_m'])
        reference=AcceptedPathReference(**reference)
    path=data['path_state']
    if path is not None: path=PathState(**path)
    separating=[]
    for row in data['separating_states']:
        value=dict(row['state']);value['material_state']=MaterialState(value['material_state'])
        value['entry_strain_direction']=tuple(value['entry_strain_direction'])
        if value['energetic_history'] is not None: value['energetic_history']=EnergeticDamageHistory(**value['energetic_history'])
        identity=row['background_element_id']
        if identity not in data['background_element_ids']: raise ValueError('checkpoint separation identity mismatch')
        separation=SeparatingState(**value)
        if (separation.material_state!=MaterialState.SEPARATING
            or not 0<=separation.s_e<=1
            or not 0<=separation.damage<=1
            or not 0<=separation.entry_softening_fraction<=1):
            raise ValueError('checkpoint separation state is invalid')
        if separation.entry_softening_fraction==1. and not (
                data['solve_mode']==SolveMode.INERTIAL_TRANSITION.value
                and separation.s_e==1. and separation.damage==1. and separation.alpha_e==0.
                and separation.nominal_equivalent_stress_Pa==0.
                and separation.remaining_fracture_energy_density_J_per_m3==0.
                and separation.separation_damage_dissipation_density_J_per_m3==0.
                and separation.energetic_history is not None
                and separation.energetic_history.damage==1.
                and separation.entry_damage_dissipation_density_J_per_m3
                    ==separation.energetic_history.fracture_energy_density):
            raise ValueError('checkpoint natural terminal separation metadata is inconsistent')
        damage=damages[data['background_element_ids'].index(identity)]
        if (damage.damage!=separation.damage or damage.status!='SEPARATING'
            or damage.nominal_equivalent_stress_Pa!=separation.nominal_equivalent_stress_Pa
            or damage.damage_dissipation_density_J_per_m3!=separation.damage_dissipation_density_J_per_m3):
            raise ValueError('checkpoint separation disagrees with material state')
        if separation.energetic_history is not None:
            from grindcae.elastoplastic.j2 import equivalent_von_mises_stress
            effective=equivalent_von_mises_stress(np.asarray(materials[data['background_element_ids'].index(identity)].stress_tensor_Pa))
            if not np.isclose(separation.effective_equivalent_stress_Pa,effective,rtol=64*np.finfo(float).eps,atol=0.):
                raise ValueError('checkpoint separation effective stress disagrees with J2 history')
        from .continuous_separation import advance_separating
        derived=advance_separating(separation,s_e=separation.s_e)
        for name in ('damage','damage_dissipation_density_J_per_m3',
            'separation_damage_dissipation_density_J_per_m3',
            'remaining_fracture_energy_density_J_per_m3'):
            expected=getattr(derived,name); actual=getattr(separation,name)
            tolerance=64*np.finfo(float).eps*max(abs(expected),abs(actual))
            if abs(expected-actual)>tolerance:
                raise ValueError('checkpoint separation progress/energy is inconsistent')
        separating.append((identity,separation))
    separating_ids=[identity for identity,_ in separating]
    if len(set(separating_ids))!=len(separating_ids):
        raise ValueError('checkpoint duplicate separation identity')
    for identity,damage in zip(data['background_element_ids'],damages,strict=True):
        if damage.status=='SEPARATING' and identity not in separating_ids:
            raise ValueError('checkpoint separation material has no control history')
    return CommittedContactStructureState(marker=float(data['marker']),displacement_vector_m=displacement.copy(),
        element_states=materials,damage_states=tuple(damages),contact_states=contacts,candidate_keys=keys,
        grain_reference_m=tuple(data['grain_reference_m']),
        grain_center_m=tuple(data['grain_center_m']) if data['grain_center_m'] is not None else None,
        background_element_ids=np.asarray(data['background_element_ids'],dtype=np.int32),
        edge_contact_states=active,archived_edge_contact_states=archived,path_state=path,
        accepted_path_reference=reference,separating_states=tuple(separating),
        solve_mode=SolveMode(data['solve_mode']),solve_failure=SolveFailure(data['solve_failure']) if data['solve_failure'] else None,
        physical_time_s=physical_time,velocity_vector_m_per_s=velocity.copy(),
        acceleration_vector_m_per_s2=acceleration.copy(),kinetic_energy_J=kinetic,
        transient_stable_step_count=stable_count,
        transient_step_count=step_count,
        transient_last_time_step_s=last_dt,
        transient_absolute_energy_error_J=transient_error,transient_maximum_energy_scale_J=transient_scale,
        exported_kinetic_energy_J=exported,exported_momentum_kg_m_per_s=momentum)
