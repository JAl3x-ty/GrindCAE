"""Strict composed input; no cached fitted force is accepted as a prediction."""
from dataclasses import dataclass, replace
import hashlib
import json
import math

from grindcae.history_pass import FixedMeshElastoplasticHistoryPassCase
from grindcae380.reconstruction import ReconstructionCase

MODE = 'literature_elastoplastic_single_pass'
RESULT_FORMAT = 'grindcae_literature_elastoplastic_history_v1'
PROJECTION = 'circumferential_strip_p1_contact_ratio_v1'


def canonical(data):
    return json.dumps(data, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))


@dataclass(frozen=True)
class LiteratureHistoryCase:
    literature_history_schema_version: int
    literature: ReconstructionCase
    history_template: FixedMeshElastoplasticHistoryPassCase
    material_provenance: str
    projection: str = PROJECTION

    def __post_init__(self):
        if type(self.literature_history_schema_version) is not int or self.literature_history_schema_version != 1:
            raise ValueError('literature_history_schema_version must be strict integer 1')
        if not isinstance(self.literature, ReconstructionCase) or not isinstance(self.history_template, FixedMeshElastoplasticHistoryPassCase):
            raise ValueError('validated literature and history inputs required')
        if self.projection != PROJECTION:
            raise ValueError('unsupported literature projection')
        if not isinstance(self.material_provenance, str) or not self.material_provenance.strip():
            raise ValueError('material_provenance is required')
        c = self.literature.reference_case
        ref = self.history_template.reference_case
        p = ref.pass_load.force_model.process
        pairs = ((c.wheel_diameter_m, p.wheel_diameter_m), (c.depth_m, p.depth_of_cut_m),
                 (c.width_m, p.grinding_width_m), (c.wheel_speed_m_s, p.wheel_surface_speed_m_per_s),
                 (c.feed_speed_m_s, p.workpiece_feed_speed_m_per_s),
                 (c.width_m, ref.analysis.thickness),
                 (c.wheel_diameter_m, ref.pass_load.trajectory.wheel.diameter_m),
                 (c.depth_m, ref.pass_load.trajectory.single_pass.depth_of_cut_m),
                 (c.yield_stress_Pa, ref.material.yield_strength))
        if any(not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0) for a, b in pairs):
            raise ValueError('literature process/yield/width must match history input; rebuild from engineering inputs')
        if c.feed_speed_m_s <= 0:
            raise ValueError('complete-pass history requires positive feed speed')
        # The host input is a geometry/material template. Placeholder force values
        # must be canonical, so stale empirical coefficients cannot affect input identity.
        calibration = ref.pass_load.force_model.to_dict()['calibration']
        if calibration != placeholder_calibration():
            raise ValueError('history template must use the declared runtime force placeholder')
        canonical(self.to_dict())

    def to_dict(self):
        return dict(literature_history_schema_version=1, literature=self.literature.to_dict(),
                    history_template=self.history_template.to_dict(), material_provenance=self.material_provenance,
                    projection=self.projection)

    @classmethod
    def from_mapping(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError('literature history requires exactly its registered fields')
        return cls(value['literature_history_schema_version'], ReconstructionCase.from_mapping(value['literature']),
                   FixedMeshElastoplasticHistoryPassCase.from_mapping(value['history_template']),
                   value['material_provenance'], value['projection'])

    @property
    def input_fingerprint(self):
        return hashlib.sha256(canonical(self.to_dict()).encode('utf-8')).hexdigest()


def placeholder_calibration():
    return dict(specific_grinding_energy_J_per_m3=1.0, normal_to_tangential_force_ratio=1.0,
                calibration_id='literature_history_runtime_placeholder',
                specific_grinding_energy_source='Replaced by recomputed literature force before any solve.',
                force_ratio_source='Replaced by recomputed literature force before any solve.',
                applicability_notes='Schema placeholder only; never an applied force or material calibration.')


def compose_case(literature, history, material_provenance):
    """Engineering process/yield override force-page values; FE E/nu/Et stay native."""
    ref = history.reference_case
    p = ref.pass_load.force_model.process
    selected = replace(literature, reference_case=replace(literature.reference_case,
        wheel_diameter_m=p.wheel_diameter_m, depth_m=p.depth_of_cut_m,
        width_m=p.grinding_width_m, wheel_speed_m_s=p.wheel_surface_speed_m_per_s,
        feed_speed_m_s=p.workpiece_feed_speed_m_per_s, yield_stress_Pa=ref.material.yield_strength))
    data = history.to_dict()
    data['moving_load_pass']['reference_case']['pass_load']['force_model']['calibration'] = placeholder_calibration()
    return LiteratureHistoryCase(1, selected, FixedMeshElastoplasticHistoryPassCase.from_mapping(data), material_provenance)
