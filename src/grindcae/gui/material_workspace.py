"""Material input and SI project schema for the 2.7 workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping


MATERIAL_SCHEMA_VERSION = 1
MATERIAL_BEHAVIORS = frozenset({"linear_elastic", "elastoplastic"})
CALIBRATION_STATUSES = frozenset(
    {
        "demonstration_not_calibrated",
        "user_input_not_calibrated",
        "literature_reference",
        "experimentally_calibrated",
    }
)
_KNOWN_FIELDS = frozenset(
    {
        "material_schema_version",
        "behavior",
        "family",
        "label",
        "elastic_modulus_Pa",
        "poisson_ratio",
        "yield_strength_Pa",
        "tangent_modulus_Pa",
        "parameter_source",
        "calibration_status",
        "source_description",
        "applicability_notes",
        "status",
    }
)


class MaterialValidationError(ValueError):
    """Chinese material validation error suitable for the GUI."""


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterialValidationError("材料数据必须是 JSON 对象。")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MaterialValidationError(f"{label}不能为空。")
    return value.strip()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialValidationError("请输入有效的有限数值。")
    number = float(value)
    if not math.isfinite(number):
        raise MaterialValidationError("请输入有效的有限数值。")
    return number


def _display_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise MaterialValidationError("请输入有效的有限数值。")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise MaterialValidationError("请输入有效的有限数值。") from exc
    if not math.isfinite(number):
        raise MaterialValidationError("请输入有效的有限数值。")
    return number


@dataclass(frozen=True)
class MaterialDisplayInput:
    label: str = "通用延性合金钢演示参数"
    behavior: str = "elastoplastic"
    family: str = "ductile_metal"
    elastic_modulus_gpa: str = "210"
    poisson_ratio: str = "0.30"
    yield_strength_mpa: str = "250"
    tangent_modulus_gpa: str = "2"
    parameter_source: str = "builtin_demo"
    calibration_status: str = "demonstration_not_calibrated"
    source_description: str = "GrindCAE 工程演示参数，不对应具体材料牌号。"
    applicability_notes: str = "仅用于工作流演示，工程应用前需要实验或可靠文献标定。"

    def build_case(self) -> "WorkbenchMaterialCase":
        label = _text(self.label, "材料名称")
        if self.family != "ductile_metal":
            raise MaterialValidationError("当前版本暂未启用该材料模型。")
        if self.behavior not in MATERIAL_BEHAVIORS:
            raise MaterialValidationError("材料行为无效。")
        elastic_modulus = _display_number(self.elastic_modulus_gpa) * 1.0e9
        if elastic_modulus <= 0.0:
            raise MaterialValidationError("弹性模量必须大于 0。")
        poisson_ratio = _display_number(self.poisson_ratio)
        if not -1.0 < poisson_ratio < 0.5:
            raise MaterialValidationError("泊松比必须大于 -1 且小于 0.5。")
        yield_strength: float | None = None
        tangent_modulus: float | None = None
        if self.behavior == "elastoplastic":
            if not isinstance(self.yield_strength_mpa, str) or not self.yield_strength_mpa.strip():
                raise MaterialValidationError("弹塑性材料必须输入屈服强度。")
            yield_strength = _display_number(self.yield_strength_mpa) * 1.0e6
            if yield_strength <= 0.0:
                raise MaterialValidationError("屈服强度必须大于 0。")
            if not isinstance(self.tangent_modulus_gpa, str) or not self.tangent_modulus_gpa.strip():
                raise MaterialValidationError("弹塑性材料必须输入切线模量。")
            tangent_modulus = _display_number(self.tangent_modulus_gpa) * 1.0e9
            if not 0.0 <= tangent_modulus < elastic_modulus:
                raise MaterialValidationError(
                    "切线模量必须大于等于 0 且小于弹性模量。"
                )
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise MaterialValidationError("标定状态无效。")
        source = self.source_description.strip() if isinstance(self.source_description, str) else ""
        if self.calibration_status in {
            "literature_reference",
            "experimentally_calibrated",
        } and not source:
            raise MaterialValidationError("文献参考或实验标定必须填写来源说明。")
        return WorkbenchMaterialCase(
            behavior=self.behavior,
            family=self.family,
            label=label,
            elastic_modulus_pa=elastic_modulus,
            poisson_ratio=poisson_ratio,
            yield_strength_pa=yield_strength,
            tangent_modulus_pa=tangent_modulus,
            parameter_source=_text(self.parameter_source, "参数来源"),
            calibration_status=self.calibration_status,
            source_description=source,
            applicability_notes=(
                self.applicability_notes.strip()
                if isinstance(self.applicability_notes, str)
                else ""
            ),
        )


@dataclass(frozen=True)
class WorkbenchMaterialCase:
    behavior: str
    family: str
    label: str
    elastic_modulus_pa: float
    poisson_ratio: float
    yield_strength_pa: float | None
    tangent_modulus_pa: float | None
    parameter_source: str
    calibration_status: str
    source_description: str
    applicability_notes: str
    status: str = "completed"
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.behavior not in MATERIAL_BEHAVIORS:
            raise MaterialValidationError("材料行为无效。")
        if self.family != "ductile_metal":
            raise MaterialValidationError("当前版本暂未启用该材料模型。")
        _text(self.label, "材料名称")
        elastic = _finite(self.elastic_modulus_pa)
        if elastic <= 0.0:
            raise MaterialValidationError("弹性模量必须大于 0。")
        poisson = _finite(self.poisson_ratio)
        if not -1.0 < poisson < 0.5:
            raise MaterialValidationError("泊松比必须大于 -1 且小于 0.5。")
        if self.behavior == "linear_elastic":
            if self.yield_strength_pa is not None or self.tangent_modulus_pa is not None:
                raise MaterialValidationError("线弹性材料的塑性参数必须为 null。")
        else:
            if self.yield_strength_pa is None:
                raise MaterialValidationError("弹塑性材料必须输入屈服强度。")
            if self.tangent_modulus_pa is None:
                raise MaterialValidationError("弹塑性材料必须输入切线模量。")
            if _finite(self.yield_strength_pa) <= 0.0:
                raise MaterialValidationError("屈服强度必须大于 0。")
            tangent = _finite(self.tangent_modulus_pa)
            if not 0.0 <= tangent < elastic:
                raise MaterialValidationError(
                    "切线模量必须大于等于 0 且小于弹性模量。"
                )
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise MaterialValidationError("标定状态无效。")
        if self.calibration_status in {
            "literature_reference",
            "experimentally_calibrated",
        } and not self.source_description.strip():
            raise MaterialValidationError("文献参考或实验标定必须填写来源说明。")
        if self.status != "completed":
            raise MaterialValidationError("材料状态必须为 completed。")

    @classmethod
    def from_mapping(cls, payload: object) -> "WorkbenchMaterialCase":
        data = _mapping(payload)
        missing = sorted(_KNOWN_FIELDS - set(data))
        if missing:
            raise MaterialValidationError(f"材料数据缺少必需字段：{', '.join(missing)}")
        version = data["material_schema_version"]
        if type(version) is not int or version != MATERIAL_SCHEMA_VERSION:
            raise MaterialValidationError("不支持的材料文件版本。")
        return cls(
            behavior=data["behavior"],
            family=data["family"],
            label=data["label"],
            elastic_modulus_pa=data["elastic_modulus_Pa"],
            poisson_ratio=data["poisson_ratio"],
            yield_strength_pa=data["yield_strength_Pa"],
            tangent_modulus_pa=data["tangent_modulus_Pa"],
            parameter_source=data["parameter_source"],
            calibration_status=data["calibration_status"],
            source_description=data["source_description"],
            applicability_notes=data["applicability_notes"],
            status=data["status"],
            extra_fields={key: value for key, value in data.items() if key not in _KNOWN_FIELDS},
        )

    def validate_material(self) -> tuple[bool, str]:
        return True, "材料参数有效。"

    def to_material_schema(self) -> dict[str, Any]:
        return self.to_dict()

    def material_summary(self) -> dict[str, str]:
        plastic_unused = "不参与当前模型"
        return {
            "材料名称": self.label,
            "材料行为": "线弹性" if self.behavior == "linear_elastic" else "弹塑性",
            "材料类别": "延性金属",
            "弹性模量": f"{self.elastic_modulus_pa / 1.0e9:g} GPa",
            "泊松比": f"{self.poisson_ratio:g}",
            "屈服强度": (
                plastic_unused
                if self.yield_strength_pa is None
                else f"{self.yield_strength_pa / 1.0e6:g} MPa"
            ),
            "切线模量": (
                plastic_unused
                if self.tangent_modulus_pa is None
                else f"{self.tangent_modulus_pa / 1.0e9:g} GPa"
            ),
            "标定状态": self.calibration_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "material_schema_version": MATERIAL_SCHEMA_VERSION,
            "behavior": self.behavior,
            "family": self.family,
            "label": self.label,
            "elastic_modulus_Pa": self.elastic_modulus_pa,
            "poisson_ratio": self.poisson_ratio,
            "yield_strength_Pa": self.yield_strength_pa,
            "tangent_modulus_Pa": self.tangent_modulus_pa,
            "parameter_source": self.parameter_source,
            "calibration_status": self.calibration_status,
            "source_description": self.source_description,
            "applicability_notes": self.applicability_notes,
            "status": self.status,
            **self.extra_fields,
        }
