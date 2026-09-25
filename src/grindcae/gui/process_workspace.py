"""Wheel, grit, and grinding-process SI schemas for the 2.7 workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Mapping

from grindcae.statistical_grain_load import (
    ManufacturerWheelSpecification,
    StatisticalGrainLoadValidationError,
    TableWheelSpecification,
    resolve_grit_specification,
)

from .material_workspace import CALIBRATION_STATUSES


WHEEL_SCHEMA_VERSION = 1
PROCESS_SCHEMA_VERSION = 1
_WHEEL_FIELDS = frozenset(
    {
        "wheel_schema_version",
        "abrasive_material",
        "display_name",
        "resolution_path",
        "grit_designation",
        "resolved_specification",
        "status",
    }
)
_PROCESS_FIELDS = frozenset(
    {
        "process_schema_version",
        "wheel_surface_speed_m_per_s",
        "workpiece_feed_speed_m_per_s",
        "grinding_width_m",
        "specific_grinding_energy_J_per_m3",
        "normal_to_tangential_force_ratio",
        "calibration_id",
        "calibration_status",
        "source_description",
        "applicability_notes",
        "geometry_reference",
        "status",
    }
)


class ProcessValidationError(ValueError):
    """Chinese wheel/process validation error suitable for the GUI."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProcessValidationError(f"{label}必须是 JSON 对象。")
    return value


def _text(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProcessValidationError(message)
    return value.strip()


def _positive_display(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ProcessValidationError(message)
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ProcessValidationError(message) from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ProcessValidationError(message)
    return number


def _positive_si(value: object, message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProcessValidationError(message)
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ProcessValidationError(message)
    return number


def geometry_fingerprint(geometry: object) -> str:
    data = _mapping(geometry, "几何引用")
    try:
        serialized = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ProcessValidationError("几何引用包含无法保存的数值。") from exc
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WheelDisplayInput:
    abrasive_material: str = "aluminum_oxide"
    resolution_path: str = "JIS_hash"
    grit_designation: str = "#5000"
    manufacturer: str = ""
    product_model: str = ""
    diameter_lower_um: str = ""
    representative_diameter_um: str = ""
    diameter_upper_um: str = ""
    manufacturer_source: str = ""
    manufacturer_data_status: str = "manufacturer_input_unverified"

    def build_case(self) -> "WorkbenchWheelCase":
        if self.abrasive_material != "aluminum_oxide":
            raise ProcessValidationError("当前版本暂未启用该磨料。")
        try:
            if self.resolution_path == "manufacturer_override":
                specification = ManufacturerWheelSpecification.from_dict(
                    {
                        "resolution_path": self.resolution_path,
                        "abrasive_material": self.abrasive_material,
                        "grit_designation": self.grit_designation,
                        "manufacturer": self.manufacturer,
                        "product_model": self.product_model,
                        "diameter_lower_m": _positive_display(
                            self.diameter_lower_um, "请输入有效的厂家粒径下限。"
                        )
                        * 1.0e-6,
                        "representative_diameter_d50_m": _positive_display(
                            self.representative_diameter_um,
                            "请输入有效的厂家代表粒径。",
                        )
                        * 1.0e-6,
                        "diameter_upper_m": _positive_display(
                            self.diameter_upper_um, "请输入有效的厂家粒径上限。"
                        )
                        * 1.0e-6,
                        "range_definition": "manufacturer_declared_lower_d50_upper",
                        "data_status": self.manufacturer_data_status,
                        "source": self.manufacturer_source,
                    }
                )
            else:
                specification = TableWheelSpecification.from_dict(
                    {
                        "resolution_path": self.resolution_path,
                        "abrasive_material": self.abrasive_material,
                        "grit_designation": self.grit_designation,
                    }
                )
            resolved = resolve_grit_specification(specification).to_dict()
        except (StatisticalGrainLoadValidationError, ProcessValidationError) as exc:
            if isinstance(exc, ProcessValidationError):
                raise
            raise ProcessValidationError(f"粒度解析失败：{exc}") from exc
        return WorkbenchWheelCase(
            abrasive_material=self.abrasive_material,
            display_name="刚玉",
            resolution_path=self.resolution_path,
            grit_designation=self.grit_designation,
            resolved_specification=resolved,
        )


@dataclass(frozen=True)
class WorkbenchWheelCase:
    abrasive_material: str
    display_name: str
    resolution_path: str
    grit_designation: str
    resolved_specification: dict[str, Any]
    status: str = "completed"
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.abrasive_material != "aluminum_oxide":
            raise ProcessValidationError("当前版本暂未启用该磨料。")
        if self.status != "completed":
            raise ProcessValidationError("砂轮状态必须为 completed。")
        resolved = _mapping(self.resolved_specification, "粒度解析结果")
        for field_name in (
            "diameter_lower_m",
            "representative_diameter_d50_m",
            "diameter_upper_m",
        ):
            _positive_si(resolved.get(field_name), "粒度解析结果必须包含有限正粒径。")

    @classmethod
    def from_mapping(cls, payload: object) -> "WorkbenchWheelCase":
        data = _mapping(payload, "砂轮数据")
        missing = sorted(_WHEEL_FIELDS - set(data))
        if missing:
            raise ProcessValidationError(f"砂轮数据缺少必需字段：{', '.join(missing)}")
        if type(data["wheel_schema_version"]) is not int or data["wheel_schema_version"] != 1:
            raise ProcessValidationError("不支持的砂轮文件版本。")
        return cls(
            abrasive_material=data["abrasive_material"],
            display_name=data["display_name"],
            resolution_path=data["resolution_path"],
            grit_designation=data["grit_designation"],
            resolved_specification=dict(_mapping(data["resolved_specification"], "粒度解析结果")),
            status=data["status"],
            extra_fields={key: value for key, value in data.items() if key not in _WHEEL_FIELDS},
        )

    def validate_wheel(self) -> tuple[bool, str]:
        return True, "砂轮与粒度参数有效。"

    def resolve_grit(self) -> dict[str, Any]:
        return dict(self.resolved_specification)

    def to_wheel_schema(self) -> dict[str, Any]:
        return self.to_dict()

    def wheel_summary(self) -> dict[str, str]:
        resolved = self.resolved_specification
        status_labels = {
            "provisional_pending_manufacturer_verification": "临时工程参考，待厂家数据核实",
            "provisional_reference_pending_standard_verification": "临时工程参考，待标准原文核实",
            "secondary_reference_unverified": "二手参考，未经标准原文核实",
            "public_secondary_reference_unverified_against_primary_standard": "公开二手换算参考，未经标准原文核实",
        }
        if resolved.get("manufacturer_override_applied") is True:
            data_status = "厂家数据覆盖"
        else:
            data_status = status_labels.get(
                str(resolved.get("data_status")), str(resolved.get("data_status", ""))
            )
        basis = resolved.get("representative_diameter_basis")
        representative_note = (
            "由粒径范围的几何平均得到的兼容代表值，不是实测 d50。"
            if basis in {
                "geometric_mean_of_secondary_reference_range_not_measured_d50",
                "geometric_mean_of_secondary_w_comparison_range_not_measured_d50",
            }
            else "公开表中的 40% 最小筛余参考值，仅为接口兼容写入代表粒径字段，不是实测 d50。"
            if basis == "washington_mills_public_40_percent_minimum_sieve_reference_not_measured_d50"
            else "采用现有严格粒度解析结果。"
        )
        return {
            "标题": f"{self.display_name} {self.grit_designation}",
            "磨料": self.display_name,
            "粒度": self.grit_designation,
            "粒度路径": self.resolution_path,
            "代表粒径": f"{float(resolved['representative_diameter_d50_m']) * 1.0e6:.3g} μm",
            "粒径下限": f"{float(resolved['diameter_lower_m']) * 1.0e6:.3g} μm",
            "粒径上限": f"{float(resolved['diameter_upper_m']) * 1.0e6:.3g} μm",
            "数据来源": str(resolved.get("source", "")),
            "数据状态": data_status,
            "代表粒径说明": representative_note,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "wheel_schema_version": WHEEL_SCHEMA_VERSION,
            "abrasive_material": self.abrasive_material,
            "display_name": self.display_name,
            "resolution_path": self.resolution_path,
            "grit_designation": self.grit_designation,
            "resolved_specification": self.resolved_specification,
            "status": self.status,
            **self.extra_fields,
        }


@dataclass(frozen=True)
class ProcessDisplayInput:
    wheel_surface_speed_m_per_s: str = "30"
    workpiece_feed_speed_m_per_s: str = "0.1"
    grinding_width_mm: str = "10"
    specific_grinding_energy_j_per_mm3: str = "16.1"
    normal_to_tangential_force_ratio: str = "2.5"
    calibration_id: str = "jiang2023_alumina_17crni2movnb_fig4_20um"
    calibration_status: str = "literature_reference"
    source_description: str = (
        "Jiang 等，Materials 2023，DOI 10.3390/ma16041720，图4氧化铝砂轮工况图读 Ft≈80 N、Fn≈200 N；反算 us≈16.1 J/mm³，Fn/Ft≈2.5。"
    )
    applicability_notes: str = (
        "公开实验参考：17CrNi2MoVNb 钢、氧化铝 150–180 μm、vs=26 m/s、vw=0.19 m/s、ae=20 μm、宽度34 mm、砂轮直径350 mm；图读和跨工况泛化均有不确定性。"
    )

    def build_case(
        self, geometry: object, wheel: WorkbenchWheelCase
    ) -> "WorkbenchProcessCase":
        geometry_data = _mapping(geometry, "几何引用")
        wheel_data = _mapping(geometry_data.get("wheel"), "几何砂轮参数")
        process_data = _mapping(geometry_data.get("process"), "几何工艺参数")
        _positive_si(wheel_data.get("diameter_m"), "几何中的砂轮直径必须为有限正数。")
        _positive_si(process_data.get("depth_of_cut_m"), "几何中的磨削深度必须为有限正数。")
        if not isinstance(wheel, WorkbenchWheelCase):
            raise ProcessValidationError("粒度尚未成功解析。")
        wheel.validate_wheel()
        calibration_id = _text(self.calibration_id, "经验参数编号不能为空。")
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ProcessValidationError("标定状态无效。")
        source = self.source_description.strip() if isinstance(self.source_description, str) else ""
        if self.calibration_status in {
            "literature_reference",
            "experimentally_calibrated",
        } and not source:
            raise ProcessValidationError("文献参考或实验标定必须填写来源说明。")
        return WorkbenchProcessCase(
            wheel_surface_speed_m_per_s=_positive_display(
                self.wheel_surface_speed_m_per_s, "砂轮线速度必须为有限正数。"
            ),
            workpiece_feed_speed_m_per_s=_positive_display(
                self.workpiece_feed_speed_m_per_s, "工件进给速度必须为有限正数。"
            ),
            grinding_width_m=_positive_display(
                self.grinding_width_mm, "磨削宽度必须为有限正数。"
            )
            * 1.0e-3,
            specific_grinding_energy_j_per_m3=_positive_display(
                self.specific_grinding_energy_j_per_mm3,
                "比磨削能必须为有限正数。",
            )
            * 1.0e9,
            normal_to_tangential_force_ratio=_positive_display(
                self.normal_to_tangential_force_ratio,
                "法向/切向力比必须为有限正数。",
            ),
            calibration_id=calibration_id,
            calibration_status=self.calibration_status,
            source_description=source,
            applicability_notes=(
                self.applicability_notes.strip()
                if isinstance(self.applicability_notes, str)
                else ""
            ),
            geometry_reference={
                "geometry_fingerprint": geometry_fingerprint(geometry_data),
                "geometry_schema_version": geometry_data.get("geometry_schema_version"),
                "geometry_source": geometry_data.get("source"),
            },
        )


@dataclass(frozen=True)
class WorkbenchProcessCase:
    wheel_surface_speed_m_per_s: float
    workpiece_feed_speed_m_per_s: float
    grinding_width_m: float
    specific_grinding_energy_j_per_m3: float
    normal_to_tangential_force_ratio: float
    calibration_id: str
    calibration_status: str
    source_description: str
    applicability_notes: str
    geometry_reference: dict[str, Any]
    status: str = "completed"
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, message in (
            (self.wheel_surface_speed_m_per_s, "砂轮线速度必须为有限正数。"),
            (self.workpiece_feed_speed_m_per_s, "工件进给速度必须为有限正数。"),
            (self.grinding_width_m, "磨削宽度必须为有限正数。"),
            (self.specific_grinding_energy_j_per_m3, "比磨削能必须为有限正数。"),
            (self.normal_to_tangential_force_ratio, "法向/切向力比必须为有限正数。"),
        ):
            _positive_si(value, message)
        _text(self.calibration_id, "经验参数编号不能为空。")
        if self.calibration_status not in CALIBRATION_STATUSES:
            raise ProcessValidationError("标定状态无效。")
        if self.calibration_status in {
            "literature_reference",
            "experimentally_calibrated",
        } and not self.source_description.strip():
            raise ProcessValidationError("文献参考或实验标定必须填写来源说明。")
        reference = _mapping(self.geometry_reference, "几何引用")
        if set(reference) != {
            "geometry_fingerprint",
            "geometry_schema_version",
            "geometry_source",
        }:
            raise ProcessValidationError("几何引用字段不完整。")
        if self.status != "completed":
            raise ProcessValidationError("工艺状态必须为 completed。")

    @classmethod
    def from_mapping(cls, payload: object) -> "WorkbenchProcessCase":
        data = _mapping(payload, "工艺数据")
        missing = sorted(_PROCESS_FIELDS - set(data))
        if missing:
            raise ProcessValidationError(f"工艺数据缺少必需字段：{', '.join(missing)}")
        if type(data["process_schema_version"]) is not int or data["process_schema_version"] != 1:
            raise ProcessValidationError("不支持的工艺文件版本。")
        return cls(
            wheel_surface_speed_m_per_s=data["wheel_surface_speed_m_per_s"],
            workpiece_feed_speed_m_per_s=data["workpiece_feed_speed_m_per_s"],
            grinding_width_m=data["grinding_width_m"],
            specific_grinding_energy_j_per_m3=data["specific_grinding_energy_J_per_m3"],
            normal_to_tangential_force_ratio=data["normal_to_tangential_force_ratio"],
            calibration_id=data["calibration_id"],
            calibration_status=data["calibration_status"],
            source_description=data["source_description"],
            applicability_notes=data["applicability_notes"],
            geometry_reference=dict(_mapping(data["geometry_reference"], "几何引用")),
            status=data["status"],
            extra_fields={key: value for key, value in data.items() if key not in _PROCESS_FIELDS},
        )

    def validate_process(self) -> tuple[bool, str]:
        return True, "磨削工艺参数有效。"

    def to_process_schema(self) -> dict[str, Any]:
        return self.to_dict()

    def process_summary(self) -> dict[str, str]:
        return {
            "砂轮线速度": f"{self.wheel_surface_speed_m_per_s:g} m/s",
            "工件进给速度": f"{self.workpiece_feed_speed_m_per_s:g} m/s",
            "磨削宽度": f"{self.grinding_width_m * 1.0e3:g} mm",
            "比磨削能": f"{self.specific_grinding_energy_j_per_m3 / 1.0e9:g} J/mm³",
            "法向/切向力比": f"{self.normal_to_tangential_force_ratio:g}",
            "经验参数编号": self.calibration_id,
            "标定状态": self.calibration_status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_schema_version": PROCESS_SCHEMA_VERSION,
            "wheel_surface_speed_m_per_s": self.wheel_surface_speed_m_per_s,
            "workpiece_feed_speed_m_per_s": self.workpiece_feed_speed_m_per_s,
            "grinding_width_m": self.grinding_width_m,
            "specific_grinding_energy_J_per_m3": self.specific_grinding_energy_j_per_m3,
            "normal_to_tangential_force_ratio": self.normal_to_tangential_force_ratio,
            "calibration_id": self.calibration_id,
            "calibration_status": self.calibration_status,
            "source_description": self.source_description,
            "applicability_notes": self.applicability_notes,
            "geometry_reference": self.geometry_reference,
            "status": self.status,
            **self.extra_fields,
        }
