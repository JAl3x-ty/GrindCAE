"""Display-unit form data and strict conversion to existing backend cases."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math

from grindcae.evolved_fem import EvolvedFemCase
from grindcae.mechanism_history_pass import MechanismHistoryPassCase
from grindcae.pass_scan import PassScanCase
from .runtime_paths import (
    default_mechanism_history_output_text,
    default_scan_output_text,
    default_single_output_text,
)



SINGLE_MODE = "single"
SCAN_MODE = "scan"
MECHANISM_HISTORY_MODE = "mechanism_history"
GUI_MODES = (SINGLE_MODE, SCAN_MODE, MECHANISM_HISTORY_MODE)


class GuiInputError(ValueError):
    """Raised before a worker starts when GUI input cannot form a valid case."""


def _number(value: str, label: str) -> float:
    try:
        number = float(value.strip())
    except (AttributeError, ValueError) as exc:
        raise GuiInputError(f"{label}：请输入有限数字。") from exc
    if not math.isfinite(number):
        raise GuiInputError(f"{label}：不允许 NaN 或无穷值。")
    return number


def _positive(value: str, label: str) -> float:
    number = _number(value, label)
    if number <= 0.0:
        raise GuiInputError(f"{label}：必须大于 0。")
    return number


def _strict_integer(value: str, label: str, minimum: int, maximum: int) -> int:
    normalized = value.strip() if isinstance(value, str) else ""
    if not normalized or normalized[0] in "+-" and not normalized[1:].isdigit():
        raise GuiInputError(f"{label}：请输入 {minimum} 到 {maximum} 的严格整数。")
    digits = normalized[1:] if normalized[:1] in "+-" else normalized
    if not digits.isdigit():
        raise GuiInputError(f"{label}：请输入 {minimum} 到 {maximum} 的严格整数。")
    number = int(normalized)
    if not minimum <= number <= maximum:
        raise GuiInputError(f"{label}：必须在 {minimum} 到 {maximum} 之间。")
    return number


def _direction(value: str, label: str) -> str:
    if value not in ("positive_x", "negative_x"):
        raise GuiInputError(f"{label}：必须为 positive_x 或 negative_x。")
    return value


@dataclass(frozen=True, slots=True)
class GuiForm:
    """Editable strings in common engineering units, independent of Tk widgets."""

    mode: str = SINGLE_MODE
    workpiece_length_mm: str = "100"
    workpiece_height_mm: str = "20"
    wheel_diameter_mm: str = "200"
    depth_of_cut_um: str = "20"
    wheel_surface_speed_m_per_s: str = "30"
    workpiece_feed_speed_m_per_s: str = "0.1"
    grinding_width_and_thickness_mm: str = "10"
    wheel_lowest_point_x_mm: str = "50"
    mesh_target_size_mm: str = "2"
    elastic_modulus_GPa: str = "210"
    poisson_ratio: str = "0.3"
    specific_grinding_energy_J_per_mm3: str = "16.1"
    normal_to_tangential_force_ratio: str = "2.5"
    relative_feed_direction: str = "positive_x"
    tangential_force_direction: str = "positive_x"
    scan_base_position_count: str = "21"
    trajectory_base_point_count: str = "401"
    calibration_id: str = "jiang2023_alumina_17crni2movnb_fig4_20um"
    specific_grinding_energy_source: str = (
        "Jiang 等，Materials 2023，DOI 10.3390/ma16041720，图4氧化铝砂轮工况图读 Ft≈80 N 后反算 us≈16.1 J/mm³。"
    )
    force_ratio_source: str = (
        "同一公开工况图读 Ft≈80 N、Fn≈200 N，取 Fn/Ft≈2.5。"
    )
    applicability_notes: str = (
        "公开实验参考：17CrNi2MoVNb 钢、氧化铝 150–180 μm、vs=26 m/s、vw=0.19 m/s、ae=20 μm、宽度34 mm、砂轮直径350 mm；图读和跨工况泛化均有不确定性。"
    )
    mechanism_history_json: str = ""
    output_directory: str = field(default_factory=default_single_output_text)

    def __post_init__(self) -> None:
        if self.mode not in GUI_MODES:
            raise GuiInputError("计算模式无效。")

    def with_mode(self, mode: str) -> GuiForm:
        if mode not in GUI_MODES:
            raise GuiInputError("计算模式无效。")
        default_output = {
            SINGLE_MODE: default_single_output_text(),
            SCAN_MODE: default_scan_output_text(),
            MECHANISM_HISTORY_MODE: default_mechanism_history_output_text(),
        }[mode]
        current_defaults = {
            "outputs/gui_single_position",
            "outputs/gui_pass_scan",
            "outputs/gui_mechanism_history",
            default_single_output_text(),
            default_scan_output_text(),
            default_mechanism_history_output_text(),
        }
        output = default_output if self.output_directory in current_defaults else self.output_directory
        return replace(self, mode=mode, output_directory=output)

    def to_evolved_fem_case(self) -> EvolvedFemCase:
        """Convert display units once, then delegate all strict model checks."""

        # 中文导读：界面常用单位在此统一转为 SI，核心模型内部只处理 SI。
        length_m = _positive(self.workpiece_length_mm, "工件长度 [mm]") * 1.0e-3
        height_m = _positive(self.workpiece_height_mm, "工件高度 [mm]") * 1.0e-3
        diameter_m = _positive(self.wheel_diameter_mm, "砂轮直径 [mm]") * 1.0e-3
        depth_m = _positive(self.depth_of_cut_um, "磨削深度 [um]") * 1.0e-6
        wheel_speed = _positive(self.wheel_surface_speed_m_per_s, "砂轮线速度 [m/s]")
        feed_speed = _positive(self.workpiece_feed_speed_m_per_s, "工件进给速度 [m/s]")
        width_m = _positive(
            self.grinding_width_and_thickness_mm,
            "磨削宽度/分析厚度 [mm]",
        ) * 1.0e-3
        position_m = _number(self.wheel_lowest_point_x_mm, "砂轮最低点 x [mm]") * 1.0e-3
        target_size_m = _positive(self.mesh_target_size_mm, "网格目标尺寸 [mm]") * 1.0e-3
        elastic_modulus_pa = _positive(self.elastic_modulus_GPa, "弹性模量 E [GPa]") * 1.0e9
        poisson_ratio = _number(self.poisson_ratio, "泊松比 nu")
        if not -1.0 < poisson_ratio < 0.5:
            raise GuiInputError("泊松比 nu：必须满足 -1 < nu < 0.5。")
        specific_energy = _positive(
            self.specific_grinding_energy_J_per_mm3,
            "比磨削能 [J/mm^3]",
        ) * 1.0e9
        force_ratio = _positive(
            self.normal_to_tangential_force_ratio,
            "法向/切向力比 Fn/Ft",
        )
        trajectory_points = _strict_integer(
            self.trajectory_base_point_count,
            "轨迹基础采样点数",
            101,
            10001,
        )
        feed_direction = _direction(self.relative_feed_direction, "轨迹方向")
        force_direction = _direction(self.tangential_force_direction, "切向力方向")
        calibration_text = {
            "calibration_id": self.calibration_id.strip(),
            "specific_grinding_energy_source": self.specific_grinding_energy_source.strip(),
            "force_ratio_source": self.force_ratio_source.strip(),
            "applicability_notes": self.applicability_notes.strip(),
        }
        for field, value in calibration_text.items():
            if not value:
                raise GuiInputError(f"经验参数来源字段 {field} 不能为空。")

        payload = {
            "evolved_fem_schema_version": 1,
            "unit_system": "SI",
            "model_type": "single_pass_evolved_surface_empirical_load_plane_stress",
            "pass_load": {
                "pass_load_schema_version": 1,
                "unit_system": "SI",
                "model_type": "single_pass_contact_ratio_empirical_load",
                "trajectory": {
                    "trajectory_schema_version": 1,
                    "unit_system": "SI",
                    "model_type": "single_pass_ideal_wheel_profile",
                    "workpiece": {
                        "length_m": length_m,
                        "original_surface_height_m": height_m,
                    },
                    "wheel": {"diameter_m": diameter_m},
                    "single_pass": {
                        "depth_of_cut_m": depth_m,
                        "relative_feed_direction": feed_direction,
                        "wheel_lowest_point_x_m": position_m,
                    },
                    "sampling": {"base_point_count": trajectory_points},
                },
                "force_model": {
                    "force_model_schema_version": 1,
                    "unit_system": "SI",
                    "model_type": "specific_grinding_energy_force_ratio",
                    "process": {
                        "wheel_surface_speed_m_per_s": wheel_speed,
                        "workpiece_feed_speed_m_per_s": feed_speed,
                        "depth_of_cut_m": depth_m,
                        "grinding_width_m": width_m,
                        "wheel_diameter_m": diameter_m,
                    },
                    "calibration": {
                        "specific_grinding_energy_J_per_m3": specific_energy,
                        "normal_to_tangential_force_ratio": force_ratio,
                        **calibration_text,
                    },
                },
                "force_mapping": {
                    "boundary": "ideal_top_surface",
                    "distribution": "uniform_over_effective_projected_interval",
                    "tangential_force_direction": force_direction,
                },
            },
            "material": {"E": elastic_modulus_pa, "nu": poisson_ratio},
            "mesh": {"target_size": target_size_m},
            "analysis": {"type": "plane_stress", "thickness": width_m},
            "boundaries": [
                {"side": "bottom", "role": "fixed"},
                {"side": "top", "role": "contact"},
                {"side": "left", "role": "free"},
                {"side": "right", "role": "free"},
            ],
        }
        try:
            # 中文导读：界面先做友好校验，最终仍由严格 Schema 构造器确认完整模型合同。
            return EvolvedFemCase.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise GuiInputError(f"单位置线弹性输入未通过严格校验：{exc}") from exc

    def to_pass_scan_case(self) -> PassScanCase:
        base_count = _strict_integer(
            self.scan_base_position_count,
            "扫描基础位置数",
            5,
            101,
        )
        payload = {
            "scan_schema_version": 1,
            "unit_system": "SI",
            "model_type": "single_pass_quasi_static_evolved_fem_scan",
            "reference_case": self.to_evolved_fem_case().to_dict(),
            "scan": {
                "range": "complete_single_pass",
                "base_position_count": base_count,
            },
        }
        try:
            return PassScanCase.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise GuiInputError(f"完整单程线弹性输入未通过严格校验：{exc}") from exc

    def to_mechanism_history_case(self) -> MechanismHistoryPassCase:
        if not self.mechanism_history_json.strip():
            raise GuiInputError("请先加载合法的机制化完整单程 JSON 配置。")
        try:
            payload = json.loads(self.mechanism_history_json)
        except json.JSONDecodeError as exc:
            raise GuiInputError(
                f"机制化完整单程 JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列，{exc.msg}。"
            ) from exc
        if not isinstance(payload, dict) or "mechanism_history_pass_schema_version" not in payload:
            raise GuiInputError(
                "机制化弹塑性完整单程模式只接受包含 "
                "mechanism_history_pass_schema_version 的合法机制化完整单程 JSON。"
            )
        try:
            return MechanismHistoryPassCase.from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise GuiInputError(f"配置未通过机制化完整单程严格 Schema 校验：{exc}") from exc

    def build_case(self) -> EvolvedFemCase | PassScanCase | MechanismHistoryPassCase:
        if self.mode == SINGLE_MODE:
            return self.to_evolved_fem_case()
        if self.mode == SCAN_MODE:
            return self.to_pass_scan_case()
        return self.to_mechanism_history_case()

    @classmethod
    def from_evolved_fem_case(
        cls,
        case: EvolvedFemCase,
        *,
        mode: str = SINGLE_MODE,
        output_directory: str | None = None,
        scan_base_position_count: int = 21,
    ) -> GuiForm:
        trajectory = case.pass_load.trajectory
        process = case.pass_load.force_model.process
        calibration = case.pass_load.force_model.calibration
        return cls(
            mode=mode,
            workpiece_length_mm=repr(trajectory.workpiece.length_m / 1.0e-3),
            workpiece_height_mm=repr(trajectory.workpiece.original_surface_height_m / 1.0e-3),
            wheel_diameter_mm=repr(trajectory.wheel.diameter_m / 1.0e-3),
            depth_of_cut_um=repr(trajectory.single_pass.depth_of_cut_m / 1.0e-6),
            wheel_surface_speed_m_per_s=f"{process.wheel_surface_speed_m_per_s:.15g}",
            workpiece_feed_speed_m_per_s=f"{process.workpiece_feed_speed_m_per_s:.15g}",
            grinding_width_and_thickness_mm=repr(process.grinding_width_m / 1.0e-3),
            wheel_lowest_point_x_mm=repr(trajectory.single_pass.wheel_lowest_point_x_m / 1.0e-3),
            mesh_target_size_mm=repr(case.mesh.target_size / 1.0e-3),
            elastic_modulus_GPa=repr(case.material.E / 1.0e9),
            poisson_ratio=f"{case.material.nu:.15g}",
            specific_grinding_energy_J_per_mm3=(
                f"{calibration.specific_grinding_energy_J_per_m3 * 1.0e-9:.15g}"
            ),
            normal_to_tangential_force_ratio=(
                f"{calibration.normal_to_tangential_force_ratio:.15g}"
            ),
            relative_feed_direction=trajectory.single_pass.relative_feed_direction,
            tangential_force_direction=case.pass_load.force_mapping.tangential_force_direction,
            scan_base_position_count=str(scan_base_position_count),
            trajectory_base_point_count=str(trajectory.sampling.base_point_count),
            calibration_id=calibration.calibration_id,
            specific_grinding_energy_source=calibration.specific_grinding_energy_source,
            force_ratio_source=calibration.force_ratio_source,
            applicability_notes=calibration.applicability_notes,
            output_directory=output_directory or default_single_output_text(),
        )

    @classmethod
    def from_pass_scan_case(
        cls,
        case: PassScanCase,
        *,
        output_directory: str | None = None,
    ) -> GuiForm:
        return cls.from_evolved_fem_case(
            case.reference_case,
            mode=SCAN_MODE,
            output_directory=output_directory or default_scan_output_text(),
            scan_base_position_count=case.scan.base_position_count,
        )

    @classmethod
    def from_mechanism_history_case(
        cls,
        case: MechanismHistoryPassCase,
        *,
        output_directory: str | None = None,
    ) -> GuiForm:
        serialized = json.dumps(
            case.to_dict(),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        return cls(
            mode=MECHANISM_HISTORY_MODE,
            mechanism_history_json=serialized,
            output_directory=(
                output_directory or default_mechanism_history_output_text()
            ),
        )
