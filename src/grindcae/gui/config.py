"""UTF-8 strict-schema configuration loading and saving for the retained phase 5A GUI."""

from __future__ import annotations

import json
from pathlib import Path

from grindcae.evolved_fem import EvolvedFemCase
from grindcae.mechanism_history_pass import MechanismHistoryPassCase
from grindcae.pass_scan import PassScanCase

from .form import (
    MECHANISM_HISTORY_MODE,
    GuiForm,
    GuiInputError,
    SCAN_MODE,
    SINGLE_MODE,
)


def load_form(path: str | Path) -> GuiForm:
    input_path = Path(path).expanduser().resolve()
    try:
        with input_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except FileNotFoundError as exc:
        raise GuiInputError(f"配置文件不存在：{input_path}") from exc
    except UnicodeDecodeError as exc:
        raise GuiInputError(f"配置文件必须为 UTF-8：{exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise GuiInputError(
            f"JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列，{exc.msg}。"
        ) from exc
    except OSError as exc:
        raise GuiInputError(f"无法读取配置文件：{exc}") from exc

    try:
        if isinstance(payload, dict) and "mechanism_history_pass_schema_version" in payload:
            return GuiForm.from_mechanism_history_case(
                MechanismHistoryPassCase.from_mapping(payload)
            )
        if isinstance(payload, dict) and "scan_schema_version" in payload:
            return GuiForm.from_pass_scan_case(PassScanCase.from_mapping(payload))
        if isinstance(payload, dict) and "evolved_fem_schema_version" in payload:
            return GuiForm.from_evolved_fem_case(EvolvedFemCase.from_mapping(payload))
    except (TypeError, ValueError) as exc:
        raise GuiInputError(f"配置未通过现有严格 Schema 校验：{exc}") from exc
    raise GuiInputError("仅支持单位置、完整单程扫描或机制化完整单程 JSON 配置。")


def save_form(form: GuiForm, path: str | Path) -> Path:
    output_path = Path(path).expanduser().resolve()
    case = form.build_case()
    payload = case.to_dict()
    try:
        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8", newline="\n")
    except (OSError, TypeError, ValueError) as exc:
        raise GuiInputError(f"无法保存 UTF-8 JSON：{exc}") from exc
    return output_path


def mode_label(mode: str) -> str:
    labels = {
        SINGLE_MODE: "单位置计算",
        SCAN_MODE: "完整单程扫描",
        MECHANISM_HISTORY_MODE: "机制化弹塑性完整单程",
    }
    return labels.get(mode, mode)
