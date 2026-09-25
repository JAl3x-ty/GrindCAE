"""User-facing names for stable GUI identifiers."""

DEMO_CALIBRATION_ID = "demo_unvalidated"
DEMO_CALIBRATION_DISPLAY_NAME = "演示参数集（未标定）"


def calibration_set_name(calibration_id: object) -> str:
    """Return a readable name without changing the persisted identifier."""

    value = str(calibration_id).strip()
    if value == DEMO_CALIBRATION_ID:
        return DEMO_CALIBRATION_DISPLAY_NAME
    return "用户自定义参数集"


def calibration_set_display_text(calibration_id: object) -> str:
    value = str(calibration_id).strip()
    if not value:
        return "尚未填写参数集 ID"
    return f"{calibration_set_name(value)}，ID：{value}"