"""Presentation-neutral progress and ETA estimates for formal analysis runs."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class AnalysisProgressEstimate:
    current: int
    total: int
    fraction: float
    percent: float
    elapsed_seconds: float
    remaining_seconds: float | None


def estimate_progress(
    *, current: int, total: int, elapsed_seconds: float
) -> AnalysisProgressEstimate:
    legal_total = max(1, int(total))
    legal_current = min(legal_total, max(0, int(current)))
    elapsed = max(0.0, float(elapsed_seconds))
    fraction = legal_current / legal_total
    remaining = (
        elapsed / legal_current * (legal_total - legal_current)
        if legal_current > 0
        else None
    )
    return AnalysisProgressEstimate(
        legal_current, legal_total, fraction, fraction * 100.0, elapsed, remaining
    )


def _format_duration(seconds: float) -> str:
    value = max(0, math.ceil(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, seconds_value = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分"
    if minutes:
        return f"{minutes}分{seconds_value:02d}秒"
    return f"{seconds_value}秒"


def format_progress_status(
    message: str, estimate: AnalysisProgressEstimate
) -> str:
    remaining = (
        _format_duration(estimate.remaining_seconds)
        if estimate.remaining_seconds is not None
        else "正在估算"
    )
    return (
        f"{message}\n"
        f"进度：{estimate.percent:.1f}%（{estimate.current}/{estimate.total}）  "
        f"已用：{_format_duration(estimate.elapsed_seconds)}  预计剩余：{remaining}"
    )


__all__ = ["AnalysisProgressEstimate", "estimate_progress", "format_progress_status"]
