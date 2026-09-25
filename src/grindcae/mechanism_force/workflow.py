"""Summary helpers for Phase 7A.1."""

from __future__ import annotations

from .core import MechanismForcePrediction


def build_summary(
    result: MechanismForcePrediction, artifact_paths: dict[str, str]
) -> dict[str, object]:
    return result.to_dict(artifact_paths)
