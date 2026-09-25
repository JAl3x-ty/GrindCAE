"""Public transactional workflow for Phase 7A.2."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .core import StatisticalGrainLoadPrediction, predict_statistical_grain_load
from .exporters import (
    write_distribution_csv,
    write_distribution_png,
    write_groups_csv,
    write_json,
)
from .models import (
    GrainPopulationSettings,
    LoadDistributionSettings,
    StatisticalGrainLoadCase,
)


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "resolved_grit_specification_json": "resolved_grit_specification.json",
    "equivalent_grain_groups_csv": "equivalent_grain_groups.csv",
    "mechanism_load_distribution_csv": "mechanism_load_distribution.csv",
    "mechanism_load_distribution_png": "mechanism_load_distribution.png",
}


class StatisticalGrainLoadWorkflowError(RuntimeError):
    """Raised when a complete statistical-load result cannot be published."""


def recommended_grain_population_settings() -> GrainPopulationSettings:
    return GrainPopulationSettings(
        active_density_factor=0.002,
        minimum_equivalent_group_count=16,
        maximum_equivalent_group_count=64,
        random_seed=5000,
    )


def recommended_load_distribution_settings() -> LoadDistributionSettings:
    return LoadDistributionSettings(
        point_count=801,
        kernel_width_factor=2.0,
        rubbing_kernel_scale=1.4,
        ploughing_kernel_scale=1.0,
        cutting_kernel_scale=0.7,
    )


def build_summary(result: StatisticalGrainLoadPrediction, artifact_paths: dict[str, str]) -> dict[str, object]:
    return result.to_dict(artifact_paths)


def artifact_paths(output_directory: str | Path) -> dict[str, Path]:
    directory = Path(output_directory).expanduser().resolve()
    return {key: directory / filename for key, filename in ARTIFACT_FILENAMES.items()}


def generate_artifacts(
    result: StatisticalGrainLoadPrediction,
    output_directory: str | Path,
    published_artifact_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    paths = artifact_paths(output_directory)
    display_paths = published_artifact_paths or paths
    write_json(result.resolved_grit_specification.to_dict(), paths["resolved_grit_specification_json"])
    write_groups_csv(result, paths["equivalent_grain_groups_csv"])
    write_distribution_csv(result, paths["mechanism_load_distribution_csv"])
    write_distribution_png(result, paths["mechanism_load_distribution_png"])
    write_json(
        build_summary(result, {key: str(path) for key, path in display_paths.items()}),
        paths["summary_json"],
    )
    validate_artifacts(paths)
    return paths


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise StatisticalGrainLoadWorkflowError("artifact set is invalid")
    for key, path in artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise StatisticalGrainLoadWorkflowError(f"{key}: missing or empty artifact")
    for key in ("summary_json", "resolved_grit_specification_json"):
        with artifacts[key].open(encoding="utf-8") as stream:
            json.load(stream)
    for key in ("equivalent_grain_groups_csv", "mechanism_load_distribution_csv"):
        with artifacts[key].open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise StatisticalGrainLoadWorkflowError(f"{key}: must contain data rows")
        for row in rows:
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise StatisticalGrainLoadWorkflowError(f"{key}: contains non-finite values")
    if artifacts["mechanism_load_distribution_png"].read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise StatisticalGrainLoadWorkflowError(
            "mechanism_load_distribution_png: invalid PNG signature"
        )


def _stage_copy(source: Path, destination: Path, label: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.parent / f".{destination.name}.{label}-{uuid.uuid4().hex}.tmp"
    shutil.copyfile(source, staged)
    return staged


def publish_artifacts(temporary: dict[str, Path], final: dict[str, Path]) -> None:
    if set(temporary) != set(final) or set(final) != set(ARTIFACT_FILENAMES):
        raise StatisticalGrainLoadWorkflowError("publication artifact set is invalid")
    validate_artifacts(temporary)
    output_directory = next(iter(final.values())).parent
    existed = output_directory.exists()
    output_directory.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in order:
            staged[key] = _stage_copy(temporary[key], final[key], "new")
        for key in order:
            if final[key].exists():
                backups[key] = _stage_copy(final[key], final[key], "backup")
        for key in order:
            os.replace(staged.pop(key), final[key])
            published.append(key)
    except (OSError, StatisticalGrainLoadWorkflowError) as exc:
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups.pop(key), final[key])
                else:
                    final[key].unlink(missing_ok=True)
            except OSError:
                pass
        raise StatisticalGrainLoadWorkflowError(f"artifact publication failed: {exc}") from exc
    finally:
        for path in (*staged.values(), *backups.values()):
            path.unlink(missing_ok=True)
        if not existed:
            try:
                output_directory.rmdir()
            except OSError:
                pass


def run_statistical_grain_load(
    case: StatisticalGrainLoadCase, output_directory: str | Path
) -> tuple[StatisticalGrainLoadPrediction, dict[str, Path], dict[str, object]]:
    """Calculate, validate, and transactionally publish one complete result."""
    final = artifact_paths(output_directory)
    result = predict_statistical_grain_load(case)
    with tempfile.TemporaryDirectory(prefix="grindcae-7a2-") as temporary_directory:
        temporary = generate_artifacts(result, temporary_directory, final)
        publish_artifacts(temporary, final)
    validate_artifacts(final)
    with final["summary_json"].open(encoding="utf-8") as stream:
        summary = json.load(stream)
    return result, final, summary
