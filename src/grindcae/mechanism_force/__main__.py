"""Transactional CLI for Phase 7A.1 mechanism force decomposition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from .core import RESULT_FORMAT, MechanismForceError, MechanismForcePrediction, predict_mechanism_force
from .exporters import CSV_FIELDS, write_fraction_csv, write_fraction_png
from .models import MechanismForceCase, MechanismForceValidationError
from grindcae.force_model import ForceModelError, ForceModelValidationError
from .workflow import build_summary


ARTIFACT_FILENAMES = {
    "summary_json": "summary.json",
    "mechanism_fractions_csv": "mechanism_fractions.csv",
    "mechanism_fractions_png": "mechanism_fractions.png",
}


class MechanismForceWorkflowError(RuntimeError):
    """Raised when the Phase 7A.1 artifact set cannot be safely published."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split public Ft/Fn totals into rubbing, ploughing, and cutting "
            "fractions for ductile-metal demonstration parameters. No FEM or history workflow is run."
        )
    )
    parser.add_argument("case", help="path to a schema version 1 UTF-8 JSON file")
    parser.add_argument("--output-dir", required=True, help="directory receiving the three mechanism-fraction artifacts")
    return parser


def artifact_paths(output_dir: str | Path) -> dict[str, Path]:
    directory = Path(output_dir).expanduser().resolve()
    return {key: directory / filename for key, filename in ARTIFACT_FILENAMES.items()}


def _serialize_summary(summary: dict[str, object]) -> str:
    try:
        return json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise MechanismForceWorkflowError(f"summary serialization failed: {exc}") from exc


def generate_artifacts(
    result: MechanismForcePrediction,
    output_dir: str | Path,
    *,
    published_artifact_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    artifacts = artifact_paths(output_dir)
    artifacts["summary_json"].parent.mkdir(parents=True, exist_ok=True)
    write_fraction_csv(result, artifacts["mechanism_fractions_csv"])
    write_fraction_png(result, artifacts["mechanism_fractions_png"])
    summary_paths = published_artifact_paths or artifacts
    path_strings = {key: str(path) for key, path in summary_paths.items()}
    artifacts["summary_json"].write_text(
        _serialize_summary(build_summary(result, path_strings)), encoding="utf-8", newline="\n"
    )
    validate_artifacts(artifacts)
    return artifacts


def validate_artifacts(artifacts: dict[str, Path]) -> None:
    if set(artifacts) != set(ARTIFACT_FILENAMES):
        raise MechanismForceWorkflowError("artifact set does not match the Phase 7A.1 contract")
    for key, path in artifacts.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise MechanismForceWorkflowError(f"artifact [{key}] is missing or empty")
    try:
        serialized = artifacts["summary_json"].read_text(encoding="utf-8")
        if "NaN" in serialized or "Infinity" in serialized:
            raise ValueError("summary contains non-finite JSON tokens")
        summary = json.loads(serialized)
        if summary.get("result_format") != RESULT_FORMAT:
            raise ValueError("summary result_format is invalid")
        with artifacts["mechanism_fractions_csv"].open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("mechanism CSV header is invalid")
        if not rows:
            raise ValueError("mechanism CSV has no rows")
        eta_values = [float(row["eta"]) for row in rows]
        if eta_values != sorted(eta_values) or eta_values[0] > 0.1 or eta_values[-1] < 10.0:
            raise ValueError("mechanism CSV eta coverage is invalid")
        for row in rows:
            numbers = [float(row[field]) for field in CSV_FIELDS]
            if not all(math.isfinite(value) for value in numbers):
                raise ValueError("mechanism CSV contains non-finite values")
            if not math.isclose(float(row["fraction_sum"]), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError("mechanism CSV fractions are not normalized")
        from matplotlib import image as matplotlib_image

        pixels = matplotlib_image.imread(artifacts["mechanism_fractions_png"])
        if pixels.size == 0 or pixels.ndim not in (2, 3):
            raise ValueError("mechanism PNG is not decodable")
        if not math.isfinite(float(pixels.min())) or not math.isfinite(float(pixels.max())):
            raise ValueError("mechanism PNG contains non-finite pixels")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MechanismForceWorkflowError(f"artifact validation failed: {exc}") from exc


def _stage_copy(source: Path, destination: Path, label: str) -> Path:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.{label}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary_path = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
            stream.flush()
            os.fsync(stream.fileno())
        return temporary_path
    except OSError as exc:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
        raise MechanismForceWorkflowError(f"artifact [{destination.name}] staging failed: {exc}") from exc


def publish_artifacts(
    temporary_artifacts: dict[str, Path], final_artifacts: dict[str, Path]
) -> None:
    if set(temporary_artifacts) != set(final_artifacts) or set(final_artifacts) != set(ARTIFACT_FILENAMES):
        raise MechanismForceWorkflowError("artifact publication set is invalid")
    validate_artifacts(temporary_artifacts)
    output_dir = final_artifacts["summary_json"].parent
    directory_existed = output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    order = [key for key in ARTIFACT_FILENAMES if key != "summary_json"] + ["summary_json"]
    staged: dict[str, Path] = {}
    backups: dict[str, Path] = {}
    published: list[str] = []
    try:
        for key in order:
            staged[key] = _stage_copy(temporary_artifacts[key], final_artifacts[key], "new")
        for key in order:
            if final_artifacts[key].exists():
                backups[key] = _stage_copy(final_artifacts[key], final_artifacts[key], "backup")
        for key in order:
            os.replace(staged[key], final_artifacts[key])
            del staged[key]
            published.append(key)
    except (OSError, MechanismForceWorkflowError) as exc:
        rollback_errors: list[str] = []
        for key in reversed(published):
            try:
                if key in backups:
                    os.replace(backups[key], final_artifacts[key])
                    del backups[key]
                else:
                    final_artifacts[key].unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"{final_artifacts[key].name} rollback failed: {rollback_exc}")
        detail = f"artifact publication failed: {exc}"
        if rollback_errors:
            detail += "; " + "; ".join(rollback_errors)
        raise MechanismForceWorkflowError(detail) from exc
    finally:
        for temporary_path in (*staged.values(), *backups.values()):
            if temporary_path.exists():
                temporary_path.unlink()
        if not directory_existed and output_dir.exists():
            try:
                output_dir.rmdir()
            except OSError:
                pass


def _validate_paths(input_path: Path, output_dir: Path) -> None:
    if output_dir == input_path:
        raise MechanismForceValidationError(
            "output_directory [directory]: --output-dir must not be the input file"
        )


def _print_result(input_path: Path, result: MechanismForcePrediction, artifacts: dict[str, Path]) -> None:
    phase4a = result.phase4a_prediction
    print(f"Input file: {input_path}")
    print(
        "Equivalent process thickness h_eq: "
        f"{result.equivalent_process_thickness_m:.12g} m; eta={result.eta:.12g}"
    )
    print(
        "Steady empirical total Ft/Fn: "
        f"{phase4a.tangential_force_magnitude_N:.12g} N / "
        f"{phase4a.normal_force_magnitude_N:.12g} N"
    )
    print(
        "Mechanism fractions rubbing/ploughing/cutting: "
        f"{result.fractions.rubbing:.12g} / {result.fractions.ploughing:.12g} / {result.fractions.cutting:.12g}"
    )
    print(
        "Force residuals Ft/Fn: "
        f"{result.tangential_force_conservation_residual_N:.12g} N / "
        f"{result.normal_force_conservation_residual_N:.12g} N"
    )
    print(f"Summary: {artifacts['summary_json']}")
    print(f"Manual acceptance figure: {artifacts['mechanism_fractions_png']}")
    print("Scope: mechanism decomposition only; no automatic grain-load chaining, FEM/history integration, GUI, or portable integration.")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        _validate_paths(input_path, output_dir)
        case = MechanismForceCase.from_json(input_path)
        result = predict_mechanism_force(case)
        final = artifact_paths(output_dir)
        with tempfile.TemporaryDirectory(prefix="grindcae-7a1-") as temporary_directory:
            temporary = generate_artifacts(
                result,
                Path(temporary_directory),
                published_artifact_paths=final,
            )
            publish_artifacts(temporary, final)
        validate_artifacts(final)
    except (
        MechanismForceValidationError,
        MechanismForceError,
        MechanismForceWorkflowError,
        ForceModelValidationError,
        ForceModelError,
    ) as exc:
        print(f"Mechanism force error: {exc}", file=sys.stderr)
        return 2
    _print_result(input_path, result, final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
