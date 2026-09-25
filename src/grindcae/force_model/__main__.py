"""Command-line entry point for the independent phase 4A force model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
from pathlib import Path
import sys
import tempfile

from .core import ForceModelError, GrindingForcePrediction, predict_grinding_force
from .models import ForceModelValidationError, GrindingForceCase


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate grinding-force magnitudes from process parameters and "
            "user-supplied empirical calibration. No FEM workflow is run."
        )
    )
    parser.add_argument("case", help="path to a force-model schema version 1 JSON file")
    parser.add_argument(
        "--output",
        required=True,
        help="output path for the force-model JSON result",
    )
    return parser


def _output_path(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    if path.suffix.lower() != ".json":
        raise ForceModelValidationError(
            "output_file [.json]: --output must use the .json extension"
        )
    return path


def _serialize_result(prediction: GrindingForcePrediction) -> str:
    try:
        return (
            json.dumps(
                prediction.to_dict(),
                indent=2,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        )
    except (TypeError, ValueError) as exc:
        raise ForceModelError(
            f"output_payload [finite JSON]: serialization failed: {exc}"
        ) from exc


def _publish_result(serialized: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise ForceModelError(
            f"output_file [writable .json]: atomic publication failed: {exc}"
        ) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _print_summary(
    input_path: Path,
    output_path: Path,
    prediction: GrindingForcePrediction,
) -> None:
    print(f"Input file: {input_path}")
    print(f"Result file: {output_path}")
    print(
        "Material removal rate Qw: "
        f"{prediction.material_removal_rate_m3_per_s:.12g} m^3/s "
        f"({1.0e9 * prediction.material_removal_rate_m3_per_s:.12g} mm^3/s)"
    )
    print(f"Estimated grinding power P: {prediction.estimated_grinding_power_W:.12g} W")
    print(
        "Tangential force magnitude Ft: "
        f"{prediction.tangential_force_magnitude_N:.12g} N"
    )
    print(
        "Normal force magnitude Fn: "
        f"{prediction.normal_force_magnitude_N:.12g} N"
    )
    print(
        "Resultant force magnitude Fr: "
        f"{prediction.resultant_force_magnitude_N:.12g} N"
    )
    print(
        "Geometric contact length lg: "
        f"{prediction.geometric_contact_length_m:.12g} m "
        f"({1.0e3 * prediction.geometric_contact_length_m:.12g} mm)"
    )
    print(
        "Calibration warning: us and Rnt require experiment or reliable "
        "literature calibration for the applicable grinding process."
    )
    print(
        "Scope warning: these are force magnitudes only and are not connected "
        "to finite-element directions or loads."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    input_path = Path(args.case).expanduser().resolve()
    try:
        output_path = _output_path(args.output)
        force_case = GrindingForceCase.from_json(input_path)
        prediction = predict_grinding_force(force_case)
        serialized = _serialize_result(prediction)
        _publish_result(serialized, output_path)
    except (ForceModelValidationError, ForceModelError) as exc:
        print(f"Force model error: {exc}", file=sys.stderr)
        return 2

    _print_summary(input_path, output_path, prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
