"""Real 4C3B, 4C4, and mechanism field smoke tests for a portable build."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time

import matplotlib.image as matplotlib_image
import meshio

from grindcae import __version__
from grindcae.evolved_fem import EvolvedFemCase, artifact_paths, run_evolved_fem_case, validate_evolved_fem_artifacts
from grindcae.mechanism_history_pass import (
    MechanismHistoryPassCase,
    run_mechanism_elastoplastic_history_pass,
)
from grindcae.mechanism_history_pass.exporters import FILENAMES as MECHANISM_FILENAMES
from grindcae.mechanism_history_pass.field_evolution import (
    RESULT_FORMAT as MECHANISM_FIELD_EVOLUTION_RESULT_FORMAT,
    validate_mechanism_field_evolution,
)
from grindcae.pass_scan import (
    PassScanCase,
    aggregate_artifact_paths,
    export_pass_scan,
    publish_pass_scan_artifacts,
    run_pass_scan,
    validate_pass_scan_artifacts,
)


SINGLE_EXAMPLE = "single_pass_evolved_fem_middle.json"
SCAN_EXAMPLE = "single_pass_quasi_static_scan.json"
MECHANISM_EXAMPLE = "mechanism_field_evolution_smoke.json"
MECHANISM_RESULT_FORMAT = (
    "grindcae_phase_7a3_statistical_mechanism_elastoplastic_history_comparison"
)
MECHANISM_ROUTES = ("baseline", "mechanism", "difference")


def _bundled_example_path(filename: str) -> Path:
    base = Path(__file__).resolve().parent / "portable_examples"
    path = base / filename
    if not path.is_file():
        raise RuntimeError(f"bundled smoke input is missing: {path}")
    return path


def _decode_pngs(root: Path) -> int:
    paths = tuple(sorted(root.rglob("*.png")))
    if not paths:
        raise RuntimeError(f"no PNG artifacts found below {root}")
    for path in paths:
        pixels = matplotlib_image.imread(path)
        if pixels.size == 0:
            raise RuntimeError(f"PNG is empty: {path}")
    return len(paths)


def _read_mesh_artifacts(root: Path) -> tuple[int, int]:
    meshes = tuple(sorted(root.rglob("*.msh")))
    vtus = tuple(sorted(root.rglob("*.vtu")))
    if not meshes or not vtus:
        raise RuntimeError(f"MSH/VTU artifacts are missing below {root}")
    for path in (*meshes, *vtus):
        loaded = meshio.read(path)
        if len(loaded.points) <= 0:
            raise RuntimeError(f"mesh artifact has no nodes: {path}")
    return len(meshes), len(vtus)


def _read_vtu_artifacts(root: Path) -> int:
    vtus = tuple(sorted(root.rglob("*.vtu")))
    if not vtus:
        raise RuntimeError(f"VTU artifacts are missing below {root}")
    for path in vtus:
        loaded = meshio.read(path)
        if len(loaded.points) <= 0:
            raise RuntimeError(f"VTU artifact has no nodes: {path}")
    return len(vtus)

def _published_artifact_counts(single_output: Path, scan_output: Path) -> tuple[int, int, int]:
    png_count = _decode_pngs(single_output) + _decode_pngs(scan_output)
    single_msh, single_vtu = _read_mesh_artifacts(single_output)
    scan_msh, scan_vtu = _read_mesh_artifacts(scan_output)
    return png_count, single_msh + scan_msh, single_vtu + scan_vtu

def _single_position_report(result: object, output: Path) -> dict[str, object]:
    solver = result.solution.solver_summary
    balance = solver.balance_residual_norm
    if not math.isfinite(balance):
        raise RuntimeError("4C3B force-balance residual is not finite")
    return {
        "output_directory": str(output),
        "artifact_count": len(artifact_paths(output)),
        "node_count": result.evolved_mesh.node_count,
        "triangle_count": result.evolved_mesh.triangle_count,
        "active_facet_count": len(result.facet_load.facets),
        "force_balance_residual_norm_N": balance,
        "maximum_displacement_m": solver.maximum_displacement,
    }


def _print_progress(message: str) -> None:
    print(message, flush=True)


def _count_pvd_datasets(path: Path) -> int:
    from xml.etree import ElementTree as ET

    return len(ET.parse(path).getroot().findall(".//DataSet"))


def _require_within(root: Path, path: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise RuntimeError(f"smoke artifact escaped its output directory: {resolved_path}")


def _finite_nonzero_difference(root: Path, manifest: dict[str, object]) -> bool:
    fields = (
        ("point", "displacement_magnitude_difference_m"),
        ("cell", "von_mises_stress_difference_Pa"),
        ("cell", "equivalent_plastic_strain_difference"),
    )
    nonzero = False
    for frame in manifest["frames"]:
        mesh = meshio.read(root / frame["difference"]["vtu_path"])
        for location, name in fields:
            if location == "point":
                values = mesh.point_data[name]
            else:
                values = mesh.cell_data_dict[name]["triangle"]
            if not math.isfinite(float(values.min())) or not math.isfinite(float(values.max())):
                raise RuntimeError(f"difference VTU field is not finite: {name}")
            nonzero = nonzero or bool((abs(values) > 0.0).any())
    return nonzero


def _run_mechanism_field_evolution_smoke(output: Path) -> dict[str, object]:
    started = time.perf_counter()
    output = output.resolve()
    case = MechanismHistoryPassCase.from_json(_bundled_example_path(MECHANISM_EXAMPLE))
    _, artifacts, _ = run_mechanism_elastoplastic_history_pass(case, output)
    for path in artifacts.values():
        _require_within(output, path)
    if set(artifacts) != set(MECHANISM_FILENAMES):
        raise RuntimeError("mechanism history retained artifact set is invalid")
    if not all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values()):
        raise RuntimeError("mechanism history retained artifact is missing or empty")
    summary_text = artifacts["summary_json"].read_text(encoding="ascii")
    if "NaN" in summary_text or "Infinity" in summary_text:
        raise RuntimeError("mechanism history summary contains non-finite JSON")
    summary = json.loads(summary_text)
    if summary.get("result_format") != MECHANISM_RESULT_FORMAT:
        raise RuntimeError("mechanism history summary result_format is invalid")
    sequence_index = summary["field_evolution"]
    for path in (
        Path(sequence_index["directory"]),
        Path(sequence_index["manifest"]),
        Path(sequence_index["position_history_csv"]),
        *(Path(sequence_index["routes"][route]["pvd"]) for route in MECHANISM_ROUTES),
    ):
        _require_within(output, path)
    root = output / "field_evolution"
    _require_within(output, root)
    manifest = validate_mechanism_field_evolution(root)
    if manifest.get("result_format") != MECHANISM_FIELD_EVOLUTION_RESULT_FORMAT:
        raise RuntimeError("mechanism field-evolution manifest format is invalid")
    frame_count = int(manifest["frame_count"])
    route_counts: dict[str, tuple[int, int]] = {}
    for route in MECHANISM_ROUTES:
        for frame in manifest["frames"]:
            _require_within(root, root / frame[route]["png_path"])
            _require_within(root, root / frame[route]["vtu_path"])
        png_count = _decode_pngs(root / route / "frames")
        vtu_count = _read_vtu_artifacts(root / route)
        pvd_path = root / route / "field_evolution.pvd"
        if _count_pvd_datasets(pvd_path) != frame_count:
            raise RuntimeError(f"{route} PVD dataset count does not match manifest")
        route_counts[route] = (png_count, vtu_count)
    if any(counts != (frame_count, frame_count) for counts in route_counts.values()):
        raise RuntimeError("mechanism field-evolution route counts do not match")
    last_frame = manifest["frames"][-1]
    final_unloaded = bool(last_frame["is_final_unloaded_target"])
    force_identical = all(
        math.isclose(frame["baseline"]["Fx_N"], frame["mechanism"]["Fx_N"], rel_tol=1e-10, abs_tol=1e-10)
        and math.isclose(frame["baseline"]["Fy_N"], frame["mechanism"]["Fy_N"], rel_tol=1e-10, abs_tol=1e-10)
        for frame in manifest["frames"]
    ) and bool(summary["comparison"]["total_force_history_identical"])
    nonzero_difference = _finite_nonzero_difference(root, manifest)
    if not final_unloaded or not force_identical or not nonzero_difference:
        raise RuntimeError("mechanism field-evolution validation flags did not pass")
    pass_states = list(dict.fromkeys(str(frame["pass_state"]) for frame in manifest["frames"]))
    return {
        "output_directory": str(output),
        "frame_count": frame_count,
        "node_count": int(summary["mesh_reuse"]["node_count"]),
        "triangle_count": int(summary["mesh_reuse"]["triangle_count"]),
        "equivalent_group_count": int(summary["resolution"]["equivalent_group_count"]),
        "pass_states": pass_states,
        "baseline_png_count": route_counts["baseline"][0],
        "mechanism_png_count": route_counts["mechanism"][0],
        "difference_png_count": route_counts["difference"][0],
        "baseline_vtu_count": route_counts["baseline"][1],
        "mechanism_vtu_count": route_counts["mechanism"][1],
        "difference_vtu_count": route_counts["difference"][1],
        "pvd_count": 3,
        "manifest_result_format": manifest["result_format"],
        "nonzero_difference_confirmed": nonzero_difference,
        "final_unloaded_confirmed": final_unloaded,
        "total_force_history_identical": force_identical,
        "retained_artifact_count": len(artifacts),
        "elapsed_seconds": time.perf_counter() - started,
    }



def run_portable_smoke(output_dir: str | Path) -> dict[str, object]:
    """Run existing public workflows and validate their published artifacts."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    single_output = root / "single_position"
    scan_output = root / "pass_scan"
    mechanism_output = root / "mechanism_field_evolution"
    workspace = root / ".scan_workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    try:
        _print_progress("[RUN] 线弹性单位置 smoke test")
        single_case = EvolvedFemCase.from_json(_bundled_example_path(SINGLE_EXAMPLE))
        single_result, single_artifacts, _ = run_evolved_fem_case(single_case, single_output, "auto")
        validate_evolved_fem_artifacts(single_artifacts)
        _print_progress("[PASS] 线弹性单位置")

        _print_progress("[RUN] 线弹性单程扫描 smoke test")
        scan_case = PassScanCase.from_json(_bundled_example_path(SCAN_EXAMPLE))
        scan_case = replace(scan_case, scan=replace(scan_case.scan, base_position_count=5))
        scan_result = run_pass_scan(scan_case, workspace / "solve")
        temporary, _ = export_pass_scan(scan_result, workspace / "artifacts", scan_output, "auto")
        final_scan_artifacts = aggregate_artifact_paths(scan_output)
        publish_pass_scan_artifacts(temporary, final_scan_artifacts)
        validate_pass_scan_artifacts(final_scan_artifacts)
        _print_progress("[PASS] 线弹性单程扫描")

        _print_progress("[RUN] 机制化弹塑性场演化 smoke test")
        mechanism_report = _run_mechanism_field_evolution_smoke(mechanism_output)
        _print_progress(
            "[PASS] 机制化弹塑性场演化 "
            f"({mechanism_report['elapsed_seconds']:.1f} s)"
        )

        png_count, msh_count, vtu_count = _published_artifact_counts(single_output, scan_output)
        report = {
            "result_format": "grindcae_phase_5b2_portable_smoke_report",
            "grindcae_version": __version__,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "output_directory": str(root),
            "single_position": _single_position_report(single_result, single_output),
            "pass_scan": {
                "output_directory": str(scan_output),
                "base_position_count": scan_case.scan.base_position_count,
                "actual_position_count": len(scan_result.history),
                "unique_snapshot_count": scan_result.unique_snapshot_count,
                "maximum_force_balance_residual_norm_N": scan_result.validations.maximum_force_balance_residual_norm_N,
                "zero_contact_endpoints_have_zero_finite_response": scan_result.validations.zero_contact_endpoints_have_zero_finite_response,
                "validations_passed": scan_result.validations.passed,
            },
            "mechanism_field_evolution": mechanism_report,
            "artifact_checks": {
                "decoded_png_count": png_count,
                "readable_msh_count": msh_count,
                "readable_vtu_count": vtu_count,
            },
            "passed": True,
        }
        report_path = root / "portable_smoke_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
            encoding="ascii",
            newline="\n",
        )
        return report
    finally:
        if workspace.exists():
            shutil.rmtree(workspace)
