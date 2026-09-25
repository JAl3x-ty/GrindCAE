"""Public solve, validate, and transactional publication API."""

from __future__ import annotations

from pathlib import Path
import tempfile
from collections.abc import Callable
from grindcae.solver import ImportedMesh

from .exporters import (
    artifact_paths,
    publish_contact_result,
    read_contact_summary,
    write_contact_artifacts,
)
from .models import SingleGrainContactCase
from .workflow import ContactTrajectoryResult, run_contact_trajectory_on_mesh
from .exporters_v2 import (
    artifact_paths_v2,
    publish_contact_result_v2,
    read_contact_summary_v2,
    write_contact_artifacts_v2,
)
from .exporters_v3 import (
    artifact_paths_v3,
    publish_contact_result_v3,
    read_contact_summary_v3,
    write_contact_artifacts_v3,
)


def run_single_grain_contact(
    case: SingleGrainContactCase,
    output_directory: str | Path,
    *,
    progress: Callable[[int, int, str | None], None] | None = None,
    imported_mesh: ImportedMesh | None = None,
    checkpoint_callback: Callable[[dict], None] | None = None,
    resume_checkpoint: dict | None = None,
) -> ContactTrajectoryResult:
    """Run and publish a complete case, optionally retaining an exact solve mesh.

    Accepted-substep callbacks and same-input resume use the trajectory kernel's
    identity, state and energy checks. A paused or failed run publishes nothing.
    """

    if not isinstance(case, SingleGrainContactCase):
        raise TypeError("case must be a SingleGrainContactCase")
    version = case.single_grain_contact_schema_version
    final = (
        artifact_paths_v3(output_directory)
        if version == 3
        else artifact_paths_v2(output_directory)
        if version == 2
        else artifact_paths(output_directory)
    )
    with tempfile.TemporaryDirectory(prefix="grindcae-single-grain-contact-") as raw:
        workspace = Path(raw)
        checkpoint_options = {}
        if checkpoint_callback is not None:
            checkpoint_options['checkpoint_callback'] = checkpoint_callback
        if resume_checkpoint is not None:
            checkpoint_options['resume_checkpoint'] = resume_checkpoint
        result = run_contact_trajectory_on_mesh(
            case, imported_mesh, workspace=workspace, progress=progress,
            **checkpoint_options,
        )
        temporary = (
            artifact_paths_v3(workspace / "artifacts")
            if version == 3
            else artifact_paths_v2(workspace / "artifacts")
            if version == 2
            else artifact_paths(workspace / "artifacts")
        )
        if version == 3:
            write_contact_artifacts_v3(result, temporary)
            publish_contact_result_v3(temporary, final)
        elif version == 2:
            write_contact_artifacts_v2(result, temporary)
            publish_contact_result_v2(temporary, final)
        else:
            write_contact_artifacts(result, temporary)
            publish_contact_result(temporary, final)
    (
        read_contact_summary_v3
        if version == 3
        else read_contact_summary_v2
        if version == 2
        else read_contact_summary
    )(final["summary_json"])
    return result
