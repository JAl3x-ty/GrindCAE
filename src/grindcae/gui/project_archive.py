"""Complete portable archive export/import for GrindCAE projects and results."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

from .project import GrindCaeProject, ProjectError, ResultRecord, load_project, save_project


ARCHIVE_SCHEMA_VERSION = 1
MAX_ARCHIVE_MEMBERS = 100_000
MAX_UNCOMPRESSED_BYTES = 20 * 1024**3


@dataclass(frozen=True)
class ImportedProjectArchive:
    project_path: Path
    result_directory: Path


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ProjectError(f"归档包含不安全或越界路径：{name}")
    return path


def export_project_archive(
    project_path: str | Path,
    archive_path: str | Path,
    result_ids: tuple[str, ...] | None = None,
) -> Path:
    source = Path(project_path).expanduser().resolve()
    project = load_project(source)
    selected = [record for record in project.results if result_ids is None or record.result_id in result_ids]
    target = Path(archive_path).expanduser().resolve()
    if not target.name.lower().endswith(".gcae-archive"):
        raise ProjectError("完整归档必须使用 .gcae-archive 扩展名。")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    checksums: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="grindcae-archive-") as temp_text:
            temp = Path(temp_text)
            portable = GrindCaeProject(**{
                **project.__dict__,
                "results": [],
            })
            for record in selected:
                output = Path(record.output_directory).resolve()
                # Persisted IDs may include ISO timestamps with Windows-invalid colons.
                # The ID stays unchanged in project metadata; only the folder is encoded.
                safe_directory = hashlib.sha256(record.result_id.encode("utf-8")).hexdigest()
                destination = temp / "project" / "results" / safe_directory
                shutil.copytree(output, destination)
                portable.results.append(ResultRecord(
                    **{
                        **record.__dict__,
                        "output_directory": str(destination),
                        "summary_path": str(destination / Path(record.summary_path).name),
                        "artifact_paths": {
                            key: str(destination / Path(value).relative_to(output))
                            for key, value in record.artifact_paths.items()
                        },
                        "location_kind": "project_relative",
                    }
                ))
            portable_path = save_project(portable, temp / "project" / "project.gcae")
            members: list[tuple[Path, str]] = [(portable_path, "project/project.gcae")]
            for path in sorted((temp / "project" / "results").rglob("*")) if (temp / "project" / "results").exists() else ():
                if path.is_file():
                    members.append((path, path.relative_to(temp).as_posix()))
            for path, name in members:
                checksums[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {
                "archive_type": "grindcae_complete_project_archive",
                "archive_schema_version": ARCHIVE_SCHEMA_VERSION,
                "project_id": project.project_id,
                "project_filename": source.name,
                "members": sorted(checksums),
            }
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
                for path, name in members:
                    archive.write(path, name)
                archive.writestr("checksums.json", json.dumps(checksums, indent=2))
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        _inspect_archive(temporary)
        temporary.replace(target)
    except (OSError, ValueError, zipfile.BadZipFile, shutil.Error) as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, ProjectError):
            raise
        raise ProjectError(f"完整归档导出失败：{exc}") from exc
    return target


def _inspect_archive(path: Path) -> tuple[dict, dict]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ProjectError("归档文件数量超过安全限制。")
            names = [info.filename for info in infos]
            if len(names) != len(set(name.casefold() for name in names)):
                raise ProjectError("归档包含重复或大小写冲突条目。")
            if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
                raise ProjectError("归档解压体积超过安全限制。")
            for info in infos:
                _safe_member(info.filename)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ProjectError("归档不能包含符号链接。")
            manifest = json.loads(archive.read("manifest.json"))
            checksums = json.loads(archive.read("checksums.json"))
            if manifest.get("archive_schema_version") != ARCHIVE_SCHEMA_VERSION:
                raise ProjectError("不支持的完整归档版本。")
            if set(manifest.get("members", [])) != set(checksums):
                raise ProjectError("归档清单和校验和不一致。")
            expected_names = set(checksums) | {"manifest.json", "checksums.json"}
            if set(names) != expected_names:
                raise ProjectError("归档实际成员与清单不一致，存在缺失或额外文件。")
            for name, digest in checksums.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                    raise ProjectError(f"归档文件校验失败：{name}")
            return manifest, checksums
    except ProjectError:
        raise
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise ProjectError(f"完整归档读取失败：{exc}") from exc


def import_project_archive(
    archive_path: str | Path, target_directory: str | Path
) -> ImportedProjectArchive:
    archive_source = Path(archive_path).expanduser().resolve()
    manifest, _ = _inspect_archive(archive_source)
    target = Path(target_directory).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise ProjectError("归档导入目标目录必须为空。")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="grindcae-import-", dir=target.parent) as temp_text:
        temp = Path(temp_text)
        with zipfile.ZipFile(archive_source) as archive:
            for info in archive.infolist():
                member = _safe_member(info.filename)
                destination = temp.joinpath(*member.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info.filename))
        project = load_project(temp / "project" / "project.gcae")
        final_project = target / Path(str(manifest.get("project_filename") or "project.gcae")).name
        if final_project.suffix.lower() != ".gcae":
            final_project = final_project.with_suffix(".gcae")
        results_root = target / f"{final_project.stem}_results"
        result_records: list[ResultRecord] = []
        for record in project.results:
            source_output = Path(record.output_directory)
            # Preserve the archive's validated relative folder, including older archives.
            destination = results_root / source_output.relative_to(temp / "project" / "results")
            result_records.append(ResultRecord(
                **{
                    **record.__dict__,
                    "output_directory": str(destination),
                    "summary_path": str(destination / Path(record.summary_path).name),
                    "artifact_paths": {
                        key: str(destination / Path(value).relative_to(source_output))
                        for key, value in record.artifact_paths.items()
                    },
                    "location_kind": "project_relative",
                }
            ))
        project.results = result_records
        target.mkdir(parents=True, exist_ok=True)
        source_results = temp / "project" / "results"
        if source_results.exists():
            shutil.copytree(source_results, results_root)
        save_project(project, final_project)
        load_project(final_project)
    return ImportedProjectArchive(final_project, results_root)
