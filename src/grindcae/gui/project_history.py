"""Application-local recent-result and verified recovery indexes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .analysis import ANALYSIS_MODE_DEFINITIONS
from .analysis_results import AnalysisResultAdapter, AnalysisResultError
from .project import GrindCaeProject, ProjectError, ResultRecord, load_project, save_project
from .result_catalog import ResultCatalog, ResultCatalogError


@dataclass(frozen=True)
class RecentResultEntry:
    result_id: str
    analysis_type: str
    result_format: str
    output_directory: str
    summary_path: str
    created_at: str
    input_fingerprint: str = ""
    project_id: str = ""
    summary_sha256: str = ""

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RecentResultEntry":
        return cls(**{name: str(value.get(name, "")) for name in cls.__dataclass_fields__})


class RecentResultStore:
    def __init__(self, path: str | Path, maximum: int = 20) -> None:
        self.path = Path(path).expanduser().resolve()
        self.maximum = maximum

    def load(self) -> tuple[RecentResultEntry, ...]:
        if not self.path.is_file():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            items = raw.get("entries", [])
            if not isinstance(items, list):
                raise ValueError("entries")
            return tuple(RecentResultEntry.from_mapping(item) for item in items if isinstance(item, dict))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProjectError(f"近期结果索引读取失败：{exc}") from exc

    def add(self, entry: RecentResultEntry) -> None:
        entries = [
            existing
            for existing in self.load()
            if Path(existing.output_directory) != Path(entry.output_directory)
        ]
        entries.insert(0, entry)
        self._write(entries[: self.maximum])

    def _write(self, entries: list[RecentResultEntry]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        payload = {"recent_results_schema_version": 1, "entries": [asdict(item) for item in entries]}
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(self.path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            temporary.unlink(missing_ok=True)
            raise ProjectError(f"近期结果索引保存失败：{exc}") from exc


@dataclass(frozen=True)
class RecentProjectEntry:
    project_id: str
    project_name: str
    path: str
    last_opened_at: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RecentProjectEntry":
        return cls(**{name: str(value.get(name, "")) for name in cls.__dataclass_fields__})


class RecentProjectStore:
    def __init__(self, path: str | Path, maximum: int = 10) -> None:
        self.path = Path(path).expanduser().resolve()
        self.maximum = maximum

    def load(self) -> tuple[RecentProjectEntry, ...]:
        if not self.path.is_file():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            entries = raw.get("entries", [])
            if not isinstance(entries, list):
                raise ValueError("entries")
            return tuple(RecentProjectEntry.from_mapping(item) for item in entries if isinstance(item, dict))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ProjectError(f"近期工程索引读取失败：{exc}") from exc

    def add(self, project: GrindCaeProject, path: str | Path) -> None:
        resolved = str(Path(path).expanduser().resolve())
        entries = [entry for entry in self.load() if Path(entry.path) != Path(resolved)]
        entries.insert(0, RecentProjectEntry(
            project.project_id,
            project.project_name,
            resolved,
            datetime.now().astimezone().isoformat(timespec="seconds"),
        ))
        payload = {"recent_projects_schema_version": 1, "entries": [asdict(item) for item in entries[: self.maximum]]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            json.loads(temporary.read_text(encoding="utf-8"))
            temporary.replace(self.path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            temporary.unlink(missing_ok=True)
            raise ProjectError(f"近期工程索引保存失败：{exc}") from exc


def recent_entry_from_record(
    record: ResultRecord, *, project_id: str = ""
) -> RecentResultEntry:
    summary = Path(record.summary_path).expanduser().resolve()
    return RecentResultEntry(
        result_id=record.result_id,
        analysis_type=record.analysis_type,
        result_format=record.result_format,
        output_directory=str(Path(record.output_directory).expanduser().resolve()),
        summary_path=str(summary),
        created_at=record.created_at,
        input_fingerprint=record.input_fingerprint,
        project_id=project_id,
        summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest(),
    )


@dataclass(frozen=True)
class RecoverySnapshot:
    path: Path
    project_id: str
    source_path: Path | None = None

    def is_newer_than_source(self) -> bool:
        if self.source_path is None or not self.source_path.is_file():
            return True
        try:
            return self.path.stat().st_mtime_ns > self.source_path.stat().st_mtime_ns
        except OSError:
            return True


class RecoveryStore:
    def __init__(self, directory: str | Path, maximum_per_project: int = 5) -> None:
        self.directory = Path(directory).expanduser().resolve()
        self.maximum_per_project = maximum_per_project

    def save(
        self,
        project: GrindCaeProject,
        *,
        source_path: str | Path | None,
        timestamp: str | None = None,
    ) -> RecoverySnapshot:
        stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        project_dir = self.directory / project.project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        path = project_dir / f"{stamp}.gcae"
        save_project(project, path)
        load_project(path)
        metadata = project_dir / f"{stamp}.json"
        metadata.write_text(
            json.dumps({"source_path": str(Path(source_path).resolve()) if source_path else None}, ensure_ascii=False),
            encoding="utf-8",
        )
        snapshots = sorted(project_dir.glob("*.gcae"), reverse=True)
        for stale in snapshots[self.maximum_per_project :]:
            stale.unlink(missing_ok=True)
            stale.with_suffix(".json").unlink(missing_ok=True)
        resolved_source = Path(source_path).expanduser().resolve() if source_path else None
        return RecoverySnapshot(path, project.project_id, resolved_source)

    def list(self, project_id: str | None = None) -> tuple[RecoverySnapshot, ...]:
        roots = [self.directory / project_id] if project_id else list(self.directory.glob("*"))
        snapshots: list[RecoverySnapshot] = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in root.glob("*.gcae"):
                source_path = None
                metadata = path.with_suffix(".json")
                try:
                    if metadata.is_file():
                        raw = json.loads(metadata.read_text(encoding="utf-8"))
                        value = raw.get("source_path") if isinstance(raw, dict) else None
                        if isinstance(value, str) and value:
                            source_path = Path(value).expanduser().resolve()
                except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                    source_path = None
                snapshots.append(RecoverySnapshot(path, root.name, source_path))
        return tuple(sorted(snapshots, key=lambda item: item.path.name, reverse=True))


def import_result_directory(
    output_directory: str | Path,
    *,
    store: RecentResultStore,
) -> RecentResultEntry:
    output = Path(output_directory).expanduser().resolve()
    matches = []
    adapter = AnalysisResultAdapter()
    for mode in ANALYSIS_MODE_DEFINITIONS:
        try:
            matches.append(adapter.read(mode, output))
        except (AnalysisResultError, OSError, ValueError):
            continue
    if len(matches) != 1:
        raise ProjectError("无法唯一识别所选目录中的正式分析结果。")
    published = matches[0]
    created_at = datetime.fromtimestamp(
        published.summary_path.stat().st_mtime
    ).astimezone().isoformat(timespec="seconds")
    entry = RecentResultEntry(
        result_id=f"recovered-{published.mode}-{created_at}",
        analysis_type=published.mode,
        result_format=published.result_format,
        output_directory=str(output),
        summary_path=str(published.summary_path),
        created_at=created_at,
        summary_sha256=hashlib.sha256(published.summary_path.read_bytes()).hexdigest(),
    )
    try:
        ResultCatalog().read(entry_as_result_record(entry))
    except (ResultCatalogError, OSError, ValueError) as exc:
        raise ProjectError(f"结果目录导入失败：严格结果校验未通过：{exc}") from exc
    store.add(entry)
    return entry


def entry_as_result_record(entry: RecentResultEntry) -> ResultRecord:
    published = AnalysisResultAdapter().read(entry.analysis_type, entry.output_directory)
    return ResultRecord(
        result_id=entry.result_id,
        analysis_type=entry.analysis_type,
        result_format=entry.result_format,
        output_directory=entry.output_directory,
        summary_path=entry.summary_path,
        artifact_paths={"representative_image": str(published.representative_image)},
        created_at=entry.created_at,
        input_fingerprint=entry.input_fingerprint,
        location_kind="recovered_external",
    )


def relink_result_record(
    record: ResultRecord, output_directory: str | Path
) -> ResultRecord:
    output = Path(output_directory).expanduser().resolve()
    try:
        published = AnalysisResultAdapter().read(record.analysis_type, output)
    except (AnalysisResultError, OSError, ValueError) as exc:
        raise ProjectError(f"结果重新定位失败：所选目录与原分析类型不匹配：{exc}") from exc
    if published.result_format != record.result_format:
        raise ProjectError("结果重新定位失败：result_format 与原记录不匹配。")
    return ResultRecord(
        result_id=record.result_id,
        analysis_type=record.analysis_type,
        result_format=record.result_format,
        output_directory=str(output),
        summary_path=str(published.summary_path),
        artifact_paths={"representative_image": str(published.representative_image)},
        created_at=record.created_at,
        status=record.status,
        input_fingerprint=record.input_fingerprint,
        location_kind="external_absolute",
    )
