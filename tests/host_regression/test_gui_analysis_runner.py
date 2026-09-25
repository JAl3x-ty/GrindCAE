from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
from threading import Event

import pytest

from grindcae.gui.analysis import ANALYSIS_MODE_DEFINITIONS, BuiltAnalysisInput
from grindcae.gui.analysis_runner import AnalysisBackend, AnalysisRunError, AnalysisRunner
from grindcae.gui.analysis_results import AnalysisPublishedResult
from grindcae.gui.background_tasks import BackgroundTaskGate


class Case: pass


def built(mode: str) -> BuiltAnalysisInput:
    return BuiltAnalysisInput(ANALYSIS_MODE_DEFINITIONS[mode], Case(), {"case": mode}, "a" * 64)


def published(mode: str, output: Path) -> AnalysisPublishedResult:
    definition = ANALYSIS_MODE_DEFINITIONS[mode]
    return AnalysisPublishedResult(mode, definition.display_name, definition.result_format, output, output / definition.summary_filename, output / definition.representative_image_filename, {}, "摘要")


def drain(queue: Queue):
    values = []
    while True:
        try: values.append(queue.get_nowait())
        except Empty: return values


@pytest.mark.parametrize("mode", tuple(m for m in ANALYSIS_MODE_DEFINITIONS if m != "literature_grinding_force"))
def test_runner_dispatches_each_mode_once_and_success_contains_adapted_result(tmp_path: Path, mode: str) -> None:
    calls = []
    def run(case, output, progress=None):
        calls.append((case, output, progress is not None))
    backend = AnalysisBackend(**{f"run_{mode}": run}, read_result=lambda selected, output: published(selected, output))
    runner = AnalysisRunner(backend=backend)
    thread = runner.start(built(mode), tmp_path / mode)
    thread.join(5)
    events = drain(runner.events)
    assert len(calls) == 1
    assert any(item.kind == "success" and item.result.mode == mode for item in events)
    assert events[-1].kind == "finished"
    assert not runner.is_running


def test_runner_rejects_second_task_and_recovers_after_backend_error(tmp_path: Path) -> None:
    entered = Event(); release = Event()
    def blocking(case, output, progress=None):
        entered.set(); release.wait(5)
    runner = AnalysisRunner(backend=AnalysisBackend(run_linear_elastic_single_position=blocking, read_result=lambda mode, output: published(mode, output)))
    thread = runner.start(built("linear_elastic_single_position"), tmp_path / "one")
    assert entered.wait(2)
    with pytest.raises(AnalysisRunError, match="正在运行"):
        runner.start(built("linear_elastic_single_position"), tmp_path / "two")
    release.set(); thread.join(5)

    def failing(case, output, progress=None): raise RuntimeError("solver boom")
    runner = AnalysisRunner(backend=AnalysisBackend(run_linear_elastic_single_position=failing))
    thread = runner.start(built("linear_elastic_single_position"), tmp_path / "bad")
    thread.join(5)
    events = drain(runner.events)
    assert any(item.kind == "error" and "计算失败" in item.message for item in events)
    assert not runner.is_running


def test_runner_shares_background_gate_with_process_preview(tmp_path: Path) -> None:
    gate = BackgroundTaskGate()
    assert gate.try_acquire("process_grain_preview")
    runner = AnalysisRunner(gate=gate)

    with pytest.raises(AnalysisRunError, match="后台任务正在运行"):
        runner.start(built("linear_elastic_single_position"), tmp_path / "blocked")

    gate.release("process_grain_preview")


@pytest.mark.parametrize(
    "mode",
    (
        "linear_elastic_single_pass",
        "elastoplastic_single_pass",
        "mechanism_elastoplastic_single_pass",
        "single_grain_high_fidelity",
    ),
)
def test_pass_runner_emits_determinate_progress_with_eta_fields(
    tmp_path: Path, mode: str
) -> None:
    clock_values = iter((100.0, 130.0, 130.0))

    def run(case, output, progress=None):
        assert progress is not None
        progress(2, 8, "已完成收敛位置 2/8")

    backend = AnalysisBackend(
        **{f"run_{mode}": run},
        read_result=lambda selected, output: published(selected, output),
    )
    runner = AnalysisRunner(backend=backend, clock=lambda: next(clock_values))

    thread = runner.start(built(mode), tmp_path / mode)
    thread.join(5)
    progress_event = next(
        event
        for event in drain(runner.events)
        if event.kind == "progress" and event.phase == "formal"
    )

    assert (progress_event.current, progress_event.total) == (2, 8)
    assert progress_event.elapsed_seconds == 30.0
    assert progress_event.remaining_seconds == 90.0


def test_requested_pilot_runs_before_formal_calculation(tmp_path: Path) -> None:
    order: list[str] = []

    def pilot(built_input, workspace, progress, run):
        order.append("pilot")
        progress(1, 4, "试算基准路线位置 1/2")

    def run(case, output, progress=None):
        order.append("formal")

    runner = AnalysisRunner(
        backend=AnalysisBackend(
            run_elastoplastic_single_pass=run,
            read_result=lambda mode, output: published(mode, output),
        ),
        pilot_runner=pilot,
    )

    thread = runner.start(
        built("elastoplastic_single_pass"), tmp_path / "pilot", run_pilot=True
    )
    thread.join(5)
    events = drain(runner.events)

    assert order == ["pilot", "formal"]
    assert any(event.kind == "pilot_success" for event in events)
    pilot_progress = next(
        event for event in events if event.kind == "progress" and event.phase == "pilot"
    )
    assert (pilot_progress.current, pilot_progress.total) == (1, 4)
    assert "试算" in pilot_progress.message


def test_formal_eta_excludes_optional_pilot_duration(tmp_path: Path) -> None:
    clock_values = iter((100.0, 220.0, 250.0, 250.0, 280.0))
    clock = lambda: next(clock_values)

    def pilot(built_input, workspace, progress, run):
        clock()
        progress(1, 4, "试算中")

    def run(case, output, progress=None):
        progress(1, 4, "正式位置 1/4")

    runner = AnalysisRunner(
        backend=AnalysisBackend(
            run_elastoplastic_single_pass=run,
            read_result=lambda mode, output: published(mode, output),
        ),
        clock=clock,
        pilot_runner=pilot,
    )

    thread = runner.start(
        built("elastoplastic_single_pass"), tmp_path / "eta", run_pilot=True
    )
    thread.join(5)
    progress_event = next(
        event
        for event in drain(runner.events)
        if event.kind == "progress" and event.phase == "formal"
    )

    assert progress_event.elapsed_seconds == 30.0
    assert progress_event.remaining_seconds == 90.0


def test_default_mechanism_pilot_forwards_route_progress_to_runner(
    monkeypatch, tmp_path: Path
) -> None:
    import grindcae.gui.analysis_runner as analysis_runner_module

    class PilotCase:
        def to_dict(self):
            return {
                "baseline_history_pass": {
                    "moving_load_pass": {"pass": {"base_position_count": 41}}
                }
            }

    class MechanismCase:
        @classmethod
        def from_mapping(cls, payload):
            assert payload["baseline_history_pass"]["moving_load_pass"]["pass"][
                "base_position_count"
            ] == 5
            return PilotCase()

    monkeypatch.setattr(
        "grindcae.mechanism_history_pass.MechanismHistoryPassCase",
        MechanismCase,
    )
    built_input = BuiltAnalysisInput(
        ANALYSIS_MODE_DEFINITIONS["mechanism_elastoplastic_single_pass"],
        PilotCase(),
        {"case": "mechanism"},
        "b" * 64,
    )
    received = []

    def run(case, output, progress=None):
        assert isinstance(case, PilotCase)
        assert progress is not None
        progress(3, 10, "试算基准路线位置 3/5")

    analysis_runner_module._run_pilot(
        built_input,
        tmp_path,
        lambda current, total, message=None: received.append(
            (current, total, message)
        ),
        run,
    )

    assert received == [(3, 10, "试算基准路线位置 3/5")]
