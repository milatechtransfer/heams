from pathlib import Path
from types import SimpleNamespace

import pytest

from heams.cli import cli


def test_run_command_prepares_config_and_executes_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/runs/smoke_test_benchmark.yaml")
    prepared_run = SimpleNamespace(
        run_config=SimpleNamespace(
            run_id="test-run",
            results_dir=Path("results/test-run"),
        ),
        experiment_configs=(object(),),
        resumed=False,
        overwritten=False,
    )
    executed: list[tuple[object, tuple[object, ...]]] = []
    prepared: list[tuple[Path, Path, bool]] = []

    def _prepare_run(path: Path, output_dir: Path, *, overwrite: bool) -> object:
        prepared.append((path, output_dir, overwrite))
        return prepared_run

    monkeypatch.setattr(cli, "prepare_run", _prepare_run)
    monkeypatch.setattr(
        cli,
        "execute_run",
        lambda run_config, experiment_configs: executed.append(
            (run_config, experiment_configs)
        ),
    )

    output_dir = Path("results/smoke_test_benchmark")
    assert (
        cli.main(
            [
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert prepared == [(config_path, output_dir, False)]
    assert executed == [(prepared_run.run_config, prepared_run.experiment_configs)]


@pytest.mark.parametrize(
    ("resumed", "overwritten", "extra_args", "expected"),
    [
        (False, False, [], "Starting run 'status-run' from scratch in results/status"),
        (True, False, [], "Resuming run 'status-run' from results/status"),
        (
            False,
            True,
            ["--overwrite"],
            "Starting run 'status-run' from scratch after overwriting results/status",
        ),
    ],
)
def test_run_command_reports_run_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    resumed: bool,
    overwritten: bool,
    extra_args: list[str],
    expected: str,
) -> None:
    run_config = SimpleNamespace(
        run_id="status-run", results_dir=Path("results/status")
    )
    prepared_run = SimpleNamespace(
        run_config=run_config,
        experiment_configs=(),
        resumed=resumed,
        overwritten=overwritten,
    )
    monkeypatch.setattr(cli, "prepare_run", lambda *args, **kwargs: prepared_run)
    monkeypatch.setattr(cli, "execute_run", lambda *args: None)

    assert (
        cli.main(
            [
                "run",
                "--config",
                "configs/runs/smoke_test_benchmark.yaml",
                "--output-dir",
                "results/status",
                *extra_args,
            ]
        )
        == 0
    )
    assert expected in capsys.readouterr().out


def test_run_command_passes_explicit_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = Path("configs/runs/smoke_test_benchmark.yaml")
    output_dir = tmp_path / "campaign-results" / "smoke_test_benchmark"
    prepared_run = SimpleNamespace(
        run_config=SimpleNamespace(
            run_id="test-run",
            results_dir=output_dir,
        ),
        experiment_configs=(object(),),
        resumed=False,
        overwritten=False,
    )
    prepared: list[tuple[Path, Path, bool]] = []

    def _prepare_run(path: Path, output_dir: Path, *, overwrite: bool) -> object:
        prepared.append((path, output_dir, overwrite))
        return prepared_run

    monkeypatch.setattr(cli, "prepare_run", _prepare_run)
    monkeypatch.setattr(
        cli,
        "execute_run",
        lambda run_config, experiment_configs: None,
    )

    assert (
        cli.main(
            [
                "run",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    assert prepared == [(config_path, output_dir, False)]


def test_run_command_passes_overwrite_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared_run = SimpleNamespace(
        run_config=SimpleNamespace(
            run_id="test-run",
            results_dir=tmp_path / "run",
        ),
        experiment_configs=(),
        resumed=False,
        overwritten=False,
    )
    calls: list[bool] = []
    monkeypatch.setattr(
        cli,
        "prepare_run",
        lambda path, output_dir, *, overwrite: calls.append(overwrite) or prepared_run,
    )
    monkeypatch.setattr(cli, "execute_run", lambda *args: None)

    assert (
        cli.main(
            [
                "run",
                "--config",
                "configs/runs/smoke_test_benchmark.yaml",
                "--output-dir",
                str(tmp_path / "run"),
                "--overwrite",
            ]
        )
        == 0
    )
    assert calls == [True]


def test_run_command_requires_config() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["run"])


def test_run_command_requires_output_directory() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["run", "--config", "configs/runs/smoke_test_benchmark.yaml"])


def test_legacy_benchmark_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--benchmark", "locomo"])

    assert exc_info.value.code == 2


def test_run_help_describes_config_driven_interface(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["run", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text
    assert "--output-dir" in help_text
    assert "--overwrite" in help_text
    assert "--benchmark" not in help_text


def test_generate_report_command_resolves_results_dir_and_writes_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results" / "smoke_test"
    written = [results_dir / "exp_a_run1" / "report.md"]
    calls: list[tuple[Path, int, bool]] = []

    def _fake_generate_experiment_reports(
        path: Path, *, max_examples: int, detailed: bool
    ) -> list[Path]:
        calls.append((path, max_examples, detailed))
        return written

    monkeypatch.setattr(
        cli, "generate_experiment_reports", _fake_generate_experiment_reports
    )

    assert cli.main(["generate-report", "--results-dir", str(results_dir)]) == 0
    # Default cap of 10 applies when --max-examples is omitted, and
    # --detailed defaults to off.
    assert calls == [(results_dir, 10, False)]


def test_generate_report_command_passes_max_examples_flag_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    results_dir = tmp_path / "results" / "smoke_test"
    calls: list[tuple[Path, int, bool]] = []

    def _fake_generate_experiment_reports(
        path: Path, *, max_examples: int, detailed: bool
    ) -> list[Path]:
        calls.append((path, max_examples, detailed))
        return [results_dir / "exp_a_run1" / "report.md"]

    monkeypatch.setattr(
        cli, "generate_experiment_reports", _fake_generate_experiment_reports
    )

    assert (
        cli.main(
            [
                "generate-report",
                "--results-dir",
                str(results_dir),
                "--max-examples",
                "3",
            ]
        )
        == 0
    )
    assert calls == [(results_dir, 3, False)]


def test_generate_report_command_passes_detailed_flag_through(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--detailed` is passed straight through to `generate_experiment_reports`
    - the CLI no longer orchestrates the CSV export itself."""
    results_dir = tmp_path / "results" / "smoke_test"
    calls: list[tuple[Path, int, bool]] = []

    def _fake_generate_experiment_reports(
        path: Path, *, max_examples: int, detailed: bool
    ) -> list[Path]:
        calls.append((path, max_examples, detailed))
        return [results_dir / "exp_a_run1" / "report.md"]

    monkeypatch.setattr(
        cli, "generate_experiment_reports", _fake_generate_experiment_reports
    )

    assert (
        cli.main(["generate-report", "--results-dir", str(results_dir), "--detailed"])
        == 0
    )
    assert calls == [(results_dir, 10, True)]


def test_generate_report_command_fails_when_nothing_was_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        cli,
        "generate_experiment_reports",
        lambda path, *, max_examples, detailed: [],
    )

    assert (
        cli.main(["generate-report", "--results-dir", str(tmp_path / "results")]) == 1
    )


def test_generate_report_command_requires_results_dir() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["generate-report"])
