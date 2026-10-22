from __future__ import annotations

import os
import re
import shlex
import subprocess
from pathlib import Path

from tests.paths import PACKAGE_ROOT, REPO_ROOT

REPOSITORY_ROOT = REPO_ROOT
SUBMIT_SCRIPT = PACKAGE_ROOT / "cluster" / "submit_campaign.sh"
CAMPAIGN = "cluster/campaigns/smoke_campaign.sh"


def test_campaigns_include_beam_token_scales() -> None:
    smoke_runs = PACKAGE_ROOT / "configs" / "runs" / "smoke_campaign"
    for name in (
        "no_memory.yaml",
        "in_context.yaml",
        "milamem.yaml",
        "mem0.yaml",
    ):
        text = (smoke_runs / name).read_text(encoding="utf-8")
        assert "smoke_campaign/beam_100k.yaml" in text
        assert "beam_1m.yaml" not in text
        assert "beam_10m.yaml" not in text

    campaigns = PACKAGE_ROOT / "cluster" / "campaigns"
    for name in (
        "no_memory_all_benchmarks",
        "milamem_all_benchmarks",
        "mem0_all_benchmarks",
    ):
        text = (campaigns / f"{name}.sh").read_text(encoding="utf-8")
        for split in ("beam_100k", "beam_500k", "beam_1m", "beam_10m"):
            assert f"{name}/{split}.yaml" in text

    in_context = (campaigns / "in_context_all_benchmarks.sh").read_text(
        encoding="utf-8"
    )
    assert "in_context_all_benchmarks/beam_100k.yaml" in in_context
    assert "in_context_all_benchmarks/beam_500k.yaml" in in_context
    assert "beam_1m.yaml" not in in_context
    assert "beam_10m.yaml" not in in_context


def test_submit_campaign_passes_stable_exact_directory_per_run(
    tmp_path: Path,
) -> None:
    result, submitted_jobs = _run_campaign_with_fake_sbatch(tmp_path)

    assert result.returncode == 0, result.stderr
    campaign_output_dir = tmp_path / "campaign-results"
    assert campaign_output_dir.is_dir()
    shell_commands = [_shell_command(submitted_job) for submitted_job in submitted_jobs]

    assert len(shell_commands) == 4
    output_directories = _output_directories(shell_commands)
    assert len(output_directories) == 4
    assert all(
        re.fullmatch(r"smoke_campaign_\d{8}-\d{6}-[0-9a-f]{8}", path.parent.name)
        for path in output_directories
    )
    assert all(path.parent.parent == campaign_output_dir for path in output_directories)
    assert {path.name for path in output_directories} == {
        "01_smoke_campaign_no_memory",
        "02_smoke_campaign_in_context",
        "03_smoke_campaign_milamem",
        "04_smoke_campaign_mem0",
    }
    assert all("--requeue" in submitted_job for submitted_job in submitted_jobs)
    assert all(
        "exec srun uv run --python 3.12 heams run" in shell_command
        for shell_command in shell_commands
    )
    assert all("--no-sync" not in shell_command for shell_command in shell_commands)
    submitted_text = "\n".join(
        argument for submitted_job in submitted_jobs for argument in submitted_job
    )
    assert submitted_text.count("--time=04:00:00") == 4
    campaign_log_dir = next(iter(output_directories)).parent / "logs"
    log_paths = [
        Path(argument.removeprefix("--output="))
        for argument in submitted_text.splitlines()
        if argument.startswith("--output=")
    ] + [
        Path(argument.removeprefix("--error="))
        for argument in submitted_text.splitlines()
        if argument.startswith("--error=")
    ]
    assert len(log_paths) == 8
    assert all(path.parent == campaign_log_dir for path in log_paths)
    assert all(path.name in {"%x_%j.out", "%x_%j.err"} for path in log_paths)


def test_submit_campaign_leaves_partition_and_account_to_the_cluster(
    tmp_path: Path,
) -> None:
    result, submitted_jobs = _run_campaign_with_fake_sbatch(tmp_path)

    assert result.returncode == 0, result.stderr
    assert len(submitted_jobs) == 4
    for submitted_job in submitted_jobs:
        assert not any(arg.startswith("--partition=") for arg in submitted_job)
        assert not any(arg.startswith("--account=") for arg in submitted_job)


def test_submit_campaign_passes_partition_and_account_from_environment(
    tmp_path: Path,
) -> None:
    result, submitted_jobs = _run_campaign_with_fake_sbatch(
        tmp_path, PARTITION="long-cpu", SLURM_ACCOUNT="my-lab"
    )

    assert result.returncode == 0, result.stderr
    assert len(submitted_jobs) == 4
    for submitted_job in submitted_jobs:
        assert "--partition=long-cpu" in submitted_job
        assert "--account=my-lab" in submitted_job


def test_submit_campaign_requires_output_directory(tmp_path: Path) -> None:
    result = _run_submit_script(tmp_path, CAMPAIGN)

    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_submit_campaign_runs_letta_like_any_other_memory_system(
    tmp_path: Path,
) -> None:
    result, submitted_jobs = _run_campaign_with_fake_sbatch(
        tmp_path, "cluster/campaigns/heams_performance_exps_letta.sh"
    )

    assert result.returncode == 0, result.stderr
    [shell_command] = [_shell_command(job) for job in submitted_jobs]
    assert (
        "exec srun uv run --python 3.12 heams run "
        "--config configs/runs/heams_performance_exps/letta_beam_1m.yaml --output-dir "
    ) in shell_command


def _run_campaign_with_fake_sbatch(
    tmp_path: Path,
    campaign: str = CAMPAIGN,
    **extra_environment: str,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    submitted_args = tmp_path / "sbatch-args.txt"
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >> \"$SBATCH_ARGS\"\nprintf '%s\\n' '---' >> \"$SBATCH_ARGS\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    _write_fake_make(bin_dir, "exit 1")
    campaign_output_dir = tmp_path / "campaign-results"

    environment = {
        "SBATCH_ARGS": str(submitted_args),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        **extra_environment,
    }
    result = _run_submit_script(
        tmp_path,
        campaign,
        str(campaign_output_dir),
        **environment,
    )
    submitted_jobs = [
        block.splitlines()
        for block in submitted_args.read_text(encoding="utf-8").split("---\n")
        if block.strip()
    ]
    return result, submitted_jobs


def _output_directories(shell_commands: list[str]) -> set[Path]:
    return {
        Path(match.group(1))
        for shell_command in shell_commands
        for match in re.finditer(r"--output-dir ([^ ]+)", shell_command)
    }


def _shell_command(submitted_job: list[str]) -> str:
    wrapper_argument = next(
        argument for argument in submitted_job if argument.startswith("--wrap=")
    )
    wrapper_parts = shlex.split(wrapper_argument.removeprefix("--wrap="))
    assert len(wrapper_parts) >= 3
    assert wrapper_parts[1] == "-lc"
    return wrapper_parts[2]


def test_submit_campaign_does_not_run_setup(
    tmp_path: Path,
) -> None:
    """Submission must not mutate the environment shared by active Slurm jobs."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    submitted_args = tmp_path / "sbatch-args.txt"
    fake_sbatch = bin_dir / "sbatch"
    fake_sbatch.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" >> "$SBATCH_ARGS"\n',
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    make_args = tmp_path / "make-args.txt"
    _write_fake_make(bin_dir, "exit 1")
    campaign_output_dir = tmp_path / "campaign-results"

    result = _run_submit_script(
        tmp_path,
        CAMPAIGN,
        str(campaign_output_dir),
        SBATCH_ARGS=str(submitted_args),
        MAKE_ARGS=str(make_args),
        PATH=f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    )

    assert result.returncode == 0, result.stderr
    assert not make_args.exists()
    assert submitted_args.exists()
    assert any(campaign_output_dir.glob("smoke_campaign_*"))


def _write_fake_make(bin_dir: Path, body: str) -> None:
    fake_make = bin_dir / "make"
    fake_make.write_text(
        f'#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" >> "$MAKE_ARGS"\n{body}\n',
        encoding="utf-8",
    )
    fake_make.chmod(0o755)


def _run_submit_script(
    tmp_path: Path,
    *arguments: str,
    **extra_environment: str,
) -> subprocess.CompletedProcess[str]:
    # Cluster settings come only from the test, never from the caller's shell.
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PARTITION", "SLURM_ACCOUNT"}
    }
    environment.update(
        {
            "REMOTE_REPO": str(REPOSITORY_ROOT),
        }
    )
    environment.update(extra_environment)
    return subprocess.run(
        [str(SUBMIT_SCRIPT), *arguments],
        capture_output=True,
        check=False,
        cwd=PACKAGE_ROOT,
        env=environment,
        text=True,
    )
