from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from dotenv import find_dotenv, load_dotenv

from heams.cli.memory_commands import list_memory_plugins, test_memory_plugins
from heams.orchestration import execute_run, prepare_run
from heams.reporting.experiment_report import generate_experiment_reports


def _main_checkout_env_path() -> Path | None:
    """Resolve the primary checkout `.env` when running from a linked worktree."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    common_dir = Path(completed.stdout.strip())
    if not common_dir:
        return None
    candidate = common_dir.parent / ".env"
    return candidate if candidate.is_file() else None


def _load_env_files() -> list[str]:
    """Load local dotenv first, then the main-checkout `.env` for worktrees."""
    loaded: list[str] = []
    local = find_dotenv(usecwd=True)
    if local:
        load_dotenv(local)
        loaded.append(local)
    main_env = _main_checkout_env_path()
    if main_env is not None and str(main_env) not in loaded:
        # Main checkout wins for keys still unset after a worktree-local .env.
        load_dotenv(main_env, override=False)
        loaded.append(str(main_env))
    return loaded


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="heams",
        description=(
            "Run benchmark campaigns from declarative configs using the refactored "
            "orchestration pipeline."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a benchmark run from a declarative run config.",
    )
    run_parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a run config under python_package/configs/runs/.",
    )
    run_parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Exact directory for this run and its resumable artifacts.",
    )
    run_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear and restart a recognized HEAMS run in the exact output directory.",
    )

    memory_parser = subparsers.add_parser(
        "memory",
        help="Inspect installed memory plugins.",
    )
    memory_subparsers = memory_parser.add_subparsers(
        dest="memory_command", required=True
    )
    memory_subparsers.add_parser(
        "list",
        help="List discovered memory plugin type names.",
    )
    memory_subparsers.add_parser(
        "test",
        help="Run an isolated memory plugin smoke diagnostic.",
    )

    generate_report_parser = subparsers.add_parser(
        "generate-report",
        help="Generate a per-experiment Markdown report from a run's saved artifacts.",
    )
    generate_report_parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help="Run directory containing per-experiment artifacts (see `heams run`).",
    )
    generate_report_parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help=(
            "Cap on how many runs get a full transcript/judge detail section "
            "per experiment report, split between failures and successes "
            "(default: 10). Pass a number >= the experiment's run count for "
            "no cap."
        ),
    )
    generate_report_parser.add_argument(
        "--detailed",
        action="store_true",
        help=(
            "Also write submetrics.csv: one row per tagged sub-metric score "
            "and efficiency metric across every benchmark in the run."
        ),
    )

    browse_parser = subparsers.add_parser(
        "browse",
        help="Open a local web GUI for navigating saved campaign results.",
    )
    browse_parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help=(
            "Run, campaign or results directory to browse; searched "
            "recursively for experiments (directories with a runs.jsonl)."
        ),
    )
    browse_parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Where to keep runs.jsonl indexes (default: "
            "<results-dir>/_results_browser_cache). Use it for shared or "
            "read-only results directories."
        ),
    )
    browse_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1, local only).",
    )
    browse_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765; 0 picks a free port).",
    )
    browse_parser.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open a browser tab automatically.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _load_env_files()
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        return _run_command(args.config, args.output_dir, args.overwrite)
    if args.command == "memory":
        if args.memory_command == "list":
            return list_memory_plugins()
        if args.memory_command == "test":
            return test_memory_plugins()
    if args.command == "generate-report":
        return _generate_report_command(
            args.results_dir, args.max_examples, args.detailed
        )

    if args.command == "browse":
        from heams.results_browser.server import serve

        return serve(
            args.results_dir,
            cache_dir=args.cache_dir,
            host=args.host,
            port=args.port,
            open_browser=not args.no_open,
        )

    print(f"Unknown command {args.command!r}.", file=sys.stderr)
    return 1


def _run_command(
    config_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    prepared_run = prepare_run(config_path, output_dir, overwrite=overwrite)
    if prepared_run.resumed:
        print(
            f"Resuming run '{prepared_run.run_config.run_id}' from "
            f"{prepared_run.run_config.results_dir}"
        )
    elif prepared_run.overwritten:
        print(
            f"Starting run '{prepared_run.run_config.run_id}' from scratch "
            f"after overwriting {prepared_run.run_config.results_dir}"
        )
    else:
        print(
            f"Starting run '{prepared_run.run_config.run_id}' from scratch in "
            f"{prepared_run.run_config.results_dir}"
        )
    execute_run(prepared_run.run_config, prepared_run.experiment_configs)
    return 0


def _generate_report_command(
    results_dir: Path, max_examples: int, detailed: bool
) -> int:
    written = generate_experiment_reports(
        results_dir, max_examples=max_examples, detailed=detailed
    )
    if not written:
        print(
            f"No experiment reports were generated under {results_dir}.",
            file=sys.stderr,
        )
        return 1
    for path in written:
        print(f"Wrote {path}")

    return 0
