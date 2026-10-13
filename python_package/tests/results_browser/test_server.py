"""HTTP server: the page, its assets, security headers and the JSON API."""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from heams.cli import cli
from heams.results_browser import server as browser_server
from tests.results_browser.fixtures import get_json, make_run, write_experiment

Serve = Callable[[Path], str]


def test_server_serves_page_and_api(campaign: Path, serve: Serve) -> None:
    base = serve(campaign)

    with urllib.request.urlopen(f"{base}/") as response:
        page = response.read().decode()
    overview = get_json(f"{base}/api/overview")
    table = get_json(f"{base}/api/table?pick=latest")
    experiment = get_json(f"{base}/api/experiment?id=run_mem0%2Fexp_mem0_locomo")
    run = get_json(f"{base}/api/run?id=run_mem0%2Fexp_mem0_locomo&run=0")

    assert '<script type="module" src="/static/js/main.js">' in page
    assert len(overview["experiments"]) == 2
    assert table["pick"] == "latest"
    assert len(experiment["runs"]) == 2
    assert experiment["headline"][0]["metric_name"] == "Task accuracy"
    assert run["count"] == 2
    assert run["run"]["scenario_id"] == "conv-1"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/", "text/html; charset=utf-8"),
        ("/static/css/browser.css", "text/css; charset=utf-8"),
        ("/static/js/main.js", "text/javascript; charset=utf-8"),
        ("/static/js/views/trace.js", "text/javascript; charset=utf-8"),
        ("/static/vendor/temml-0.13.5/Temml.woff2", "font/woff2"),
    ],
)
def test_assets_are_served_with_security_headers(
    campaign: Path, serve: Serve, path: str, content_type: str
) -> None:
    with urllib.request.urlopen(f"{serve(campaign)}{path}") as response:
        headers = response.headers

    assert headers["Content-Type"] == content_type
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert "unsafe-inline" not in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize(
    ("path", "cache"),
    [
        ("/static/js/main.js", "no-store"),
        (
            "/static/vendor/temml-0.13.5/temml.min.js",
            "public, max-age=31536000, immutable",
        ),
    ],
)
def test_only_versioned_vendor_files_are_cached(
    campaign: Path, serve: Serve, path: str, cache: str
) -> None:
    with urllib.request.urlopen(f"{serve(campaign)}{path}") as response:
        assert response.headers["Cache-Control"] == cache


@pytest.mark.parametrize(
    "path",
    [
        "/api/experiment?id=..%2F..",
        "/api/run?id=run_mem0%2Fexp_mem0_locomo&run=9",
        "/../catalog.py",
        "/static/../server.py",
        "/static/%2e%2e/server.py",
        "/static/js/package.json",
        "/static/missing.js",
    ],
)
def test_server_rejects_unknown_targets(
    campaign: Path, serve: Serve, path: str
) -> None:
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{serve(campaign)}{path}")
    assert error.value.code == 404


def test_experiment_lists_each_sample_once_like_the_table(
    tmp_path: Path, serve: Serve
) -> None:
    # A resumed run appended conv-1 again; its last record is authoritative.
    write_experiment(
        tmp_path / "run",
        "exp",
        benchmark="locomo",
        memory="mem0",
        runs=[
            make_run("conv-1", passed=False, category="temporal"),
            make_run("conv-2", passed=True, category="temporal"),
            make_run("conv-1", passed=True, category="temporal"),
        ],
    )

    runs = get_json(f"{serve(tmp_path)}/api/experiment?id=run%2Fexp")["runs"]

    assert [(run["scenario_id"], run["i"], run["accuracy"]) for run in runs] == [
        ("conv-1", 2, 1.0),
        ("conv-2", 1, 1.0),
    ]


def test_raw_download_name_is_header_safe(tmp_path: Path, serve: Serve) -> None:
    write_experiment(
        tmp_path / "run",
        'exp"\\x',
        benchmark="locomo",
        memory="mem0",
        runs=[make_run("conv-1", passed=True, category="temporal")],
    )
    experiment_id = urllib.parse.quote('run/exp"\\x', safe="")

    with urllib.request.urlopen(
        f"{serve(tmp_path)}/api/raw?id={experiment_id}&run=0"
    ) as response:
        disposition = response.headers["Content-Disposition"]

    assert disposition == 'attachment; filename="exp_x-run0.json"'


def test_server_rejects_unknown_pick(campaign: Path, serve: Serve) -> None:
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(f"{serve(campaign)}/api/table?pick=best")
    assert error.value.code == 400


def test_server_serves_trace_timeline_and_events(traced: Path, serve: Serve) -> None:
    base = serve(traced)

    trace = get_json(f"{base}/api/trace?id=run%2Fexp&run=0")
    event = get_json(f"{base}/api/trace_event?id=run%2Fexp&run=0&event=3")
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(f"{base}/api/trace?id=run%2Fexp&run=2")
    with pytest.raises(urllib.error.HTTPError) as bad_event:
        urllib.request.urlopen(f"{base}/api/trace_event?id=run%2Fexp&run=0&event=x")

    assert len(trace["events"]) == 4
    assert event["event"]["result"] == {
        "memories": ["Melanie painted a sunrise in 2022."]
    }
    assert missing.value.code == 404
    assert bad_event.value.code == 400


def test_browse_command_passes_options_to_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Path, dict]] = []

    def fake_serve(results_dir: Path, **kwargs: object) -> int:
        calls.append((results_dir, kwargs))
        return 0

    monkeypatch.setattr(browser_server, "serve", fake_serve)

    assert (
        cli.main(["browse", "--results-dir", str(tmp_path), "--port", "0", "--no-open"])
        == 0
    )
    assert calls == [
        (
            tmp_path,
            {"cache_dir": None, "host": "127.0.0.1", "port": 0, "open_browser": False},
        )
    ]
