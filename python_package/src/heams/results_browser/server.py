"""Stdlib HTTP server behind `heams browse`."""

from __future__ import annotations

import json
import math
import re
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from heams.results_browser.catalog import (
    DisplayLimits,
    RecordTooLargeError,
    ResultsCatalog,
    TraceNotFoundError,
)
from heams.results_browser.table import Pick, build_table, latest_summaries

STATIC_DIR = Path(__file__).resolve().parent / "static"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".woff2": "font/woff2",
}
#: Third-party files, in versioned directories: their content never changes
#: at a given path, so browsers may keep them.
VENDOR_DIR = STATIC_DIR / "vendor"
_VENDOR_CACHE = "public, max-age=31536000, immutable"
#: The page loads only its own scripts and styles and calls only this API.
#: Dynamic styles are set through the CSSOM, so no inline style or script is
#: allowed.
_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


class _ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def _make_handler(catalog: ResultsCatalog) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "heams-browse"

        def log_message(self, format: str, *args: Any) -> None:
            pass  # keep the terminal quiet; errors are reported in responses

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = {key: values[-1] for key, values in parse_qs(parsed.query).items()}
            try:
                if parsed.path.startswith("/api/"):
                    self._api(parsed.path.removeprefix("/api/"), query)
                else:
                    self._static(parsed.path)
            except _ApiError as exc:
                self._json({"error": str(exc)}, exc.status)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
                self._json({"error": repr(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # -- routes ---------------------------------------------------------

        def _api(self, route: str, query: dict[str, str]) -> None:
            if route == "overview":
                if query.get("refresh"):
                    catalog.refresh()
                self._json(
                    {
                        "results_dir": str(catalog.root),
                        "title": catalog.root.name,
                        "experiments": [
                            experiment.to_json()
                            for experiment in catalog.experiments.values()
                        ],
                    }
                )
            elif route == "table":
                try:
                    pick = Pick(query.get("pick") or Pick.LARGEST)
                except ValueError as exc:
                    raise _ApiError(HTTPStatus.BAD_REQUEST, "unknown pick") from exc
                self._json(build_table(catalog, pick))
            elif route == "experiment":
                experiment = self._experiment(query)
                index = catalog.run_index(experiment.id)
                report = experiment.path / "report.md"
                run_status = experiment.path / "run_status.json"
                self._json(
                    {
                        "experiment": experiment.to_json(),
                        "config": experiment.config_path.read_text(encoding="utf-8")
                        if experiment.config_path
                        else None,
                        "run_status": run_status.read_text(encoding="utf-8")
                        if run_status.is_file()
                        else None,
                        "has_report": report.is_file(),
                        "headline": catalog.headline_rows(experiment.id),
                        "runs": latest_summaries(index.summaries),
                    }
                )
            elif route == "run":
                experiment = self._experiment(query)
                full = query.get("full") == "1"
                try:
                    payload = catalog.run(
                        experiment.id,
                        self._run_number(query),
                        limits=None if full else DisplayLimits(),
                    )
                except IndexError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, "run not found") from exc
                except RecordTooLargeError as exc:
                    raise _ApiError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"This sample is {exc.size / 2**20:,.0f} MB, too large to "
                        "load untruncated; use Download raw JSON instead.",
                    ) from exc
                payload["count"] = len(catalog.run_index(experiment.id).offsets)
                self._json(payload)
            elif route == "raw":
                experiment = self._experiment(query)
                number = self._run_number(query)
                try:
                    size, chunks = catalog.raw_run(experiment.id, number)
                except IndexError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, "run not found") from exc
                filename = _download_name(f"{experiment.path.name}-run{number}.json")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(size))
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{filename}"'
                )
                for key, value in _SECURITY_HEADERS.items():
                    self.send_header(key, value)
                self.end_headers()
                for chunk in chunks:
                    self.wfile.write(chunk)
            elif route == "trace":
                experiment = self._experiment(query)
                try:
                    payload = catalog.trace(experiment.id, self._run_number(query))
                except IndexError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, "run not found") from exc
                except TraceNotFoundError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
                self._json(payload)
            elif route == "trace_event":
                experiment = self._experiment(query)
                try:
                    position = int(query.get("event", ""))
                except ValueError as exc:
                    raise _ApiError(
                        HTTPStatus.BAD_REQUEST, "event must be an int"
                    ) from exc
                try:
                    payload = catalog.trace_event(
                        experiment.id, self._run_number(query), position
                    )
                except IndexError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, "event not found") from exc
                except TraceNotFoundError as exc:
                    raise _ApiError(HTTPStatus.NOT_FOUND, str(exc)) from exc
                self._json(payload)
            elif route == "report":
                experiment = self._experiment(query)
                report = experiment.path / "report.md"
                if not report.is_file():
                    raise _ApiError(HTTPStatus.NOT_FOUND, "no report.md")
                self._send(report.read_bytes(), "text/plain; charset=utf-8")
            else:
                raise _ApiError(HTTPStatus.NOT_FOUND, f"unknown route {route!r}")

        def _experiment(self, query: dict[str, str]):
            try:
                return catalog.experiment(query.get("id", ""))
            except KeyError as exc:
                raise _ApiError(HTTPStatus.NOT_FOUND, "experiment not found") from exc

        @staticmethod
        def _run_number(query: dict[str, str]) -> int:
            try:
                return int(query.get("run", ""))
            except ValueError as exc:
                raise _ApiError(HTTPStatus.BAD_REQUEST, "run must be an int") from exc

        def _static(self, path: str) -> None:
            """Serve the page (`/`) and its assets (`/static/...`)."""
            if path in ("/", "/index.html"):
                target = STATIC_DIR / "index.html"
            elif path.startswith("/static/"):
                target = (STATIC_DIR / path.removeprefix("/static/")).resolve()
            else:
                raise _ApiError(HTTPStatus.NOT_FOUND, "not found")
            content_type = _CONTENT_TYPES.get(target.suffix)
            if (
                content_type is None
                or not target.is_relative_to(STATIC_DIR)
                or not target.is_file()
            ):
                raise _ApiError(HTTPStatus.NOT_FOUND, "not found")
            cache = _VENDOR_CACHE if target.is_relative_to(VENDOR_DIR) else None
            self._send(target.read_bytes(), content_type, cache=cache)

        # -- responses -------------------------------------------------------

        def _json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            try:
                body = json.dumps(payload, allow_nan=False, default=str)
            except ValueError:  # NaN/inf somewhere in a saved record
                body = json.dumps(_finite(payload), default=str)
            self._send(body.encode("utf-8"), "application/json", status=status)

        def _send(
            self,
            body: bytes,
            content_type: str,
            headers: dict[str, str] | None = None,
            status: HTTPStatus = HTTPStatus.OK,
            cache: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache or "no-store")
            for key, value in {**_SECURITY_HEADERS, **(headers or {})}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _download_name(name: str) -> str:
    """``name`` reduced to characters safe in a ``Content-Disposition`` value."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    return value


def make_server(
    catalog: ResultsCatalog, host: str = "127.0.0.1", port: int = 8765
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), _make_handler(catalog))


def serve(
    results_dir: Path,
    *,
    cache_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> int:
    catalog = ResultsCatalog(
        results_dir, cache_dir=cache_dir.expanduser() if cache_dir else None
    )
    build_table(catalog)  # start indexing the default table's runs right away
    server = make_server(catalog, host, port)
    url = f"http://{host}:{server.server_address[1]}/"
    print(
        f"Browsing {len(catalog.experiments)} experiments under {catalog.root}\n"
        f"Open {url} (Ctrl+C to stop)",
        file=sys.stderr,
    )
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


__all__ = ["make_server", "serve"]
