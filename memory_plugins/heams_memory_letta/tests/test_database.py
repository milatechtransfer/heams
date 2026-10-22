"""The private database: the migrations download, and (integration) a real start."""

from __future__ import annotations

import hashlib
import io
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import pytest

from heams_memory_letta import database


def _archive(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"letta-{database.LETTA_VERSION}/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture()
def download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Serves `data` in place of the GitHub archive, with a matching checksum."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    requests: list[str] = []

    def serve(data: bytes, *, checksum: str | None = None) -> list[str]:
        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def urlopen(url: str, timeout: float) -> Response:
            requests.append(url)
            return Response(data)

        monkeypatch.setattr(database.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(
            database, "_SOURCE_SHA256", checksum or hashlib.sha256(data).hexdigest()
        )
        return requests

    return serve


def test_migrations_keep_only_alembic_files_and_download_once(download) -> None:
    requests = download(
        _archive(
            {
                "alembic.ini": "[alembic]",
                "alembic/env.py": "# env",
                "alembic/versions/0001_initial.py": "# migration",
                "letta/server/app.py": "# not a migration",
            }
        )
    )

    path = database.ensure_migrations()
    files = sorted(str(f.relative_to(path)) for f in path.rglob("*") if f.is_file())
    assert files == ["alembic.ini", "alembic/env.py", "alembic/versions/0001_initial.py"]

    assert database.ensure_migrations() == path
    assert requests == [database._SOURCE_URL]


def test_a_checksum_mismatch_is_refused_and_caches_nothing(download) -> None:
    download(_archive({"alembic.ini": "[alembic]"}), checksum="0" * 64)

    with pytest.raises(RuntimeError, match="SHA-256"):
        database.ensure_migrations()
    assert not database.migrations_dir().exists()


@pytest.fixture()
def temp_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Where tempfile creates folders, isolated per test."""
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp))
    return temp


def test_a_failed_download_leaves_no_folders(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    def fail() -> Path:
        raise RuntimeError("offline")

    monkeypatch.setattr(database, "_postgres_bin", lambda: temp_dir)
    monkeypatch.setattr(database, "ensure_migrations", fail)
    monkeypatch.delenv("SLURM_TMPDIR", raising=False)
    with pytest.raises(RuntimeError, match="offline"):
        database.LocalPostgres()
    assert list(temp_dir.iterdir()) == []


def test_an_empty_slurm_tmpdir_falls_back_to_the_temp_dir(
    monkeypatch: pytest.MonkeyPatch, temp_dir: Path
) -> None:
    monkeypatch.setattr(database, "_postgres_bin", lambda: temp_dir)
    monkeypatch.setattr(database, "ensure_migrations", lambda: temp_dir)
    monkeypatch.setenv("SLURM_TMPDIR", "")
    postgres = database.LocalPostgres()
    try:
        assert postgres._root.parent == temp_dir
    finally:
        postgres.stop()


@pytest.fixture()
def default_sigterm():
    """Starting a database installs a SIGTERM handler; restore the original."""
    original = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    yield
    signal.signal(signal.SIGTERM, original)


@pytest.mark.integration
@pytest.mark.usefixtures("default_sigterm")
def test_a_private_database_starts_with_letta_schema_and_is_removed() -> None:
    postgres = database.LocalPostgres()
    root, socket_dir = postgres._root, postgres._socket_dir
    try:
        postgres.start()
        assert postgres.uri.endswith(f"?host={socket_dir}")
        revision = postgres._run(
            "psql", *postgres._tcp_args(), "-d", "letta",
            "-Atc", "select version_num from alembic_version",
        )  # fmt: skip
        assert revision.strip() == database._SCHEMA_REVISION
    finally:
        postgres.stop()
    assert not root.exists()
    assert not socket_dir.exists()


# "running": SIGTERM after startup. "startup": SIGTERM while migrations run, after
# Postgres has started.
_SIGTERM_SCRIPTS = {
    "running": (
        "from heams_memory_letta import database\n"
        "database.local_database_uri()\n"
        "print(database._local._root, database._local._socket_dir, flush=True)\n"
        "import time; time.sleep(120)\n"
    ),
    "startup": (
        "import time\n"
        "from heams_memory_letta import database\n"
        "def migrate(self, migrations):\n"
        "    print(self._root, self._socket_dir, flush=True)\n"
        "    time.sleep(120)\n"
        "database.LocalPostgres._migrate = migrate\n"
        "database.local_database_uri()\n"
    ),
}


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["running", "startup"])
def test_sigterm_stops_and_removes_the_database(phase: str) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", _SIGTERM_SCRIPTS[phase]],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        root, socket_dir = map(Path, process.stdout.readline().split())
        assert root.is_dir()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == -signal.SIGTERM
    finally:
        process.kill()
    deadline = time.monotonic() + 10
    while (root.exists() or socket_dir.exists()) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not root.exists()
    assert not socket_dir.exists()
    running = subprocess.run(
        ["pgrep", "-f", f"postgres -D {root}"], capture_output=True, text=True
    )
    assert running.stdout == ""
