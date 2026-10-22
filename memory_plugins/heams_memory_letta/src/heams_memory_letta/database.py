"""A private Postgres for Letta, started on demand.

Letta 0.16.8 runs only on Postgres with pgvector. When the variable named in the
config's `pg_uri_env` is unset, the plugin starts a throwaway database for the
process: PostgreSQL 17 and pgvector come from the `pgembed` wheel, and the schema from
Letta's Alembic migrations at the same tag, which the pip package omits. The migrations
are downloaded once into a cache, ahead of time by `make setup`.

Each process gets its own database, so concurrent runs never share one, and the
database is stopped and deleted when the process exits.

`python -m heams_memory_letta` downloads the migrations and checks that a database
starts.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import io
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path

LETTA_VERSION = "0.16.8"

_SOURCE_URL = f"https://github.com/letta-ai/letta/archive/refs/tags/{LETTA_VERSION}.tar.gz"
_SOURCE_SHA256 = "405580dc8373fc81af7769cc1563224091959949e2d51d64d4c830e73b39909d"
# alembic_version after `alembic upgrade head` at the tag.
_SCHEMA_REVISION = "1c28e167b74f"
_USER = "letta"
_DATABASE = "letta"

_local: LocalPostgres | None = None
_local_lock = threading.Lock()


def migrations_dir() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "heams-memory-letta" / f"letta-{LETTA_VERSION}-migrations"


def ensure_migrations() -> Path:
    """Letta's `alembic.ini` and `alembic/` at the pinned tag, downloaded once."""
    target = migrations_dir()
    if (target / "alembic.ini").is_file():
        return target
    try:
        with urllib.request.urlopen(_SOURCE_URL, timeout=120) as response:
            data = response.read()
    except OSError as exc:
        raise RuntimeError(
            f"Could not download Letta {LETTA_VERSION}'s migrations from {_SOURCE_URL}: "
            f"{exc}. Run `make -C memory_plugins/heams_memory_letta setup` where there "
            "is network access, or set LETTA_PG_URI to an existing database."
        ) from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != _SOURCE_SHA256:
        raise RuntimeError(
            f"Letta {LETTA_VERSION}'s source archive has SHA-256 {digest}, "
            f"expected {_SOURCE_SHA256}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".migrations-", dir=target.parent))
    try:
        prefix = f"letta-{LETTA_VERSION}/"
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                name = member.name.removeprefix(prefix)
                if name != "alembic.ini" and not name.startswith("alembic/"):
                    continue
                member.name = name
                archive.extract(member, staging, filter="data")
        try:
            staging.rename(target)
        except OSError:
            # Another process finished first; its copy is identical.
            if not (target / "alembic.ini").is_file():
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return target


def local_database_uri() -> str:
    """The URI of this process's private database, started on first use."""
    global _local
    with _local_lock:
        if _local is None:
            database = LocalPostgres()
            database.start()
            _local = database
        return _local.uri


class LocalPostgres:
    """A throwaway Postgres with Letta's schema, deleted when the process exits."""

    def __init__(self) -> None:
        self._bin = _postgres_bin()
        # Before any folder exists, so a failed download leaves nothing behind.
        self._migrations = ensure_migrations()
        # SLURM_TMPDIR is node-local on the cluster, which avoids network-scratch stalls.
        self._root = Path(
            tempfile.mkdtemp(
                prefix="heams-letta-pg-", dir=os.environ.get("SLURM_TMPDIR") or None
            )
        )
        # A Unix socket path must fit in about 100 bytes, which macOS temp dirs exceed.
        self._socket_dir = Path(
            tempfile.mkdtemp(prefix="hlpg-", dir="/tmp" if os.path.isdir("/tmp") else None)
        )
        self._data = self._root / "data"
        self._log = self._root / "postgres.log"
        self._password = secrets.token_hex(24)
        self._port = _free_port()
        self._running = False
        self.uri = ""

    def start(self) -> None:
        # Before anything starts, so an exit or SIGTERM during startup cleans up too.
        atexit.register(self.stop)
        _stop_on_sigterm(self)
        try:
            password_file = self._root / "password"
            password_file.write_text(self._password)
            # Compute nodes are shared, so the server requires a password even locally.
            self._run(
                "initdb",
                "-D", str(self._data),
                "-U", _USER,
                "--auth=scram-sha-256",
                f"--pwfile={password_file}",
            )  # fmt: skip
            password_file.unlink()
            # Set first, so a stop during `pg_ctl start` still stops the server.
            self._running = True
            # Crash safety is off: the database is discarded when the process exits.
            self._run(
                "pg_ctl",
                "-D", str(self._data),
                "-l", str(self._log),
                "-w", "start",
                "-o",
                f"-p {self._port} -k {self._socket_dir} -c listen_addresses=127.0.0.1 "
                "-c fsync=off -c synchronous_commit=off -c full_page_writes=off",
            )  # fmt: skip
            self._run("createdb", *self._tcp_args(), _DATABASE)
            self._run(
                "psql", *self._tcp_args(), "-d", _DATABASE,
                "-qc", "create extension if not exists vector",
            )  # fmt: skip
            self._migrate(self._migrations)
        except BaseException:
            self.stop()
            raise
        # Letta opens many short-lived connections, so it uses the Unix socket. The path
        # stays unencoded: Letta saves the URI through configparser, which rejects `%`.
        self.uri = (
            f"postgresql://{_USER}:{self._password}@localhost:{self._port}/{_DATABASE}"
            f"?host={self._socket_dir}"
        )

    def stop(self) -> None:
        if self._running:
            self._running = False
            subprocess.run(
                [str(self._bin / "pg_ctl"), "-D", str(self._data), "-m", "fast", "stop"],
                capture_output=True,
                check=False,
            )
        shutil.rmtree(self._root, ignore_errors=True)
        shutil.rmtree(self._socket_dir, ignore_errors=True)

    def _migrate(self, migrations: Path) -> None:
        # Migrations connect over TCP, as the cluster setup script's did.
        tcp_uri = (
            f"postgresql://{_USER}:{self._password}@127.0.0.1:{self._port}/{_DATABASE}"
        )
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            cwd=migrations,
            env={**os.environ, "LETTA_PG_URI": tcp_uri},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Letta's migrations failed:\n{_tail(result.stderr or result.stdout)}"
            )
        revision = self._run(
            "psql", *self._tcp_args(), "-d", _DATABASE,
            "-Atc", "select version_num from alembic_version",
        ).strip()  # fmt: skip
        if revision != _SCHEMA_REVISION:
            raise RuntimeError(
                f"Letta's schema is at revision {revision!r}, expected {_SCHEMA_REVISION}."
            )

    def _tcp_args(self) -> list[str]:
        return ["-h", "127.0.0.1", "-p", str(self._port), "-U", _USER]

    def _run(self, program: str, *args: str) -> str:
        result = subprocess.run(
            [str(self._bin / program), *args],
            env={**os.environ, "PGPASSWORD": self._password},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr or result.stdout
            if self._log.exists():
                detail += f"\nPostgres log:\n{_tail(self._log.read_text())}"
            raise RuntimeError(f"Letta's database: {program} failed:\n{detail.strip()}")
        return result.stdout


def _stop_on_sigterm(database: LocalPostgres) -> None:
    """Stop the database on SIGTERM too, then end the process as the signal would have.

    atexit does not run when SIGTERM ends a process, as on a Slurm time limit. Python
    installs signal handlers only from the main thread, and an existing handler is left
    alone."""
    if threading.current_thread() is not threading.main_thread():
        return
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        return

    def stop_then_terminate(signum: int, frame: object) -> None:
        database.stop()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, stop_then_terminate)


def _postgres_bin() -> Path:
    """The binaries the `pgembed` wheel ships, found without importing it."""
    spec = importlib.util.find_spec("pgembed")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "pgembed is not installed; run `make -C memory_plugins/heams_memory_letta "
            "setup`, or set LETTA_PG_URI to an existing database."
        )
    path = Path(spec.submodule_search_locations[0]) / "pginstall" / "bin"
    if not (path / "postgres").is_file():
        raise RuntimeError(f"pgembed's PostgreSQL binaries are missing from {path}.")
    return path


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def main() -> None:
    print(f"Letta {LETTA_VERSION} migrations: {ensure_migrations()}")
    database = LocalPostgres()
    try:
        database.start()
    finally:
        database.stop()
    print(f"A private Postgres started with Letta's schema ({_SCHEMA_REVISION}).")
