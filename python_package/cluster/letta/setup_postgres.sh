#!/usr/bin/env bash
# One-time Letta database setup: Postgres 17 + pgvector under $LETTA_PG_ROOT and a
# template database with Letta's schema, copied by run_with_postgres.sh for each job.
# Run on a compute node (salloc), after `make setup`:  bash cluster/letta/setup_postgres.sh
set -euo pipefail

LETTA_PG_ROOT="${LETTA_PG_ROOT:-${SCRATCH:-$HOME}/letta_pg}"
LETTA_VERSION="0.16.8"
PG="$LETTA_PG_ROOT/env/bin"
TEMPLATE="$LETTA_PG_ROOT/template"
PASSWORD_FILE="$LETTA_PG_ROOT/password"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALEMBIC="$PACKAGE_DIR/.venv/bin/alembic"

[ -x "$ALEMBIC" ] || { echo "Missing $ALEMBIC; run 'make setup' in python_package first." >&2; exit 2; }
[ ! -e "$TEMPLATE" ] || { echo "$TEMPLATE already exists; delete it to rebuild." >&2; exit 2; }
mkdir -p "$LETTA_PG_ROOT"

if [ ! -x "$PG/postgres" ]; then
  if [ ! -x "$LETTA_PG_ROOT/bin/micromamba" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C "$LETTA_PG_ROOT" bin/micromamba
  fi
  MAMBA_ROOT_PREFIX="$LETTA_PG_ROOT/mamba" "$LETTA_PG_ROOT/bin/micromamba" create -y \
    -p "$LETTA_PG_ROOT/env" -c conda-forge postgresql=17 pgvector
fi
"$PG/postgres" --version | grep -q ' 17\.' \
  || { echo "Expected PostgreSQL 17 in $LETTA_PG_ROOT/env." >&2; exit 1; }

if [ ! -s "$PASSWORD_FILE" ]; then
  (umask 077 && python3 -c 'import secrets; print(secrets.token_hex(24))' > "$PASSWORD_FILE")
fi
chmod 600 "$PASSWORD_FILE"

work="$(mktemp -d "$LETTA_PG_ROOT/setup.XXXXXX")"
cleanup() {
  local status=$?
  "$PG/pg_ctl" -D "$TEMPLATE" -m fast stop >/dev/null 2>&1 || true
  rm -rf "$work"
  if [ "$status" -ne 0 ]; then
    rm -rf "$TEMPLATE"
  fi
}
trap cleanup EXIT

# Compute nodes are shared, so the server requires a password even on localhost.
"$PG/initdb" -D "$TEMPLATE" -U letta --auth=scram-sha-256 --pwfile="$PASSWORD_FILE" >/dev/null

port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
"$PG/pg_ctl" -D "$TEMPLATE" -l "$work/postgres.log" -w start \
  -o "-p $port -k $work -c listen_addresses=127.0.0.1"

PGPASSWORD="$(cat "$PASSWORD_FILE")"
export PGPASSWORD
"$PG/createdb" -h 127.0.0.1 -p "$port" -U letta letta
"$PG/psql" -h 127.0.0.1 -p "$port" -U letta -d letta -qc 'create extension if not exists vector'

# The letta package omits its migrations, so they come from the source at the same tag.
letta_src="$LETTA_PG_ROOT/letta-$LETTA_VERSION"
[ -d "$letta_src" ] || git clone --depth 1 -b "$LETTA_VERSION" https://github.com/letta-ai/letta.git "$letta_src"
(cd "$letta_src" && LETTA_PG_URI="postgresql://letta:$PGPASSWORD@127.0.0.1:$port/letta" "$ALEMBIC" upgrade head)

schema="$("$PG/psql" -h 127.0.0.1 -p "$port" -U letta -d letta -Atc 'select version_num from alembic_version')"
vector="$("$PG/psql" -h 127.0.0.1 -p "$port" -U letta -d letta -Atc "select extversion from pg_extension where extname = 'vector'")"
echo "Letta template ready at $TEMPLATE (schema $schema, pgvector $vector)."
