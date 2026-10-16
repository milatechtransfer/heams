#!/usr/bin/env bash
# Runs `heams run "$@"` against a throwaway Postgres copied from the template built by
# setup_postgres.sh, so concurrent Letta jobs never share a database.
# Used as a campaign's RUN_WRAPPER; see cluster/campaigns/heams_performance_exps_letta.sh.
set -euo pipefail

LETTA_PG_ROOT="${LETTA_PG_ROOT:-${SCRATCH:-$HOME}/letta_pg}"
PG="$LETTA_PG_ROOT/env/bin"
TEMPLATE="$LETTA_PG_ROOT/template"
PASSWORD_FILE="$LETTA_PG_ROOT/password"

for required in "$PG/pg_ctl" "$TEMPLATE/PG_VERSION" "$PASSWORD_FILE"; do
  [ -e "$required" ] || {
    echo "Missing $required; run cluster/letta/setup_postgres.sh first." >&2
    exit 2
  }
done

job_id="${SLURM_JOB_ID:-local-$$}"
job_root="${SLURM_TMPDIR:-$LETTA_PG_ROOT/jobs}"
log_dir="$LETTA_PG_ROOT/logs"
mkdir -p "$job_root" "$log_dir"
# Node-local storage avoids network-scratch stalls. Unique copies also isolate requeues.
job_dir="$(mktemp -d "$job_root/letta_pg_$job_id.XXXXXX")"

cleanup() {
  "$PG/pg_ctl" -D "$job_dir/data" -m fast stop >/dev/null 2>&1 || true
  if [ -f "$job_dir/postgres.log" ]; then
    cp "$job_dir/postgres.log" "$log_dir/$job_id.log" \
      || echo "Could not preserve Postgres log from $job_dir." >&2
  fi
  rm -rf "$job_dir"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

cp -a "$TEMPLATE" "$job_dir/data"

port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
# Crash safety is off: the database is discarded, and fsync on network scratch is slow.
"$PG/pg_ctl" -D "$job_dir/data" -l "$job_dir/postgres.log" -w start \
  -o "-p $port -k '$job_dir' -c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off -c full_page_writes=off"

# Letta opens many short-lived connections; use the job's Unix socket instead of TCP.
socket_host="$(python3 -c 'import sys; from urllib.parse import quote; print(quote(sys.argv[1], safe=""))' "$job_dir")"
LETTA_PG_URI="postgresql://letta:$(cat "$PASSWORD_FILE")@localhost:$port/letta?host=$socket_host"
export LETTA_PG_URI

uv run --python 3.12 heams run "$@"
