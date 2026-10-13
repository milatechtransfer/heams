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

job_dir="$LETTA_PG_ROOT/jobs/${SLURM_JOB_ID:-local-$$}"
# A requeued job keeps its id; start it from a clean copy.
rm -rf "$job_dir"
mkdir -p "$job_dir"
cp -a "$TEMPLATE" "$job_dir/data"

cleanup() {
  "$PG/pg_ctl" -D "$job_dir/data" -m fast stop >/dev/null 2>&1 || true
  rm -rf "$job_dir"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

port="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
# Crash safety is off: the database is discarded, and fsync on network scratch is slow.
"$PG/pg_ctl" -D "$job_dir/data" -l "$job_dir/postgres.log" -w start \
  -o "-p $port -k $job_dir -c listen_addresses=127.0.0.1 -c fsync=off -c synchronous_commit=off -c full_page_writes=off"

LETTA_PG_URI="postgresql://letta:$(cat "$PASSWORD_FILE")@127.0.0.1:$port/letta"
export LETTA_PG_URI

uv run --python 3.12 heams run "$@"
