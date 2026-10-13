#!/usr/bin/env bash
# Open the results browser (`heams browse`) running on a Slurm cluster from
# your laptop.
#
# Runs on your laptop. `start` (the default) opens one shared SSH connection
# to the cluster's login host, prepares a dedicated checkout on first use,
# reuses the running dashboard job or submits one, waits for its address,
# tunnels it to http://localhost:$LOCAL_PORT and opens your browser.
#
# Site settings are never hard-coded: they are the BROWSE_* keys of the
# repository's .env (copied from .env_template), or exported variables. See
# docs/results-browser.md, "Viewing cluster results from your laptop".
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# The .env to read: this checkout's, else the main checkout's when running
# from a linked worktree (the same lookup as the `heams` CLI).
env_file() {
  if [ -f "$REPO_ROOT/.env" ]; then
    echo "$REPO_ROOT/.env"
    return
  fi
  local common
  common="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)" || return 0
  [ -f "$(dirname "$common")/.env" ] && echo "$(dirname "$common")/.env"
  return 0
}

# Load the BROWSE_* lines of .env. Other keys (API credentials, ...) are
# never read. Values are taken literally (never executed; `$VARIABLES` are
# expanded later, on the cluster), empty values are skipped so defaults
# apply, and variables already set in the environment win over the file.
load_settings() {
  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(BROWSE_[A-Z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    value="${BASH_REMATCH[2]}"
    if [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]]; then
      value="${BASH_REMATCH[1]}"
    fi
    if [ -n "$value" ] && [ -z "${!key:-}" ]; then
      printf -v "$key" '%s' "$value"
    fi
  done <"$1"
}

ENV_FILE="$(env_file)"
if [ -n "$ENV_FILE" ]; then
  load_settings "$ENV_FILE"
fi

# Required: the SSH host (from your ~/.ssh/config) of the cluster's login
# node, and the results directory on the cluster's shared file system.
CLUSTER_HOST="${BROWSE_CLUSTER_HOST:-}"
RESULTS_DIR="${BROWSE_RESULTS_DIR:-}"
# Optional. Paths are expanded on the cluster, so `$` stays literal here.
LOCAL_PORT="${BROWSE_LOCAL_PORT:-8767}"
REMOTE_BASE='${SCRATCH:-$HOME}'
REMOTE_CHECKOUT="${BROWSE_REMOTE_CHECKOUT:-$REMOTE_BASE/mozilla-memory-browse}"
REMOTE_CACHE="${BROWSE_REMOTE_CACHE:-$REMOTE_BASE/mozilla-memory-browse-cache}"
REPO_URL="${BROWSE_REPO_URL:-https://github.com/milatechtransfer/mozilla-memory.git}"
BRANCH="${BROWSE_BRANCH:-}"
SLURM_PARTITION="${BROWSE_SLURM_PARTITION:-}"
SLURM_ACCOUNT="${BROWSE_SLURM_ACCOUNT:-}"
CPUS="${BROWSE_CPUS:-2}"
MEM="${BROWSE_MEM:-8G}"
TIME_LIMIT="${BROWSE_TIME_LIMIT:-08:00:00}"
JOB_NAME="${JOB_NAME:-heams-browse}"
WAIT_MINUTES="${WAIT_MINUTES:-30}"
CONNECTION_HOURS="${CONNECTION_HOURS:-12}"

usage() {
  cat >&2 <<EOF
Usage: $0 [start|stop|status|update]

  start   Connect, start or reuse the dashboard job, tunnel it to
          http://localhost:$LOCAL_PORT and open the browser (default).
  stop    Cancel the dashboard job and close the tunnel.
  status  Show the dashboard job and whether the tunnel is open.
  update  Pull and sync the cluster checkout, and restart a running
          dashboard so it serves the updated code.

Settings are the BROWSE_* keys of .env (see .env_template), or exported:
required BROWSE_CLUSTER_HOST and BROWSE_RESULTS_DIR; optional
BROWSE_SLURM_PARTITION, BROWSE_SLURM_ACCOUNT, BROWSE_CPUS, BROWSE_MEM,
BROWSE_TIME_LIMIT, BROWSE_LOCAL_PORT, BROWSE_REMOTE_CHECKOUT,
BROWSE_REMOTE_CACHE, BROWSE_REPO_URL, BROWSE_BRANCH. NO_BROWSER=1 skips
opening the browser.
EOF
}

log() { printf '%s\n' "$*" >&2; }

require_settings() {
  local missing=""
  [ -n "$CLUSTER_HOST" ] || missing="$missing BROWSE_CLUSTER_HOST"
  [ -n "$RESULTS_DIR" ] || missing="$missing BROWSE_RESULTS_DIR"
  if [ -n "$missing" ]; then
    log "Missing setting(s):$missing."
    log "Add them to ${ENV_FILE:-$REPO_ROOT/.env} (see $REPO_ROOT/.env_template), or export them."
    exit 2
  fi
}

STATE_DIR="$HOME/.cache/heams-browse"
# One shared connection per cluster host; every command below reuses it, so
# a cluster that asks for a one-time password asks only once.
socket() { printf '%s/%s.sock' "$STATE_DIR" "$CLUSTER_HOST"; }
# Your user name on the cluster, as resolved from your SSH config.
cluster_user() { ssh -G "$CLUSTER_HOST" 2>/dev/null | awk '$1 == "user" { print $2 }'; }

connected() { ssh -S "$(socket)" -O check "$CLUSTER_HOST" >/dev/null 2>&1; }

ensure_connection() {
  connected && return
  log "Connecting to $CLUSTER_HOST (answer any login prompt; the connection lasts ${CONNECTION_HOURS} h)."
  mkdir -p "$STATE_DIR"
  # A socket left by a connection that has ended blocks a new one.
  rm -f "$(socket)"
  # Prompts read from and write to the terminal directly; other output goes
  # to a log so the background process does not hold this script's pipes.
  ssh -M -S "$(socket)" -o ControlPersist="${CONNECTION_HOURS}h" -Nf "$CLUSTER_HOST" \
    >/dev/null 2>"$STATE_DIR/connection.log" || true
  connected || {
    log "Could not connect to $CLUSTER_HOST:"
    cat "$STATE_DIR/connection.log" >&2
    exit 1
  }
}

# Run a command on the login host in a login shell (tools such as `uv` are
# often only on a login shell's PATH).
remote() {
  ssh -S "$(socket)" "$CLUSTER_HOST" "bash -lc $(printf '%q' "$1")"
}

ensure_checkout() {
  local branch_flag=""
  [ -n "$BRANCH" ] && branch_flag="--branch $BRANCH"
  remote "
    if [ ! -x \"$REMOTE_CHECKOUT/python_package/.venv/bin/heams\" ]; then
      echo 'First use: preparing $REMOTE_CHECKOUT' >&2
      [ -d \"$REMOTE_CHECKOUT\" ] || git clone -q $branch_flag $REPO_URL \"$REMOTE_CHECKOUT\"
      cd \"$REMOTE_CHECKOUT/python_package\" && uv sync -q --python 3.12
    fi"
}

# Login shells can print banners, so keep only a trailing numeric job ID.
job_id() { tail -n 1 | grep -E '^[0-9]+$' || true; }

running_job() {
  remote "squeue -u \$USER -n $JOB_NAME -h -t RUNNING,PENDING -o %i | head -1" | job_id
}

submit_job() {
  local options="--job-name=$JOB_NAME --cpus-per-task=$CPUS --mem=$MEM --time=$TIME_LIMIT"
  [ -n "$SLURM_PARTITION" ] && options="$options --partition=$SLURM_PARTITION"
  [ -n "$SLURM_ACCOUNT" ] && options="$options --account=$SLURM_ACCOUNT"
  remote "
    mkdir -p \"$REMOTE_CHECKOUT/logs\" && cd \"$REMOTE_CHECKOUT/python_package\" &&
    sbatch --parsable $options \
      --output=\"$REMOTE_CHECKOUT/logs/browse-%j.out\" \
      --wrap=\"exec .venv/bin/heams browse --results-dir '$RESULTS_DIR' --cache-dir $REMOTE_CACHE --no-open --port 0\""
}

# Print "<node> <port>" once the job serves, or fail if it ended.
wait_for_address() {
  local job="$1" deadline=$((SECONDS + WAIT_MINUTES * 60)) state node port last=""
  while [ "$SECONDS" -lt "$deadline" ]; do
    read -r state node port < <(
      remote "
        read -r state node < <(squeue -j $job -h -o '%T %N')
        port=\$(grep -o 'http://127.0.0.1:[0-9]*' \"$REMOTE_CHECKOUT/logs/browse-$job.out\" 2>/dev/null | head -1 | cut -d: -f3)
        echo \"\${state:-ENDED} \${node:--} \${port:--}\""
    )
    if [ "$state" = "RUNNING" ] && [ "$port" != "-" ]; then
      echo "$node $port"
      return
    fi
    if [ "$state" = "ENDED" ]; then
      log "Job $job ended before serving. Last log lines:"
      remote "tail -n 20 \"$REMOTE_CHECKOUT/logs/browse-$job.out\"" >&2 || true
      return 1
    fi
    if [ "$state" != "$last" ]; then
      log "Job $job is $state; waiting for the dashboard address..."
      last="$state"
    fi
    sleep 10
  done
  log "Timed out after $WAIT_MINUTES minutes; job $job is still $state."
  return 1
}

# Matches this script's tunnel and one opened by hand (`-L <port>:127.0.0.1:`).
tunnel_pattern() { printf -- '-L (127\\.0\\.0\\.1:)?%s:127\\.0\\.0\\.1:' "$LOCAL_PORT"; }

close_tunnel() { pkill -f -- "$(tunnel_pattern)" 2>/dev/null || true; }

# The server listens on the node's loopback only, so the tunnel ends on the
# node itself: SSH to the node by the name Slurm reports, through the login
# host (this needs SSH access to nodes running your jobs).
open_tunnel() {
  local node="$1" port="$2" user
  user="$(cluster_user)"
  close_tunnel
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
    -o StrictHostKeyChecking=accept-new \
    -o "ProxyCommand=ssh -S '$(socket)' -W %h:%p $CLUSTER_HOST" \
    -L "127.0.0.1:$LOCAL_PORT:127.0.0.1:$port" ${user:+-l "$user"} "$node" \
    </dev/null >/dev/null 2>"$STATE_DIR/tunnel.log" || {
    log "Could not open the tunnel on port $LOCAL_PORT:"
    cat "$STATE_DIR/tunnel.log" >&2
    exit 1
  }
  for _ in $(seq 1 30); do
    curl -s -o /dev/null --max-time 5 "http://127.0.0.1:$LOCAL_PORT/" && return
    sleep 1
  done
  log "The tunnel is open but the dashboard does not answer yet."
}

open_browser() {
  [ "${NO_BROWSER:-}" = "1" ] && return
  if command -v open >/dev/null; then
    open "$1"
  elif command -v xdg-open >/dev/null; then
    xdg-open "$1" >/dev/null 2>&1
  fi
}

start() {
  require_settings
  ensure_connection
  ensure_checkout
  local job address node port
  job="$(running_job)"
  if [ -n "$job" ]; then
    log "Reusing dashboard job $job."
  else
    job="$(submit_job | job_id)"
    [ -n "$job" ] || { log "Submitting the dashboard job failed."; exit 1; }
    log "Submitted dashboard job $job ($CPUS CPUs, $MEM, $TIME_LIMIT)."
  fi
  address="$(wait_for_address "$job")" || exit 1
  read -r node port <<<"$address"
  open_tunnel "$node" "$port"
  local url="http://localhost:$LOCAL_PORT/"
  log ""
  log "Dashboard: $url  (job $job on $node)"
  log "Stop it with: $0 stop"
  open_browser "$url"
}

stop() {
  require_settings
  ensure_connection
  local jobs
  jobs="$(remote "squeue -u \$USER -n $JOB_NAME -h -o %i | tr '\n' ' '")"
  if [ -n "${jobs// /}" ]; then
    remote "scancel $jobs"
    log "Cancelled dashboard job(s): $jobs"
  else
    log "No dashboard job is running."
  fi
  close_tunnel
  log "Tunnel on port $LOCAL_PORT closed."
}

status() {
  require_settings
  if connected; then
    remote "squeue -u \$USER -n $JOB_NAME -o '%.10i %.9T %.10M %.10l %N'"
  else
    log "Not connected to $CLUSTER_HOST (run: $0 start)."
  fi
  if pgrep -f -- "$(tunnel_pattern)" >/dev/null; then
    log "Tunnel: open on http://localhost:$LOCAL_PORT/"
  else
    log "Tunnel: closed"
  fi
}

update() {
  require_settings
  ensure_connection
  ensure_checkout
  # The checkout is dedicated to the dashboard, so it follows its upstream
  # branch even if that branch was rewritten, unless files were edited there.
  remote "
    cd \"$REMOTE_CHECKOUT\" &&
    if [ -n \"\$(git status --porcelain --untracked-files=no)\" ]; then
      echo 'The cluster checkout has local changes; commit or discard them first:' >&2
      git status --short --untracked-files=no >&2
      exit 1
    fi &&
    git fetch -q && git reset -q --hard '@{upstream}' && git log --oneline -1 &&
    cd python_package && uv sync -q --python 3.12"
  # A running server keeps the code it started with.
  if [ -n "$(running_job)" ]; then
    log "Restarting the dashboard on the updated checkout."
    stop
    start
  fi
}

case "${1:-start}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  update) update ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac
