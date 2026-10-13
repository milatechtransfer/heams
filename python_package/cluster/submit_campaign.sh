#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_REPO="${REMOTE_REPO:-$(cd "$HERE/../.." && pwd)}"

usage() {
  echo "Usage: $0 cluster/campaigns/<campaign>.sh <output-directory>" >&2
}

[ "$#" -eq 2 ] || { usage; exit 2; }
campaign_definition="$1"
campaign_output_dir="$2"

[ -n "$campaign_output_dir" ] || {
  echo "Output directory must not be empty." >&2
  exit 2
}
mkdir -p "$campaign_output_dir"
campaign_output_dir="$(cd "$campaign_output_dir" && pwd)"

campaign_definition="${campaign_definition#./}"
case "$campaign_definition" in
  cluster/campaigns/*.sh|campaigns/*.sh) ;;
  *) echo "Campaign definitions must live under cluster/campaigns/." >&2; exit 2 ;;
esac
case "$campaign_definition" in
  campaigns/*) campaign_definition="cluster/$campaign_definition" ;;
esac
[ -f "$REMOTE_REPO/python_package/$campaign_definition" ] || {
  echo "Campaign definition not found: $campaign_definition" >&2
  exit 2
}

# Campaign definitions are trusted, tracked shell data. They set the campaign
# name, shared Slurm resources, and the list of run-config filenames. Cluster
# settings (PARTITION, SLURM_ACCOUNT) come from the environment, so the same
# campaign runs on any Slurm cluster.
# shellcheck source=/dev/null
source "$REMOTE_REPO/python_package/$campaign_definition"

: "${CAMPAIGN_NAME:?Campaign definition must set CAMPAIGN_NAME}"
: "${SLURM_TIME:?Campaign definition must set SLURM_TIME}"
: "${CPUS_PER_TASK:?Campaign definition must set CPUS_PER_TASK}"
: "${MEM_PER_TASK:?Campaign definition must set MEM_PER_TASK}"
: "${RUN_CONFIGS:?Campaign definition must set RUN_CONFIGS}"

case "$CPUS_PER_TASK" in
  ''|*[!0-9]*) echo "CPUS_PER_TASK must be a positive integer." >&2; exit 2 ;;
esac
[ "$CPUS_PER_TASK" -gt 0 ] || { echo "CPUS_PER_TASK must be positive." >&2; exit 2; }

run_command="uv run --python 3.12 heams run"
if [ -n "${RUN_WRAPPER:-}" ]; then
  case "$RUN_WRAPPER" in
    cluster/*.sh) ;;
    *) echo "RUN_WRAPPER must be a .sh script under cluster/: $RUN_WRAPPER" >&2; exit 2 ;;
  esac
  if [[ "$RUN_WRAPPER" =~ (^|/)\.\.($|/) ]]; then
    echo "RUN_WRAPPER must not contain parent-directory paths: $RUN_WRAPPER" >&2
    exit 2
  fi
  [ -f "$REMOTE_REPO/python_package/$RUN_WRAPPER" ] || {
    echo "RUN_WRAPPER not found: $RUN_WRAPPER" >&2
    exit 2
  }
  # The wrapper receives heams run's arguments and runs heams itself.
  run_command="bash $(printf '%q' "$RUN_WRAPPER")"
fi

campaign_instance="${CAMPAIGN_NAME}_$(date -u +%Y%m%d-%H%M%S)-$(uuidgen | tr '[:upper:]' '[:lower:]' | tr -d '-' | cut -c1-8)"
campaign_run_dir="$campaign_output_dir/$campaign_instance"
campaign_log_dir="$campaign_run_dir/logs"
mkdir -p "$campaign_log_dir"

echo "Campaign definition: $campaign_definition"
echo "Campaign output directory: $campaign_run_dir"
echo "Submitting ${#RUN_CONFIGS[@]} run config(s)."
run_index=0
for run_config in "${RUN_CONFIGS[@]}"; do
  case "$run_config" in
    /*|*.yaml|*.yml) ;;
    *) echo "Run configs must be YAML files: $run_config" >&2; exit 2 ;;
  esac
  if [[ "$run_config" =~ (^|/)\.\.($|/) ]]; then
    echo "Run configs must not contain parent-directory paths: $run_config" >&2
    exit 2
  fi
  [ -n "$run_config" ] && [ "${run_config#/}" = "$run_config" ] || {
    echo "Run configs must be relative paths: $run_config" >&2
    exit 2
  }
  config_path="$REMOTE_REPO/python_package/configs/runs/$run_config"
  [ -f "$config_path" ] || {
    echo "Run config not found: configs/runs/$run_config" >&2
    exit 2
  }
  [ ! -L "$config_path" ] || {
    echo "Run config must not be a symlink: $run_config" >&2
    exit 2
  }
  resolved_config_path="$(realpath "$config_path")"
  case "$resolved_config_path" in
    "$REMOTE_REPO/python_package/configs/runs"/*) ;;
    *) echo "Run config must resolve under configs/runs/: $run_config" >&2; exit 2 ;;
  esac
  run_stem="${run_config%.*}"
  run_stem="${run_stem//\//_}"
  run_index=$((run_index + 1))
  run_output_dir="$campaign_run_dir/$(printf '%02d' "$run_index")_$run_stem"
  job_name="mozmem-${CAMPAIGN_NAME}-${run_stem}"
  printf '%s\n' "  $run_config -> $job_name (output=$run_output_dir partition=${PARTITION:-default} time=$SLURM_TIME cpus=$CPUS_PER_TASK mem=$MEM_PER_TASK gres=${GRES:-none})"

  job_command="set -euo pipefail; cd $(printf '%q' "$REMOTE_REPO/python_package"); exec srun $run_command --config $(printf '%q' "configs/runs/$run_config") --output-dir $(printf '%q' "$run_output_dir")"
  # bash 3.2 with set -u treats "${empty[@]}" as unbound, so the partition,
  # account, and gres flags are appended only when set; otherwise Slurm uses the
  # cluster's defaults (or SBATCH_PARTITION / SBATCH_ACCOUNT).
  sbatch_args=(
    --job-name="$job_name"
    --time="$SLURM_TIME"
    --ntasks=1
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM_PER_TASK"
    --output="$campaign_log_dir/%x_%j.out"
    --error="$campaign_log_dir/%x_%j.err"
  )
  if [ -n "${PARTITION:-}" ]; then
    sbatch_args+=(--partition="$PARTITION")
  fi
  if [ -n "${SLURM_ACCOUNT:-}" ]; then
    sbatch_args+=(--account="$SLURM_ACCOUNT")
  fi
  if [ -n "${GRES:-}" ]; then
    sbatch_args+=(--gres="$GRES")
  fi
  sbatch_args+=(
    --requeue
    --wrap="bash -lc $(printf '%q' "$job_command")"
  )
  sbatch "${sbatch_args[@]}"
done
