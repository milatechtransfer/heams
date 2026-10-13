# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# BEAM 1M with deterministic recall for Letta, each job on its own throwaway Postgres.
# One-time prerequisite: bash cluster/letta/setup_postgres.sh (on a compute node).

CAMPAIGN_NAME="heams_performance_exps_letta"
SLURM_TIME="72:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"
RUN_WRAPPER="cluster/letta/run_with_postgres.sh"

RUN_CONFIGS=(
  "heams_performance_exps/letta_beam_1m.yaml"
)
