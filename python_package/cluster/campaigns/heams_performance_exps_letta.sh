# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# BEAM 1M with deterministic recall for Letta. Each job starts its own throwaway
# Postgres; see memory_plugins/heams_memory_letta/README.md.

CAMPAIGN_NAME="heams_performance_exps_letta"
SLURM_TIME="72:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "heams_performance_exps/letta_beam_1m.yaml"
)
