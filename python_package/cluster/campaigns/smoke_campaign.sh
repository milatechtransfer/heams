# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# Each run config covers all benchmarks for one memory system.

CAMPAIGN_NAME="smoke_campaign"
SLURM_TIME="04:00:00"
CPUS_PER_TASK=1
MEM_PER_TASK="16G"

RUN_CONFIGS=(
  "smoke_campaign/no_memory.yaml"
  "smoke_campaign/in_context.yaml"
  "smoke_campaign/milamem.yaml"
  "smoke_campaign/mem0.yaml"
)
