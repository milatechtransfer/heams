# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# Each run config contains one full benchmark experiment with in-context memory.

CAMPAIGN_NAME="in_context_all_benchmarks"
SLURM_TIME="48:00:00"
CPUS_PER_TASK=1
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "in_context_all_benchmarks/locomo.yaml"
  "in_context_all_benchmarks/long_mem_eval.yaml"
  "in_context_all_benchmarks/transparency.yaml"
  "in_context_all_benchmarks/contextual_integrity.yaml"
  "in_context_all_benchmarks/stale.yaml"
  "in_context_all_benchmarks/mpbench.yaml"
  "in_context_all_benchmarks/beam_100k.yaml"
  "in_context_all_benchmarks/beam_500k.yaml"
)
