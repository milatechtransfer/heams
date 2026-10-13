# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# Each run config contains one full benchmark experiment without memory.

CAMPAIGN_NAME="no_memory_all_benchmarks"
SLURM_TIME="48:00:00"
CPUS_PER_TASK=1
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "no_memory_all_benchmarks/locomo.yaml"
  "no_memory_all_benchmarks/long_mem_eval.yaml"
  "no_memory_all_benchmarks/transparency.yaml"
  "no_memory_all_benchmarks/contextual_integrity.yaml"
  "no_memory_all_benchmarks/stale.yaml"
  "no_memory_all_benchmarks/mpbench.yaml"
  "no_memory_all_benchmarks/beam_100k.yaml"
  "no_memory_all_benchmarks/beam_500k.yaml"
  "no_memory_all_benchmarks/beam_1m.yaml"
  "no_memory_all_benchmarks/beam_10m.yaml"
)
