# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# Each run config contains one full benchmark experiment with MilaMem memory.

CAMPAIGN_NAME="milamem_all_benchmarks"
SLURM_TIME="48:00:00"
CPUS_PER_TASK=1
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "milamem_all_benchmarks/locomo.yaml"
  "milamem_all_benchmarks/long_mem_eval.yaml"
  "milamem_all_benchmarks/transparency.yaml"
  "milamem_all_benchmarks/contextual_integrity.yaml"
  "milamem_all_benchmarks/stale.yaml"
  "milamem_all_benchmarks/mpbench.yaml"
  "milamem_all_benchmarks/beam_100k.yaml"
  "milamem_all_benchmarks/beam_500k.yaml"
  "milamem_all_benchmarks/beam_1m.yaml"
  "milamem_all_benchmarks/beam_10m.yaml"
)
