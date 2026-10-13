# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# Smoke (limit: 1) for Qwen3-Embedding on BEAM 1M; submit with GRES=gpu:1.

CAMPAIGN_NAME="milamem_recall_ablation_smoke_gpu"
SLURM_TIME="12:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "milamem_recall_ablation/smoke_qwen3_0_6b_beam_1m.yaml"
)
