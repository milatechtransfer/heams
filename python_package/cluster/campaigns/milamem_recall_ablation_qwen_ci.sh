# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# MilaMem recall ablation: Qwen3-Embedding-0.6B on CI (small pools, CPU is enough).

CAMPAIGN_NAME="milamem_recall_ablation_qwen_ci"
SLURM_TIME="12:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="32G"

RUN_CONFIGS=(
  "milamem_recall_ablation/qwen3_0_6b_k5_contextual_integrity.yaml"
  "milamem_recall_ablation/qwen3_0_6b_k10_contextual_integrity.yaml"
)
