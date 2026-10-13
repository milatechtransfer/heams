# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# MilaMem recall ablation: Qwen3-Embedding-0.6B on BEAM 1M; submit with GRES=gpu:1.

CAMPAIGN_NAME="milamem_recall_ablation_qwen_beam"
SLURM_TIME="48:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "milamem_recall_ablation/qwen3_0_6b_k5_beam_1m.yaml"
  "milamem_recall_ablation/qwen3_0_6b_k10_beam_1m.yaml"
)
