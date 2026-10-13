# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# MilaMem recall ablation: OpenAI text-embedding-3-small, top_k 5/10, BEAM 1M + CI.

CAMPAIGN_NAME="milamem_recall_ablation_openai"
SLURM_TIME="48:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "milamem_recall_ablation/openai_k5_beam_1m.yaml"
  "milamem_recall_ablation/openai_k10_beam_1m.yaml"
  "milamem_recall_ablation/openai_k5_contextual_integrity.yaml"
  "milamem_recall_ablation/openai_k10_contextual_integrity.yaml"
)
