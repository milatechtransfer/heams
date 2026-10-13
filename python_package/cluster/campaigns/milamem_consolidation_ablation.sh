# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# MilaMem consolidation ablation (C1): off / on with pending_threshold None / on
# with pending_threshold 10, on EE (edit), STALE, and LoCoMo. One job per run.

CAMPAIGN_NAME="milamem_consolidation_ablation"
# STALE with consolidation on stores ~237k memories (~50h+); pick a partition allowing 96h.
SLURM_TIME="96:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "milamem_ablations/consolidation_off_ee_edit.yaml"
  "milamem_ablations/consolidation_off_stale.yaml"
  "milamem_ablations/consolidation_off_locomo.yaml"
  "milamem_ablations/consolidation_on_none_ee_edit.yaml"
  "milamem_ablations/consolidation_on_none_stale.yaml"
  "milamem_ablations/consolidation_on_none_locomo.yaml"
  "milamem_ablations/consolidation_on_10_ee_edit.yaml"
  "milamem_ablations/consolidation_on_10_stale.yaml"
  "milamem_ablations/consolidation_on_10_locomo.yaml"
)
