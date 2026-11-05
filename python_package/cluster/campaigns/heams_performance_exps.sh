# Campaign definitions are sourced by ../submit_campaign.sh.
# Cluster-specific settings (PARTITION, SLURM_ACCOUNT) come from the environment;
# see docs/cluster.md.
# BEAM 1M with deterministic recall, one Slurm job per memory system (run in parallel).
# Letta runs from heams_performance_exps_letta.sh.

CAMPAIGN_NAME="heams_performance_exps"
# HippoRAG ingests ~6s/passage (~85 min per conversation, ~50h for 35); resubmit to resume.
SLURM_TIME="72:00:00"
CPUS_PER_TASK=4
MEM_PER_TASK="64G"

RUN_CONFIGS=(
  "heams_performance_exps/hipporag_beam_1m.yaml"
  "heams_performance_exps/mem0_beam_1m.yaml"
  "heams_performance_exps/in_context_beam_1m.yaml"
  "heams_performance_exps/no_memory_beam_1m.yaml"
)
