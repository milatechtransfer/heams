# Cluster workflow

The Slurm workflow schedules complete, declarative run configs. A campaign
manifest chooses which run configs start in parallel; each selected run config
then executes its `experiments` list serially through `heams run`.

```text
cluster/campaigns/<campaign>.sh
  -> one Slurm job per listed configs/runs/<run>.yaml
    -> heams run --config configs/runs/<run>.yaml --output-dir <campaign-root>/<campaign-instance>/<index>_<run-stem>
      -> experiments execute serially
```

The cluster scripts supply only the artifact output directory. Model, memory,
benchmark, limit, and concurrency settings remain authoritative in the referenced
YAML component and run configs.

## Define a campaign

Create a file under `cluster/campaigns/`. This is the single, human-editable
list of run configs that should execute in parallel. The bundled
`cluster/campaigns/smoke_campaign.sh`:

```bash
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
```

Campaign submissions always pass `--requeue` to `sbatch`. This enables the
Slurm requeue behavior supported by the site. HEAMS does not add time-limit
signal handling or automatic retries for application errors.

`RUN_CONFIGS` entries reference YAML files only under `configs/runs/`. Paths
such as `../run.yaml` and missing files are rejected. Each entry is checked just
before its job is submitted, so an invalid entry stops the loop after the
earlier entries were already submitted. The
campaign definition sets one shared resource profile for all of its run configs:
`SLURM_TIME`, `CPUS_PER_TASK`, `MEM_PER_TASK`, and optional `GRES` (for example
`gpu:1`). Define a separate campaign when runs need different resource
profiles.

## Cluster settings

Settings that depend on your cluster are not stored in campaign files. Set them
in your environment, for example in your shell profile on the cluster:

| Variable | Meaning |
|----------|---------|
| `PARTITION` | Slurm partition for every job. When unset, Slurm uses the cluster's default (or `SBATCH_PARTITION`) |
| `SLURM_ACCOUNT` | Slurm account to charge. When unset, no `--account` is passed |
| `GRES` | Generic resources passed as `--gres`, for example `gpu:1` for runs that embed on a GPU. When unset, no `--gres` is passed |

A campaign file may still set these itself; a value set there overrides the
environment.

## Prepare the cluster checkout

On the cluster, clone the repository and create the untracked `.env`
file in that checkout.

```sh
git clone <repository-url> ~/heams
cd ~/heams
cp .env_template .env
$EDITOR .env

cd python_package
make setup
# Letta campaigns only: download Letta's migrations while there is network access,
# and check that a private database starts
uv run --python 3.12 python -m heams_memory_letta
```

The `.env` file supplies credentials such as `OPENAI_API_KEY`; it is ignored by
Git and belongs only on the cluster checkout that will submit and execute the
jobs. Complete setup before submitting a campaign, then do not run `make setup`,
`uv sync`, or `uv pip install` in that checkout while its jobs are pending or
running. Campaign submission and jobs use the existing environment without
synchronizing it.

Before submitting a full campaign, run a small config-driven job from the login node to
verify credentials, dependency installation, and download access:

```sh
uv run --python 3.12 heams run --config configs/runs/testing/locomo_smoke.yaml --output-dir results/locomo_smoke
```

The Slurm compute nodes must have outbound access to the model provider and the
LoCoMo dataset URL if the dataset is not already cached. On clusters where
compute nodes cannot download data, ask the cluster administrator for a shared
dataset location or materialize the dataset from a networked login node before
submitting jobs.

## Submit a campaign

```sh
cd /path/to/heams/python_package
PARTITION=my-partition cluster/submit_campaign.sh cluster/campaigns/smoke_campaign.sh /scratch/$USER/heams
```

Or use the Make wrapper, which defaults to the `smoke_campaign` campaign
and an output root of `results/campaigns` (under `python_package/`):

```sh
PARTITION=my-partition CAMPAIGN_OUTPUT_DIR=/scratch/$USER/heams make launch-campaign
```

Both the Make target and direct invocations of `submit_campaign.sh` use the
same unconditional requeue behavior.

Set `CAMPAIGN` or `CAMPAIGN_OUTPUT_DIR` to choose a different campaign or output
root.

The script submits one Slurm job per run-config entry, checking each entry just
before submitting it. It does not install or synchronize dependencies. The second argument
is a campaign output root. At submission time, the script creates one shared
campaign directory named
`<campaign-name>_<UTC timestamp>-<short UUID>` beneath that root. Each listed
config receives one distinct stable exact run directory named
`<index>_<run-stem>` below the campaign instance. Requeued or manually repeated
attempts must use that same command and directory.

Each job runs:

```sh
exec srun uv run --python 3.12 heams run --config configs/runs/<run>.yaml --output-dir /scratch/$USER/heams/smoke_campaign_20260727-163603-a1b2c3d4/01_<run-stem>
```

For example, a campaign submitted to `/scratch/$USER/heams`
produces this layout:

```text
/scratch/$USER/heams/
└── smoke_campaign_20260727-163603-a1b2c3d4/
  ├── 01_smoke_campaign_no_memory/
  │   ├── run_config.20260727-163611-8df325f0.yaml
  │   └── gpt56_luna_no_memory_locomo/
  │       ├── experiment_config.20260727-163611-8df325f0.yaml
  │       └── workspace/
  ├── 02_smoke_campaign_in_context/
  │   ├── run_config.20260727-163614-3c7e91ab.yaml
  │   └── gpt56_luna_in_context_locomo/
  │       ├── experiment_config.20260727-163614-3c7e91ab.yaml
  │       └── workspace/
  ├── 03_smoke_campaign_milamem/
  ├── 04_smoke_campaign_mem0/
  └── logs/
    └── ...
```

The smoke campaign is organized around memory systems rather than benchmarks.
It submits four run-config jobs: `no_memory`, `in_context`, `milamem`, and
`mem0`. Each job runs nine benchmark experiments, for 36 experiments total.

Slurm stdout and stderr logs are stored in the campaign instance's `logs/`
directory alongside the run artifacts. Override `REMOTE_REPO` when the checkout
layout differs. Credentials
remain in the repository-root `.env` and are never copied into a campaign
definition or Slurm command.

## Results and retention

Scratch directories are temporary. Copy final result directories and logs to
durable storage according to cluster policy after the campaign completes.

## Browse results

To review campaign results without copying them off the cluster, run
`heams browse` as a CPU job and open it from your laptop through an SSH tunnel:
see [Viewing cluster results from your laptop](results-browser.md#viewing-cluster-results-from-your-laptop).
