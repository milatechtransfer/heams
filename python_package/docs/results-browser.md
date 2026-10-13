# Results browser

`heams browse` serves a local, read-only web GUI over saved results. It is for
reviewing runs, not a leaderboard: start from the campaign-wide criterion
table and drill down to the transcript and judge output of any sample.
Nothing is re-run and no API keys are needed.

```sh
cd python_package
uv run heams browse --results-dir ../results
```

The command prints a local URL (default `http://127.0.0.1:8765/`) and opens it
in your browser. Stop it with `Ctrl+C`. Flags are listed in the
[CLI reference](cli-reference.md#heams-browse).

## What it reads

Point `--results-dir` at a single run directory, a campaign snapshot, or the
whole `results/` folder; it is searched recursively:

- Every directory with a `runs.jsonl` is an **experiment**. Its benchmark,
  memory system, agent model and `limit` come from `experiment_config.*.yaml`;
  its start date from the `run_id` in that file name; its status from
  `run_status.json` (`complete`, `failed`, `incomplete` when the file is
  missing, e.g. a job that was still running, or `unknown` when it cannot be
  read); its sample count from the records in `runs.jsonl`.
- Scores are computed from `runs.jsonl` itself, so runs never processed by
  `heams generate-report` are included. When the run directory has a
  `submetrics.csv` (written at run level by `heams generate-report --detailed`),
  each experiment's rows are also shown on its experiment page.
- Directories whose name starts with `_` (`_rollup/`, the browser's own cache)
  and agent `workspace/` folders are ignored.

BEAM experiments share `type: beam`; the browser splits them by the config's
`split` (`BEAM 100K`, `500K`, `1M`, `10M`).

## Views

Every view and filter is encoded in the URL, so a link to a filtered table or
a single failing sample can be pasted into a review thread for anyone
browsing the same directory.

### Criterion table

Memory systems as columns; rows grouped by dimension (Performance, Privacy,
Efficiency), benchmark and criterion. Performance benchmarks are ordered by
context size.

Results for one memory system and benchmark are often spread over several
runs: smoke tests, subsets, retries, full campaigns, and runs still in
progress. Each column therefore takes, per benchmark, **one run**, chosen by
the **Source run** menu:

| Source run | Picks |
|---|---|
| Largest run (default) | The run with the most samples recorded in `runs.jsonl`, finished or not; the newest breaks ties |
| Latest finished run | The newest finished run, whatever its size |
| Latest finished full-dataset run | The newest finished run over its whole dataset: no `limit` and, for Retention Policy, no `policy_ids` selection |

A run is **finished** when its `run_status.json` reports success. Runs
that are still going, were cut off, or failed (`incomplete`, `failed`) only
count for "Largest", and their cells are flagged with that status. Their
numbers cover the samples recorded so far; use **rescan** to recount and
re-read runs that have grown.

Columns can therefore mix runs from different campaigns and dates. Every
cell shows its sample count and run date (`n=500 · 09-14`), and hovering shows
the run directory, start time, status, sample count and `limit`. The
collapsible **Sources** panel lists, for every memory system × benchmark, the
selected run (linked), its date, size and status, and how many other runs
exist. When no run matches, the cell shows the newest run's status instead.

!!! warning "Check the sources before comparing columns"
    "Largest" can select an unfinished run, a subset (`limit: 75`) or a run
    of the same benchmark type made with a different dataset or evaluator
    settings; "Latest" can select a smoke run (`limit: 1`). Use "Latest
    finished full-dataset run" for whole-dataset numbers, and the Sources
    panel to see where each column came from.

Cell values:

- Each judge record is assigned to a criterion: its `criterion` tag if the
  judge sets one, else the first tag group the benchmark declares in
  `criterion_tags` (for example BEAM `question_type`, MPBench
  `attack_class`, Erasability & Editability `task_type / phase`); see
  `heams.reporting.criteria`. A sample's criterion score is the mean of its
  records; the cell is the mean over samples, with `n` samples.
- The `Overall` row is the task accuracy: the mean `accuracy_score` of the
  scored samples.
- Samples without a valid score (`score_available: false`, for example a
  judge or runtime failure) are left out of every value, not counted as 0,
  and reported as `excl.` on the `Overall` row. A run with no scored sample
  shows `N/A`.
- Efficiency rows are per-sample means of duration, LLM tokens, memory
  operations and memory-operation error rate.
- The best value in each row is underlined (lowest for time, tokens and
  error rates). Percentages are heat-shaded.

Retention Policy runs over every policy (`policy_ids: null`) are table
sources like any other benchmark. Runs restricted to some policies (for
example a smoke run on `RP-01`, or older one-experiment-per-policy
campaigns) cannot stand for the benchmark, so they are listed under
**Experiments** only.

Click a cell to open the experiment behind it, filtered to that criterion.
The **Experiments** tab lists every experiment, newest first, with its date,
status, sample count and scope (full dataset, `limit`, or subset).

### Experiment

Status, agent model, mean accuracy and fully successful samples, followed by:

- the headline aggregates from `submetrics.csv`, the resolved experiment
  config and `run_status.json`;
- a pass-rate breakdown for every judge-record tag (question type, attack
  class, difficulty, …), each linking to the matching samples;
- the list of samples, searchable by scenario, question and tag, filterable
  by result (all correct, any failure, all wrong, unscored) and sortable by
  accuracy, failures, turns, duration and tokens.

### Sample

One `RunResult`: accuracy and efficiency figures, the judge's audit notes,
every judge record (expected, observed, reason) with a pass/fail filter, the
transcript with each tool call's arguments and output, system and component
metrics, and the memory ledger. **Prev / Next** step through samples.

Recorded values are laid out for reading without changing what they say:
JSON or Python `str()` written as text (a tool output such as
`['{"content": ...}']`) is parsed and shown as indented `key: value` lines,
text stored escaped (a model answer as a Python repr inside a judge reason)
gets its line breaks back, and long cells expand in place on click. Timeline
rows preview an event's text values (`content: ...`) rather than its JSON.
**Download raw JSON** keeps the exact recorded form.

LaTeX between `\(…\)`, `\[…\]`, `$$…$$` or `$…$` is typeset as math.
The renderer ([Temml](https://temml.org), about 170 KB) is fetched only when
a page holds math, then cached by the browser, and each text block is
typeset as it scrolls into view, so pages without math pay nothing.
Expressions that do not parse stay as written; **Typeset math** on the sample
page switches back to the source (remembered per browser).

Benchmarks make this easier by putting the compared texts in a judge
record's `expected` and `observed` fields and keeping `reason` to the
verdict, for example ``extracted AIME answer '214', expected '204'``.

Full-context runs can hold megabytes of recalled history per tool call, so
long strings, long lists and oversized tool or ledger payloads are shortened
for display. **Load untruncated** shows the full record, and **Download raw
JSON** saves the exact `runs.jsonl` line. Samples over 64 MB (some
EvoMemory records are about 2 GB) are only shown shortened; download them
raw to see everything.

### Execution trace timeline

When a run wrote an `execution_traces.jsonl` sidecar (one trace per sample,
joined through `RunResult.trace_id`), the sample page shows that sample's
**execution trace**: every framework and agent action in the order it
happened.

- Summary cards: outcome, event count, wall time, LLM calls and tokens, tool
  calls, memory operations and errors, plus how often each tool was called.
- A waterfall with one row per event: start offset, duration, type, name,
  step, status, a bar on the sample's time axis, and a content preview
  (with the tool arguments). Rows are grouped by phase and user stimulus,
  colored by type (priming, manipulation, user stimulus, memory operation,
  LLM call, tool call), and failed events are highlighted.
- Filters by event type (click the chips), phase, stimulus, free text, and
  errors only. With a filter on, the bars zoom to the matching events' time
  window. Filters are kept in the URL.
- Click a row to open the event in full: timing and coordinates, token
  usage, error, content, arguments, result, and for LLM calls every input
  message sent to the model.

LLM-call events carry the whole prompt, so a trace is often much larger than
its run record. The timeline reads only each event's small fields and
previews; an event's full content is loaded when you open it. Samples
without a trace, or from runs made before traces were recorded, say so in
place of the timeline. Traces hold model-visible prompts, recalled memory
and tool output; treat them as private run artifacts.

## Where names, sections and order come from

The browser has no list of benchmarks or memory systems. Everything it shows
about them is declared by the framework, so results of a new benchmark or a
new memory plugin appear correctly without changing the browser:

| Shown | Declared by |
|---|---|
| Benchmark name, section (Performance or Privacy), order, criteria | The benchmark's `report_info` (`BenchmarkInfo`, see [Adding a benchmark](adding-benchmark.md#declare-how-the-benchmark-is-reported)) |
| Separate rows per variant (BEAM splits) and their order | `BenchmarkInfo.variant` and `variant_context_tokens` |
| Which runs cover the whole dataset | `BenchmarkInfo.scope` |
| Memory system names | The plugin's `MemoryPlugin.display_name`; two configurations of one plugin keep their config ids apart |
| Trace event colors and counts | The event kinds in `heams.execution_traces` (`TRACE_EVENT_KINDS`) |

Results of a benchmark the installed version does not declare are still
listed, under their config `type` in an "Other" section. Memory systems whose
plugin is not installed are shown by their config `id`; install the plugin
(for example `uv sync --extra all-memory-plugins`) to get its display name.

## Performance and caching

Each experiment's `runs.jsonl` is read once to record where every sample
starts plus a short summary of it (scores, tags, metrics). That takes about
20 s for a 3.6 GB file and is cached as JSON under
`<results-dir>/_results_browser_cache/`; the cache is rebuilt when
`runs.jsonl` changes size or modification time. Opening a sample then reads
only that one line. When the results directory is read-only, indexes are
kept in memory for the session instead.

The server starts indexing the runs the table needs as soon as it starts,
on a background thread, so the page is usable immediately: cells show
`indexing` and fill in as runs are processed (the page refreshes itself).
The default selection is indexed first, then the other Source run choices so
switching is quick. The first start on the whole `results/` folder takes a
few minutes; later starts read the cache.

Use **rescan** on the table page after new runs land while the server is
running.

Records over 64 MB are never loaded whole. The file is memory-mapped and
such records are walked in place: only the small fields a summary needs
(scores, metrics, the first question) are decoded, and the sample view is
built from the same walk. A 7.7 GB file of four EvoMemory records indexes in
about two minutes within a 4 GB Slurm job. Indexes are cached in
`--cache-dir` when given, in one subdirectory per results directory so
one cache can serve several; use it when the results directory is shared or
read-only so the browser does not write into it.

## Viewing cluster results from your laptop

Campaign results usually stay on the cluster's shared file system, where they
are too large to copy. The browser then runs **on the cluster**, as a small
Slurm CPU job next to the data, and your laptop's browser reaches it through
an SSH tunnel:

```text
laptop browser ── http://localhost:8767
   └─ SSH tunnel ── <login host> ── <compute node>:127.0.0.1:<port>
                                      └─ heams browse (Slurm job)
                                           └─ reads <results-dir>
```

The server listens on the compute node's loopback interface only, so it is
reachable only through an SSH session to that node, which only you hold. All
commands below run **on your laptop**; the ones that act on the cluster go
through SSH.

Requirements:

- An SSH host entry for the cluster's login node in your `~/.ssh/config`
  (called `<cluster>` below), with your cluster user name.
- SSH access, through the login node, to compute nodes running your jobs
  (the usual Slurm setup). The tunnel ends on the node itself.
- `uv` on the cluster, and a way for the cluster to clone this repository.

### Quick start

`python_package/cluster/browse_dashboard.sh` does every step of the manual
procedure below.

!!! note "First, enter your cluster's values in `.env`"
    The launcher reads its settings from the repository's `.env`, the same
    git-ignored file that holds your API keys, so your cluster's host, paths
    and partition never end up in the repository. `.env_template` lists the
    `BROWSE_*` keys with empty values. If you don't have a `.env` yet, create
    it from the template at the repository root; then fill in at least
    `BROWSE_CLUSTER_HOST` and `BROWSE_RESULTS_DIR`:

    ```sh
    cp .env_template .env    # only if .env does not exist yet
    $EDITOR .env
    ```

    If your `.env` predates these keys, copy the `BROWSE_*` block from
    `.env_template` into it. The script reads only the `BROWSE_*` lines, as
    literal values (never executed); empty values fall back to the defaults
    below, and a variable exported in your shell takes precedence over the
    file. From a linked worktree without its own `.env`, it uses the main
    checkout's, like the `heams` CLI. Without the two required keys the script
    stops and says which one is missing.

Then start the dashboard:

```sh
python_package/cluster/browse_dashboard.sh
```

It opens one shared SSH connection (answer your cluster's login prompt,
including any one-time password, once; the connection is reused for 12
hours), prepares a dedicated checkout on the cluster on first use, reuses the
running dashboard job or submits one, waits for it, opens the tunnel and
opens **http://localhost:8767** in your browser.

| Command | Does |
|---|---|
| `browse_dashboard.sh` or `browse_dashboard.sh start` | Connect, start or reuse the job, tunnel, open the browser |
| `browse_dashboard.sh status` | Show the dashboard job and whether the tunnel is open |
| `browse_dashboard.sh stop` | Cancel the job and close the tunnel |
| `browse_dashboard.sh update` | Pull and re-sync the cluster checkout, and restart a running dashboard on it |

A fresh start takes about a minute, depending on the queue; running it again
while the job is up reopens the tunnel in a few seconds.

| `.env` key | Meaning | Default |
|---|---|---|
| `BROWSE_CLUSTER_HOST` | SSH host of the login node, from your `~/.ssh/config` | required |
| `BROWSE_RESULTS_DIR` | Results directory on the cluster | required |
| `BROWSE_SLURM_PARTITION`, `BROWSE_SLURM_ACCOUNT` | Where the job runs | cluster defaults |
| `BROWSE_CPUS`, `BROWSE_MEM`, `BROWSE_TIME_LIMIT` | Job resources | `2`, `8G`, `08:00:00` |
| `BROWSE_LOCAL_PORT` | Port on your laptop | `8767` |
| `BROWSE_REMOTE_CHECKOUT`, `BROWSE_REMOTE_CACHE` | Dedicated checkout and index cache on the cluster | under `$SCRATCH` (or `$HOME`) |
| `BROWSE_REPO_URL`, `BROWSE_BRANCH` | What the first use clones | this repository's HTTPS URL, default branch |

Any of them can also be set for one run, for example
`BROWSE_LOCAL_PORT=8780 BROWSE_MEM=16G python_package/cluster/browse_dashboard.sh`;
`NO_BROWSER=1` skips opening the browser.

### Manual procedure

The same steps by hand, for when you need to adapt or debug one. Replace
`<cluster>`, `<results-dir>` and the other placeholders with your values.

#### 1. Open one shared connection

Open a background "master" connection that the following commands reuse, so
any login prompt (password, one-time code) is answered once:

```sh
mkdir -p ~/.cache/heams-browse
SOCKET=~/.cache/heams-browse/<cluster>.sock
ssh -M -S "$SOCKET" -o ControlPersist=12h -Nf <cluster>
ssh -S "$SOCKET" -O check <cluster>
```

`Master running` means you are connected.

#### 2. Prepare a checkout on the cluster (once)

Use a checkout that no campaign job runs from: campaign checkouts must not
be synced while their jobs are pending or running. `make setup` is not
needed: the browser uses no memory plugins and no `.env`. The commands run
through `bash -lc` because tools such as `uv` are often only on a login
shell's `PATH`.

```sh
ssh -S "$SOCKET" <cluster> "bash -lc 'git clone https://github.com/milatechtransfer/heams.git \$SCRATCH/heams-browse && cd \$SCRATCH/heams-browse/python_package && uv sync --python 3.12'"
```

To update it later (the checkout is dedicated to the dashboard, so it can
follow its branch even after the branch is rewritten):

```sh
ssh -S "$SOCKET" <cluster> "bash -lc 'cd \$SCRATCH/heams-browse && git fetch && git reset --hard @{upstream} && cd python_package && uv sync --python 3.12'"
```

#### 3. Start the server job

The job reads the results directory and keeps its index cache in your own
space (`--cache-dir`), so it never writes into a shared folder. `--port 0`
picks a free port on the node. Add `--partition` or `--account` if your
cluster needs them.

```sh
ssh -S "$SOCKET" <cluster> "bash -lc 'mkdir -p \$SCRATCH/heams-browse/logs && cd \$SCRATCH/heams-browse/python_package && sbatch --parsable --job-name=heams-browse --cpus-per-task=2 --mem=8G --time=08:00:00 --output=\$SCRATCH/heams-browse/logs/browse-%j.out --wrap=\"exec .venv/bin/heams browse --results-dir <results-dir> --cache-dir \$SCRATCH/heams-browse-cache --no-open --port 0\"'"
JOB=<job id printed above>
```

#### 4. Find the node and port

Once the job is `RUNNING`, its log shows the address:

```sh
ssh -S "$SOCKET" <cluster> "bash -lc 'squeue -j $JOB -h -o \"%T %N\"; cat \$SCRATCH/heams-browse/logs/browse-$JOB.out'"
```

```text
RUNNING <node>
Browsing <n> experiments under <results-dir>
Open http://127.0.0.1:<port>/ (Ctrl+C to stop)
```

#### 5. Open the tunnel

Forward a local port to the server on the node, hopping through the login
host over the shared connection:

```sh
ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=60 \
  -o "ProxyCommand=ssh -S $SOCKET -W %h:%p <cluster>" \
  -L 127.0.0.1:8767:127.0.0.1:<port> -l <cluster user> <node>
```

Then open **http://localhost:8767**. On the first start the table shows
`indexing` cells that fill in by themselves; later jobs reuse the cache.

#### 6. Stop everything

```sh
ssh -S "$SOCKET" <cluster> "bash -lc 'scancel $JOB'"
pkill -f "127.0.0.1:8767:127.0.0.1:"
ssh -S "$SOCKET" -O exit <cluster>
```

The job also ends at its time limit, and the tunnel closes with the job or
the shared connection.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `Missing setting(s): BROWSE_CLUSTER_HOST BROWSE_RESULTS_DIR` | Add those keys to `.env` (see the `BROWSE_*` block of `.env_template`), or export them. |
| `Could not connect to <cluster>` | Check that `ssh <cluster>` works on its own; the message includes SSH's error. |
| `Could not open the tunnel on port 8767: ... Address already in use` | Another program uses that local port. Close it, or set another `BROWSE_LOCAL_PORT`. |
| The tunnel fails with `Permission denied` | Your cluster does not allow SSH to compute nodes, or only to nodes running your jobs: check that the job is `RUNNING`. |
| `uv: command not found` | The command ran outside a login shell. Wrap it in `bash -lc '...'` as above. |
| Cloning fails on the cluster | The cluster cannot reach GitHub over HTTPS, or uses an SSH key under a host alias: set `BROWSE_REPO_URL`. |
| The job stays `PENDING` | The partition is busy. Wait, or set another CPU `BROWSE_SLURM_PARTITION`. |
| The job state is `OUT_OF_MEMORY` | Set `BROWSE_MEM=16G`. Records of several GB are read in place, but mapped file pages still count towards the job's memory. |
| Cells stay `indexing` for minutes | First start on a large folder: big `runs.jsonl` files are read once over the network file system, then cached. |
| New runs don't appear | Click **rescan** on the table page. |

## Development

The server is `heams.results_browser`:

| Module | Role |
|---|---|
| `catalog.py` | Finds experiments, builds and caches `runs.jsonl` indexes, serves bounded run records |
| `records.py` | Reads records in place over a memory map (large records are never loaded whole) |
| `table.py` | Chooses source runs and computes criterion-table cells |
| `traces.py` | Indexes `execution_traces.jsonl` and builds timeline rows |
| `server.py` | HTTP server: JSON API under `/api/`, the page and its assets under `/static/` |

Criterion rules live in `heams.reporting.criteria`, shared with other
criterion-level reports.

The page is plain HTML, CSS and JavaScript modules in
`heams/results_browser/static/`, with no build step:

| File | Role |
|---|---|
| `js/dom.js` | `html` tagged template that escapes every interpolated value |
| `js/format.js`, `js/filters.js`, `js/router.js` | Pure logic: labels and formats, sample and event filters, URL state |
| `js/readable.js` | Pure logic: decodes escaped text, JSON and Python literals, and lays values out for reading |
| `js/math.js` | Finds LaTeX in text blocks and typesets it lazily with Temml |
| `vendor/temml-<version>/` | Temml, unmodified from its npm release (MIT); excluded from pre-commit hooks and cached by browsers |
| `js/api.js`, `js/ui.js` | API client and shared page components |
| `js/views/*.js` | Criterion table, experiment list, experiment, sample, trace timeline |

The server sends a Content-Security-Policy that allows only the page's own
scripts and styles, so markup may not contain inline scripts or `style`
attributes; dynamic sizes and colors are set through data attributes (see
`mount` in `dom.js`).

Tests are in `tests/results_browser/`. The JavaScript unit tests
(`tests/results_browser/frontend/*.test.mjs`) use Node's built-in runner and
run as part of `pytest` when `node` is installed:

```sh
node --test tests/results_browser/frontend/*.test.mjs
```

## Access

The server binds to `127.0.0.1` by default, so only your machine can reach it.
`--host 0.0.0.0` exposes it, and the raw results with it, to your network; on
the cluster use the SSH tunnel described in
[Viewing cluster results from your laptop](#viewing-cluster-results-from-your-laptop).
