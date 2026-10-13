"""Local web GUI for browsing saved `heams` campaign results.

Backs `heams browse`: a stdlib HTTP server that reads a results directory
in place (``submetrics.csv`` aggregates, ``experiment_config.*.yaml``,
``runs.jsonl``) and serves a single-page app for navigating from the
campaign-wide criterion table down to individual benchmark samples.
"""
