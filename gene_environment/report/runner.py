"""
runner.py

Runs every "report" module (Word tables + figures + CSVs) in one shot,
via `run_all_reports()`. Each report is registered in REPORTS as a
(name, callable, description) triple.

Adding a new report module later only means: write its module with a
`run_<name>(...)` entry point (see generate_table1.py / generate_table2.py
for the pattern), add one `_run_<name>()` wrapper below (a one-liner
calling `_run_subprocess(...)`, see below) with the defaults you want for
a routine "generate everything" run, and append it to REPORTS. cli.py's
`generate-reports` command does not need to change.

Why subprocess, not a plain in-process function call
------------------------------------------------------
Each report module opens its own DB connection(s) through the shared
pool in gene_environment.db.connection (get_annotated_results,
call_stored_routine_to_df, the per-exposure tested-variant-count
queries...). When every report used to be launched as its own
`python -m gene_environment.report.<module>` process, each one got a
completely fresh connection pool for free. Running them back-to-back
in a single long-lived process instead (as a plain in-process call)
shares one pool across all of them for the whole run, and can leave it
in a bad state by the time a later report runs -- surfacing as
`mysql.connector.errors.OperationalError: MySQL Connection not
available` on whichever report happens to run last. `_run_subprocess`
below runs each report exactly as it would run standalone (its own
process, its own fresh connection pool), so `generate-reports` gives
you one command without reintroducing that failure mode.

`c9_check` and `generate_c9_stats` are intentionally NOT part of REPORTS
either: they're a separate exploratory analysis chain (restricted
genotype/environment/C9orf72 merge, then per-exposure stratified stats),
not a paper table, and `generate_c9_stats` depends on files that
`c9_check` writes to disk rather than on the DB astores the four reports
above pull from. Both are reachable as their own CLI subcommands
(`run-c9-check`, `generate-c9-stats`).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportSpec:
    name: str
    run: Callable[[], None]
    description: str


def _run_subprocess(module: str) -> None:
    """Run `python -m <module>` as its own process and raise if it fails.
    See the module docstring for why this is a subprocess and not a
    plain in-process function call."""
    cmd = [sys.executable, "-m", module]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")


def _run_table1() -> None:
    from gene_environment.report.generate_table1 import COHORT_MAPPING_CSV

    # Table 1 needs the id -> generation mapping CSV that
    # build_cohort_mapping.py produces by reading VCF headers. Generate it
    # automatically if it's missing, rather than failing -- that mapping
    # only needs to be rebuilt when the VCF sample sets change, so if it's
    # already there we leave it alone and reuse it as-is.
    if not Path(COHORT_MAPPING_CSV).exists():
        log.info("Cohort mapping CSV not found (%s) -- generating it first.", COHORT_MAPPING_CSV)
        print(f"Cohort mapping CSV not found ({COHORT_MAPPING_CSV}) -- generating it first...")
        _run_subprocess("gene_environment.report.build_cohort_mapping")

    _run_subprocess("gene_environment.report.generate_table1")


def _run_table2() -> None:
    _run_subprocess("gene_environment.report.generate_table2")


def _run_table2b() -> None:
    _run_subprocess("gene_environment.report.generate_table2b")


def _run_annotated_tables() -> None:
    _run_subprocess("gene_environment.report.build_annotated_tables")


REPORTS: List[ReportSpec] = [
    ReportSpec("table1", _run_table1, "Table 1: cohort descriptive statistics"),
    ReportSpec("table2", _run_table2, "Table 2: significant variants + chromosome enrichment"),
    ReportSpec("table2b", _run_table2b, "Table 2b: gene annotations for significant variants"),
    ReportSpec("annotated-tables", _run_annotated_tables, "Supplementary/main-text gene annotation tables"),
]


def run_all_reports(only: Optional[Iterable[str]] = None) -> None:
    """Run every registered report, in REPORTS order, or only the ones
    named in `only`. Raises ValueError if `only` names an unknown report.
    Stops at the first report that fails (its exception propagates) --
    reports after it in the list are not attempted."""
    known = {r.name for r in REPORTS}
    if only:
        only = list(only)
        unknown = set(only) - known
        if unknown:
            raise ValueError(f"Unknown report name(s): {sorted(unknown)}. Available: {sorted(known)}")
        selected = [r for r in REPORTS if r.name in set(only)]
    else:
        selected = REPORTS

    for spec in selected:
        log.info("=== Running report: %s (%s) ===", spec.name, spec.description)
        print(f"\n=== Running report: {spec.name} ({spec.description}) ===")
        spec.run()
        log.info("=== Done: %s ===", spec.name)

    print(f"\nAll requested reports completed ({len(selected)}/{len(REPORTS)}).")
