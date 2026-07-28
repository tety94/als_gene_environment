"""
runner.py

Runs every "report" module (Word tables + figures + CSVs) in one shot,
via `run_all_reports()`. Each report is registered in REPORTS as a
(name, callable, description) triple.

Adding a new report module later only means: write its module with a
`run_<name>(...)` entry point (see generate_table1.py / generate_table2.py
for the pattern), add one `_run_<name>()` wrapper below with the defaults
you want for a routine "generate everything" run, and append it to
REPORTS. cli.py's `generate-reports` command does not need to change.

`build_cohort_mapping` and `c9_check` are intentionally NOT part of
REPORTS: the former is a one-off prerequisite for Table 1 (only needs
re-running when the VCF sample sets change, not every time reports are
regenerated) and the latter is a standalone diagnostic/merge script, not
a paper table. Both are still reachable as their own CLI subcommands.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportSpec:
    name: str
    run: Callable[[], None]
    description: str


def _run_table1() -> None:
    from gene_environment.report.generate_table1 import run_table1
    run_table1()


def _run_table2() -> None:
    from gene_environment.report.generate_table2 import run_table2
    run_table2()


def _run_table2b() -> None:
    from gene_environment.report.generate_table2b import run_table2b
    run_table2b()


def _run_annotated_tables() -> None:
    from gene_environment.report.build_annotated_tables import run_annotated_tables
    run_annotated_tables()


REPORTS: List[ReportSpec] = [
    ReportSpec("table1", _run_table1, "Table 1: cohort descriptive statistics"),
    ReportSpec("table2", _run_table2, "Table 2: significant variants + chromosome enrichment"),
    ReportSpec("table2b", _run_table2b, "Table 2b: gene annotations for significant variants"),
    ReportSpec("annotated-tables", _run_annotated_tables, "Supplementary/main-text gene annotation tables"),
]


def run_all_reports(only: Optional[Iterable[str]] = None) -> None:
    """Run every registered report, in REPORTS order, or only the ones
    named in `only`. Raises ValueError if `only` names an unknown report."""
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
