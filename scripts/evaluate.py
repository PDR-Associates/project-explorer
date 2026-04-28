"""Evaluation runner — scores Project Explorer across intent, inference, and response quality.

Usage:
    uv run python scripts/evaluate.py
    uv run python scripts/evaluate.py --project myproj         # live agent tests
    uv run python scripts/evaluate.py --no-live                # classification only
    uv run python scripts/evaluate.py --output report.json     # save JSON report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


# ── golden dataset ─────────────────────────────────────────────────────────────

@dataclass
class QueryCase:
    query: str
    expected_intent: str
    # For live tests: keywords that must appear in the agent response
    expected_keywords: list[str] = field(default_factory=list)
    # Optional project slug — None means cross-project
    project_slug: str | None = None
    # Label for grouping in the report
    category: str = ""


GOLDEN_DATASET: list[QueryCase] = [
    # ── statistical ──────────────────────────────────────────────────────────
    QueryCase("how many stars does this project have",   "statistical", ["star"], category="statistical"),
    QueryCase("how many commits in the last 30 days",    "statistical", ["commit"], category="statistical"),
    QueryCase("who are the top committers",              "statistical", ["committer", "commit"], category="statistical"),
    QueryCase("show me the top contributors",            "statistical", ["contributor", "commit"], category="statistical"),
    QueryCase("list contributors to the project",        "statistical", ["contributor"], category="statistical"),
    QueryCase("commit history over the last year",       "statistical", ["commit"], category="statistical"),
    QueryCase("release cadence for this project",        "statistical", ["release"], category="statistical"),
    QueryCase("lines of code in the repo",               "statistical", ["line", "loc", "code"], category="statistical"),
    QueryCase("growth over time",                        "statistical", category="statistical"),
    QueryCase("weekly commits for this project",         "statistical", category="statistical"),
    QueryCase("how many forks does this have",           "statistical", ["fork"], category="statistical"),
    QueryCase("who committed to this project",           "statistical", ["commit"], category="statistical"),
    QueryCase("show a chart of star growth",             "statistical", category="statistical"),
    QueryCase("committers in the last 90 days",          "statistical", category="statistical"),

    # ── comparison ───────────────────────────────────────────────────────────
    QueryCase("compare project-a and project-b",         "comparison", category="comparison"),
    QueryCase("compare alpha vs beta",                   "comparison", category="comparison"),
    QueryCase("difference between foo and bar",          "comparison", category="comparison"),
    QueryCase("which project is more popular",           "comparison", category="comparison"),
    QueryCase("which has more stars, alpha or beta",     "comparison", category="comparison"),
    QueryCase("alpha versus beta",                       "comparison", category="comparison"),
    QueryCase("alpha vs beta",                           "comparison", category="comparison"),
    QueryCase("side-by-side comparison of alpha and beta", "comparison", category="comparison"),
    QueryCase("head-to-head alpha and beta",             "comparison", category="comparison"),
    QueryCase("more commits than beta",                  "comparison", category="comparison"),
    QueryCase("which has fewer open issues",             "comparison", category="comparison"),

    # ── health ────────────────────────────────────────────────────────────────
    QueryCase("is this project actively maintained",     "health", category="health"),
    QueryCase("how actively maintained is it",           "health", category="health"),
    QueryCase("community health of the project",         "health", category="health"),
    QueryCase("what is the bus factor",                  "health", category="health"),
    QueryCase("contributor diversity",                   "health", category="health"),
    QueryCase("is this project abandoned",               "health", category="health"),
    QueryCase("last activity on the project",            "health", category="health"),

    # ── code_search ───────────────────────────────────────────────────────────
    QueryCase("how do I use the authentication module",  "code_search", category="code_search"),
    QueryCase("show me an example of creating a client", "code_search", category="code_search"),
    QueryCase("how is the parser implemented",           "code_search", category="code_search"),
    QueryCase("source code for the main function",       "code_search", category="code_search"),
    QueryCase("how can I call the retry method",         "code_search", category="code_search"),

    # ── conceptual ────────────────────────────────────────────────────────────
    QueryCase("what is the overall architecture",        "conceptual", category="conceptual"),
    QueryCase("how does the routing work",               "conceptual", category="conceptual"),
    QueryCase("explain the configuration system",        "conceptual", category="conceptual"),
    QueryCase("getting started guide",                   "conceptual", category="conceptual"),
    QueryCase("how to install",                          "conceptual", category="conceptual"),
    QueryCase("overview of the system",                  "conceptual", category="conceptual"),

    # ── general fallback ──────────────────────────────────────────────────────
    QueryCase("tell me something interesting",           "general", category="general"),
    QueryCase("",                                        "general", category="general"),
]


# ── result tracking ───────────────────────────────────────────────────────────

@dataclass
class TestResult:
    query: str
    category: str
    test_name: str
    passed: bool
    detail: str = ""
    latency_ms: int = 0


@dataclass
class EvalReport:
    results: list[TestResult] = field(default_factory=list)

    def add(self, r: TestResult) -> None:
        self.results.append(r)

    def passed(self) -> list[TestResult]:
        return [r for r in self.results if r.passed]

    def failed(self) -> list[TestResult]:
        return [r for r in self.results if not r.passed]

    def score_by_category(self) -> dict[str, tuple[int, int]]:
        cats: dict[str, list[bool]] = {}
        for r in self.results:
            cats.setdefault(r.category, []).append(r.passed)
        return {cat: (sum(v), len(v)) for cat, v in cats.items()}

    def overall_score(self) -> tuple[int, int]:
        return len(self.passed()), len(self.results)


# ── test functions ────────────────────────────────────────────────────────────

def run_intent_classification(report: EvalReport) -> None:
    console.print("\n[bold cyan]Intent Classification[/bold cyan]")
    from explorer.query_processor import QueryProcessor
    qp = QueryProcessor()

    for case in GOLDEN_DATASET:
        t0 = time.monotonic()
        got = qp.classify(case.query).value
        ms = int((time.monotonic() - t0) * 1000)
        passed = got == case.expected_intent
        detail = "" if passed else f"got={got}"
        report.add(TestResult(
            query=case.query[:60],
            category=case.category,
            test_name="intent",
            passed=passed,
            detail=detail,
            latency_ms=ms,
        ))
        sym = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {sym} [{case.category}] {case.query[:55]!r}"
                      + (f"  [red]→ {detail}[/red]" if not passed else ""))


def run_project_inference(report: EvalReport, registry_slug: str) -> None:
    console.print(f"\n[bold cyan]Project Inference (against '{registry_slug}')[/bold cyan]")
    from explorer.registry import ProjectRegistry
    from explorer.agents.base import BaseExplorerAgent

    class _DummyAgent(BaseExplorerAgent):
        def system_prompt(self): return ""
        def tools(self): return []
        def handle(self, *a, **kw): return ""

    reg = ProjectRegistry()
    projects = reg.list_all()
    if not any(p.slug == registry_slug for p in projects):
        console.print(f"  [yellow]⚠[/yellow] Project '{registry_slug}' not found — skipping inference tests")
        return

    agent = _DummyAgent()
    inference_cases = [
        (f"how many stars does {registry_slug} have", registry_slug, True),
        (f"who are the top committers for {registry_slug}", registry_slug, True),
        ("how does the architecture work", registry_slug, False),  # no slug → None expected
    ]

    for query, expected_slug, expect_match in inference_cases:
        t0 = time.monotonic()
        got = agent._infer_project_slug(query)
        ms = int((time.monotonic() - t0) * 1000)
        if expect_match:
            passed = got == expected_slug
            detail = "" if passed else f"got={got!r}"
        else:
            passed = got is None
            detail = "" if passed else f"expected None, got={got!r}"
        report.add(TestResult(
            query=query[:60],
            category="inference",
            test_name="project_inference",
            passed=passed,
            detail=detail,
            latency_ms=ms,
        ))
        sym = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {sym} {query[:55]!r}" + (f"  [red]{detail}[/red]" if not passed else ""))


def run_live_agent_tests(report: EvalReport, project_slug: str) -> None:
    console.print(f"\n[bold cyan]Live Agent Tests (project='{project_slug}')[/bold cyan]")
    console.print("  [dim]These call real agents and require indexed data.[/dim]")

    from explorer.rag_system import RAGSystem

    live_cases = [
        ("how many stars does this project have",  "statistical", ["star"]),
        ("who are the top committers",             "statistical", ["commit"]),
        ("is this project actively maintained",    "health",      ["maintai", "activ", "commit", "contributor"]),
        ("how many releases has this had",         "statistical", ["release"]),
    ]

    rag = RAGSystem()
    for query, intent_label, keywords in live_cases:
        t0 = time.monotonic()
        try:
            response = rag.query(query, project_slug=project_slug)
            ms = int((time.monotonic() - t0) * 1000)
            response_lower = response.lower()
            missing = [kw for kw in keywords if kw not in response_lower]
            passed = len(missing) == 0
            detail = f"missing keywords: {missing}" if missing else ""
        except Exception as exc:
            ms = int((time.monotonic() - t0) * 1000)
            passed = False
            detail = f"exception: {exc}"
            response = ""

        report.add(TestResult(
            query=query[:60],
            category="live_" + intent_label,
            test_name="live_agent",
            passed=passed,
            detail=detail,
            latency_ms=ms,
        ))
        sym = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(
            f"  {sym} [{intent_label}] {query[:45]!r}  [dim]{ms}ms[/dim]"
            + (f"\n       [red]{detail}[/red]" if not passed else "")
        )


def run_compare_agent_tests(report: EvalReport, slug_a: str, slug_b: str) -> None:
    console.print(f"\n[bold cyan]Comparison Tests ({slug_a} vs {slug_b})[/bold cyan]")
    from explorer.agents.compare_agent import CompareAgent

    agent = CompareAgent()
    query = f"compare {slug_a} and {slug_b}"

    # slug extraction
    t0 = time.monotonic()
    slugs = agent._extract_project_slugs(query)
    ms = int((time.monotonic() - t0) * 1000)
    passed = slug_a in slugs and slug_b in slugs
    detail = f"extracted={slugs}" if not passed else ""
    report.add(TestResult(
        query=query,
        category="comparison",
        test_name="slug_extraction",
        passed=passed,
        detail=detail,
        latency_ms=ms,
    ))
    sym = "[green]✓[/green]" if passed else "[red]✗[/red]"
    console.print(f"  {sym} slug extraction: {slugs}" + (f"  [red]{detail}[/red]" if not passed else ""))

    # both slugs route to COMPARISON intent
    from explorer.query_processor import QueryProcessor, QueryIntent
    intent = QueryProcessor().classify(query)
    passed = intent == QueryIntent.COMPARISON
    report.add(TestResult(
        query=query,
        category="comparison",
        test_name="compare_intent",
        passed=passed,
        detail="" if passed else f"got={intent.value}",
        latency_ms=0,
    ))
    sym = "[green]✓[/green]" if passed else "[red]✗[/red]"
    console.print(f"  {sym} intent routing: {intent.value}")


# ── report rendering ──────────────────────────────────────────────────────────

def print_report(report: EvalReport) -> None:
    console.print()
    console.rule("[bold]Evaluation Report")

    # Per-category table
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Category", style="bold")
    table.add_column("Passed", justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Bar", min_width=20)

    for category, (passed, total) in sorted(report.score_by_category().items()):
        pct = passed / total if total else 0
        color = "green" if pct == 1.0 else "yellow" if pct >= 0.8 else "red"
        bar_len = int(pct * 20)
        bar = f"[{color}]{'█' * bar_len}[/{color}]{'░' * (20 - bar_len)}"
        table.add_row(
            category,
            str(passed),
            str(total),
            f"[{color}]{pct:.0%}[/{color}]",
            bar,
        )

    # Overall totals
    total_pass, total = report.overall_score()
    pct = total_pass / total if total else 0
    color = "green" if pct == 1.0 else "yellow" if pct >= 0.8 else "red"
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_pass}[/bold]",
        f"[bold]{total}[/bold]",
        f"[bold {color}]{pct:.0%}[/bold {color}]",
        "",
    )
    console.print(table)

    # Failed tests
    failed = report.failed()
    if failed:
        console.print(f"\n[bold red]Failed tests ({len(failed)}):[/bold red]")
        for r in failed:
            console.print(f"  [red]✗[/red] [{r.category}] {r.query!r}")
            if r.detail:
                console.print(f"      [dim]{r.detail}[/dim]")
    else:
        console.print("\n[bold green]All tests passed![/bold green]")

    # Average latency for live tests
    live = [r for r in report.results if r.test_name == "live_agent"]
    if live:
        avg = sum(r.latency_ms for r in live) / len(live)
        console.print(f"\n[dim]Average live agent latency: {avg:.0f}ms[/dim]")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Project Explorer evaluation runner")
    parser.add_argument("--project", "-p", help="Project slug for live agent tests")
    parser.add_argument("--compare", nargs=2, metavar=("SLUG_A", "SLUG_B"),
                        help="Two project slugs for comparison tests")
    parser.add_argument("--no-live", action="store_true",
                        help="Skip live agent tests (classification + inference only)")
    parser.add_argument("--output", "-o", help="Write JSON report to this file")
    args = parser.parse_args()

    report = EvalReport()

    # Always run classification
    run_intent_classification(report)

    # Project inference requires a known slug
    if args.project:
        run_project_inference(report, args.project)

    # Live agent tests
    if args.project and not args.no_live:
        run_live_agent_tests(report, args.project)

    # Comparison tests
    if args.compare:
        run_compare_agent_tests(report, args.compare[0], args.compare[1])
    elif args.project and not args.no_live:
        console.print("\n[dim]Tip: add --compare SLUG_A SLUG_B to run comparison tests.[/dim]")

    print_report(report)

    if args.output:
        data = {
            "overall": {"passed": report.overall_score()[0], "total": report.overall_score()[1]},
            "by_category": {
                cat: {"passed": p, "total": t}
                for cat, (p, t) in report.score_by_category().items()
            },
            "failures": [
                {"query": r.query, "category": r.category, "detail": r.detail}
                for r in report.failed()
            ],
        }
        Path(args.output).write_text(json.dumps(data, indent=2))
        console.print(f"\n[dim]Report saved to {args.output}[/dim]")

    _, total = report.overall_score()
    sys.exit(0 if not report.failed() else 1)


if __name__ == "__main__":
    main()
