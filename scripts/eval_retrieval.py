#!/usr/bin/env python3
"""
Retrieval quality evaluation script.

Runs the golden Q&A dataset against a live RAGSystem and scores each query by:
  - keyword_recall: fraction of expected keywords found in the response
  - latency_ms: end-to-end query time
  - intent_match: whether the classified intent matches the expected intent

Logs all results to MLflow experiment 'project-explorer-eval'.
Exits non-zero if mean keyword_recall falls below --min-recall threshold.

Usage:
    uv run python scripts/eval_retrieval.py --project <slug>
    uv run python scripts/eval_retrieval.py --project <slug> --min-recall 0.6
    uv run python scripts/eval_retrieval.py --project <slug> --no-mlflow
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _keyword_recall(response: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found (case-insensitive) in the response."""
    if not expected_keywords:
        return 1.0
    r = response.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in r)
    return hits / len(expected_keywords)


def run_eval(
    project_slug: str | None = None,
    golden_path: Path = Path(__file__).parent.parent / "tests" / "fixtures" / "golden_qa.json",
    min_recall: float = 0.70,
    use_mlflow: bool = True,
) -> int:
    from explorer.rag_system import RAGSystem
    from explorer.query_processor import QueryProcessor

    golden = json.loads(golden_path.read_text())
    rag = RAGSystem()
    processor = QueryProcessor()

    results = []
    print(f"\nRunning {len(golden)} eval queries" + (f" scoped to '{project_slug}'" if project_slug else "") + "...\n")

    for i, entry in enumerate(golden, 1):
        query = entry["query"]
        expected_intent = entry.get("intent", "general")
        expected_keywords = entry.get("expected_keywords", [])
        notes = entry.get("notes", "")

        t0 = time.monotonic()
        try:
            response = rag.query(query, project_slug=project_slug)
        except Exception as exc:
            response = f"ERROR: {exc}"
        latency_ms = int((time.monotonic() - t0) * 1000)

        actual_intent = processor.classify(query).value
        intent_match = actual_intent == expected_intent
        recall = _keyword_recall(response, expected_keywords)

        results.append({
            "query": query,
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_match": intent_match,
            "keyword_recall": recall,
            "latency_ms": latency_ms,
            "notes": notes,
        })

        status = "✓" if recall >= 0.5 else "✗"
        print(
            f"  {i:2d}. [{status}] {query[:55]:<55} "
            f"recall={recall:.0%}  {latency_ms}ms  intent={actual_intent}"
        )

    mean_recall = sum(r["keyword_recall"] for r in results) / len(results)
    mean_latency = sum(r["latency_ms"] for r in results) / len(results)
    intent_accuracy = sum(1 for r in results if r["intent_match"]) / len(results)

    print(f"\n{'='*70}")
    print(f"  Queries:          {len(results)}")
    print(f"  Mean recall:      {mean_recall:.1%}  (threshold: {min_recall:.1%})")
    print(f"  Intent accuracy:  {intent_accuracy:.1%}")
    print(f"  Mean latency:     {mean_latency:.0f}ms")
    print(f"{'='*70}\n")

    if use_mlflow:
        _log_to_mlflow(results, mean_recall, mean_latency, intent_accuracy, project_slug)

    if mean_recall < min_recall:
        print(f"FAIL: mean keyword recall {mean_recall:.1%} < threshold {min_recall:.1%}")
        return 1
    print("PASS")
    return 0


def _log_to_mlflow(
    results: list[dict],
    mean_recall: float,
    mean_latency: float,
    intent_accuracy: float,
    project_slug: str | None,
) -> None:
    try:
        import mlflow
        from explorer.config import get_config
        cfg = get_config().observability.mlflow
        mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment("project-explorer-eval")
        with mlflow.start_run(run_name=f"eval-{project_slug or 'all'}"):
            mlflow.log_params({
                "project_slug": project_slug or "all",
                "num_queries": len(results),
            })
            mlflow.log_metrics({
                "mean_keyword_recall": mean_recall,
                "mean_latency_ms": mean_latency,
                "intent_accuracy": intent_accuracy,
            })
            for i, r in enumerate(results):
                mlflow.log_metric("keyword_recall", r["keyword_recall"], step=i)
                mlflow.log_metric("latency_ms", r["latency_ms"], step=i)
        print(f"MLflow results logged to {cfg.tracking_uri} (experiment: project-explorer-eval)")
    except Exception as exc:
        print(f"MLflow logging skipped: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate retrieval quality against golden Q&A")
    parser.add_argument("--project", "-p", default=None, help="Project slug to scope queries to")
    parser.add_argument("--min-recall", type=float, default=0.70, help="Minimum acceptable mean recall (0-1)")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging")
    parser.add_argument("--golden", default=None, help="Path to golden Q&A JSON file")
    args = parser.parse_args()

    golden_path = Path(args.golden) if args.golden else Path(__file__).parent.parent / "tests" / "fixtures" / "golden_qa.json"
    exit_code = run_eval(
        project_slug=args.project,
        golden_path=golden_path,
        min_recall=args.min_recall,
        use_mlflow=not args.no_mlflow,
    )
    sys.exit(exit_code)
