#!/usr/bin/env python3
"""One-shot pipeline execution for testing. Runs a single pipeline pass and prints results."""

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import structlog

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
)

from paper_scout.db import ensure_indexes, candidates_col
from paper_scout.pipeline import run_pipeline

log = structlog.get_logger()


def main():
    log.info("run_once_starting")

    ensure_indexes()
    stats = run_pipeline()

    print("\n" + "=" * 60)
    print("PIPELINE RUN COMPLETE")
    print("=" * 60)
    print(f"  Duration:    {(stats.finished_at - stats.started_at).total_seconds():.1f}s")
    print(f"  Fetched:     {stats.papers_fetched}")
    print(f"  New papers:  {stats.new_papers}")
    print(f"  Duplicates:  {stats.duplicates_skipped}")
    print(f"  Candidates:  {stats.candidates_found}")
    if stats.errors:
        print(f"  Errors:      {len(stats.errors)}")
        for err in stats.errors:
            print(f"    - {err}")
    print("=" * 60)

    # Print top candidates
    if stats.candidates_found > 0:
        print("\nTOP CANDIDATES:")
        print("-" * 60)
        candidates = list(
            candidates_col()
            .find({"status": "new"})
            .sort("scores.composite", -1)
            .limit(10)
        )
        for i, c in enumerate(candidates, 1):
            scores = c.get("scores", {})
            print(f"\n{i}. [{scores.get('composite', 0):.3f}] {c['title']}")
            print(f"   Embedding: {scores.get('embedding_similarity', 0):.3f}  "
                  f"Keyword: {scores.get('keyword_match', 0):.3f}  "
                  f"Citation: {scores.get('citation_signal', 0):.3f}  "
                  f"Community: {scores.get('community_signal', 0):.3f}  "
                  f"Recency: {scores.get('recency_bonus', 0):.3f}")
            if c.get("arxiv_url"):
                print(f"   {c['arxiv_url']}")
            authors = c.get("authors", [])
            if authors:
                names = ", ".join(a.get("name", "") for a in authors[:5])
                if len(authors) > 5:
                    names += f" (+{len(authors) - 5} more)"
                print(f"   Authors: {names}")
        print()


if __name__ == "__main__":
    main()
