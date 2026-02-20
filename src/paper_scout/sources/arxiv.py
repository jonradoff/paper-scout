"""arXiv API client. Fetches recent papers from configured categories."""

import time
from datetime import datetime, timedelta, timezone

import arxiv
import structlog

from paper_scout.config import settings
from paper_scout.models import Author, Paper

log = structlog.get_logger()

# arXiv asks for no more than 1 request per 3 seconds
ARXIV_DELAY = 3.0


def fetch_recent_papers() -> list[Paper]:
    """Fetch papers from arXiv published within the fetch window."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.fetch_window_hours)
    all_papers: list[Paper] = []

    for category in settings.arxiv_categories:
        try:
            papers = _fetch_category(category, cutoff)
            all_papers.extend(papers)
            log.info("arxiv_category_fetched", category=category, count=len(papers))
        except Exception as e:
            log.error("arxiv_category_error", category=category, error=str(e))
        time.sleep(ARXIV_DELAY)

    # Dedupe within the arXiv batch by arxiv_id
    seen: set[str] = set()
    deduped: list[Paper] = []
    for p in all_papers:
        if p.arxiv_id and p.arxiv_id not in seen:
            seen.add(p.arxiv_id)
            deduped.append(p)

    log.info("arxiv_fetch_complete", total=len(deduped), raw=len(all_papers))
    return deduped


def _fetch_category(category: str, cutoff: datetime) -> list[Paper]:
    """Fetch papers from a single arXiv category."""
    client = arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY,
        num_retries=3,
    )
    search = arxiv.Search(
        query=f"cat:{category}",
        max_results=settings.arxiv_max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    for result in client.results(search):
        published = result.published.replace(tzinfo=timezone.utc) if result.published.tzinfo is None else result.published
        if published < cutoff:
            break

        arxiv_id = result.entry_id.split("/abs/")[-1]
        # Strip version suffix for consistent IDs
        if "v" in arxiv_id:
            arxiv_id = arxiv_id.rsplit("v", 1)[0]

        authors = [
            Author(name=a.name)
            for a in result.authors
        ]

        paper = Paper(
            arxiv_id=arxiv_id,
            title=result.title.strip().replace("\n", " "),
            abstract=result.summary.strip().replace("\n", " "),
            authors=authors,
            categories=[category] + [c for c in (result.categories or []) if c != category],
            published=published,
            pdf_url=result.pdf_url,
            arxiv_url=result.entry_id,
            source="arxiv",
        )
        papers.append(paper)

    return papers
